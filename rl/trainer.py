from __future__ import annotations

import math
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import Tensor
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader, Sampler

from rl.actor.actor import PolicyActor, load_policy_actor
from rl.actor.policy_knn import build_policy_knn
from rl.critic.critic import StateActionCritic, build_critic, build_critic_optimizer, sarsa_td_loss
from rl.reward.global_knn_cache import GlobalKNNCache, load_global_knn_cache
from rl.reward.reward import RLNLCReward, RewardOutput
from rl.state import LabelState
from setting.config import Config
from setting.dataset import NPYPathDataset, build_transforms
from setup.global_knn import extract_embeddings
from setup.warmup import loader_worker_options, resolve_device, seed_everything, worker_init_fn


TRAINER_CHECKPOINT_VERSION = 5
REQUIRED_CHECKPOINT_KEYS = frozenset(
    {
        "version",
        "epoch",
        "actor",
        "critic",
        "actor_optimizer",
        "critic_optimizer",
        "actor_scheduler",
        "critic_scheduler",
        "amp_scaler",
        "label_state",
        "history",
        "class_names",
        "model_name",
        "global_knn_cache",
        "global_knn_provenance_sha256",
        "training_signature",
        "cpu_rng_state",
        "cuda_rng_states",
    }
)


class MutableIndexSampler(Sampler[int]):
    """Reuse one persistent-worker loader for full scans and small image banks."""

    def __init__(self, sample_count: int) -> None:
        if sample_count <= 0:
            raise ValueError("sample_count must be positive.")
        self.sample_count = sample_count
        self._indices: range | tuple[int, ...] = range(sample_count)

    def set_sequential(self) -> None:
        self._indices = range(self.sample_count)

    def set_indices(self, indices: Tensor | Sequence[int]) -> None:
        if isinstance(indices, Tensor):
            if indices.ndim != 1:
                raise ValueError("Sampler indices must be one-dimensional.")
            values = tuple(int(index) for index in indices.cpu().tolist())
        else:
            values = tuple(int(index) for index in indices)
        if not values:
            raise ValueError("Sampler indices must not be empty.")
        if min(values) < 0 or max(values) >= self.sample_count:
            raise ValueError("Sampler indices contain an out-of-range value.")
        if len(set(values)) != len(values):
            raise ValueError("Sampler indices must be unique.")
        self._indices = values

    def __iter__(self) -> Iterator[int]:
        return iter(self._indices)

    def __len__(self) -> int:
        return len(self._indices)


@dataclass(frozen=True, slots=True)
class RLStepMetrics:
    epoch: int
    trajectory_step: int
    reward: float
    label_consistency: float
    noisy_label_alignment: float
    q_value: float
    actor_loss: float
    critic_loss: float | None
    action_rate: float
    correction_rate: float


@dataclass(frozen=True, slots=True)
class RLTrainingResult:
    final_state: LabelState
    history: tuple[RLStepMetrics, ...]
    checkpoint_path: Path


def build_rl_loader(
    actor: PolicyActor, cfg: Config, device: torch.device
) -> tuple[NPYPathDataset, MutableIndexSampler, DataLoader]:
    _, eval_transform = build_transforms(actor.feature_extractor, cfg.data)
    dataset = NPYPathDataset(cfg.data, cfg.global_knn.split, transform=eval_transform)
    sampler = MutableIndexSampler(len(dataset))
    loader_generator = torch.Generator().manual_seed(cfg.runtime.seed)
    loader = DataLoader(
        dataset,
        batch_size=cfg.loader.rl_feature_batch_size,
        sampler=sampler,
        num_workers=cfg.loader.num_workers,
        pin_memory=cfg.loader.pin_memory and device.type == "cuda",
        drop_last=False,
        worker_init_fn=worker_init_fn,
        generator=loader_generator,
        **loader_worker_options(cfg),
    )
    return dataset, sampler, loader


def build_actor_policy_graph(
    actor: PolicyActor,
    sampler: MutableIndexSampler,
    loader: DataLoader,
    sample_count: int,
    device: torch.device,
    cfg: Config,
) -> tuple[Tensor, Tensor]:
    """Extract the current actor embeddings and their exact KNN graph."""
    actor.eval()
    sampler.set_sequential()
    embeddings = extract_embeddings(actor.feature_extractor, loader, device, cfg, progress=False).to(device)
    if embeddings.size(0) != sample_count:
        raise RuntimeError("Policy embedding count does not match the dataset.")
    return embeddings, build_policy_knn(embeddings, cfg)


def build_rl_scheduler(optimizer: Optimizer, cfg: Config) -> MultiStepLR:
    if cfg.rl_train.scheduler_name != "step_halfway":
        raise ValueError("The RL scheduler must be step_halfway.")
    milestone = max(1, math.ceil(cfg.rl_train.epochs * cfg.rl_train.lr_decay_fraction))
    return MultiStepLR(optimizer, milestones=[milestone], gamma=cfg.rl_train.lr_decay_factor)


def build_rl_training_signature(cfg: Config) -> dict[str, object]:
    """Settings that must stay fixed when an epoch-boundary run is resumed."""
    train = cfg.rl_train
    return {
        "model_name": cfg.model.name,
        "class_names": cfg.data.class_names,
        "image_size": cfg.data.image_size,
        "letterbox_fill": cfg.data.letterbox_fill,
        "rl_feature_batch_size": cfg.loader.rl_feature_batch_size,
        "global_k": cfg.global_knn.k,
        "global_query_chunk_size": cfg.global_knn.query_chunk_size,
        "global_reference_chunk_size": cfg.global_knn.reference_chunk_size,
        "policy_temperature": cfg.policy.temperature,
        "policy_correction_chunk_size": cfg.policy.correction_chunk_size,
        "reward_nla_weight": cfg.reward.nla_weight,
        "use_amp": cfg.runtime.use_amp,
        "use_channels_last": cfg.runtime.use_channels_last,
        "epochs": train.epochs,
        "trajectory_length": train.trajectory_length,
        "discount_factor": train.discount_factor,
        "critic_num_bins": train.critic_num_bins,
        "initial_state_randomization_rate": (train.initial_state_randomization_rate),
        "actor_optimizer_name": train.actor_optimizer_name,
        "actor_lr": train.actor_lr,
        "actor_weight_decay": train.actor_weight_decay,
        "actor_adamw_betas": train.actor_adamw_betas,
        "actor_adamw_eps": train.actor_adamw_eps,
        "critic_optimizer_name": train.critic_optimizer_name,
        "critic_lr": train.critic_lr,
        "critic_momentum": train.critic_momentum,
        "critic_weight_decay": train.critic_weight_decay,
        "scheduler_name": train.scheduler_name,
        "lr_decay_fraction": train.lr_decay_fraction,
        "lr_decay_factor": train.lr_decay_factor,
        "policy_update_mode": train.policy_update_mode,
        "policy_update_subset_size": (
            train.policy_update_subset_size if train.policy_update_mode == "subset" else None
        ),
        "policy_update_batch_size": train.policy_update_batch_size,
    }


def validate_rl_destination(cfg: Config) -> None:
    """Prevent an accidental overwrite before loading the expensive model."""
    output_path = cfg.rl_train.output_dir / cfg.rl_train.checkpoint_filename
    resume_path = cfg.rl_train.resume_checkpoint_path
    if resume_path is not None:
        if not resume_path.is_file():
            raise FileNotFoundError(f"RL resume checkpoint not found: {resume_path}")
        different_existing_output = output_path.is_file() and output_path.resolve() != resume_path.resolve()
        if different_existing_output and not cfg.rl_train.overwrite:
            raise FileExistsError(
                f"RL output checkpoint already exists: {output_path}. "
                "Set rl_train.overwrite=True to replace it."
            )
        return
    if output_path.is_file() and not cfg.rl_train.overwrite:
        raise FileExistsError(
            f"RL output checkpoint already exists: {output_path}. "
            "Set rl_train.resume_checkpoint_path to resume it or "
            "rl_train.overwrite=True to start over."
        )


class RLTrainer:
    def __init__(
        self,
        *,
        actor: PolicyActor,
        critic: StateActionCritic,
        reward: RLNLCReward,
        cache: GlobalKNNCache,
        dataset: NPYPathDataset,
        sampler: MutableIndexSampler,
        loader: DataLoader,
        actor_optimizer: Optimizer,
        critic_optimizer: Optimizer,
        actor_scheduler: MultiStepLR,
        critic_scheduler: MultiStepLR,
        device: torch.device,
        cfg: Config,
    ) -> None:
        cfg.validate()
        if len(dataset) != cache.sample_count:
            raise ValueError("RL dataset and global KNN cache must have equal N.")
        if not torch.equal(dataset.targets, cache.labels.cpu()):
            raise ValueError("RL dataset labels do not match the cache order.")
        if cache.k != cfg.global_knn.k:
            raise ValueError("RL cache k does not match the config.")
        if len(dataset) <= cache.k:
            raise ValueError("RL training requires more than k samples.")
        if loader.dataset is not dataset or loader.sampler is not sampler:
            raise ValueError("RL loader must use the supplied dataset and sampler.")

        self.actor = actor
        self.critic = critic
        self.reward = reward
        self.cache = cache
        self.dataset = dataset
        self.sampler = sampler
        self.loader = loader
        self.actor_optimizer = actor_optimizer
        self.critic_optimizer = critic_optimizer
        self.actor_scheduler = actor_scheduler
        self.critic_scheduler = critic_scheduler
        self.device = device
        self.cfg = cfg
        self.amp_enabled = cfg.runtime.use_amp and device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp_enabled)
        self.completed_epochs = 0
        self._resume_history: list[RLStepMetrics] = []
        self._resume_state: LabelState | None = None
        self._resume_checkpoint_path: Path | None = None

        self.actor.to(device)
        self.critic.to(device)
        self.reward.to(device)
        try:
            self.fixed_embeddings = cache.fixed_embeddings.to(device=device, dtype=torch.float32)
        except RuntimeError as exc:
            raise RuntimeError("Could not cache fixed reward embeddings on the RL device.") from exc

    def _decode_history(self, value: object, completed_epochs: int) -> list[RLStepMetrics]:
        if not isinstance(value, list):
            raise TypeError("RL checkpoint history must be a list.")
        expected_count = completed_epochs * self.cfg.rl_train.trajectory_length
        if len(value) != expected_count:
            raise ValueError("RL checkpoint history length does not match its epoch.")

        history: list[RLStepMetrics] = []
        for position, raw_metrics in enumerate(value):
            if not isinstance(raw_metrics, Mapping):
                raise TypeError("Each RL history item must be a mapping.")
            try:
                metrics = RLStepMetrics(**dict(raw_metrics))
            except TypeError as exc:
                raise ValueError("Invalid RL history item fields.") from exc
            expected_epoch = position // self.cfg.rl_train.trajectory_length + 1
            expected_step = position % self.cfg.rl_train.trajectory_length + 1
            if metrics.epoch != expected_epoch or metrics.trajectory_step != expected_step:
                raise ValueError("RL checkpoint history order is invalid.")
            history.append(metrics)
        return history

    def load_checkpoint(self, checkpoint_path: Path) -> int:
        """Restore a checkpoint saved after a complete outer RL epoch."""
        if not checkpoint_path.is_absolute():
            raise ValueError("RL resume checkpoint path must be absolute.")
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"RL resume checkpoint not found: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if not isinstance(checkpoint, dict):
            raise TypeError("RL checkpoint must contain a dictionary.")
        missing = REQUIRED_CHECKPOINT_KEYS.difference(checkpoint)
        if missing:
            raise ValueError(f"RL checkpoint is missing fields: {sorted(missing)}.")
        if checkpoint["version"] != TRAINER_CHECKPOINT_VERSION:
            raise ValueError(
                f"Unsupported RL checkpoint version: {checkpoint['version']} != {TRAINER_CHECKPOINT_VERSION}."
            )
        if tuple(checkpoint["class_names"]) != self.cfg.data.class_names:
            raise ValueError("RL checkpoint class names do not match config.")
        if checkpoint["model_name"] != self.cfg.model.name:
            raise ValueError("RL checkpoint model does not match config.")

        saved_cache_path = Path(str(checkpoint["global_knn_cache"]))
        if saved_cache_path.resolve() != self.cache.cache_path.resolve():
            raise ValueError("RL checkpoint global KNN cache does not match the loaded cache.")
        if checkpoint["global_knn_provenance_sha256"] != self.cache.provenance_sha256:
            raise ValueError("RL checkpoint was trained with a different Global KNN cache.")
        saved_signature = checkpoint["training_signature"]
        expected_signature = build_rl_training_signature(self.cfg)
        if not isinstance(saved_signature, Mapping):
            raise TypeError("RL checkpoint training signature must be a mapping.")
        mismatches = sorted(
            key
            for key in set(saved_signature) | set(expected_signature)
            if saved_signature.get(key) != expected_signature.get(key)
        )
        if mismatches:
            raise ValueError(f"RL resume settings differ from the checkpoint: {mismatches}.")

        completed_epochs = checkpoint["epoch"]
        if (
            not isinstance(completed_epochs, int)
            or isinstance(completed_epochs, bool)
            or not 0 < completed_epochs <= self.cfg.rl_train.epochs
        ):
            raise ValueError("RL checkpoint epoch is invalid.")
        history = self._decode_history(checkpoint["history"], completed_epochs)
        label_state_value = checkpoint["label_state"]
        if not isinstance(label_state_value, Mapping):
            raise TypeError("RL checkpoint label_state must be a mapping.")
        label_state = LabelState.from_state_dict(label_state_value, device=self.device)
        if (
            label_state.sample_count != len(self.dataset)
            or label_state.num_classes != self.cfg.num_classes
            or label_state.step != self.cfg.rl_train.trajectory_length
            or not torch.equal(label_state.noisy_labels, self.cache.labels.to(self.device))
        ):
            raise ValueError("RL checkpoint label state is incompatible.")

        cpu_rng_state = checkpoint["cpu_rng_state"]
        cuda_rng_states = checkpoint["cuda_rng_states"]
        if (
            not isinstance(cpu_rng_state, Tensor)
            or cpu_rng_state.dtype != torch.uint8
            or cpu_rng_state.ndim != 1
        ):
            raise TypeError("RL checkpoint CPU RNG state is invalid.")
        if not isinstance(cuda_rng_states, list) or not all(
            isinstance(state, Tensor) and state.dtype == torch.uint8 and state.ndim == 1
            for state in cuda_rng_states
        ):
            raise TypeError("RL checkpoint CUDA RNG states are invalid.")
        if self.device.type == "cuda" and len(cuda_rng_states) != torch.cuda.device_count():
            raise ValueError("RL checkpoint CUDA RNG state count does not match visible GPUs.")

        self.actor.load_state_dict(checkpoint["actor"], strict=True)
        self.critic.load_state_dict(checkpoint["critic"], strict=True)
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
        self.actor_scheduler.load_state_dict(checkpoint["actor_scheduler"])
        self.critic_scheduler.load_state_dict(checkpoint["critic_scheduler"])
        self.scaler.load_state_dict(checkpoint["amp_scaler"])
        torch.set_rng_state(cpu_rng_state)
        if self.device.type == "cuda":
            torch.cuda.set_rng_state_all(cuda_rng_states)

        self.completed_epochs = completed_epochs
        self._resume_history = history
        self._resume_state = label_state
        self._resume_checkpoint_path = checkpoint_path
        return completed_epochs

    def _initial_state(self) -> LabelState:
        noisy_labels = self.cache.labels.to(self.device)
        state = LabelState.from_noisy_labels(noisy_labels, self.cfg.num_classes)
        rate = self.cfg.rl_train.initial_state_randomization_rate
        random_count = min(state.sample_count, max(1, round(state.sample_count * rate)))
        selected = torch.randperm(state.sample_count, device=self.device)[:random_count]
        randomized_hard_labels = state.noisy_labels.clone()
        original = randomized_hard_labels[selected]
        alternatives = torch.randint(self.cfg.num_classes - 1, (random_count,), device=self.device)
        alternatives += alternatives.ge(original)
        randomized_hard_labels[selected] = alternatives
        randomized_labels = torch.nn.functional.one_hot(
            randomized_hard_labels, num_classes=self.cfg.num_classes
        ).to(torch.float32)
        return LabelState(noisy_labels=state.noisy_labels, current_labels=randomized_labels, step=0)

    def _extract_policy_graph(self) -> tuple[Tensor, Tensor]:
        return build_actor_policy_graph(
            self.actor, self.sampler, self.loader, len(self.dataset), self.device, self.cfg
        )

    def _select_policy_samples(self) -> Tensor:
        dataset_size = len(self.dataset)
        if self.cfg.rl_train.policy_update_mode == "full":
            return torch.arange(dataset_size)
        sample_count = min(dataset_size, self.cfg.rl_train.policy_update_subset_size)
        if sample_count == dataset_size:
            return torch.arange(dataset_size)
        selected = torch.randperm(len(self.dataset))[:sample_count]
        return selected.sort().values

    def _load_query_images(self, requested_indices: Tensor) -> Tensor:
        if requested_indices.ndim != 1:
            raise ValueError("requested_indices must be [B].")
        requested_indices = requested_indices.to(dtype=torch.long, device="cpu")
        self.sampler.set_indices(requested_indices)

        image_chunks: list[Tensor] = []
        samples_seen = 0
        for images, _, sample_indices in self.loader:
            next_samples_seen = samples_seen + sample_indices.numel()
            if not torch.equal(sample_indices, requested_indices[samples_seen:next_samples_seen]):
                raise RuntimeError("Policy image loader changed the requested order.")
            image_chunks.append(images)
            samples_seen = next_samples_seen
        if not image_chunks:
            raise RuntimeError("The policy image loader returned no images.")
        if samples_seen != requested_indices.numel():
            raise RuntimeError("Policy image loader returned too few images.")
        if len(image_chunks) == 1:
            return image_chunks[0]
        return torch.cat(image_chunks)

    def _update_actor(
        self,
        state: LabelState,
        policy_embeddings: Tensor,
        policy_neighbor_indices: Tensor,
        actions: Tensor,
        q_value: Tensor,
    ) -> float:
        if policy_embeddings.ndim != 2 or policy_embeddings.size(0) != len(self.dataset):
            raise ValueError("policy_embeddings must be [N, D] for the full dataset.")
        if policy_embeddings.device != self.device:
            raise ValueError("policy_embeddings must already be on the actor device.")
        selected_cpu = self._select_policy_samples()
        selected = selected_cpu.to(self.device)
        selected_neighbors = policy_neighbor_indices[selected]
        image_bank = (
            self._load_query_images(selected_cpu)
            if self.cfg.rl_train.policy_update_mode == "subset"
            else None
        )

        self.actor.eval()
        self.actor_optimizer.zero_grad(set_to_none=True)
        selected_count = selected.numel()
        batch_size = self.cfg.rl_train.policy_update_batch_size
        total_loss = torch.zeros((), device=self.device)

        # The forward policy always sees the full cached KNN graph. Gradients
        # are collected for every selected query and its neighbors, but only
        # rows inside the chosen update set are propagated through the actor
        # backbone. In subset mode this strictly bounds backbone backward to
        # the subset while preserving neighbor-gradient contributions inside
        # that subset.
        embedding_leaf = policy_embeddings.detach().clone().requires_grad_(True)
        for start in range(0, selected_count, batch_size):
            end = min(start + batch_size, selected_count)
            batch_indices = selected[start:end]
            batch_neighbors = selected_neighbors[start:end]
            policy_step = self.actor.policy(
                embedding_leaf[batch_indices],
                embedding_leaf[batch_neighbors],
                state.current_labels[batch_indices],
                state.current_labels[batch_neighbors],
                actions=actions[batch_indices],
            )
            loss = -(q_value.detach() * policy_step.log_probabilities.sum() / selected_count)
            loss.backward()
            total_loss += loss.detach()

        if embedding_leaf.grad is None:
            raise RuntimeError("Policy loss produced no embedding gradient.")
        selected_embedding_gradient = embedding_leaf.grad[selected].detach().clone()
        del embedding_leaf

        for start in range(0, selected_count, batch_size):
            end = min(start + batch_size, selected_count)
            if image_bank is not None:
                images = image_bank[start:end]
            else:
                images = self._load_query_images(selected_cpu[start:end])
            images = images.to(self.device, non_blocking=self.device.type == "cuda")
            with torch.autocast(device_type=self.device.type, enabled=self.amp_enabled):
                encoded = self.actor.encode(images)
                surrogate = (encoded.float() * selected_embedding_gradient[start:end].float()).sum()
            self.scaler.scale(surrogate).backward()

        self.scaler.step(self.actor_optimizer)
        self.scaler.update()
        return float(total_loss)

    def _update_critic(self, encoding: Tensor, reward: Tensor, next_encoding: Tensor | None) -> float:
        self.critic_optimizer.zero_grad(set_to_none=True)
        current_q = self.critic.value_from_encoding(encoding)
        if next_encoding is None:
            next_q = torch.zeros_like(current_q)
        else:
            next_q = self.critic.value_from_encoding(next_encoding)
        td = sarsa_td_loss(
            current_q,
            reward,
            next_q,
            discount_factor=self.cfg.rl_train.discount_factor,
            terminal=next_encoding is None,
        )
        td.loss.backward()
        self.critic_optimizer.step()
        return float(td.loss.detach())

    def _save_checkpoint(self, epoch: int, state: LabelState, history: list[RLStepMetrics]) -> Path:
        output_dir = self.cfg.rl_train.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = output_dir / self.cfg.rl_train.checkpoint_filename
        temporary_path = output_dir / f".{checkpoint_path.name}.tmp"
        try:
            torch.save(
                {
                    "version": TRAINER_CHECKPOINT_VERSION,
                    "epoch": epoch,
                    "actor": self.actor.state_dict(),
                    "critic": self.critic.state_dict(),
                    "actor_optimizer": self.actor_optimizer.state_dict(),
                    "critic_optimizer": self.critic_optimizer.state_dict(),
                    "actor_scheduler": self.actor_scheduler.state_dict(),
                    "critic_scheduler": self.critic_scheduler.state_dict(),
                    "amp_scaler": self.scaler.state_dict(),
                    "label_state": state.state_dict(),
                    "history": [asdict(item) for item in history],
                    "class_names": self.cfg.data.class_names,
                    "model_name": self.cfg.model.name,
                    "global_knn_cache": str(self.cache.cache_path),
                    "global_knn_provenance_sha256": (self.cache.provenance_sha256),
                    "training_signature": build_rl_training_signature(self.cfg),
                    "cpu_rng_state": torch.get_rng_state(),
                    "cuda_rng_states": (torch.cuda.get_rng_state_all() if self.device.type == "cuda" else []),
                },
                temporary_path,
            )
            temporary_path.replace(checkpoint_path)
        finally:
            temporary_path.unlink(missing_ok=True)
        return checkpoint_path

    @staticmethod
    def _print_metrics(metrics: RLStepMetrics) -> None:
        critic_loss = "-" if metrics.critic_loss is None else f"{metrics.critic_loss:.6f}"
        print(
            f"[RL] epoch={metrics.epoch} step={metrics.trajectory_step} "
            f"reward={metrics.reward:.6f} q={metrics.q_value:.6f} "
            f"actor={metrics.actor_loss:.6f} critic={critic_loss} "
            f"action={metrics.action_rate:.4f} "
            f"changed={metrics.correction_rate:.4f}"
        )

    def fit(self) -> RLTrainingResult:
        history = list(self._resume_history)
        checkpoint_path = (
            self._resume_checkpoint_path
            or self.cfg.rl_train.output_dir / self.cfg.rl_train.checkpoint_filename
        )
        final_state = self._resume_state

        for epoch_index in range(self.completed_epochs, self.cfg.rl_train.epochs):
            epoch = epoch_index + 1
            state = self._initial_state()
            previous_encoding: Tensor | None = None
            previous_reward: Tensor | None = None

            for trajectory_index in range(self.cfg.rl_train.trajectory_length):
                policy_embeddings, policy_neighbors = self._extract_policy_graph()
                correction = self.actor.policy.correct_all(
                    policy_embeddings, state.current_labels, policy_neighbors
                )
                reward_output: RewardOutput = self.reward(
                    correction.corrected_labels,
                    correction.actions,
                    self.fixed_embeddings,
                    self.cache.neighbor_indices,
                    self.cache.neighbor_cosine_similarities,
                )
                encoding = self.critic.encode(reward_output.per_sample_consistency).detach()
                with torch.no_grad():
                    q_value = self.critic.value_from_encoding(encoding)
                actor_loss = self._update_actor(
                    state, policy_embeddings, policy_neighbors, correction.actions, q_value
                )
                del policy_embeddings, policy_neighbors

                critic_losses: list[float] = []
                if previous_encoding is not None and previous_reward is not None:
                    critic_losses.append(self._update_critic(previous_encoding, previous_reward, encoding))

                state = state.transition(correction.corrected_labels)
                is_terminal = trajectory_index + 1 == self.cfg.rl_train.trajectory_length
                if is_terminal:
                    critic_losses.append(self._update_critic(encoding, reward_output.total_reward, None))
                else:
                    previous_encoding = encoding
                    previous_reward = reward_output.total_reward.detach()

                metrics = RLStepMetrics(
                    epoch=epoch,
                    trajectory_step=trajectory_index + 1,
                    reward=float(reward_output.total_reward),
                    label_consistency=float(reward_output.label_consistency),
                    noisy_label_alignment=float(reward_output.noisy_label_alignment),
                    q_value=float(q_value.detach()),
                    actor_loss=actor_loss,
                    critic_loss=(sum(critic_losses) / len(critic_losses) if critic_losses else None),
                    action_rate=float(correction.actions.float().mean()),
                    correction_rate=state.correction_rate,
                )
                history.append(metrics)
                self._print_metrics(metrics)

            self.actor_scheduler.step()
            self.critic_scheduler.step()
            final_state = state
            if epoch % self.cfg.rl_train.checkpoint_interval == 0 or epoch == self.cfg.rl_train.epochs:
                checkpoint_path = self._save_checkpoint(epoch, state, history)
                self._resume_checkpoint_path = checkpoint_path
            self.completed_epochs = epoch
            self._resume_history = list(history)
            self._resume_state = state

        if final_state is None:
            raise RuntimeError("RL training completed without a label state.")
        return RLTrainingResult(
            final_state=final_state, history=tuple(history), checkpoint_path=checkpoint_path
        )


def build_rl_trainer(cfg: Config) -> RLTrainer:
    cfg.validate()
    validate_rl_destination(cfg)
    seed_everything(cfg.runtime.seed)
    device = resolve_device(cfg)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = cfg.runtime.cudnn_benchmark

    actor = load_policy_actor(cfg, device)
    critic = build_critic(cfg).to(device)
    reward = RLNLCReward(cfg).to(device)
    cache = load_global_knn_cache(cfg)
    dataset, sampler, loader = build_rl_loader(actor, cfg, device)

    if cfg.rl_train.actor_optimizer_name.lower() != "adamw":
        raise ValueError("The RL actor optimizer must be AdamW.")
    actor_optimizer = AdamW(
        actor.parameters(),
        lr=cfg.rl_train.actor_lr,
        weight_decay=cfg.rl_train.actor_weight_decay,
        betas=cfg.rl_train.actor_adamw_betas,
        eps=cfg.rl_train.actor_adamw_eps,
    )
    critic_optimizer = build_critic_optimizer(critic, cfg)
    trainer = RLTrainer(
        actor=actor,
        critic=critic,
        reward=reward,
        cache=cache,
        dataset=dataset,
        sampler=sampler,
        loader=loader,
        actor_optimizer=actor_optimizer,
        critic_optimizer=critic_optimizer,
        actor_scheduler=build_rl_scheduler(actor_optimizer, cfg),
        critic_scheduler=build_rl_scheduler(critic_optimizer, cfg),
        device=device,
        cfg=cfg,
    )
    resume_path = cfg.rl_train.resume_checkpoint_path
    if resume_path is not None:
        trainer.load_checkpoint(resume_path)
    return trainer
