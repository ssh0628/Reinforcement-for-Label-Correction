from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader

from rl.actor.actor import PolicyActor, load_policy_actor
from rl.reward.global_knn_cache import load_global_knn_cache
from rl.state import LabelState
from rl.trainer import (
    TRAINER_CHECKPOINT_VERSION,
    MutableIndexSampler,
    build_actor_policy_graph,
    build_rl_loader,
    build_rl_training_signature,
)
from setting.config import CFG, Config
from setting.dataset import NPYPathDataset, dataset_manifest_fingerprint
from setup.warmup import resolve_device, seed_everything


CLEANING_ARTIFACT_VERSION = 2
REQUIRED_POLICY_CHECKPOINT_KEYS = frozenset(
    {
        "version",
        "epoch",
        "actor",
        "class_names",
        "model_name",
        "global_knn_cache",
        "global_knn_provenance_sha256",
        "training_signature",
    }
)


@dataclass(frozen=True, slots=True)
class CleaningStepMetrics:
    step: int
    mean_correction_probability: float
    action_rate: float
    cumulative_changed_rate: float


@dataclass(frozen=True, slots=True)
class CleaningResult:
    final_state: LabelState
    history: tuple[CleaningStepMetrics, ...]
    artifact_path: Path
    corrected_labels_path: Path


def validate_cleaning_destination(cfg: Config) -> None:
    output_paths = (
        cfg.cleaning.output_dir / cfg.cleaning.artifact_filename,
        cfg.cleaning.output_dir / cfg.cleaning.corrected_labels_filename,
    )
    existing = [path for path in output_paths if path.exists()]
    if existing and not cfg.cleaning.overwrite:
        raise FileExistsError(
            "Cleaning output already exists: "
            f"{existing}. Set cleaning.overwrite=True to replace it."
        )


def load_cleaning_checkpoint(
    cfg: Config,
    global_cache_path: Path,
    global_cache_provenance_sha256: str,
) -> dict[str, object]:
    checkpoint_path = cfg.cleaning.checkpoint_path
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"RL cleaning checkpoint not found: {checkpoint_path}"
        )
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )
    if not isinstance(checkpoint, dict):
        raise TypeError("RL cleaning checkpoint must contain a dictionary.")
    missing = REQUIRED_POLICY_CHECKPOINT_KEYS.difference(checkpoint)
    if missing:
        raise ValueError(
            f"RL cleaning checkpoint is missing fields: {sorted(missing)}."
        )
    if checkpoint["version"] != TRAINER_CHECKPOINT_VERSION:
        raise ValueError(
            "Unsupported RL cleaning checkpoint version: "
            f"{checkpoint['version']} != {TRAINER_CHECKPOINT_VERSION}."
        )
    if checkpoint["epoch"] != cfg.rl_train.epochs:
        raise ValueError(
            "Final cleaning requires a checkpoint from the last RL epoch."
        )
    if tuple(checkpoint["class_names"]) != cfg.data.class_names:
        raise ValueError("RL cleaning checkpoint class names do not match config.")
    if checkpoint["model_name"] != cfg.model.name:
        raise ValueError("RL cleaning checkpoint model does not match config.")

    checkpoint_cache_path = Path(str(checkpoint["global_knn_cache"]))
    if checkpoint_cache_path.resolve() != global_cache_path.resolve():
        raise ValueError(
            "RL cleaning checkpoint global KNN cache does not match config."
        )
    if (
        checkpoint["global_knn_provenance_sha256"]
        != global_cache_provenance_sha256
    ):
        raise ValueError(
            "RL cleaning checkpoint was trained with a different Global KNN cache."
        )
    saved_signature = checkpoint["training_signature"]
    expected_signature = build_rl_training_signature(cfg)
    if not isinstance(saved_signature, Mapping):
        raise TypeError("RL checkpoint training signature must be a mapping.")
    mismatches = sorted(
        key
        for key in set(saved_signature) | set(expected_signature)
        if saved_signature.get(key) != expected_signature.get(key)
    )
    if mismatches:
        raise ValueError(
            "Cleaning settings differ from the trained policy: "
            f"{mismatches}."
        )
    actor_state = checkpoint["actor"]
    if not isinstance(actor_state, Mapping):
        raise TypeError("RL cleaning checkpoint actor state must be a mapping.")
    return checkpoint


class LabelCleaner:
    def __init__(
        self,
        *,
        actor: PolicyActor,
        dataset: NPYPathDataset,
        sampler: MutableIndexSampler,
        loader: DataLoader,
        checkpoint_path: Path,
        global_cache_path: Path,
        global_cache_provenance_sha256: str,
        device: torch.device,
        cfg: Config,
    ) -> None:
        cfg.validate()
        if loader.dataset is not dataset or loader.sampler is not sampler:
            raise ValueError(
                "Cleaning loader must use the supplied dataset and sampler."
            )
        self.actor = actor
        self.dataset = dataset
        self.sampler = sampler
        self.loader = loader
        self.checkpoint_path = checkpoint_path
        self.global_cache_path = global_cache_path
        self.global_cache_provenance_sha256 = global_cache_provenance_sha256
        self.device = device
        self.cfg = cfg

        self.actor.to(device).eval()
        for parameter in self.actor.parameters():
            parameter.requires_grad = False

    def _save_outputs(
        self,
        state: LabelState,
        history: list[CleaningStepMetrics],
        last_actions: Tensor,
        last_probabilities: Tensor,
    ) -> tuple[Path, Path]:
        output_dir = self.cfg.cleaning.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        artifact_path = output_dir / self.cfg.cleaning.artifact_filename
        labels_path = output_dir / self.cfg.cleaning.corrected_labels_filename
        artifact_temp = output_dir / f".{artifact_path.name}.tmp"
        labels_temp = output_dir / f".{labels_path.name}.tmp"

        hard_labels = state.hard_labels.detach().cpu().numpy().astype(
            np.int64,
            copy=False,
        )
        artifact = {
            "version": CLEANING_ARTIFACT_VERSION,
            "split": self.cfg.global_knn.split,
            "trajectory_length": self.cfg.cleaning.trajectory_length,
            "action_mode": "stochastic",
            "sample_count": state.sample_count,
            "class_names": self.cfg.data.class_names,
            "model_name": self.cfg.model.name,
            "policy_checkpoint": str(self.checkpoint_path),
            "global_knn_cache": str(self.global_cache_path),
            "global_knn_provenance_sha256": (
                self.global_cache_provenance_sha256
            ),
            "dataset_manifest_sha256": dataset_manifest_fingerprint(
                self.dataset.paths
            ),
            "label_state": state.state_dict(),
            "last_actions": last_actions.detach().cpu().clone(),
            "last_correction_probabilities": (
                last_probabilities.detach().cpu().clone()
            ),
            "history": [asdict(item) for item in history],
            "corrected_labels_file": labels_path.name,
        }
        try:
            with labels_temp.open("wb") as file:
                np.save(file, hard_labels, allow_pickle=False)
            torch.save(artifact, artifact_temp)
            labels_temp.replace(labels_path)
            artifact_temp.replace(artifact_path)
        finally:
            labels_temp.unlink(missing_ok=True)
            artifact_temp.unlink(missing_ok=True)
        return artifact_path, labels_path

    def run(self) -> CleaningResult:
        state = LabelState.from_noisy_labels(
            self.dataset.targets,
            self.cfg.num_classes,
            device=self.device,
        )
        embeddings, policy_neighbors = build_actor_policy_graph(
            self.actor,
            self.sampler,
            self.loader,
            len(self.dataset),
            self.device,
            self.cfg,
        )
        history: list[CleaningStepMetrics] = []
        last_actions: Tensor | None = None
        last_probabilities: Tensor | None = None

        for step in range(1, self.cfg.cleaning.trajectory_length + 1):
            correction = self.actor.policy.correct_all(
                embeddings,
                state.current_labels,
                policy_neighbors,
            )
            state = state.transition(correction.corrected_labels)
            last_actions = correction.actions
            last_probabilities = correction.correction_probabilities
            metrics = CleaningStepMetrics(
                step=step,
                mean_correction_probability=float(
                    correction.correction_probabilities.mean()
                ),
                action_rate=float(correction.actions.float().mean()),
                cumulative_changed_rate=state.correction_rate,
            )
            history.append(metrics)
            print(
                f"[CLEAN] step={step}/{self.cfg.cleaning.trajectory_length} "
                f"mean_p={metrics.mean_correction_probability:.6f} "
                f"action={metrics.action_rate:.4f} "
                f"changed={metrics.cumulative_changed_rate:.4f}"
            )

        if last_actions is None or last_probabilities is None:
            raise RuntimeError("Cleaning completed without a correction step.")
        del embeddings, policy_neighbors
        artifact_path, labels_path = self._save_outputs(
            state,
            history,
            last_actions,
            last_probabilities,
        )
        return CleaningResult(
            final_state=state,
            history=tuple(history),
            artifact_path=artifact_path,
            corrected_labels_path=labels_path,
        )


def build_label_cleaner(cfg: Config) -> LabelCleaner:
    cfg.validate()
    validate_cleaning_destination(cfg)
    seed_everything(cfg.runtime.seed)
    device = resolve_device(cfg)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = cfg.runtime.cudnn_benchmark

    global_cache = load_global_knn_cache(cfg)
    global_cache_path = global_cache.cache_path
    global_cache_provenance_sha256 = global_cache.provenance_sha256
    cache_sample_count = global_cache.sample_count
    cache_labels = global_cache.labels
    del global_cache

    checkpoint = load_cleaning_checkpoint(
        cfg,
        global_cache_path,
        global_cache_provenance_sha256,
    )
    actor = load_policy_actor(cfg, device)
    actor.load_state_dict(checkpoint["actor"], strict=True)
    del checkpoint
    dataset, sampler, loader = build_rl_loader(actor, cfg, device)
    if len(dataset) != cache_sample_count or not torch.equal(
        dataset.targets,
        cache_labels,
    ):
        raise ValueError("Cleaning dataset does not match the global KNN cache.")
    del cache_labels
    return LabelCleaner(
        actor=actor,
        dataset=dataset,
        sampler=sampler,
        loader=loader,
        checkpoint_path=cfg.cleaning.checkpoint_path,
        global_cache_path=global_cache_path,
        global_cache_provenance_sha256=global_cache_provenance_sha256,
        device=device,
        cfg=cfg,
    )


def main(cfg: Config = CFG) -> CleaningResult:
    cleaner = build_label_cleaner(cfg)
    print(f"device={cleaner.device}")
    print(f"policy_checkpoint={cleaner.checkpoint_path}")
    print(f"samples={len(cleaner.dataset)}")
    result = cleaner.run()
    print(f"[OK] Cleaning artifact saved: {result.artifact_path}")
    print(f"[OK] Corrected labels saved: {result.corrected_labels_path}")
    return result


if __name__ == "__main__":
    main()
