from __future__ import annotations

import random
import time
from dataclasses import replace
from pathlib import Path
from typing import Callable, TypeVar

import numpy as np
import timm
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import MultiStepLR
from torchvision.datasets import MNIST

from rl.actor.policy import LabelCorrectionPolicy
from rl.critic.critic import (
    build_critic,
    build_critic_optimizer,
    sarsa_td_loss,
)
from rl.reward.reward import RLNLCReward
from setting.config import Config


# H100 speed benchmark matching the current RLNLC training configuration.
MNIST_ROOT = Path.home() / ".cache" / "rlnlc" / "mnist"
DOWNLOAD_MNIST = True
EXPECTED_SAMPLES = 60_000
DIGITS = tuple(range(10))
NUM_CLASSES = len(DIGITS)
NOISE_RATE = 0.40

MODEL_NAME = "convnextv2_tiny.fcmae_ft_in22k_in1k"
PRETRAINED = False
IMAGE_SIZE = 224
DROP_RATE = 0.1
DROP_PATH_RATE = 0.2

RL_EPOCHS = 1
TRAJECTORY_LENGTH = 10
FEATURE_BATCH_SIZE = 1_024
POLICY_UPDATE_SUBSET_SIZE = 10_000
POLICY_UPDATE_BATCH_SIZE = 512

K = 10
TEMPERATURE = 0.5

ACTOR_LR = 3e-5
ACTOR_WEIGHT_DECAY = 0.1
ACTOR_BETAS = (0.9, 0.999)
ACTOR_EPS = 1e-8
CRITIC_LR = 1e-2
CRITIC_MOMENTUM = 0.9
CRITIC_WEIGHT_DECAY = 5e-4
DISCOUNT_FACTOR = 0.9
NLA_WEIGHT = 0.5
LR_DECAY_FACTOR = 0.1

USE_AMP = True
USE_CHANNELS_LAST = True
CUDNN_BENCHMARK = True
SEED = 0

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
T = TypeVar("T")
Timings = dict[str, list[float]]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_h100_device() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "mnist_test_h100.py requires CUDA, but torch.cuda.is_available() "
            "is False. Check the NVIDIA driver and CUDA-enabled PyTorch build."
        )
    return torch.device("cuda", 0)


def synchronize(device: torch.device) -> None:
    torch.cuda.synchronize(device)


def pin_for_cuda(tensor: Tensor) -> Tensor:
    return tensor.pin_memory() if torch.cuda.is_available() else tensor


def measure(
    name: str,
    device: torch.device,
    timings: Timings,
    operation: Callable[[], T],
    *,
    step: int | None = None,
) -> T:
    label = name if step is None else f"step_{step:02d}/{name}"
    print(f"[RUN]  {label}")
    synchronize(device)
    started = time.perf_counter()
    result = operation()
    synchronize(device)
    elapsed = time.perf_counter() - started
    timings.setdefault(name, []).append(elapsed)
    print(f"[TIME] {label:<36} {elapsed:>10.3f} sec")
    return result


def load_full_mnist_train() -> tuple[Tensor, Tensor]:
    dataset = MNIST(
        root=MNIST_ROOT,
        train=True,
        download=DOWNLOAD_MNIST,
    )
    images = pin_for_cuda(dataset.data.contiguous())
    labels = pin_for_cuda(dataset.targets.to(torch.long).contiguous())
    if images.size(0) != EXPECTED_SAMPLES or labels.size(0) != EXPECTED_SAMPLES:
        raise RuntimeError(
            "Unexpected MNIST train size: "
            f"images={images.size(0)}, labels={labels.size(0)}."
        )
    if set(labels.tolist()) != set(DIGITS):
        raise RuntimeError("MNIST train must contain every digit from 0 through 9.")
    return images, labels


def class_counts(labels: Tensor) -> tuple[int, ...]:
    counts = torch.bincount(labels, minlength=NUM_CLASSES)
    return tuple(int(value) for value in counts.tolist())


def inject_stratified_symmetric_noise(
    clean_labels: Tensor,
) -> tuple[Tensor, Tensor]:
    """Corrupt a fixed fraction per class using the global seed."""
    if clean_labels.ndim != 1 or clean_labels.numel() == 0:
        raise ValueError("clean_labels must be a non-empty one-dimensional tensor.")
    if not 0.0 <= NOISE_RATE < 1.0:
        raise ValueError("NOISE_RATE must be in [0, 1).")

    generator = torch.Generator().manual_seed(SEED)
    noisy_labels = clean_labels.clone()
    noise_mask = torch.zeros_like(clean_labels, dtype=torch.bool)

    for digit in DIGITS:
        class_indices = clean_labels.eq(digit).nonzero(as_tuple=False).flatten()
        noise_count = round(class_indices.numel() * NOISE_RATE)
        if noise_count == 0:
            continue
        selected = class_indices[
            torch.randperm(class_indices.numel(), generator=generator)[:noise_count]
        ]
        alternatives = torch.randint(
            NUM_CLASSES - 1,
            (noise_count,),
            generator=generator,
        )
        original = clean_labels[selected]
        alternatives += alternatives.ge(original)
        noisy_labels[selected] = alternatives
        noise_mask[selected] = True

    if not torch.equal(noisy_labels.ne(clean_labels), noise_mask):
        raise RuntimeError("Noise mask and corrupted labels do not agree.")
    return pin_for_cuda(noisy_labels), pin_for_cuda(noise_mask)


def preprocess(
    images: Tensor,
    device: torch.device,
    mean: Tensor,
    std: Tensor,
) -> Tensor:
    images = images.to(device=device, dtype=torch.float32, non_blocking=True)
    images = images.unsqueeze(1).div_(255.0)
    images = F.interpolate(
        images,
        size=(IMAGE_SIZE, IMAGE_SIZE),
        mode="bicubic",
        align_corners=False,
        antialias=True,
    )
    images = images.expand(-1, 3, -1, -1).contiguous(
        memory_format=(
            torch.channels_last if USE_CHANNELS_LAST else torch.contiguous_format
        )
    )
    return (images - mean) / std


def encode(model: nn.Module, images: Tensor) -> Tensor:
    feature_map = model.forward_features(images)
    embeddings = model.forward_head(feature_map, pre_logits=True)
    if embeddings.ndim != 2:
        raise RuntimeError(f"Expected [B, D] embeddings, got {embeddings.shape}.")
    return embeddings


def warm_device_kernels(
    model: nn.Module,
    raw_images: Tensor,
    device: torch.device,
    mean: Tensor,
    std: Tensor,
) -> None:
    model.eval()
    model.zero_grad(set_to_none=True)
    inference_images = preprocess(
        raw_images[:FEATURE_BATCH_SIZE],
        device,
        mean,
        std,
    )
    with torch.inference_mode(), torch.autocast(
        device_type="cuda",
        enabled=USE_AMP,
    ):
        encode(model, inference_images)

    update_images = preprocess(
        raw_images[:POLICY_UPDATE_BATCH_SIZE],
        device,
        mean,
        std,
    )
    with torch.autocast(device_type="cuda", enabled=USE_AMP):
        encode(model, update_images).mean().backward()
    model.zero_grad(set_to_none=True)


@torch.inference_mode()
def extract_all_embeddings(
    model: nn.Module,
    raw_images: Tensor,
    device: torch.device,
    mean: Tensor,
    std: Tensor,
) -> Tensor:
    model.eval()
    sample_count = raw_images.size(0)
    output: Tensor | None = None
    for start in range(0, sample_count, FEATURE_BATCH_SIZE):
        end = min(start + FEATURE_BATCH_SIZE, sample_count)
        images = preprocess(
            raw_images[start:end],
            device,
            mean,
            std,
        )
        with torch.autocast(device_type="cuda", enabled=USE_AMP):
            embeddings = encode(model, images)
        if output is None:
            output = torch.empty(
                (sample_count, embeddings.size(1)),
                dtype=torch.float32,
                device=device,
            )
        output[start:end].copy_(embeddings)
    if output is None:
        raise RuntimeError("Embedding extraction received an empty dataset.")
    return output


@torch.inference_mode()
def build_neighbor_indices(embeddings: Tensor) -> Tensor:
    sample_count = embeddings.size(0)
    if embeddings.ndim != 2 or sample_count <= K:
        raise ValueError("Full KNN requires [N, D] embeddings with N > K.")
    features = embeddings.float()
    squared_norms = features.square().sum(dim=1)
    distances = torch.mm(features, features.transpose(0, 1))
    distances.mul_(-2.0)
    distances.add_(squared_norms.unsqueeze(1))
    distances.add_(squared_norms.unsqueeze(0))
    distances.clamp_min_(0.0)
    distances.fill_diagonal_(float("inf"))
    neighbor_indices = distances.topk(
        K,
        dim=1,
        largest=False,
        sorted=True,
    ).indices
    return neighbor_indices


@torch.inference_mode()
def build_global_graph(embeddings: Tensor) -> tuple[Tensor, Tensor]:
    neighbor_indices = build_neighbor_indices(embeddings)
    normalized = F.normalize(embeddings.float(), dim=1)
    neighbor_cosines = (
        normalized.unsqueeze(1) * normalized[neighbor_indices]
    ).sum(dim=2)
    return neighbor_indices, neighbor_cosines


def select_policy_subset(sample_count: int, step: int) -> Tensor:
    generator = torch.Generator().manual_seed(SEED + step)
    selected = torch.randperm(sample_count, generator=generator)[
        :POLICY_UPDATE_SUBSET_SIZE
    ]
    return selected.sort().values


def update_query_only_actor(
    model: nn.Module,
    policy: LabelCorrectionPolicy,
    optimizer: AdamW,
    scaler: torch.amp.GradScaler,
    raw_images: Tensor,
    selected_cpu: Tensor,
    cached_embeddings: Tensor,
    label_state: Tensor,
    neighbor_indices: Tensor,
    actions: Tensor,
    q_value: Tensor,
    device: torch.device,
    mean: Tensor,
    std: Tensor,
) -> float:
    """Update query features only; neighbor features stay fixed for this step."""
    model.eval()
    optimizer.zero_grad(set_to_none=True)
    selected = selected_cpu.to(device=device, non_blocking=True)
    selected_images = pin_for_cuda(raw_images[selected_cpu])
    selected_count = selected.numel()
    total_loss = torch.zeros((), device=device)
    total_batches = (
        selected_count + POLICY_UPDATE_BATCH_SIZE - 1
    ) // POLICY_UPDATE_BATCH_SIZE

    for batch_number, start in enumerate(
        range(0, selected_count, POLICY_UPDATE_BATCH_SIZE),
        start=1,
    ):
        end = min(start + POLICY_UPDATE_BATCH_SIZE, selected_count)
        batch_indices = selected[start:end]
        batch_neighbors = neighbor_indices[batch_indices]
        images = preprocess(
            selected_images[start:end],
            device,
            mean,
            std,
        )
        with torch.autocast(device_type="cuda", enabled=USE_AMP):
            query_embeddings = encode(model, images)
            neighbor_embeddings = cached_embeddings[batch_neighbors].detach()
            policy_step = policy(
                query_embeddings,
                neighbor_embeddings,
                label_state[batch_indices],
                label_state[batch_neighbors],
                actions=actions[batch_indices],
            )
            loss = -(
                q_value.detach()
                * policy_step.log_probabilities.sum()
                / selected_count
            )
        scaler.scale(loss).backward()
        total_loss += loss.detach()
        if batch_number % 10 == 0 or batch_number == total_batches:
            print(
                f"[ACTOR] batch={batch_number}/{total_batches} "
                f"samples={end}/{selected_count}"
            )

    scaler.step(optimizer)
    scaler.update()
    return float(total_loss)


def update_critic(
    critic: nn.Module,
    optimizer: torch.optim.Optimizer,
    encoding: Tensor,
    reward: Tensor,
    next_encoding: Tensor | None,
) -> float:
    optimizer.zero_grad(set_to_none=True)
    current_q = critic.value_from_encoding(encoding)
    terminal = next_encoding is None
    next_q = torch.zeros_like(current_q)
    if next_encoding is not None:
        next_q = critic.value_from_encoding(next_encoding)
    td = sarsa_td_loss(
        current_q,
        reward,
        next_q,
        discount_factor=DISCOUNT_FACTOR,
        terminal=terminal,
    )
    td.loss.backward()
    optimizer.step()
    return float(td.loss.detach())


def build_benchmark_config() -> Config:
    cfg = Config()
    return replace(
        cfg,
        model=replace(
            cfg.model,
            name=MODEL_NAME,
            pretrained=PRETRAINED,
            drop_rate=DROP_RATE,
            drop_path_rate=DROP_PATH_RATE,
        ),
        global_knn=replace(
            cfg.global_knn,
            k=K,
            query_chunk_size=EXPECTED_SAMPLES,
            reference_chunk_size=EXPECTED_SAMPLES,
            cache_features_on_device=True,
        ),
        policy=replace(
            cfg.policy,
            temperature=TEMPERATURE,
            correction_chunk_size=EXPECTED_SAMPLES,
        ),
        reward=replace(cfg.reward, nla_weight=NLA_WEIGHT),
        rl_train=replace(
            cfg.rl_train,
            epochs=RL_EPOCHS,
            trajectory_length=TRAJECTORY_LENGTH,
            discount_factor=DISCOUNT_FACTOR,
            actor_lr=ACTOR_LR,
            actor_weight_decay=ACTOR_WEIGHT_DECAY,
            actor_adamw_betas=ACTOR_BETAS,
            actor_adamw_eps=ACTOR_EPS,
            critic_lr=CRITIC_LR,
            critic_momentum=CRITIC_MOMENTUM,
            critic_weight_decay=CRITIC_WEIGHT_DECAY,
            use_policy_update_subset=True,
            policy_update_subset_size=POLICY_UPDATE_SUBSET_SIZE,
            policy_update_batch_size=POLICY_UPDATE_BATCH_SIZE,
        ),
        runtime=replace(
            cfg.runtime,
            device="cuda",
            use_amp=USE_AMP,
            use_channels_last=USE_CHANNELS_LAST,
            cudnn_benchmark=CUDNN_BENCHMARK,
            seed=SEED,
        ),
    )


def print_configuration(
    device: torch.device,
    clean_labels: Tensor,
    noisy_labels: Tensor,
    noise_mask: Tensor,
) -> None:
    properties = torch.cuda.get_device_properties(device)
    print("MNIST H100 RLNLC speed benchmark")
    print(f"device={device} ({properties.name})")
    print(f"device_memory_gib={properties.total_memory / 1024**3:.2f}")
    print(f"samples={clean_labels.numel()} digits={DIGITS}")
    print(f"clean_class_counts={class_counts(clean_labels)}")
    print(f"noisy_class_counts={class_counts(noisy_labels)}")
    print(
        f"noise_type=stratified_symmetric noise_rate={NOISE_RATE:.2f} "
        f"noise_seed={SEED} corrupted={int(noise_mask.sum())}"
    )
    print(f"rl_epochs={RL_EPOCHS} trajectory_length={TRAJECTORY_LENGTH}")
    print(
        f"policy_update_subset={POLICY_UPDATE_SUBSET_SIZE} "
        f"policy_update_batch={POLICY_UPDATE_BATCH_SIZE}"
    )
    print(
        f"feature_batch={FEATURE_BATCH_SIZE} k={K} "
        "knn_mode=full_60000x60000"
    )
    print(
        "correction_mode=full_60000 "
        f"amp={USE_AMP} channels_last={USE_CHANNELS_LAST}"
    )
    print(
        f"actor=AdamW(lr={ACTOR_LR}, wd={ACTOR_WEIGHT_DECAY}) "
        f"critic=SGD(lr={CRITIC_LR}, momentum={CRITIC_MOMENTUM})"
    )
    print("pretrained=False (architecture and compute path are unchanged)")


def print_timing_summary(timings: Timings) -> None:
    measured_total = sum(sum(values) for values in timings.values())
    print("\n[TIMING SUMMARY]")
    for name, values in timings.items():
        total = sum(values)
        mean = total / len(values)
        percentage = 100.0 * total / measured_total
        print(
            f"{name:<30} total={total:>10.3f} sec  "
            f"mean={mean:>9.3f}  calls={len(values):>2}  "
            f"({percentage:>6.2f}%)"
        )
    print(f"{'measured_total':<30} total={measured_total:>10.3f} sec")


def main() -> None:
    device = resolve_h100_device()
    seed_everything(SEED)
    torch.backends.cudnn.benchmark = CUDNN_BENCHMARK
    torch.cuda.reset_peak_memory_stats(device)
    timings: Timings = {}

    raw_images, clean_labels_cpu = measure(
        "mnist_load",
        device,
        timings,
        load_full_mnist_train,
    )
    noisy_labels_cpu, noise_mask_cpu = measure(
        "noise_injection",
        device,
        timings,
        lambda: inject_stratified_symmetric_noise(clean_labels_cpu),
    )
    print_configuration(
        device,
        clean_labels_cpu,
        noisy_labels_cpu,
        noise_mask_cpu,
    )
    cfg = build_benchmark_config()

    model = measure(
        "model_init",
        device,
        timings,
        lambda: timm.create_model(
            MODEL_NAME,
            pretrained=PRETRAINED,
            num_classes=0,
            drop_rate=DROP_RATE,
            drop_path_rate=DROP_PATH_RATE,
        ).to(
            device=device,
            memory_format=(
                torch.channels_last
                if USE_CHANNELS_LAST
                else torch.contiguous_format
            ),
        ),
    )
    mean = torch.tensor(IMAGENET_MEAN, device=device).reshape(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).reshape(1, 3, 1, 1)
    measure(
        "kernel_warmup",
        device,
        timings,
        lambda: warm_device_kernels(model, raw_images, device, mean, std),
    )

    policy = LabelCorrectionPolicy(cfg.policy).to(device)
    reward_function = RLNLCReward(cfg).to(device)
    critic = build_critic(cfg).to(device)
    critic_optimizer = build_critic_optimizer(critic, cfg)
    actor_optimizer = AdamW(
        model.parameters(),
        lr=ACTOR_LR,
        weight_decay=ACTOR_WEIGHT_DECAY,
        betas=ACTOR_BETAS,
        eps=ACTOR_EPS,
    )
    actor_scheduler = MultiStepLR(
        actor_optimizer,
        milestones=[1],
        gamma=LR_DECAY_FACTOR,
    )
    critic_scheduler = MultiStepLR(
        critic_optimizer,
        milestones=[1],
        gamma=LR_DECAY_FACTOR,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=USE_AMP)

    fixed_embeddings = measure(
        "global_cache_feature_extraction",
        device,
        timings,
        lambda: extract_all_embeddings(model, raw_images, device, mean, std),
    ).detach()
    global_neighbors, global_cosines = measure(
        "global_cache_exact_knn",
        device,
        timings,
        lambda: build_global_graph(fixed_embeddings),
    )

    label_state = F.one_hot(
        noisy_labels_cpu.to(device, non_blocking=True),
        num_classes=NUM_CLASSES,
    ).to(torch.float32)
    clean_labels = clean_labels_cpu.to(device, non_blocking=True)
    initial_noisy_labels = noisy_labels_cpu.to(device, non_blocking=True)
    noise_mask = noise_mask_cpu.to(device, non_blocking=True)
    previous_encoding: Tensor | None = None
    previous_reward: Tensor | None = None
    last_action_rate = 0.0
    last_actor_loss = 0.0
    critic_losses: list[float] = []

    for step in range(1, TRAJECTORY_LENGTH + 1):
        print(f"\n[TRAJECTORY] epoch=1/{RL_EPOCHS} step={step}/{TRAJECTORY_LENGTH}")
        policy_embeddings = measure(
            "actor_feature_extraction",
            device,
            timings,
            lambda: extract_all_embeddings(model, raw_images, device, mean, std),
            step=step,
        )
        policy_neighbors = measure(
            "exact_policy_knn",
            device,
            timings,
            lambda: build_neighbor_indices(policy_embeddings),
            step=step,
        )
        correction = measure(
            "full_correction",
            device,
            timings,
            lambda: policy.correct_all(
                policy_embeddings,
                label_state,
                policy_neighbors,
            ),
            step=step,
        )
        if correction.actions.all():
            correction.actions[0] = False
            correction.corrected_labels[0] = label_state[0]

        reward_output = measure(
            "reward_including_clean_knn",
            device,
            timings,
            lambda: reward_function(
                correction.corrected_labels,
                correction.actions,
                fixed_embeddings,
                global_neighbors,
                global_cosines,
            ),
            step=step,
        )

        def encode_critic_state() -> tuple[Tensor, Tensor]:
            encoding = critic.encode(
                reward_output.per_sample_consistency
            ).detach()
            with torch.no_grad():
                q_value = critic.value_from_encoding(encoding)
            return encoding, q_value

        encoding, q_value = measure(
            "critic_state_encoding",
            device,
            timings,
            encode_critic_state,
            step=step,
        )
        selected_cpu = select_policy_subset(EXPECTED_SAMPLES, step)
        last_actor_loss = measure(
            "query_only_actor_update",
            device,
            timings,
            lambda: update_query_only_actor(
                model,
                policy,
                actor_optimizer,
                scaler,
                raw_images,
                selected_cpu,
                policy_embeddings,
                label_state,
                policy_neighbors,
                correction.actions,
                q_value,
                device,
                mean,
                std,
            ),
            step=step,
        )

        step_critic_losses: list[float] = []

        def perform_critic_updates() -> tuple[float, ...]:
            if previous_encoding is not None and previous_reward is not None:
                step_critic_losses.append(
                    update_critic(
                        critic,
                        critic_optimizer,
                        previous_encoding,
                        previous_reward,
                        encoding,
                    )
                )
            if step == TRAJECTORY_LENGTH:
                step_critic_losses.append(
                    update_critic(
                        critic,
                        critic_optimizer,
                        encoding,
                        reward_output.total_reward,
                        None,
                    )
                )
            return tuple(step_critic_losses)

        if previous_encoding is not None or step == TRAJECTORY_LENGTH:
            measure(
                "critic_update",
                device,
                timings,
                perform_critic_updates,
                step=step,
            )
            critic_losses.extend(step_critic_losses)

        label_state = correction.corrected_labels
        last_action_rate = float(correction.actions.float().mean())
        current_hard_labels = label_state.argmax(dim=1)
        changed_from_noisy_rate = float(
            current_hard_labels.ne(initial_noisy_labels).float().mean()
        )
        clean_accuracy = float(
            current_hard_labels.eq(clean_labels).float().mean()
        )
        print(
            f"[RL] epoch=1 step={step} "
            f"reward={float(reward_output.total_reward):.6f} "
            f"q={float(q_value):.6f} actor={last_actor_loss:.6f} "
            f"action={last_action_rate:.4f} "
            f"changed_from_noisy={changed_from_noisy_rate:.4f} "
            f"clean_accuracy={clean_accuracy:.4f}"
        )

        if step < TRAJECTORY_LENGTH:
            previous_encoding = encoding
            previous_reward = reward_output.total_reward.detach()
        del policy_embeddings, policy_neighbors, correction, reward_output

    actor_scheduler.step()
    critic_scheduler.step()
    print_timing_summary(timings)

    measured_total = sum(sum(values) for values in timings.values())
    setup_names = {
        "mnist_load",
        "noise_injection",
        "model_init",
        "kernel_warmup",
        "global_cache_feature_extraction",
        "global_cache_exact_knn",
    }
    setup_total = sum(
        sum(values) for name, values in timings.items() if name in setup_names
    )
    epoch_total = measured_total - setup_total
    final_hard_labels = label_state.argmax(dim=1)
    final_changed_from_noisy_rate = float(
        final_hard_labels.ne(initial_noisy_labels).float().mean()
    )
    final_clean_accuracy = float(
        final_hard_labels.eq(clean_labels).float().mean()
    )
    noisy_recovery_rate = float(
        final_hard_labels[noise_mask].eq(clean_labels[noise_mask]).float().mean()
    )
    peak_allocated = torch.cuda.max_memory_allocated(device) / 1024**3
    peak_reserved = torch.cuda.max_memory_reserved(device) / 1024**3

    print("\n[RESULT]")
    print(f"samples={EXPECTED_SAMPLES}, classes={NUM_CLASSES}, rl_epochs=1")
    print(
        f"noise_type=stratified_symmetric, noise_rate={NOISE_RATE:.2f}, "
        f"noise_seed={SEED}"
    )
    print(
        f"trajectory_steps={TRAJECTORY_LENGTH}, "
        f"actor_updates_per_step={POLICY_UPDATE_SUBSET_SIZE}"
    )
    print(f"k={K}, neighbor_gradient=False, cached_fixed_reward_graph=True")
    print(
        f"last_action_rate={last_action_rate:.4f}, "
        f"final_changed_from_noisy_rate={final_changed_from_noisy_rate:.4f}"
    )
    print(
        f"final_clean_accuracy={final_clean_accuracy:.4f}, "
        f"noisy_recovery_rate={noisy_recovery_rate:.4f}"
    )
    mean_critic_loss = (
        sum(critic_losses) / len(critic_losses) if critic_losses else float("nan")
    )
    print(
        f"last_actor_loss={last_actor_loss:.6f}, "
        f"mean_critic_loss={mean_critic_loss:.6f}"
    )
    print(f"setup_seconds={setup_total:.3f}")
    print(f"one_epoch_seconds={epoch_total:.3f}")
    print(f"seconds_per_trajectory_step={epoch_total / TRAJECTORY_LENGTH:.3f}")
    print(
        f"peak_cuda_allocated_gib={peak_allocated:.3f}, "
        f"peak_cuda_reserved_gib={peak_reserved:.3f}"
    )


if __name__ == "__main__":
    main()
