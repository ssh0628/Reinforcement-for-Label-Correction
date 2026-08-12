"""RTX 5080 speed-only benchmark for the current RLNLC training path.

This benchmark intentionally excludes supervised warmup, validation/test
evaluation, correction-quality metrics, and checkpoints. It profiles one RL
epoch over all 60,000 MNIST training images so that ten trajectory steps expose
the runtime share of feature extraction, exact KNN, correction/reward, and
actor/critic updates.

The model uses random weights because pretrained values do not change tensor
shapes or compute cost. This script is for systems profiling only and must not
be used to draw conclusions about label-correction accuracy.
"""

from __future__ import annotations

import csv
import math
import random
import sys
import time
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import replace
from pathlib import Path
from typing import Callable, TextIO, TypeVar

import numpy as np
import timm
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import MultiStepLR
from torchvision.datasets import MNIST

from rl.actor.policy import LabelCorrectionPolicy
from rl.actor.policy_knn import build_exact_policy_knn
from rl.critic.critic import (
    build_critic,
    build_critic_optimizer,
    sarsa_td_loss,
)
from rl.reward.reward import RLNLCReward
from setting.config import Config


# Paths
PROJECT_ROOT = Path(__file__).resolve().parent
MNIST_ROOT = PROJECT_ROOT / "data" / "mnist"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "mnist_speed_rtx5080"
RUN_LOG_FILENAME = "run.log"
TIMING_CSV_FILENAME = "timing.csv"
STEP_TIMING_CSV_FILENAME = "step_timing.csv"
SUMMARY_CSV_FILENAME = "speed_summary.csv"

# Workload
EXPECTED_SAMPLES = 60_000
NUM_CLASSES = 10
NOISE_RATE = 0.40
SPEED_EPOCHS = 1
TRAJECTORY_LENGTH = 10
K = 10
TEMPERATURE = 0.5

# RTX 5080 starting point. Tune one value at a time between runs.
IMAGE_SIZE = 224
FEATURE_BATCH_SIZE = 512
POLICY_UPDATE_SUBSET_SIZE = 6_000
POLICY_UPDATE_BATCH_SIZE = 128
KNN_QUERY_CHUNK_SIZE = 2_048
KNN_REFERENCE_CHUNK_SIZE = 32_768
CORRECTION_CHUNK_SIZE = 16_384

# RL settings kept aligned with the current project profile.
MODEL_NAME = "convnextv2_tiny.fcmae_ft_in22k_in1k"
PRETRAINED = False
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

# Runtime
DOWNLOAD_MNIST = True
USE_AMP = True
AMP_DTYPE = torch.bfloat16
USE_CHANNELS_LAST = True
CUDNN_BENCHMARK = True
MATMUL_PRECISION = "high"
SEED = 0
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

STAGE_NAMES = (
    "actor_feature_extraction",
    "exact_policy_knn",
    "full_correction",
    "reward_including_clean_knn",
    "critic_state_encoding",
    "query_only_actor_update",
    "critic_update",
)

TIMING_FIELDS = (
    "stage",
    "calls",
    "total_seconds",
    "mean_seconds",
    "min_seconds",
    "max_seconds",
    "percentage",
)

STEP_TIMING_FIELDS = (
    "epoch",
    "trajectory_step",
    "global_step",
    *STAGE_NAMES,
    "total_seconds",
)

SUMMARY_FIELDS = (
    "device",
    "device_memory_gib",
    "samples",
    "noise_rate",
    "speed_epochs",
    "trajectory_length",
    "total_rl_steps",
    "model",
    "pretrained",
    "embedding_dim",
    "feature_batch_size",
    "policy_update_subset_size",
    "policy_update_batch_size",
    "k",
    "knn_query_chunk_size",
    "knn_reference_chunk_size",
    "correction_chunk_size",
    "amp_dtype",
    "setup_seconds",
    "rl_seconds",
    "mean_rl_step_seconds",
    "samples_per_rl_step_second",
    "measured_stage_seconds",
    "wall_seconds",
    "peak_cuda_allocated_gib",
    "peak_cuda_reserved_gib",
)

T = TypeVar("T")
Timings = dict[str, list[float]]


class TeeStream:
    def __init__(self, *streams: TextIO) -> None:
        self.streams = streams

    def write(self, text: str) -> int:
        for stream in self.streams:
            stream.write(text)
            stream.flush()
        return len(text)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("mnist_speed_test.py requires a CUDA GPU.")
    return torch.device("cuda", 0)


def synchronize(device: torch.device) -> None:
    torch.cuda.synchronize(device)


def measure(
    name: str,
    device: torch.device,
    timings: Timings,
    operation: Callable[[], T],
    *,
    step_record: dict[str, object] | None = None,
) -> T:
    synchronize(device)
    started = time.perf_counter()
    result = operation()
    synchronize(device)
    elapsed = time.perf_counter() - started
    timings.setdefault(name, []).append(elapsed)
    if step_record is not None:
        step_record[name] = elapsed
    print(f"[TIME] {name:<32} {elapsed:>10.3f} sec")
    return result


def write_csv(
    path: Path,
    rows: list[dict[str, object]],
    fieldnames: tuple[str, ...],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_mnist_train() -> tuple[Tensor, Tensor]:
    dataset = MNIST(
        root=MNIST_ROOT,
        train=True,
        download=DOWNLOAD_MNIST,
    )
    images = dataset.data.contiguous().pin_memory()
    labels = dataset.targets.to(torch.long).contiguous().pin_memory()
    if images.shape != (EXPECTED_SAMPLES, 28, 28):
        raise RuntimeError(f"Unexpected MNIST image shape: {images.shape}.")
    if labels.shape != (EXPECTED_SAMPLES,):
        raise RuntimeError(f"Unexpected MNIST label shape: {labels.shape}.")
    return images, labels


def inject_stratified_symmetric_noise(clean_labels: Tensor) -> Tensor:
    generator = torch.Generator().manual_seed(SEED)
    noisy_labels = clean_labels.clone()
    class_sizes = [
        int(clean_labels.eq(class_id).sum())
        for class_id in range(NUM_CLASSES)
    ]
    exact_counts = [size * NOISE_RATE for size in class_sizes]
    noise_counts = [math.floor(value) for value in exact_counts]
    target_count = round(clean_labels.numel() * NOISE_RATE)
    remainder = target_count - sum(noise_counts)
    allocation_order = sorted(
        range(NUM_CLASSES),
        key=lambda index: exact_counts[index] - noise_counts[index],
        reverse=True,
    )
    for index in allocation_order[:remainder]:
        noise_counts[index] += 1

    for class_id, noise_count in enumerate(noise_counts):
        class_indices = clean_labels.eq(class_id).nonzero().flatten()
        selected = class_indices[
            torch.randperm(class_indices.numel(), generator=generator)[
                :noise_count
            ]
        ]
        alternatives = torch.randint(
            NUM_CLASSES - 1,
            (noise_count,),
            generator=generator,
        )
        alternatives += alternatives.ge(clean_labels[selected])
        noisy_labels[selected] = alternatives

    actual_count = int(noisy_labels.ne(clean_labels).sum())
    if actual_count != target_count:
        raise RuntimeError(
            f"Noise injection produced {actual_count}, expected {target_count}."
        )
    return noisy_labels.pin_memory()


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
            torch.channels_last
            if USE_CHANNELS_LAST
            else torch.contiguous_format
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
) -> int:
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
        dtype=AMP_DTYPE,
        enabled=USE_AMP,
    ):
        inference_embeddings = encode(model, inference_images)

    update_images = preprocess(
        raw_images[:POLICY_UPDATE_BATCH_SIZE],
        device,
        mean,
        std,
    )
    with torch.autocast(
        device_type="cuda",
        dtype=AMP_DTYPE,
        enabled=USE_AMP,
    ):
        encode(model, update_images).mean().backward()
    model.zero_grad(set_to_none=True)
    return inference_embeddings.size(1)


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
        images = preprocess(raw_images[start:end], device, mean, std)
        with torch.autocast(
            device_type="cuda",
            dtype=AMP_DTYPE,
            enabled=USE_AMP,
        ):
            embeddings = encode(model, images)
        if output is None:
            output = torch.empty(
                (sample_count, embeddings.size(1)),
                dtype=torch.float32,
                device=device,
            )
        output[start:end].copy_(embeddings)
    if output is None:
        raise RuntimeError("Cannot extract embeddings from an empty dataset.")
    return output


@torch.inference_mode()
def build_neighbor_indices(embeddings: Tensor) -> Tensor:
    return build_exact_policy_knn(
        embeddings,
        k=K,
        query_chunk_size=KNN_QUERY_CHUNK_SIZE,
        reference_chunk_size=KNN_REFERENCE_CHUNK_SIZE,
    )


@torch.inference_mode()
def build_global_graph(embeddings: Tensor) -> tuple[Tensor, Tensor]:
    neighbor_indices = build_neighbor_indices(embeddings)
    normalized = F.normalize(embeddings.float(), dim=1)
    neighbor_cosines = torch.empty(
        neighbor_indices.shape,
        dtype=torch.float32,
        device=embeddings.device,
    )
    for start in range(0, embeddings.size(0), CORRECTION_CHUNK_SIZE):
        end = min(start + CORRECTION_CHUNK_SIZE, embeddings.size(0))
        indices = neighbor_indices[start:end]
        neighbor_cosines[start:end] = (
            normalized[start:end].unsqueeze(1) * normalized[indices]
        ).sum(dim=2)
    return neighbor_indices, neighbor_cosines


def select_policy_subset(sample_count: int, global_step: int) -> Tensor:
    generator = torch.Generator().manual_seed(SEED + global_step)
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
) -> None:
    model.eval()
    optimizer.zero_grad(set_to_none=True)
    selected = selected_cpu.to(device=device, non_blocking=True)
    selected_images = raw_images[selected_cpu].pin_memory()
    selected_count = selected.numel()

    for start in range(0, selected_count, POLICY_UPDATE_BATCH_SIZE):
        end = min(start + POLICY_UPDATE_BATCH_SIZE, selected_count)
        batch_indices = selected[start:end]
        batch_neighbors = neighbor_indices[batch_indices]
        images = preprocess(
            selected_images[start:end],
            device,
            mean,
            std,
        )
        with torch.autocast(
            device_type="cuda",
            dtype=AMP_DTYPE,
            enabled=USE_AMP,
        ):
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

    scaler.step(optimizer)
    scaler.update()


def update_critic(
    critic: nn.Module,
    optimizer: torch.optim.Optimizer,
    encoding: Tensor,
    reward: Tensor,
    next_encoding: Tensor | None,
) -> None:
    optimizer.zero_grad(set_to_none=True)
    current_q = critic.value_from_encoding(encoding)
    next_q = (
        torch.zeros_like(current_q)
        if next_encoding is None
        else critic.value_from_encoding(next_encoding)
    )
    td = sarsa_td_loss(
        current_q,
        reward,
        next_q,
        discount_factor=DISCOUNT_FACTOR,
        terminal=next_encoding is None,
    )
    td.loss.backward()
    optimizer.step()


def build_speed_config() -> Config:
    cfg = Config()
    return replace(
        cfg,
        model=replace(
            cfg.model,
            name=MODEL_NAME,
            pretrained=PRETRAINED,
            drop_rate=0.0,
            drop_path_rate=0.0,
        ),
        global_knn=replace(
            cfg.global_knn,
            k=K,
            query_chunk_size=KNN_QUERY_CHUNK_SIZE,
            reference_chunk_size=KNN_REFERENCE_CHUNK_SIZE,
            cache_features_on_device=True,
        ),
        policy=replace(
            cfg.policy,
            temperature=TEMPERATURE,
            correction_chunk_size=CORRECTION_CHUNK_SIZE,
        ),
        reward=replace(cfg.reward, nla_weight=NLA_WEIGHT),
        rl_train=replace(
            cfg.rl_train,
            epochs=SPEED_EPOCHS,
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


def build_timing_rows(timings: Timings) -> list[dict[str, object]]:
    measured_total = sum(sum(values) for values in timings.values())
    rows: list[dict[str, object]] = []
    for stage, values in timings.items():
        total = sum(values)
        rows.append(
            {
                "stage": stage,
                "calls": len(values),
                "total_seconds": total,
                "mean_seconds": total / len(values),
                "min_seconds": min(values),
                "max_seconds": max(values),
                "percentage": 100.0 * total / measured_total,
            }
        )
    return rows


def main() -> None:
    if POLICY_UPDATE_SUBSET_SIZE > EXPECTED_SAMPLES:
        raise ValueError("Policy subset cannot exceed the dataset size.")
    if POLICY_UPDATE_BATCH_SIZE > POLICY_UPDATE_SUBSET_SIZE:
        raise ValueError("Actor batch cannot exceed the policy subset size.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timing_path = OUTPUT_DIR / TIMING_CSV_FILENAME
    step_timing_path = OUTPUT_DIR / STEP_TIMING_CSV_FILENAME
    summary_path = OUTPUT_DIR / SUMMARY_CSV_FILENAME

    device = resolve_device()
    seed_everything(SEED)
    torch.backends.cudnn.benchmark = CUDNN_BENCHMARK
    torch.set_float32_matmul_precision(MATMUL_PRECISION)
    torch.cuda.reset_peak_memory_stats(device)
    wall_started = time.perf_counter()
    timings: Timings = {}
    step_rows: list[dict[str, object]] = []

    properties = torch.cuda.get_device_properties(device)
    print("MNIST RLNLC speed-only benchmark")
    print(f"device={properties.name} memory={properties.total_memory / 1024**3:.2f} GiB")
    print(f"samples={EXPECTED_SAMPLES} rl_steps={SPEED_EPOCHS * TRAJECTORY_LENGTH}")
    print(
        f"feature_batch={FEATURE_BATCH_SIZE} "
        f"policy_subset={POLICY_UPDATE_SUBSET_SIZE} "
        f"actor_batch={POLICY_UPDATE_BATCH_SIZE}"
    )
    print(
        f"knn_query={KNN_QUERY_CHUNK_SIZE} "
        f"knn_reference={KNN_REFERENCE_CHUNK_SIZE} "
        f"correction_chunk={CORRECTION_CHUNK_SIZE}"
    )

    raw_images, clean_labels = measure(
        "mnist_load",
        device,
        timings,
        load_mnist_train,
    )
    noisy_labels_cpu = measure(
        "noise_injection",
        device,
        timings,
        lambda: inject_stratified_symmetric_noise(clean_labels),
    )
    del clean_labels

    model = measure(
        "model_init",
        device,
        timings,
        lambda: timm.create_model(
            MODEL_NAME,
            pretrained=PRETRAINED,
            num_classes=0,
            drop_rate=0.0,
            drop_path_rate=0.0,
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
    embedding_dim = measure(
        "kernel_warmup",
        device,
        timings,
        lambda: warm_device_kernels(model, raw_images, device, mean, std),
    )

    cfg = build_speed_config()
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
        milestones=[max(1, math.ceil(SPEED_EPOCHS / 2))],
        gamma=LR_DECAY_FACTOR,
    )
    critic_scheduler = MultiStepLR(
        critic_optimizer,
        milestones=[max(1, math.ceil(SPEED_EPOCHS / 2))],
        gamma=LR_DECAY_FACTOR,
    )
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=USE_AMP and AMP_DTYPE == torch.float16,
    )

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

    initial_noisy_labels = noisy_labels_cpu.to(device, non_blocking=True)
    setup_names = {
        "mnist_load",
        "noise_injection",
        "model_init",
        "kernel_warmup",
        "global_cache_feature_extraction",
        "global_cache_exact_knn",
    }
    setup_seconds = sum(
        sum(values)
        for name, values in timings.items()
        if name in setup_names
    )

    synchronize(device)
    rl_started = time.perf_counter()
    global_step = 0
    for epoch in range(1, SPEED_EPOCHS + 1):
        label_state = F.one_hot(
            initial_noisy_labels,
            num_classes=NUM_CLASSES,
        ).to(torch.float32)
        previous_encoding: Tensor | None = None
        previous_reward: Tensor | None = None

        for trajectory_step in range(1, TRAJECTORY_LENGTH + 1):
            global_step += 1
            step_record: dict[str, object] = {
                "epoch": epoch,
                "trajectory_step": trajectory_step,
                "global_step": global_step,
            }
            print(
                f"\n[STEP] epoch={epoch}/{SPEED_EPOCHS} "
                f"trajectory={trajectory_step}/{TRAJECTORY_LENGTH}"
            )
            synchronize(device)
            step_started = time.perf_counter()

            policy_embeddings = measure(
                "actor_feature_extraction",
                device,
                timings,
                lambda: extract_all_embeddings(
                    model,
                    raw_images,
                    device,
                    mean,
                    std,
                ),
                step_record=step_record,
            )
            policy_neighbors = measure(
                "exact_policy_knn",
                device,
                timings,
                lambda: build_neighbor_indices(policy_embeddings),
                step_record=step_record,
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
                step_record=step_record,
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
                step_record=step_record,
            )

            def encode_critic_state() -> tuple[Tensor, Tensor]:
                encoding_value = critic.encode(
                    reward_output.per_sample_consistency
                ).detach()
                with torch.no_grad():
                    q_value_value = critic.value_from_encoding(encoding_value)
                return encoding_value, q_value_value

            encoding, q_value = measure(
                "critic_state_encoding",
                device,
                timings,
                encode_critic_state,
                step_record=step_record,
            )
            selected_cpu = select_policy_subset(EXPECTED_SAMPLES, global_step)
            measure(
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
                step_record=step_record,
            )

            def perform_critic_updates() -> None:
                if previous_encoding is not None and previous_reward is not None:
                    update_critic(
                        critic,
                        critic_optimizer,
                        previous_encoding,
                        previous_reward,
                        encoding,
                    )
                if trajectory_step == TRAJECTORY_LENGTH:
                    update_critic(
                        critic,
                        critic_optimizer,
                        encoding,
                        reward_output.total_reward,
                        None,
                    )

            measure(
                "critic_update",
                device,
                timings,
                perform_critic_updates,
                step_record=step_record,
            )

            label_state = correction.corrected_labels
            if trajectory_step < TRAJECTORY_LENGTH:
                previous_encoding = encoding
                previous_reward = reward_output.total_reward.detach()
            del policy_embeddings, policy_neighbors, correction, reward_output

            synchronize(device)
            step_record["total_seconds"] = time.perf_counter() - step_started
            step_rows.append(step_record)
            print(
                f"[STEP TOTAL] {float(step_record['total_seconds']):.3f} sec"
            )

        actor_scheduler.step()
        critic_scheduler.step()
        del label_state

    synchronize(device)
    rl_seconds = time.perf_counter() - rl_started
    measured_stage_seconds = sum(sum(values) for values in timings.values())
    wall_seconds = time.perf_counter() - wall_started
    mean_rl_step_seconds = rl_seconds / global_step
    peak_allocated = torch.cuda.max_memory_allocated(device) / 1024**3
    peak_reserved = torch.cuda.max_memory_reserved(device) / 1024**3

    timing_rows = build_timing_rows(timings)
    write_csv(timing_path, timing_rows, TIMING_FIELDS)
    write_csv(step_timing_path, step_rows, STEP_TIMING_FIELDS)
    write_csv(
        summary_path,
        [
            {
                "device": properties.name,
                "device_memory_gib": properties.total_memory / 1024**3,
                "samples": EXPECTED_SAMPLES,
                "noise_rate": NOISE_RATE,
                "speed_epochs": SPEED_EPOCHS,
                "trajectory_length": TRAJECTORY_LENGTH,
                "total_rl_steps": global_step,
                "model": MODEL_NAME,
                "pretrained": PRETRAINED,
                "embedding_dim": embedding_dim,
                "feature_batch_size": FEATURE_BATCH_SIZE,
                "policy_update_subset_size": POLICY_UPDATE_SUBSET_SIZE,
                "policy_update_batch_size": POLICY_UPDATE_BATCH_SIZE,
                "k": K,
                "knn_query_chunk_size": KNN_QUERY_CHUNK_SIZE,
                "knn_reference_chunk_size": KNN_REFERENCE_CHUNK_SIZE,
                "correction_chunk_size": CORRECTION_CHUNK_SIZE,
                "amp_dtype": str(AMP_DTYPE),
                "setup_seconds": setup_seconds,
                "rl_seconds": rl_seconds,
                "mean_rl_step_seconds": mean_rl_step_seconds,
                "samples_per_rl_step_second": (
                    EXPECTED_SAMPLES / mean_rl_step_seconds
                ),
                "measured_stage_seconds": measured_stage_seconds,
                "wall_seconds": wall_seconds,
                "peak_cuda_allocated_gib": peak_allocated,
                "peak_cuda_reserved_gib": peak_reserved,
            }
        ],
        SUMMARY_FIELDS,
    )

    print("\n[TIMING SUMMARY]")
    for row in sorted(
        timing_rows,
        key=lambda item: float(item["total_seconds"]),
        reverse=True,
    ):
        print(
            f"{str(row['stage']):<32} "
            f"total={float(row['total_seconds']):>10.3f} sec "
            f"mean={float(row['mean_seconds']):>9.3f} sec "
            f"share={float(row['percentage']):>6.2f}%"
        )
    print(f"setup_seconds={setup_seconds:.3f}")
    print(f"rl_seconds={rl_seconds:.3f}")
    print(f"mean_rl_step_seconds={mean_rl_step_seconds:.3f}")
    print(f"wall_seconds={wall_seconds:.3f}")
    print(
        f"peak_cuda_allocated_gib={peak_allocated:.3f} "
        f"peak_cuda_reserved_gib={peak_reserved:.3f}"
    )
    print(f"timing_csv={timing_path}")
    print(f"step_timing_csv={step_timing_path}")
    print(f"speed_summary_csv={summary_path}")


def run_with_file_logging() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    log_path = OUTPUT_DIR / RUN_LOG_FILENAME
    with log_path.open("w", encoding="utf-8", buffering=1) as log_handle:
        stdout = TeeStream(sys.stdout, log_handle)
        stderr = TeeStream(sys.stderr, log_handle)
        with redirect_stdout(stdout), redirect_stderr(stderr):
            print(f"run_log={log_path}")
            main()


if __name__ == "__main__":
    run_with_file_logging()
