"""CIFAR-10 RLNLC training engine.

Dataset loading, preprocessing, model construction, and experiment settings
are supplied by ``cifar_rl.py``. This module contains only the reusable
warm-up, KNN, actor-critic, cleaning, metrics, checkpoint, and timing logic.
"""

from __future__ import annotations

import csv
import json
import math
import random
import sys
import time
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import replace
from pathlib import Path
from typing import Callable, TextIO, TypeVar

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.optim import Optimizer, SGD
from torch.optim.lr_scheduler import MultiStepLR

from rl.actor.policy import LabelCorrectionPolicy
from rl.actor.policy_knn import build_exact_policy_knn
from rl.critic.critic import (
    build_critic,
    build_critic_optimizer,
    sarsa_td_loss,
)
from rl.reward.reward import RLNLCReward
from setting.config import Config


DATASET_NAME = "CIFAR-10"
DATASET_STAGE_PREFIX = "cifar10"
RUN_LOG_FILENAME = "run.log"
CHECK_LOG_FILENAME = "check.log"
ACTOR_BEST_CHECKPOINT_FILENAME = "actor_best.pt"
ACTOR_LAST_CHECKPOINT_FILENAME = "actor_last.pt"
CRITIC_BEST_CHECKPOINT_FILENAME = "critic_best.pt"
CRITIC_LAST_CHECKPOINT_FILENAME = "critic_last.pt"
TRAIN_CSV_FILENAME = "train.csv"
TEST_CSV_FILENAME = "test.csv"
TEST_PER_CLASS_CSV_FILENAME = "test_per_class.csv"
TIMING_CSV_FILENAME = "timing.csv"
RUN_SUMMARY_CSV_FILENAME = "run_summary.csv"
CLASS_IDS = tuple(range(10))
NUM_CLASSES = len(CLASS_IDS)
EXPECTED_SAMPLES = 50_000
NOISE_RATE = 0.40

EXTERNAL_NOISY_LABELS_PATH: Path | None = None
EXTERNAL_NOISE_MASK_PATH: Path | None = None
EXTERNAL_WARMUP_CHECKPOINT_PATH: Path | None = None
CLEANING_TRAJECTORY_LENGTH = 0
CORRECTED_LABELS_OUTPUT_PATH: Path | None = None
CLEANING_CSV_FILENAME = "cleaning.csv"
CLEANING_SUMMARY_FILENAME = "cleaning_summary.csv"
CLEANING_PER_CLASS_FILENAME = "cleaning_per_class.csv"

ACTOR_UPDATE_MODE = "full"  # "full" or "subset"
POLICY_UPDATE_SUBSET_SIZE = 5_000
if ACTOR_UPDATE_MODE not in {"full", "subset"}:
    raise ValueError("ACTOR_UPDATE_MODE must be 'full' or 'subset'.")
POLICY_UPDATE_SAMPLES = (
    EXPECTED_SAMPLES
    if ACTOR_UPDATE_MODE == "full"
    else min(EXPECTED_SAMPLES, POLICY_UPDATE_SUBSET_SIZE)
)
OUTPUT_DIR = (
    Path("outputs") / f"cifar10_rl_{ACTOR_UPDATE_MODE}"
)

MODEL_NAME = "cifar_resnet18"
WARMUP_MODEL_ID = ""
PRETRAINED = False
IMAGE_SIZE = 32
MODEL_FACTORY: Callable[[bool, int], nn.Module] | None = None
TRAIN_DATA_LOADER: Callable[[], tuple[Tensor, Tensor]] | None = None
EVALUATION_DATA_LOADER: (
    Callable[[], dict[str, tuple[Tensor, Tensor]]] | None
) = None
PREPROCESS_FUNCTION: (
    Callable[[Tensor, torch.device, Tensor, Tensor], Tensor] | None
) = None

WARMUP_EPOCHS = 1
WARMUP_BATCH_SIZE = 64
WARMUP_EVAL_BATCH_SIZE = 256
WARMUP_LR = 1e-2
WARMUP_WEIGHT_DECAY = 0.05
WARMUP_MIN_NOISY_VALIDATION_ACCURACY = 0.45
WARMUP_MOMENTUM = 0.9
WARMUP_LR_DECAY_FRACTION = 0.5
WARMUP_LR_DECAY_FACTOR = 0.1

RL_EPOCHS = 1
TRAJECTORY_LENGTH = 10
INITIAL_STATE_RANDOMIZATION_RATE = 0.10
FEATURE_BATCH_SIZE = 256
POLICY_UPDATE_BATCH_SIZE = 64

K = 10
TEMPERATURE = 0.5
KNN_QUERY_CHUNK_SIZE = 2_048
KNN_REFERENCE_CHUNK_SIZE = 32_768
CORRECTION_CHUNK_SIZE = 16_384

ACTOR_LR = 3e-5
ACTOR_WEIGHT_DECAY = 0.1
ACTOR_MOMENTUM = 0.9
CRITIC_LR = 1e-2
CRITIC_MOMENTUM = 0.9
CRITIC_WEIGHT_DECAY = 5e-4
CRITIC_NUM_BINS = 100
DISCOUNT_FACTOR = 0.9
NLA_WEIGHT = 0.5
LR_DECAY_FACTOR = 0.1
LR_DECAY_FRACTION = 0.5

USE_AMP = True
AMP_DTYPE = torch.bfloat16
USE_CHANNELS_LAST = True
CUDNN_BENCHMARK = True
SEED = 0
DIAGNOSTICS_ENABLED = False
DIAGNOSTIC_PROBE_SIZE = 1_024

DATA_MEAN = (0.4914, 0.4822, 0.4465)
DATA_STD = (0.2470, 0.2435, 0.2616)
T = TypeVar("T")
Timings = dict[str, list[float]]

WARMUP_FIELDS = (
    "epoch",
    "learning_rate",
    "train_loss",
    "train_noisy_accuracy",
    "validation_loss",
    "validation_noisy_accuracy",
    "validation_clean_accuracy",
    "validation_clean_macro_f1",
    "elapsed_seconds",
)

SUMMARY_FIELDS = (
    "epoch",
    "split",
    "samples",
    "loss",
    "accuracy",
    "balanced_accuracy",
    "macro_recall",
    "macro_precision",
    "macro_f1",
    "noise_count",
    "noise_rate",
    "correction_count",
    "correction_rate",
    "correct_correction_count",
    "correction_precision",
    "incorrect_correction_count",
    "false_correction_count",
    "false_correction_rate",
    "noisy_recovery_count",
    "noisy_recovery_rate",
    "clean_preservation_rate",
    "action_count",
    "action_rate",
    "reward",
    "actor_loss",
    "critic_loss",
    "elapsed_seconds",
)

PER_CLASS_FIELDS = (
    "split",
    "class_id",
    "support",
    "loss",
    "accuracy",
    "precision",
    "recall",
    "f1",
    "noise_count",
    "noise_rate",
    "correction_count",
    "correction_rate",
    "correct_correction_count",
    "correction_precision",
    "incorrect_correction_count",
    "false_correction_count",
    "false_correction_rate",
    "noisy_recovery_count",
    "noisy_recovery_rate",
    "clean_preservation_rate",
)

CLEANING_FIELDS = (
    "step",
    "action_count",
    "action_rate",
    "cumulative_changed_count",
    "cumulative_changed_rate",
    "clean_accuracy",
)

TIMING_FIELDS = (
    "stage",
    "calls",
    "total_seconds",
    "mean_seconds",
    "percentage",
)

RUN_SUMMARY_FIELDS = (
    "dataset",
    "model_name",
    "warmup_model_id",
    "image_size",
    "seed",
    "noise_rate",
    "train_samples",
    "validation_samples",
    "test_samples",
    "pretrained",
    "warmup_epochs",
    "warmup_batch_size",
    "warmup_optimizer",
    "warmup_learning_rate",
    "warmup_weight_decay",
    "warmup_best_epoch",
    "warmup_best_noisy_validation_accuracy",
    "warmup_best_clean_validation_accuracy",
    "warmup_deployment_mode",
    "warmup_deployment_epoch",
    "warmup_seconds",
    "rl_epochs",
    "trajectory_length",
    "initial_state_randomization_rate",
    "feature_batch_size",
    "policy_update_mode",
    "policy_update_samples",
    "policy_update_batch_size",
    "actor_optimizer",
    "actor_learning_rate",
    "actor_momentum",
    "actor_weight_decay",
    "diagnostics_enabled",
    "diagnostic_probe_size",
    "diagnostic_log",
    "rl_epoch0_validation_accuracy",
    "rl_epoch0_validation_macro_f1",
    "k",
    "temperature",
    "discount_factor",
    "reward_nla_weight",
    "critic_num_bins",
    "lr_decay_fraction",
    "lr_decay_factor",
    "knn_query_chunk_size",
    "knn_reference_chunk_size",
    "correction_chunk_size",
    "cleaning_trajectory_length",
    "corrected_labels_path",
    "cleaning_accuracy",
    "cleaning_noisy_recovery_rate",
    "cleaning_false_correction_rate",
    "rl_best_epoch",
    "rl_best_validation_accuracy",
    "rl_best_validation_balanced_accuracy",
    "rl_best_validation_macro_f1",
    "rl_best_validation_loss",
    "actor_best_checkpoint",
    "actor_last_checkpoint",
    "critic_best_checkpoint",
    "critic_last_checkpoint",
    "actor_deployment_checkpoint",
    "actor_deployment_epoch",
    "setup_seconds",
    "train_seconds",
    "mean_epoch_seconds",
    "validation_seconds",
    "checkpoint_seconds",
    "test_seconds",
    "measured_total_seconds",
    "peak_cuda_allocated_gib",
    "peak_cuda_reserved_gib",
)


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


def initialize_randomized_label_state(
    noisy_labels: Tensor,
    *,
    num_classes: int,
    randomization_rate: float,
    epoch: int,
) -> tuple[Tensor, int]:
    """Create the paper-style randomized initial state for one RL epoch."""
    if noisy_labels.ndim != 1 or noisy_labels.numel() == 0:
        raise ValueError("noisy_labels must be a non-empty [N] tensor.")
    if noisy_labels.dtype != torch.long:
        raise TypeError("noisy_labels must use torch.long.")
    if num_classes <= 1:
        raise ValueError("num_classes must be at least two.")
    if not 0 < randomization_rate < 1:
        raise ValueError("randomization_rate must be in (0, 1).")
    if epoch <= 0:
        raise ValueError("epoch must be positive.")

    random_count = min(
        noisy_labels.numel(),
        max(1, round(noisy_labels.numel() * randomization_rate)),
    )
    generator = torch.Generator(device=noisy_labels.device).manual_seed(
        SEED + epoch - 1
    )
    selected = torch.randperm(
        noisy_labels.numel(),
        device=noisy_labels.device,
        generator=generator,
    )[:random_count]
    randomized_labels = noisy_labels.clone()
    original = randomized_labels[selected]
    alternatives = torch.randint(
        num_classes - 1,
        (random_count,),
        device=noisy_labels.device,
        generator=generator,
    )
    alternatives += alternatives.ge(original)
    randomized_labels[selected] = alternatives
    label_state = F.one_hot(
        randomized_labels,
        num_classes=num_classes,
    ).to(torch.float32)
    return label_state, random_count


def resolve_local_device() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError(
            f"The {DATASET_NAME} RL experiment requires CUDA, but "
            "torch.cuda.is_available() "
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


def load_training_data() -> tuple[Tensor, Tensor]:
    if TRAIN_DATA_LOADER is None:
        raise RuntimeError(
            "TRAIN_DATA_LOADER must be configured by cifar_rl.py before "
            "starting the experiment."
        )
    return TRAIN_DATA_LOADER()


def load_evaluation_data() -> dict[str, tuple[Tensor, Tensor]]:
    if EVALUATION_DATA_LOADER is None:
        raise RuntimeError(
            "EVALUATION_DATA_LOADER must be configured by cifar_rl.py before "
            "starting the experiment."
        )
    return EVALUATION_DATA_LOADER()


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

    class_sizes = [int(clean_labels.eq(digit).sum()) for digit in CLASS_IDS]
    exact_counts = [size * NOISE_RATE for size in class_sizes]
    noise_counts = [math.floor(value) for value in exact_counts]
    target_noise_count = round(clean_labels.numel() * NOISE_RATE)
    remainder = target_noise_count - sum(noise_counts)
    allocation_order = sorted(
        range(NUM_CLASSES),
        key=lambda index: exact_counts[index] - noise_counts[index],
        reverse=True,
    )
    for index in allocation_order[:remainder]:
        noise_counts[index] += 1

    for digit, noise_count in zip(CLASS_IDS, noise_counts):
        class_indices = clean_labels.eq(digit).nonzero(as_tuple=False).flatten()
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
    if int(noise_mask.sum()) != target_noise_count:
        raise RuntimeError("Noise injection did not reach the requested rate.")
    return pin_for_cuda(noisy_labels), pin_for_cuda(noise_mask)


def load_noisy_label_artifacts(
    clean_labels: Tensor,
) -> tuple[Tensor, Tensor]:
    """Load and validate one shared noisy-label artifact when configured."""
    paths = (EXTERNAL_NOISY_LABELS_PATH, EXTERNAL_NOISE_MASK_PATH)
    if paths == (None, None):
        return inject_stratified_symmetric_noise(clean_labels)
    if any(path is None for path in paths):
        raise ValueError(
            "EXTERNAL_NOISY_LABELS_PATH and EXTERNAL_NOISE_MASK_PATH must "
            "either both be set or both be None."
        )
    noisy_path = Path(paths[0])
    mask_path = Path(paths[1])
    if not noisy_path.is_file():
        raise FileNotFoundError(f"Noisy-label artifact not found: {noisy_path}")
    if not mask_path.is_file():
        raise FileNotFoundError(f"Noise-mask artifact not found: {mask_path}")

    noisy_array = np.load(noisy_path, allow_pickle=False)
    mask_array = np.load(mask_path, allow_pickle=False)
    noisy_labels = torch.from_numpy(np.asarray(noisy_array)).to(torch.long)
    noise_mask = torch.from_numpy(np.asarray(mask_array)).to(torch.bool)
    expected_shape = tuple(clean_labels.shape)
    if tuple(noisy_labels.shape) != expected_shape:
        raise ValueError(
            f"Noisy labels must have shape {expected_shape}, got "
            f"{tuple(noisy_labels.shape)}."
        )
    if tuple(noise_mask.shape) != expected_shape:
        raise ValueError(
            f"Noise mask must have shape {expected_shape}, got "
            f"{tuple(noise_mask.shape)}."
        )
    if noisy_labels.numel() and (
        int(noisy_labels.min()) < 0 or int(noisy_labels.max()) >= NUM_CLASSES
    ):
        raise ValueError("Noisy labels contain an out-of-range class ID.")
    if not torch.equal(noisy_labels.ne(clean_labels), noise_mask):
        raise ValueError(
            "Saved noise mask does not match clean/noisy label differences."
        )
    expected_noise_count = round(clean_labels.numel() * NOISE_RATE)
    if int(noise_mask.sum()) != expected_noise_count:
        raise ValueError(
            "Saved noise count does not match NOISE_RATE: "
            f"{int(noise_mask.sum())} != {expected_noise_count}."
        )
    return (
        pin_for_cuda(noisy_labels.contiguous()),
        pin_for_cuda(noise_mask.contiguous()),
    )


def safe_ratio(numerator: Tensor, denominator: Tensor) -> Tensor:
    return numerator.float() / denominator.clamp_min(1).float()


def correction_summary(
    soft_labels: Tensor,
    clean_labels: Tensor,
    initial_noisy_labels: Tensor,
    noise_mask: Tensor,
    *,
    epoch: int,
    split: str,
    action_count: int = 0,
    action_rate: float = 0.0,
    reward: float | None = None,
    actor_loss: float | None = None,
    critic_loss: float | None = None,
    elapsed_seconds: float = 0.0,
) -> dict[str, object]:
    sample_count = clean_labels.numel()
    if soft_labels.shape != (sample_count, NUM_CLASSES):
        raise ValueError("soft_labels must have shape [N, NUM_CLASSES].")
    if (
        initial_noisy_labels.shape != clean_labels.shape
        or noise_mask.shape != clean_labels.shape
    ):
        raise ValueError("Metric label tensors must have equal [N] shapes.")

    hard_labels = soft_labels.argmax(dim=1)
    flat_pairs = clean_labels * NUM_CLASSES + hard_labels
    confusion = torch.bincount(
        flat_pairs,
        minlength=NUM_CLASSES * NUM_CLASSES,
    ).reshape(NUM_CLASSES, NUM_CLASSES)
    true_positive = confusion.diag()
    support = confusion.sum(dim=1)
    predicted = confusion.sum(dim=0)
    recall = safe_ratio(true_positive, support)
    precision = safe_ratio(true_positive, predicted)
    f1 = 2.0 * precision * recall / (precision + recall).clamp_min(1e-12)

    clean_probabilities = soft_labels.float().gather(
        1,
        clean_labels.unsqueeze(1),
    ).squeeze(1)
    loss = -clean_probabilities.clamp_min(1e-8).log().mean()

    changed = hard_labels.ne(initial_noisy_labels)
    correct_correction = changed & hard_labels.eq(clean_labels)
    incorrect_correction = changed & hard_labels.ne(clean_labels)
    clean_mask = ~noise_mask
    false_correction = clean_mask & changed
    recovered = noise_mask & hard_labels.eq(clean_labels)
    clean_preserved = clean_mask & hard_labels.eq(clean_labels)

    correction_count = changed.sum()
    noise_count = noise_mask.sum()
    clean_count = clean_mask.sum()
    return {
        "epoch": epoch,
        "split": split,
        "samples": sample_count,
        "loss": float(loss),
        "accuracy": float(true_positive.sum().float() / sample_count),
        "balanced_accuracy": float(recall.mean()),
        "macro_recall": float(recall.mean()),
        "macro_precision": float(precision.mean()),
        "macro_f1": float(f1.mean()),
        "noise_count": int(noise_count),
        "noise_rate": float(noise_count.float() / sample_count),
        "correction_count": int(correction_count),
        "correction_rate": float(correction_count.float() / sample_count),
        "correct_correction_count": int(correct_correction.sum()),
        "correction_precision": float(
            safe_ratio(correct_correction.sum(), correction_count)
        ),
        "incorrect_correction_count": int(incorrect_correction.sum()),
        "false_correction_count": int(false_correction.sum()),
        "false_correction_rate": float(
            safe_ratio(false_correction.sum(), clean_count)
        ),
        "noisy_recovery_count": int(recovered.sum()),
        "noisy_recovery_rate": float(safe_ratio(recovered.sum(), noise_count)),
        "clean_preservation_rate": float(
            safe_ratio(clean_preserved.sum(), clean_count)
        ),
        "action_count": action_count,
        "action_rate": action_rate,
        "reward": reward,
        "actor_loss": actor_loss,
        "critic_loss": critic_loss,
        "elapsed_seconds": elapsed_seconds,
    }


def correction_per_class(
    soft_labels: Tensor,
    clean_labels: Tensor,
    initial_noisy_labels: Tensor,
    noise_mask: Tensor,
    *,
    split: str,
) -> list[dict[str, object]]:
    hard_labels = soft_labels.argmax(dim=1)
    rows: list[dict[str, object]] = []
    for class_id in CLASS_IDS:
        class_mask = clean_labels.eq(class_id)
        predicted_mask = hard_labels.eq(class_id)
        class_noise = class_mask & noise_mask
        class_clean = class_mask & ~noise_mask
        changed = class_mask & hard_labels.ne(initial_noisy_labels)
        correct_correction = changed & hard_labels.eq(clean_labels)
        incorrect_correction = changed & hard_labels.ne(clean_labels)
        false_correction = class_clean & hard_labels.ne(initial_noisy_labels)
        recovered = class_noise & hard_labels.eq(clean_labels)
        preserved = class_clean & hard_labels.eq(clean_labels)

        support = class_mask.sum()
        true_positive = (class_mask & predicted_mask).sum()
        predicted_count = predicted_mask.sum()
        recall = safe_ratio(true_positive, support)
        precision = safe_ratio(true_positive, predicted_count)
        f1 = 2.0 * precision * recall / (precision + recall).clamp_min(1e-12)
        class_loss = -soft_labels[class_mask, class_id].float().clamp_min(
            1e-8
        ).log().mean()
        correction_count = changed.sum()
        noise_count = class_noise.sum()
        clean_count = class_clean.sum()
        rows.append(
            {
                "split": split,
                "class_id": class_id,
                "support": int(support),
                "loss": float(class_loss),
                "accuracy": float(recall),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "noise_count": int(noise_count),
                "noise_rate": float(safe_ratio(noise_count, support)),
                "correction_count": int(correction_count),
                "correction_rate": float(
                    safe_ratio(correction_count, support)
                ),
                "correct_correction_count": int(correct_correction.sum()),
                "correction_precision": float(
                    safe_ratio(correct_correction.sum(), correction_count)
                ),
                "incorrect_correction_count": int(incorrect_correction.sum()),
                "false_correction_count": int(false_correction.sum()),
                "false_correction_rate": float(
                    safe_ratio(false_correction.sum(), clean_count)
                ),
                "noisy_recovery_count": int(recovered.sum()),
                "noisy_recovery_rate": float(
                    safe_ratio(recovered.sum(), noise_count)
                ),
                "clean_preservation_rate": float(
                    safe_ratio(preserved.sum(), clean_count)
                ),
            }
        )
    return rows


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


def append_csv(
    path: Path,
    rows: list[dict[str, object]],
    fieldnames: tuple[str, ...],
) -> None:
    needs_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if needs_header:
            writer.writeheader()
        writer.writerows(rows)


def append_check_event(
    path: Path,
    event: str,
    **values: object,
) -> None:
    """Append one machine-readable diagnostic event to ``check.log``."""
    def json_safe(value: object) -> object:
        if isinstance(value, float):
            return value if math.isfinite(value) else str(value)
        if isinstance(value, dict):
            return {str(key): json_safe(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [json_safe(item) for item in value]
        return value

    record = json_safe({"event": event, **values})
    with path.open("a", encoding="utf-8", buffering=1) as handle:
        handle.write(
            json.dumps(
                record,
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
            )
        )
        handle.write("\n")


@torch.no_grad()
def tensor_distribution(tensor: Tensor) -> dict[str, float | int]:
    """Return compact finite-value statistics without retaining GPU tensors."""
    values = tensor.detach().float()
    if values.numel() == 0:
        return {
            "count": 0,
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
            "l2_norm": 0.0,
            "max_abs": 0.0,
            "finite_fraction": 1.0,
            "nonzero_fraction": 0.0,
        }
    finite = torch.isfinite(values)
    safe = torch.where(finite, values, torch.zeros_like(values))
    metrics = torch.stack(
        (
            safe.mean(),
            safe.std(unbiased=False),
            safe.min(),
            safe.max(),
            safe.norm(),
            safe.abs().max(),
            finite.float().mean(),
            safe.ne(0).float().mean(),
        )
    ).cpu().tolist()
    return {
        "count": values.numel(),
        "mean": metrics[0],
        "std": metrics[1],
        "min": metrics[2],
        "max": metrics[3],
        "l2_norm": metrics[4],
        "max_abs": metrics[5],
        "finite_fraction": metrics[6],
        "nonzero_fraction": metrics[7],
    }


@torch.no_grad()
def snapshot_parameters(model: nn.Module) -> dict[str, Tensor]:
    return {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }


@torch.no_grad()
def snapshot_batchnorm_buffers(model: nn.Module) -> dict[str, Tensor]:
    snapshots: dict[str, Tensor] = {}
    for module_name, module in model.named_modules():
        if not isinstance(module, nn.modules.batchnorm._BatchNorm):
            continue
        prefix = f"{module_name}." if module_name else ""
        for buffer_name in ("running_mean", "running_var", "num_batches_tracked"):
            buffer = getattr(module, buffer_name, None)
            if buffer is not None:
                snapshots[f"{prefix}{buffer_name}"] = buffer.detach().clone()
    return snapshots


def parameter_group(name: str) -> str:
    return name.split(".", maxsplit=1)[0]


@torch.no_grad()
def parameter_update_diagnostics(
    model: nn.Module,
    optimizer: Optimizer,
    before: dict[str, Tensor],
    batchnorm_before: dict[str, Tensor],
) -> dict[str, object]:
    """Summarize gradients and the actual optimizer parameter displacement."""
    batchnorm_parameter_ids = {
        id(parameter)
        for module in model.modules()
        if isinstance(module, nn.modules.batchnorm._BatchNorm)
        for parameter in module.parameters(recurse=False)
    }
    rows: list[Tensor] = []
    metadata: list[tuple[str, bool, int, bool]] = []
    for name, parameter in model.named_parameters():
        if name not in before:
            continue
        previous = before[name].float()
        current = parameter.detach().float()
        delta = current - previous
        gradient = parameter.grad
        if gradient is None:
            gradient_square = current.new_zeros(())
            gradient_max = current.new_zeros(())
            gradient_finite = current.new_ones(())
        else:
            gradient_float = gradient.detach().float()
            gradient_square = gradient_float.square().sum()
            gradient_max = gradient_float.abs().max()
            gradient_finite = torch.isfinite(gradient_float).all().float()
        rows.append(
            torch.stack(
                (
                    previous.square().sum(),
                    current.square().sum(),
                    delta.square().sum(),
                    (previous * current).sum(),
                    gradient_square,
                    gradient_max,
                    delta.abs().max(),
                    gradient_finite,
                )
            )
        )
        metadata.append(
            (
                parameter_group(name),
                id(parameter) in batchnorm_parameter_ids,
                parameter.numel(),
                gradient is not None,
            )
        )

    if not rows:
        raise RuntimeError("No trainable parameters were available for diagnostics.")
    values = torch.stack(rows).cpu().tolist()

    def aggregate(indices: list[int]) -> dict[str, float | int | bool]:
        before_sq = sum(values[index][0] for index in indices)
        after_sq = sum(values[index][1] for index in indices)
        delta_sq = sum(values[index][2] for index in indices)
        dot = sum(values[index][3] for index in indices)
        gradient_sq = sum(values[index][4] for index in indices)
        before_norm = math.sqrt(max(0.0, before_sq))
        after_norm = math.sqrt(max(0.0, after_sq))
        delta_norm = math.sqrt(max(0.0, delta_sq))
        gradient_norm = math.sqrt(max(0.0, gradient_sq))
        return {
            "parameter_tensors": len(indices),
            "parameter_elements": sum(metadata[index][2] for index in indices),
            "parameters_with_gradient": sum(
                int(metadata[index][3]) for index in indices
            ),
            "parameter_norm_before": before_norm,
            "parameter_norm_after": after_norm,
            "parameter_delta_norm": delta_norm,
            "relative_parameter_delta": delta_norm / max(before_norm, 1e-12),
            "parameter_cosine_before_after": dot
            / max(before_norm * after_norm, 1e-12),
            "gradient_norm": gradient_norm,
            "gradient_max_abs": max(values[index][5] for index in indices),
            "parameter_delta_max_abs": max(
                values[index][6] for index in indices
            ),
            "gradient_all_finite": all(
                values[index][7] == 1.0 for index in indices
            ),
            "delta_to_gradient_ratio": delta_norm
            / max(gradient_norm, 1e-12),
        }

    all_indices = list(range(len(metadata)))
    groups = sorted({item[0] for item in metadata})
    group_diagnostics = {
        group: aggregate(
            [index for index, item in enumerate(metadata) if item[0] == group]
        )
        for group in groups
    }
    batchnorm_indices = [
        index for index, item in enumerate(metadata) if item[1]
    ]

    buffer_delta_sq = 0.0
    buffer_delta_max = 0.0
    buffer_changed = 0
    for name, module in model.named_modules():
        if not isinstance(module, nn.modules.batchnorm._BatchNorm):
            continue
        prefix = f"{name}." if name else ""
        for buffer_name in ("running_mean", "running_var", "num_batches_tracked"):
            current = getattr(module, buffer_name, None)
            key = f"{prefix}{buffer_name}"
            if current is None or key not in batchnorm_before:
                continue
            delta = current.detach().float() - batchnorm_before[key].float()
            delta_square = float(delta.square().sum())
            delta_max = float(delta.abs().max()) if delta.numel() else 0.0
            buffer_delta_sq += delta_square
            buffer_delta_max = max(buffer_delta_max, delta_max)
            buffer_changed += int(delta_max > 0.0)

    momentum_square = 0.0
    momentum_max = 0.0
    momentum_tensors = 0
    for state in optimizer.state.values():
        momentum = state.get("momentum_buffer")
        if momentum is None:
            continue
        momentum_float = momentum.detach().float()
        momentum_square += float(momentum_float.square().sum())
        momentum_max = max(momentum_max, float(momentum_float.abs().max()))
        momentum_tensors += 1

    return {
        "overall": aggregate(all_indices),
        "groups": group_diagnostics,
        "batchnorm_affine": (
            aggregate(batchnorm_indices) if batchnorm_indices else None
        ),
        "batchnorm_buffers": {
            "buffer_count": len(batchnorm_before),
            "changed_buffer_count": buffer_changed,
            "delta_norm": math.sqrt(max(0.0, buffer_delta_sq)),
            "delta_max_abs": buffer_delta_max,
        },
        "momentum": {
            "buffer_count": momentum_tensors,
            "l2_norm": math.sqrt(max(0.0, momentum_square)),
            "max_abs": momentum_max,
        },
    }


def preprocess(
    images: Tensor,
    device: torch.device,
    mean: Tensor,
    std: Tensor,
) -> Tensor:
    if PREPROCESS_FUNCTION is None:
        raise RuntimeError(
            "PREPROCESS_FUNCTION must be configured by cifar_rl.py before "
            "starting the experiment."
        )
    return PREPROCESS_FUNCTION(images, device, mean, std)


def create_experiment_model() -> nn.Module:
    if MODEL_FACTORY is None:
        raise RuntimeError(
            "MODEL_FACTORY must be configured by cifar_rl.py before starting "
            "the experiment."
        )
    return MODEL_FACTORY(PRETRAINED, NUM_CLASSES)


def encode(model: nn.Module, images: Tensor) -> Tensor:
    feature_map = model.forward_features(images)
    embeddings = model.forward_head(feature_map, pre_logits=True)
    if embeddings.ndim != 2:
        raise RuntimeError(f"Expected [B, D] embeddings, got {embeddings.shape}.")
    return embeddings


def _build_warmup_optimizer(model: nn.Module) -> SGD:
    return SGD(
        model.parameters(),
        lr=WARMUP_LR,
        momentum=WARMUP_MOMENTUM,
        weight_decay=WARMUP_WEIGHT_DECAY,
    )


def _build_halfway_scheduler(
    optimizer: Optimizer,
    epochs: int,
    decay_fraction: float,
    decay_factor: float,
) -> MultiStepLR:
    milestone = max(1, math.ceil(epochs * decay_fraction))
    return MultiStepLR(
        optimizer,
        milestones=[milestone],
        gamma=decay_factor,
    )


def _classification_metrics(confusion_matrix: Tensor) -> tuple[float, float]:
    matrix = confusion_matrix.to(torch.float64)
    true_positives = matrix.diag()
    actual = matrix.sum(dim=1)
    predicted = matrix.sum(dim=0)
    recall = true_positives / actual.clamp_min(1)
    precision = true_positives / predicted.clamp_min(1)
    f1_denominator = precision + recall
    per_class_f1 = torch.where(
        f1_denominator > 0,
        2.0 * precision * recall / f1_denominator,
        torch.zeros_like(f1_denominator),
    )
    accuracy = true_positives.sum() / matrix.sum().clamp_min(1)
    return float(accuracy), float(per_class_f1.mean())


def _update_classification_confusion(
    confusion_matrix: Tensor,
    predictions: Tensor,
    targets: Tensor,
) -> None:
    flat_indices = targets.to(torch.long) * NUM_CLASSES + predictions.to(torch.long)
    confusion_matrix += torch.bincount(
        flat_indices,
        minlength=NUM_CLASSES * NUM_CLASSES,
    ).reshape(NUM_CLASSES, NUM_CLASSES)


@torch.inference_mode()
def evaluate_warmup_model(
    model: nn.Module,
    raw_images: Tensor,
    noisy_labels_cpu: Tensor,
    clean_labels_cpu: Tensor,
    device: torch.device,
    mean: Tensor,
    std: Tensor,
    criterion: nn.CrossEntropyLoss,
) -> dict[str, float]:
    model.eval()
    sample_count = raw_images.size(0)
    loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    noisy_confusion = torch.zeros(
        (NUM_CLASSES, NUM_CLASSES),
        dtype=torch.long,
        device=device,
    )
    clean_confusion = torch.zeros_like(noisy_confusion)
    for start in range(0, sample_count, WARMUP_EVAL_BATCH_SIZE):
        end = min(start + WARMUP_EVAL_BATCH_SIZE, sample_count)
        images = preprocess(raw_images[start:end], device, mean, std)
        noisy_targets = noisy_labels_cpu[start:end].to(
            device=device,
            non_blocking=True,
        )
        clean_targets = clean_labels_cpu[start:end].to(
            device=device,
            non_blocking=True,
        )
        with torch.autocast(
            device_type="cuda",
            dtype=AMP_DTYPE,
            enabled=USE_AMP,
        ):
            logits = model(images)
            loss = criterion(logits, noisy_targets)
        predictions = logits.argmax(dim=1)
        loss_sum += loss.to(torch.float64) * (end - start)
        _update_classification_confusion(
            noisy_confusion,
            predictions,
            noisy_targets,
        )
        _update_classification_confusion(
            clean_confusion,
            predictions,
            clean_targets,
        )

    noisy_accuracy, _ = _classification_metrics(noisy_confusion)
    clean_accuracy, clean_macro_f1 = _classification_metrics(clean_confusion)
    return {
        "loss": float(loss_sum / sample_count),
        "noisy_accuracy": noisy_accuracy,
        "clean_accuracy": clean_accuracy,
        "clean_macro_f1": clean_macro_f1,
    }


def _save_warmup_checkpoint(
    model: nn.Module,
    path: Path,
    *,
    epoch: int,
    noisy_validation_accuracy: float,
    clean_validation_accuracy: float,
) -> None:
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    try:
        torch.save(
            {
                "epoch": epoch,
                "model": model.state_dict(),
                "noisy_validation_accuracy": noisy_validation_accuracy,
                "clean_validation_accuracy": clean_validation_accuracy,
                "noise_rate": NOISE_RATE,
                "pretrained": PRETRAINED,
                "warmup_model_id": WARMUP_MODEL_ID,
                "selection": "best",
                "best_epoch": epoch,
                "best_noisy_validation_accuracy": noisy_validation_accuracy,
                "best_clean_validation_accuracy": clean_validation_accuracy,
            },
            temporary_path,
        )
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)


def load_warmup_checkpoint(
    model: nn.Module,
    checkpoint_path: Path,
    device: torch.device,
) -> dict[str, object]:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Warmup checkpoint not found: {checkpoint_path}"
        )
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )
    if not isinstance(checkpoint, dict):
        raise TypeError("Warmup checkpoint must contain a dictionary.")
    required = {
        "epoch",
        "model",
        "noisy_validation_accuracy",
        "clean_validation_accuracy",
        "noise_rate",
        "pretrained",
    }
    missing = required.difference(checkpoint)
    if missing:
        raise KeyError(
            f"Warmup checkpoint is missing fields: {sorted(missing)}"
        )
    if not math.isclose(float(checkpoint["noise_rate"]), NOISE_RATE):
        raise ValueError("Warmup checkpoint noise rate does not match this run.")
    if bool(checkpoint["pretrained"]) != PRETRAINED:
        raise ValueError(
            "Warmup checkpoint pretrained setting does not match this run."
        )
    checkpoint_model_id = str(checkpoint.get("warmup_model_id", ""))
    if WARMUP_MODEL_ID and checkpoint_model_id != WARMUP_MODEL_ID:
        raise ValueError(
            "Warmup model ID does not match this run: "
            f"{checkpoint_model_id!r} != {WARMUP_MODEL_ID!r}."
        )
    deployment_mode = str(checkpoint.get("selection", "best"))
    if deployment_mode != "best":
        raise ValueError("CIFAR warmup checkpoint must use best selection.")
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(
        device=device,
        memory_format=(
            torch.channels_last
            if USE_CHANNELS_LAST
            else torch.contiguous_format
        ),
    )
    model.eval()
    return {
        "best_epoch": int(checkpoint.get("best_epoch", checkpoint["epoch"])),
        "best_noisy_validation_accuracy": float(
            checkpoint.get(
                "best_noisy_validation_accuracy",
                checkpoint["noisy_validation_accuracy"],
            )
        ),
        "best_clean_validation_accuracy": float(
            checkpoint.get(
                "best_clean_validation_accuracy",
                checkpoint["clean_validation_accuracy"],
            )
        ),
        "deployment_mode": deployment_mode,
        "deployment_epoch": int(checkpoint["epoch"]),
    }


def _rl_selection_key(
    validation_summary: dict[str, object],
) -> tuple[float, float, float]:
    """Rank RL checkpoints by macro F1, balanced accuracy, then loss."""
    macro_f1 = float(validation_summary["macro_f1"])
    balanced_accuracy = float(validation_summary["balanced_accuracy"])
    loss = float(validation_summary["loss"])
    metrics = (macro_f1, balanced_accuracy, loss)
    if not all(math.isfinite(value) for value in metrics):
        raise ValueError("RL validation metrics must be finite.")
    return macro_f1, balanced_accuracy, -loss


def _save_rl_checkpoints(
    actor_path: Path,
    critic_path: Path,
    *,
    epoch: int,
    model: nn.Module,
    critic: nn.Module,
    validation_summary: dict[str, object],
) -> None:
    """Save matching actor and critic states for one RL epoch."""
    actor_path.parent.mkdir(parents=True, exist_ok=True)
    critic_path.parent.mkdir(parents=True, exist_ok=True)
    validation_metrics = {
        "accuracy": float(validation_summary["accuracy"]),
        "balanced_accuracy": float(validation_summary["balanced_accuracy"]),
        "macro_f1": float(validation_summary["macro_f1"]),
        "loss": float(validation_summary["loss"]),
    }
    actor_payload = {
        "epoch": epoch,
        "model_name": MODEL_NAME,
        "num_classes": NUM_CLASSES,
        "validation": validation_metrics,
        "model": model.state_dict(),
    }
    critic_payload = {
        "epoch": epoch,
        "num_bins": CRITIC_NUM_BINS,
        "validation": validation_metrics,
        "critic": critic.state_dict(),
    }
    for path, payload in (
        (actor_path, actor_payload),
        (critic_path, critic_payload),
    ):
        temporary_path = path.with_suffix(f"{path.suffix}.tmp")
        try:
            torch.save(payload, temporary_path)
            temporary_path.replace(path)
        finally:
            temporary_path.unlink(missing_ok=True)


def _restore_actor(
    model: nn.Module,
    checkpoint_path: Path,
    device: torch.device,
) -> dict[str, object]:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Actor checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )
    if not isinstance(checkpoint, dict):
        raise TypeError("Actor checkpoint must contain a dictionary.")
    required = {"epoch", "model_name", "num_classes", "validation", "model"}
    missing = required.difference(checkpoint)
    if missing:
        raise KeyError(f"Actor checkpoint is missing fields: {sorted(missing)}")
    if checkpoint["model_name"] != MODEL_NAME:
        raise ValueError("Actor checkpoint model name does not match this run.")
    if checkpoint["num_classes"] != NUM_CLASSES:
        raise ValueError("Actor checkpoint class count does not match this run.")
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(
        device=device,
        memory_format=(
            torch.channels_last
            if USE_CHANNELS_LAST
            else torch.contiguous_format
        ),
    )
    model.eval()
    return checkpoint


def train_supervised_warmup(
    model: nn.Module,
    train_images: Tensor,
    train_noisy_labels_cpu: Tensor,
    validation_images: Tensor,
    validation_noisy_labels_cpu: Tensor,
    validation_clean_labels_cpu: Tensor,
    device: torch.device,
    mean: Tensor,
    std: Tensor,
    warmup_csv_path: Path,
    checkpoint_path: Path,
) -> dict[str, object]:
    """Warm up semantic features using only the 40%-noisy labels."""
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=USE_AMP and AMP_DTYPE == torch.float16,
    )
    optimizer = _build_warmup_optimizer(model)
    scheduler = _build_halfway_scheduler(
        optimizer,
        WARMUP_EPOCHS,
        WARMUP_LR_DECAY_FRACTION,
        WARMUP_LR_DECAY_FACTOR,
    )
    history: list[dict[str, object]] = []
    best_noisy_accuracy = float("-inf")
    best_clean_accuracy = float("nan")
    best_epoch = 0

    for epoch in range(1, WARMUP_EPOCHS + 1):
        epoch_started = time.perf_counter()
        model.train()

        generator = torch.Generator().manual_seed(SEED + epoch)
        permutation = torch.randperm(
            train_images.size(0),
            generator=generator,
        )
        loss_sum = torch.zeros((), device=device, dtype=torch.float64)
        correct_count = torch.zeros((), device=device, dtype=torch.long)
        samples_seen = 0
        for start in range(0, permutation.numel(), WARMUP_BATCH_SIZE):
            end = min(start + WARMUP_BATCH_SIZE, permutation.numel())
            batch_indices = permutation[start:end]
            images = preprocess(
                train_images[batch_indices],
                device,
                mean,
                std,
            )
            targets = train_noisy_labels_cpu[batch_indices].to(
                device=device,
                non_blocking=True,
            )
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type="cuda",
                dtype=AMP_DTYPE,
                enabled=USE_AMP,
            ):
                logits = model(images)
                loss = criterion(logits, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            batch_count = end - start
            loss_sum += loss.detach().to(torch.float64) * batch_count
            correct_count += logits.argmax(dim=1).eq(targets).sum()
            samples_seen += batch_count

        validation = evaluate_warmup_model(
            model,
            validation_images,
            validation_noisy_labels_cpu,
            validation_clean_labels_cpu,
            device,
            mean,
            std,
            criterion,
        )
        row: dict[str, object] = {
            "epoch": epoch,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "train_loss": float(loss_sum / samples_seen),
            "train_noisy_accuracy": float(correct_count / samples_seen),
            "validation_loss": validation["loss"],
            "validation_noisy_accuracy": validation["noisy_accuracy"],
            "validation_clean_accuracy": validation["clean_accuracy"],
            "validation_clean_macro_f1": validation["clean_macro_f1"],
            "elapsed_seconds": time.perf_counter() - epoch_started,
        }
        history.append(row)
        write_csv(warmup_csv_path, history, WARMUP_FIELDS)
        print(
            f"[WARMUP] epoch={epoch}/{WARMUP_EPOCHS} "
            f"lr={float(row['learning_rate']):.6g} "
            f"train_loss={row['train_loss']:.4f} "
            f"train_noisy_acc={row['train_noisy_accuracy']:.4f} "
            f"val_noisy_acc={row['validation_noisy_accuracy']:.4f} "
            f"val_clean_acc={row['validation_clean_accuracy']:.4f} "
            f"val_clean_f1={row['validation_clean_macro_f1']:.4f}"
        )

        noisy_accuracy = float(validation["noisy_accuracy"])
        if noisy_accuracy > best_noisy_accuracy:
            best_noisy_accuracy = noisy_accuracy
            best_clean_accuracy = float(validation["clean_accuracy"])
            best_epoch = epoch
            _save_warmup_checkpoint(
                model,
                checkpoint_path,
                epoch=epoch,
                noisy_validation_accuracy=best_noisy_accuracy,
                clean_validation_accuracy=best_clean_accuracy,
            )
        scheduler.step()

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device)
    model.eval()
    deployed_epoch = int(checkpoint["epoch"])
    print(
        "[WARMUP] deployment=best "
        f"deployment_epoch={deployed_epoch} best_epoch={best_epoch} "
        f"val_noisy_acc={best_noisy_accuracy:.4f} "
        f"val_clean_acc={best_clean_accuracy:.4f}"
    )
    if best_noisy_accuracy < WARMUP_MIN_NOISY_VALIDATION_ACCURACY:
        raise RuntimeError(
            "Warmup quality is too low to build a semantic RL reward graph: "
            f"{best_noisy_accuracy:.4f} < "
            f"{WARMUP_MIN_NOISY_VALIDATION_ACCURACY:.4f}."
        )
    return {
        "best_epoch": best_epoch,
        "best_noisy_validation_accuracy": best_noisy_accuracy,
        "best_clean_validation_accuracy": best_clean_accuracy,
        "deployment_mode": "best",
        "deployment_epoch": deployed_epoch,
    }


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
        dtype=AMP_DTYPE,
        enabled=USE_AMP,
    ):
        encode(model, inference_images)

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
        raise RuntimeError("Embedding extraction received an empty dataset.")
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


def evaluate_correction_split(
    *,
    split: str,
    epoch: int,
    model: nn.Module,
    policy: LabelCorrectionPolicy,
    raw_images: Tensor,
    clean_labels_cpu: Tensor,
    noisy_labels_cpu: Tensor,
    noise_mask_cpu: Tensor,
    device: torch.device,
    mean: Tensor,
    std: Tensor,
    timings: Timings,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    synchronize(device)
    started = time.perf_counter()
    device_index = device.index if device.index is not None else 0
    with torch.random.fork_rng(devices=[device_index]):
        torch.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        embeddings = measure(
            f"{split}_feature_extraction",
            device,
            timings,
            lambda: extract_all_embeddings(model, raw_images, device, mean, std),
        )
        neighbors = measure(
            f"{split}_exact_knn",
            device,
            timings,
            lambda: build_neighbor_indices(embeddings),
        )
        clean_labels = clean_labels_cpu.to(device, non_blocking=True)
        initial_noisy_labels = noisy_labels_cpu.to(device, non_blocking=True)
        noise_mask = noise_mask_cpu.to(device, non_blocking=True)
        label_state = F.one_hot(
            initial_noisy_labels,
            num_classes=NUM_CLASSES,
        ).to(torch.float32)
        action_count = 0
        for evaluation_step in range(1, TRAJECTORY_LENGTH + 1):
            correction = measure(
                f"{split}_correction",
                device,
                timings,
                lambda: policy.correct_all(
                    embeddings,
                    label_state,
                    neighbors,
                ),
                step=evaluation_step,
            )
            action_count += int(correction.actions.sum())
            label_state = correction.corrected_labels

    synchronize(device)
    elapsed = time.perf_counter() - started
    action_rate = action_count / (
        clean_labels.numel() * TRAJECTORY_LENGTH
    )
    summary = correction_summary(
        label_state,
        clean_labels,
        initial_noisy_labels,
        noise_mask,
        epoch=epoch,
        split=split,
        action_count=action_count,
        action_rate=action_rate,
        elapsed_seconds=elapsed,
    )
    per_class = correction_per_class(
        label_state,
        clean_labels,
        initial_noisy_labels,
        noise_mask,
        split=split,
    )
    del embeddings, neighbors, label_state
    return summary, per_class


def clean_full_training_labels(
    *,
    model: nn.Module,
    policy: LabelCorrectionPolicy,
    raw_images: Tensor,
    clean_labels_cpu: Tensor,
    noisy_labels_cpu: Tensor,
    noise_mask_cpu: Tensor,
    device: torch.device,
    mean: Tensor,
    std: Tensor,
    timings: Timings,
    corrected_labels_path: Path,
    cleaning_csv_path: Path,
    cleaning_summary_path: Path,
    cleaning_per_class_path: Path,
    checkpoint_epoch: int,
) -> dict[str, object]:
    """Deploy the selected frozen actor for T' steps over the full train split."""
    if CLEANING_TRAJECTORY_LENGTH <= 0:
        raise ValueError("CLEANING_TRAJECTORY_LENGTH must be positive.")
    synchronize(device)
    started = time.perf_counter()
    device_index = device.index if device.index is not None else 0
    with torch.random.fork_rng(devices=[device_index]):
        torch.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        embeddings = measure(
            "cleaning_feature_extraction",
            device,
            timings,
            lambda: extract_all_embeddings(model, raw_images, device, mean, std),
        )
        neighbors = measure(
            "cleaning_exact_knn",
            device,
            timings,
            lambda: build_neighbor_indices(embeddings),
        )
        clean_labels = clean_labels_cpu.to(device, non_blocking=True)
        initial_noisy_labels = noisy_labels_cpu.to(device, non_blocking=True)
        noise_mask = noise_mask_cpu.to(device, non_blocking=True)
        label_state = F.one_hot(
            initial_noisy_labels,
            num_classes=NUM_CLASSES,
        ).to(torch.float32)
        history: list[dict[str, object]] = []
        total_action_count = 0

        for cleaning_step in range(1, CLEANING_TRAJECTORY_LENGTH + 1):
            correction = measure(
                "cleaning_correction",
                device,
                timings,
                lambda: policy.correct_all(
                    embeddings,
                    label_state,
                    neighbors,
                ),
                step=cleaning_step,
            )
            step_action_count = int(correction.actions.sum())
            total_action_count += step_action_count
            label_state = correction.corrected_labels
            hard_labels = label_state.argmax(dim=1)
            cumulative_changed_count = int(
                hard_labels.ne(initial_noisy_labels).sum()
            )
            row = {
                "step": cleaning_step,
                "action_count": step_action_count,
                "action_rate": step_action_count / EXPECTED_SAMPLES,
                "cumulative_changed_count": cumulative_changed_count,
                "cumulative_changed_rate": (
                    cumulative_changed_count / EXPECTED_SAMPLES
                ),
                "clean_accuracy": float(
                    hard_labels.eq(clean_labels).float().mean()
                ),
            }
            history.append(row)
            print(
                f"[CLEAN] step={cleaning_step}/"
                f"{CLEANING_TRAJECTORY_LENGTH} "
                f"action={float(row['action_rate']):.4f} "
                f"changed={float(row['cumulative_changed_rate']):.4f} "
                f"clean_accuracy={float(row['clean_accuracy']):.4f}"
            )

    synchronize(device)
    elapsed = time.perf_counter() - started
    summary = correction_summary(
        label_state,
        clean_labels,
        initial_noisy_labels,
        noise_mask,
        epoch=checkpoint_epoch,
        split="train_cleaning",
        action_count=total_action_count,
        action_rate=(
            total_action_count
            / (EXPECTED_SAMPLES * CLEANING_TRAJECTORY_LENGTH)
        ),
        elapsed_seconds=elapsed,
    )
    per_class = correction_per_class(
        label_state,
        clean_labels,
        initial_noisy_labels,
        noise_mask,
        split="train_cleaning",
    )
    corrected_array = (
        label_state.argmax(dim=1).detach().cpu().numpy().astype(np.int64)
    )
    corrected_labels_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = corrected_labels_path.with_suffix(
        f"{corrected_labels_path.suffix}.tmp"
    )
    try:
        with temporary_path.open("wb") as handle:
            np.save(handle, corrected_array, allow_pickle=False)
        temporary_path.replace(corrected_labels_path)
    finally:
        temporary_path.unlink(missing_ok=True)
    write_csv(cleaning_csv_path, history, CLEANING_FIELDS)
    write_csv(cleaning_summary_path, [summary], SUMMARY_FIELDS)
    write_csv(cleaning_per_class_path, per_class, PER_CLASS_FIELDS)
    print(f"[CLEAN] corrected_labels={corrected_labels_path}")
    del embeddings, neighbors, label_state
    return summary


def select_policy_queries(sample_count: int, step: int) -> Tensor:
    if POLICY_UPDATE_SAMPLES == sample_count:
        return torch.arange(sample_count)
    generator = torch.Generator().manual_seed(SEED + step)
    selected = torch.randperm(sample_count, generator=generator)[
        :POLICY_UPDATE_SAMPLES
    ]
    return selected.sort().values


def compute_policy_embedding_gradient(
    policy: LabelCorrectionPolicy,
    cached_embeddings: Tensor,
    label_state: Tensor,
    neighbor_indices: Tensor,
    actions: Tensor,
    q_value: Tensor,
    query_indices_cpu: Tensor,
    device: torch.device,
) -> tuple[Tensor, float, dict[str, object] | None]:
    """Differentiate policy loss on queries using the full cached KNN graph.

    KNN indices are discrete and fixed within the current RL step. Query and
    neighbor embedding gradients are accumulated here; the following backbone
    VJP restricts parameter backpropagation to the selected update set.
    """
    sample_count = cached_embeddings.size(0)
    if cached_embeddings.ndim != 2 or sample_count != EXPECTED_SAMPLES:
        raise ValueError("cached_embeddings must have shape [N, D].")
    if neighbor_indices.size(0) != sample_count:
        raise ValueError("neighbor_indices must cover all N samples.")
    if query_indices_cpu.ndim != 1 or query_indices_cpu.numel() == 0:
        raise ValueError("query_indices_cpu must be a non-empty vector.")
    if query_indices_cpu.device.type != "cpu":
        raise ValueError("query_indices_cpu must remain on CPU.")

    # Embeddings extracted under inference_mode cannot directly participate in
    # autograd. Cloning creates a regular leaf while preserving their values.
    embedding_leaf = cached_embeddings.detach().clone().requires_grad_(True)
    query_indices = query_indices_cpu.to(device=device, non_blocking=True)
    query_count = query_indices.numel()
    total_loss = torch.zeros((), device=device)
    total_batches = math.ceil(query_count / POLICY_UPDATE_BATCH_SIZE)

    for batch_number, start in enumerate(
        range(0, query_count, POLICY_UPDATE_BATCH_SIZE),
        start=1,
    ):
        end = min(start + POLICY_UPDATE_BATCH_SIZE, query_count)
        batch_indices = query_indices[start:end]
        batch_neighbors = neighbor_indices[batch_indices]
        policy_step = policy(
            embedding_leaf[batch_indices],
            embedding_leaf[batch_neighbors],
            label_state[batch_indices],
            label_state[batch_neighbors],
            actions=actions[batch_indices],
        )
        loss = -(
            q_value.detach()
            * policy_step.log_probabilities.sum()
            / query_count
        )
        loss.backward()
        total_loss += loss.detach()
        if batch_number % 100 == 0 or batch_number == total_batches:
            print(
                f"[ACTOR POLICY GRAD] batch={batch_number}/{total_batches} "
                f"queries={end}/{query_count}"
            )

    embedding_gradient = embedding_leaf.grad
    if embedding_gradient is None:
        raise RuntimeError("Policy loss did not produce an embedding gradient.")
    selected_embedding_gradient = (
        embedding_gradient[query_indices].detach().clone()
    )
    diagnostics: dict[str, object] | None = None
    if DIAGNOSTICS_ENABLED:
        full_distribution = tensor_distribution(embedding_gradient)
        selected_distribution = tensor_distribution(selected_embedding_gradient)
        full_norm = float(full_distribution["l2_norm"])
        selected_norm = float(selected_distribution["l2_norm"])
        outside_norm = math.sqrt(
            max(0.0, full_norm * full_norm - selected_norm * selected_norm)
        )
        diagnostics = {
            "full_graph_gradient": full_distribution,
            "selected_vjp_gradient": selected_distribution,
            "full_graph_row_norm": tensor_distribution(
                embedding_gradient.float().norm(dim=1)
            ),
            "selected_vjp_row_norm": tensor_distribution(
                selected_embedding_gradient.float().norm(dim=1)
            ),
            "outside_selected_gradient_norm": outside_norm,
            "selected_gradient_norm_fraction": selected_norm
            / max(full_norm, 1e-12),
        }
    del embedding_leaf
    return selected_embedding_gradient, float(total_loss), diagnostics


def update_backbone_from_embedding_gradient(
    model: nn.Module,
    optimizer: Optimizer,
    scaler: torch.amp.GradScaler,
    raw_images: Tensor,
    embedding_gradient: Tensor,
    update_indices_cpu: Tensor,
    device: torch.device,
    mean: Tensor,
    std: Tensor,
) -> dict[str, object] | None:
    """Map selected embedding gradients through the actor backbone.

    The cached policy graph may use every dataset sample, but this VJP sends
    gradients through only the chosen full/subset actor-update set.
    """
    model.eval()
    parameter_before = (
        snapshot_parameters(model) if DIAGNOSTICS_ENABLED else None
    )
    batchnorm_before = (
        snapshot_batchnorm_buffers(model) if DIAGNOSTICS_ENABLED else None
    )
    optimizer.zero_grad(set_to_none=True)
    sample_count = raw_images.size(0)
    if update_indices_cpu.ndim != 1 or update_indices_cpu.numel() == 0:
        raise ValueError("update_indices_cpu must be a non-empty vector.")
    if update_indices_cpu.device.type != "cpu":
        raise ValueError("update_indices_cpu must remain on CPU.")
    update_count = update_indices_cpu.numel()
    if (
        embedding_gradient.ndim != 2
        or embedding_gradient.size(0) != update_count
    ):
        raise ValueError("embedding_gradient must have shape [update_count, D].")
    use_full_order = update_count == sample_count
    selected_images = (
        raw_images
        if use_full_order
        else pin_for_cuda(raw_images[update_indices_cpu])
    )
    total_batches = math.ceil(update_count / POLICY_UPDATE_BATCH_SIZE)

    for batch_number, start in enumerate(
        range(0, update_count, POLICY_UPDATE_BATCH_SIZE),
        start=1,
    ):
        end = min(start + POLICY_UPDATE_BATCH_SIZE, update_count)
        images = preprocess(selected_images[start:end], device, mean, std)
        with torch.autocast(
            device_type=device.type,
            dtype=AMP_DTYPE,
            enabled=USE_AMP,
        ):
            current_embeddings = encode(model, images)
            surrogate = (
                current_embeddings.float()
                * embedding_gradient[start:end].float()
            ).sum()
        scaler.scale(surrogate).backward()
        if batch_number % 100 == 0 or batch_number == total_batches:
            print(
                f"[ACTOR BACKBONE VJP] batch={batch_number}/{total_batches} "
                f"samples={end}/{update_count}"
            )

    if DIAGNOSTICS_ENABLED:
        scaler.unscale_(optimizer)
    scaler.step(optimizer)
    scaler.update()
    if parameter_before is None or batchnorm_before is None:
        return None
    diagnostics = parameter_update_diagnostics(
        model,
        optimizer,
        parameter_before,
        batchnorm_before,
    )
    del parameter_before, batchnorm_before
    return diagnostics


@torch.inference_mode()
def embedding_drift_diagnostics(
    model: nn.Module,
    probe_images: Tensor,
    embeddings_before: Tensor,
    device: torch.device,
    mean: Tensor,
    std: Tensor,
) -> dict[str, object]:
    """Measure representation drift on a fixed probe set after one update."""
    model.eval()
    images = preprocess(probe_images, device, mean, std)
    with torch.autocast(
        device_type=device.type,
        dtype=AMP_DTYPE,
        enabled=USE_AMP,
    ):
        embeddings_after = encode(model, images).float()
    before = embeddings_before.detach().float()
    cosine = F.cosine_similarity(before, embeddings_after, dim=1)
    before_norm = before.norm(dim=1)
    after_norm = embeddings_after.norm(dim=1)
    absolute_l2 = (embeddings_after - before).norm(dim=1)
    relative_l2 = absolute_l2 / before_norm.clamp_min(1e-12)
    return {
        "probe_samples": before.size(0),
        "cosine_similarity": tensor_distribution(cosine),
        "absolute_l2_change": tensor_distribution(absolute_l2),
        "relative_l2_change": tensor_distribution(relative_l2),
        "embedding_norm_before": tensor_distribution(before_norm),
        "embedding_norm_after": tensor_distribution(after_norm),
    }


@torch.inference_mode()
def knn_graph_diagnostics(
    embeddings: Tensor,
    neighbors: Tensor,
    clean_labels: Tensor,
    noisy_labels: Tensor,
    state_labels: Tensor,
) -> dict[str, object]:
    """Report semantic KNN purity, feature scale, and graph hubness."""
    clean_purity = clean_labels[neighbors].eq(
        clean_labels.unsqueeze(1)
    ).float().mean()
    noisy_purity = noisy_labels[neighbors].eq(
        noisy_labels.unsqueeze(1)
    ).float().mean()
    state_purity = state_labels[neighbors].eq(
        state_labels.unsqueeze(1)
    ).float().mean()
    purities = torch.stack((clean_purity, noisy_purity, state_purity)).cpu()
    indegree = torch.bincount(
        neighbors.flatten(),
        minlength=embeddings.size(0),
    ).float()
    return {
        "clean_label_purity": float(purities[0]),
        "noisy_label_purity": float(purities[1]),
        "current_state_purity": float(purities[2]),
        "embedding_values": tensor_distribution(embeddings),
        "embedding_row_norm": tensor_distribution(
            embeddings.float().norm(dim=1)
        ),
        "neighbor_indegree": tensor_distribution(indegree),
    }


def update_critic(
    critic: nn.Module,
    optimizer: torch.optim.Optimizer,
    encoding: Tensor,
    reward: Tensor,
    next_encoding: Tensor | None,
) -> tuple[float, dict[str, object] | None]:
    parameter_before = (
        snapshot_parameters(critic) if DIAGNOSTICS_ENABLED else None
    )
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
    loss = float(td.loss.detach())
    if parameter_before is None:
        return loss, None
    diagnostics = {
        "terminal": terminal,
        "current_q": float(current_q.detach()),
        "reward": float(reward.detach()),
        "next_q": float(next_q.detach()),
        "td_target": float(td.target.detach()),
        "td_error": float(td.error.detach()),
        "td_loss": loss,
        "parameter_update": parameter_update_diagnostics(
            critic,
            optimizer,
            parameter_before,
            {},
        ),
    }
    del parameter_before
    return loss, diagnostics


def build_engine_config() -> Config:
    cfg = Config()
    return replace(
        cfg,
        model=replace(
            cfg.model,
            name=MODEL_NAME,
            pretrained=PRETRAINED,
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
            epochs=RL_EPOCHS,
            trajectory_length=TRAJECTORY_LENGTH,
            initial_state_randomization_rate=(
                INITIAL_STATE_RANDOMIZATION_RATE
            ),
            discount_factor=DISCOUNT_FACTOR,
            actor_lr=ACTOR_LR,
            actor_weight_decay=ACTOR_WEIGHT_DECAY,
            critic_lr=CRITIC_LR,
            critic_momentum=CRITIC_MOMENTUM,
            critic_weight_decay=CRITIC_WEIGHT_DECAY,
            critic_num_bins=CRITIC_NUM_BINS,
            policy_update_mode=ACTOR_UPDATE_MODE,
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
    print(f"{DATASET_NAME} RLNLC experiment")
    print(f"device={device} ({properties.name})")
    print(f"device_memory_gib={properties.total_memory / 1024**3:.2f}")
    print(f"samples={clean_labels.numel()} classes={CLASS_IDS}")
    print(f"clean_class_counts={class_counts(clean_labels)}")
    print(f"noisy_class_counts={class_counts(noisy_labels)}")
    print(
        f"noise_type=stratified_symmetric noise_rate={NOISE_RATE:.2f} "
        f"noise_seed={SEED} corrupted={int(noise_mask.sum())}"
    )
    print(
        f"rl_epochs={RL_EPOCHS} trajectory_length={TRAJECTORY_LENGTH} "
        "initial_state_randomization_rate="
        f"{INITIAL_STATE_RANDOMIZATION_RATE:.2f}"
    )
    print(
        f"actor_update_mode={ACTOR_UPDATE_MODE} "
        f"policy_update_samples={POLICY_UPDATE_SAMPLES} "
        f"policy_update_batch={POLICY_UPDATE_BATCH_SIZE}"
    )
    print(
        f"feature_batch={FEATURE_BATCH_SIZE} k={K} "
        "knn_mode=exact_chunked "
        f"query_chunk={KNN_QUERY_CHUNK_SIZE} "
        f"reference_chunk={KNN_REFERENCE_CHUNK_SIZE}"
    )
    print(
        f"correction_mode=full_{clean_labels.numel()} "
        f"correction_chunk={CORRECTION_CHUNK_SIZE} "
        f"amp={USE_AMP} amp_dtype={AMP_DTYPE} "
        f"channels_last={USE_CHANNELS_LAST}"
    )
    print(
        f"actor=SGD(lr={ACTOR_LR}, "
        f"momentum={ACTOR_MOMENTUM}, wd={ACTOR_WEIGHT_DECAY}) "
        f"critic=SGD(lr={CRITIC_LR}, momentum={CRITIC_MOMENTUM})"
    )
    print(
        f"rl_diagnostics={DIAGNOSTICS_ENABLED} "
        f"diagnostic_probe_size={DIAGNOSTIC_PROBE_SIZE} "
        f"check_log={OUTPUT_DIR / CHECK_LOG_FILENAME}"
    )
    print(
        f"pretrained={PRETRAINED} warmup_epochs={WARMUP_EPOCHS} "
        f"warmup_model_id={WARMUP_MODEL_ID or 'unset'} "
        f"warmup_batch={WARMUP_BATCH_SIZE} "
        "warmup_min_noisy_val_acc="
        f"{WARMUP_MIN_NOISY_VALIDATION_ACCURACY}"
    )
    print(
        "warmup_optimizer=SGD warmup_scheduler=step_halfway "
        "warmup_deploy_checkpoint=best rl_deploy_checkpoint=last"
    )
    print(
        "warmup_best_metric=noisy_validation_accuracy "
        "warmup_deployment=best "
        "clean_labels=reporting_only"
    )


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


def build_timing_rows(timings: Timings) -> list[dict[str, object]]:
    measured_total = sum(sum(values) for values in timings.values())
    return [
        {
            "stage": name,
            "calls": len(values),
            "total_seconds": sum(values),
            "mean_seconds": sum(values) / len(values),
            "percentage": 100.0 * sum(values) / measured_total,
        }
        for name, values in timings.items()
    ]


def main() -> None:
    if RL_EPOCHS <= 0:
        raise ValueError("RL_EPOCHS must be positive.")
    if EXTERNAL_WARMUP_CHECKPOINT_PATH is None:
        raise RuntimeError(
            "CIFAR RL requires the warmup checkpoint configured in "
            "config_resent18.py. Run cifar_warmup.py first."
        )
    if not 0 < POLICY_UPDATE_SAMPLES <= EXPECTED_SAMPLES:
        raise ValueError(
            "POLICY_UPDATE_SAMPLES must be in [1, EXPECTED_SAMPLES]."
        )
    if POLICY_UPDATE_BATCH_SIZE > POLICY_UPDATE_SAMPLES:
        raise ValueError(
            "POLICY_UPDATE_BATCH_SIZE cannot exceed POLICY_UPDATE_SAMPLES."
        )
    if DIAGNOSTICS_ENABLED and DIAGNOSTIC_PROBE_SIZE <= 0:
        raise ValueError("DIAGNOSTIC_PROBE_SIZE must be positive when enabled.")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    actor_best_checkpoint_path = OUTPUT_DIR / ACTOR_BEST_CHECKPOINT_FILENAME
    actor_last_checkpoint_path = OUTPUT_DIR / ACTOR_LAST_CHECKPOINT_FILENAME
    critic_best_checkpoint_path = OUTPUT_DIR / CRITIC_BEST_CHECKPOINT_FILENAME
    critic_last_checkpoint_path = OUTPUT_DIR / CRITIC_LAST_CHECKPOINT_FILENAME
    train_csv_path = OUTPUT_DIR / TRAIN_CSV_FILENAME
    test_csv_path = OUTPUT_DIR / TEST_CSV_FILENAME
    test_per_class_path = OUTPUT_DIR / TEST_PER_CLASS_CSV_FILENAME
    timing_csv_path = OUTPUT_DIR / TIMING_CSV_FILENAME
    run_summary_path = OUTPUT_DIR / RUN_SUMMARY_CSV_FILENAME
    check_log_path = OUTPUT_DIR / CHECK_LOG_FILENAME
    corrected_labels_path = (
        CORRECTED_LABELS_OUTPUT_PATH
        if CORRECTED_LABELS_OUTPUT_PATH is not None
        else OUTPUT_DIR / "train_corrected_labels.npy"
    )
    cleaning_csv_path = OUTPUT_DIR / CLEANING_CSV_FILENAME
    cleaning_summary_path = OUTPUT_DIR / CLEANING_SUMMARY_FILENAME
    cleaning_per_class_path = OUTPUT_DIR / CLEANING_PER_CLASS_FILENAME
    write_csv(train_csv_path, [], SUMMARY_FIELDS)
    if DIAGNOSTICS_ENABLED:
        check_log_path.write_text("", encoding="utf-8")

    device = resolve_local_device()
    seed_everything(SEED)
    torch.backends.cudnn.benchmark = CUDNN_BENCHMARK
    torch.cuda.reset_peak_memory_stats()
    timings: Timings = {}
    print(f"output_dir={OUTPUT_DIR}")

    raw_images, clean_labels_cpu = measure(
        f"{DATASET_STAGE_PREFIX}_load",
        device,
        timings,
        load_training_data,
    )
    evaluation_splits = measure(
        f"{DATASET_STAGE_PREFIX}_eval_load",
        device,
        timings,
        load_evaluation_data,
    )
    noisy_labels_cpu, noise_mask_cpu = measure(
        (
            "noise_artifact_load"
            if EXTERNAL_NOISY_LABELS_PATH is not None
            else "noise_injection"
        ),
        device,
        timings,
        lambda: load_noisy_label_artifacts(clean_labels_cpu),
    )
    evaluation_data: dict[str, tuple[Tensor, Tensor, Tensor, Tensor]] = {}
    for split, (split_images, split_clean_labels) in evaluation_splits.items():
        split_noisy_labels, split_noise_mask = measure(
            f"{split}_noise_injection",
            device,
            timings,
            lambda labels=split_clean_labels: inject_stratified_symmetric_noise(
                labels
            ),
        )
        evaluation_data[split] = (
            split_images,
            split_clean_labels,
            split_noisy_labels,
            split_noise_mask,
        )

    print_configuration(
        device,
        clean_labels_cpu,
        noisy_labels_cpu,
        noise_mask_cpu,
    )
    print(
        f"validation_samples={evaluation_data['val'][0].size(0)} "
        f"test_samples={evaluation_data['test'][0].size(0)}"
    )
    cfg = build_engine_config()

    model = measure(
        "model_init",
        device,
        timings,
        lambda: create_experiment_model().to(
            device=device,
            memory_format=(
                torch.channels_last
                if USE_CHANNELS_LAST
                else torch.contiguous_format
            ),
        ),
    )
    mean = torch.tensor(DATA_MEAN, device=device).reshape(1, 3, 1, 1)
    std = torch.tensor(DATA_STD, device=device).reshape(1, 3, 1, 1)
    measure(
        "kernel_warmup",
        device,
        timings,
        lambda: warm_device_kernels(model, raw_images, device, mean, std),
    )

    warmup_checkpoint_path = EXTERNAL_WARMUP_CHECKPOINT_PATH
    warmup_result = measure(
        "warmup_checkpoint_load",
        device,
        timings,
        lambda: load_warmup_checkpoint(
            model,
            warmup_checkpoint_path,
            device,
        ),
    )
    for parameter in model.parameters():
        parameter.requires_grad = True
    model.to(
        device=device,
        memory_format=(
            torch.channels_last
            if USE_CHANNELS_LAST
            else torch.contiguous_format
        ),
    )
    model.eval()

    policy = LabelCorrectionPolicy(cfg.policy).to(device)
    reward_function = RLNLCReward(cfg).to(device)
    critic = build_critic(cfg).to(device)
    critic_optimizer = build_critic_optimizer(critic, cfg)
    actor_optimizer = SGD(
        model.parameters(),
        lr=ACTOR_LR,
        momentum=ACTOR_MOMENTUM,
        weight_decay=ACTOR_WEIGHT_DECAY,
    )
    scheduler_milestone = max(
        1,
        math.ceil(RL_EPOCHS * LR_DECAY_FRACTION),
    )
    actor_scheduler = MultiStepLR(
        actor_optimizer,
        milestones=[scheduler_milestone],
        gamma=LR_DECAY_FACTOR,
    )
    critic_scheduler = MultiStepLR(
        critic_optimizer,
        milestones=[scheduler_milestone],
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

    clean_labels = clean_labels_cpu.to(device, non_blocking=True)
    initial_noisy_labels = noisy_labels_cpu.to(device, non_blocking=True)
    noise_mask = noise_mask_cpu.to(device, non_blocking=True)
    diagnostic_probe_indices_cpu: Tensor | None = None
    diagnostic_probe_indices_device: Tensor | None = None
    diagnostic_probe_images: Tensor | None = None
    if DIAGNOSTICS_ENABLED:
        probe_size = min(DIAGNOSTIC_PROBE_SIZE, EXPECTED_SAMPLES)
        diagnostic_probe_indices_cpu = (
            torch.arange(probe_size, dtype=torch.long)
            * EXPECTED_SAMPLES
            // probe_size
        )
        diagnostic_probe_indices_device = diagnostic_probe_indices_cpu.to(
            device=device,
            non_blocking=True,
        )
        diagnostic_probe_images = pin_for_cuda(
            raw_images[diagnostic_probe_indices_cpu].contiguous()
        )
        initial_actor_parameters = snapshot_parameters(model)
        initial_actor_batchnorm = snapshot_batchnorm_buffers(model)
        initial_critic_parameters = snapshot_parameters(critic)
        append_check_event(
            check_log_path,
            "run_start",
            schema_version=1,
            seed=SEED,
            actor_learning_rate=ACTOR_LR,
            actor_momentum=ACTOR_MOMENTUM,
            actor_weight_decay=ACTOR_WEIGHT_DECAY,
            critic_learning_rate=CRITIC_LR,
            critic_momentum=CRITIC_MOMENTUM,
            critic_weight_decay=CRITIC_WEIGHT_DECAY,
            discount_factor=DISCOUNT_FACTOR,
            reward_nla_weight=NLA_WEIGHT,
            update_mode=ACTOR_UPDATE_MODE,
            update_samples=POLICY_UPDATE_SAMPLES,
            update_batch_size=POLICY_UPDATE_BATCH_SIZE,
            probe_samples=probe_size,
            amp=USE_AMP,
            amp_dtype=str(AMP_DTYPE),
            model_training=model.training,
            batchnorm_training_modules=sum(
                int(module.training)
                for module in model.modules()
                if isinstance(module, nn.modules.batchnorm._BatchNorm)
            ),
            fixed_knn=knn_graph_diagnostics(
                fixed_embeddings,
                global_neighbors,
                clean_labels,
                initial_noisy_labels,
                initial_noisy_labels,
            ),
            fixed_knn_cosine=tensor_distribution(global_cosines),
            actor_initial=parameter_update_diagnostics(
                model,
                actor_optimizer,
                initial_actor_parameters,
                initial_actor_batchnorm,
            ),
            critic_initial=parameter_update_diagnostics(
                critic,
                critic_optimizer,
                initial_critic_parameters,
                {},
            ),
            cuda_allocated_gib=torch.cuda.memory_allocated(device) / 1024**3,
            cuda_reserved_gib=torch.cuda.memory_reserved(device) / 1024**3,
        )
        del (
            initial_actor_parameters,
            initial_actor_batchnorm,
            initial_critic_parameters,
        )
    epoch_seconds: list[float] = []
    validation_seconds: list[float] = []
    final_train_summary: dict[str, object] | None = None
    best_rl_epoch = 0
    best_rl_key: tuple[float, float, float] | None = None
    best_validation_summary: dict[str, object] | None = None

    val_images, val_clean, val_noisy, val_noise_mask = evaluation_data["val"]
    epoch_zero_summary, _ = evaluate_correction_split(
        split="val",
        epoch=0,
        model=model,
        policy=policy,
        raw_images=val_images,
        clean_labels_cpu=val_clean,
        noisy_labels_cpu=val_noisy,
        noise_mask_cpu=val_noise_mask,
        device=device,
        mean=mean,
        std=std,
        timings=timings,
    )
    validation_seconds.append(float(epoch_zero_summary["elapsed_seconds"]))
    append_csv(train_csv_path, [epoch_zero_summary], SUMMARY_FIELDS)
    print(
        "[RL EPOCH 0] reporting_only=True "
        f"val_acc={float(epoch_zero_summary['accuracy']):.6f} "
        f"val_macro_f1={float(epoch_zero_summary['macro_f1']):.6f}"
    )
    seed_everything(SEED)

    for epoch in range(1, RL_EPOCHS + 1):
        synchronize(device)
        epoch_started = time.perf_counter()
        label_state, randomized_count = measure(
            "initial_state_randomization",
            device,
            timings,
            lambda current_epoch=epoch: initialize_randomized_label_state(
                initial_noisy_labels,
                num_classes=NUM_CLASSES,
                randomization_rate=INITIAL_STATE_RANDOMIZATION_RATE,
                epoch=current_epoch,
            ),
            step=epoch,
        )
        print(
            f"[STATE] epoch={epoch} randomized={randomized_count}/"
            f"{EXPECTED_SAMPLES} "
            f"rate={randomized_count / EXPECTED_SAMPLES:.4f}"
        )
        previous_encoding: Tensor | None = None
        previous_reward: Tensor | None = None
        epoch_action_count = 0
        epoch_rewards: list[float] = []
        epoch_actor_losses: list[float] = []
        epoch_critic_losses: list[float] = []

        for step in range(1, TRAJECTORY_LENGTH + 1):
            global_step = (epoch - 1) * TRAJECTORY_LENGTH + step
            print(
                f"\n[TRAJECTORY] epoch={epoch}/{RL_EPOCHS} "
                f"step={step}/{TRAJECTORY_LENGTH}"
            )
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
                step=global_step,
            )
            policy_neighbors = measure(
                "exact_policy_knn",
                device,
                timings,
                lambda: build_neighbor_indices(policy_embeddings),
                step=global_step,
            )
            step_knn_diagnostics = (
                knn_graph_diagnostics(
                    policy_embeddings,
                    policy_neighbors,
                    clean_labels,
                    initial_noisy_labels,
                    label_state.argmax(dim=1),
                )
                if DIAGNOSTICS_ENABLED
                else None
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
                step=global_step,
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
                step=global_step,
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
                step=global_step,
            )
            query_indices_cpu = select_policy_queries(
                EXPECTED_SAMPLES,
                global_step,
            )
            (
                embedding_gradient,
                actor_loss,
                policy_gradient_diagnostics,
            ) = measure(
                "actor_policy_gradient",
                device,
                timings,
                lambda: compute_policy_embedding_gradient(
                    policy,
                    policy_embeddings,
                    label_state,
                    policy_neighbors,
                    correction.actions,
                    q_value,
                    query_indices_cpu,
                    device,
                ),
                step=global_step,
            )
            probe_embeddings_before = (
                policy_embeddings[diagnostic_probe_indices_device].detach()
                if diagnostic_probe_indices_device is not None
                else None
            )
            actor_update_diagnostics = measure(
                "actor_backbone_update",
                device,
                timings,
                lambda: update_backbone_from_embedding_gradient(
                    model,
                    actor_optimizer,
                    scaler,
                    raw_images,
                    embedding_gradient,
                    query_indices_cpu,
                    device,
                    mean,
                    std,
                ),
                step=global_step,
            )
            step_embedding_drift = None
            if (
                diagnostic_probe_images is not None
                and probe_embeddings_before is not None
            ):
                step_embedding_drift = measure(
                    "actor_embedding_drift_diagnostics",
                    device,
                    timings,
                    lambda: embedding_drift_diagnostics(
                        model,
                        diagnostic_probe_images,
                        probe_embeddings_before,
                        device,
                        mean,
                        std,
                    ),
                    step=global_step,
                )
            del embedding_gradient

            step_critic_losses: list[float] = []
            step_critic_diagnostics: list[dict[str, object]] = []

            def perform_critic_updates() -> tuple[float, ...]:
                if previous_encoding is not None and previous_reward is not None:
                    critic_loss, critic_diagnostics = update_critic(
                        critic,
                        critic_optimizer,
                        previous_encoding,
                        previous_reward,
                        encoding,
                    )
                    step_critic_losses.append(critic_loss)
                    if critic_diagnostics is not None:
                        step_critic_diagnostics.append(critic_diagnostics)
                if step == TRAJECTORY_LENGTH:
                    critic_loss, critic_diagnostics = update_critic(
                        critic,
                        critic_optimizer,
                        encoding,
                        reward_output.total_reward,
                        None,
                    )
                    step_critic_losses.append(critic_loss)
                    if critic_diagnostics is not None:
                        step_critic_diagnostics.append(critic_diagnostics)
                return tuple(step_critic_losses)

            if previous_encoding is not None or step == TRAJECTORY_LENGTH:
                measure(
                    "critic_update",
                    device,
                    timings,
                    perform_critic_updates,
                    step=global_step,
                )
                epoch_critic_losses.extend(step_critic_losses)

            label_state = correction.corrected_labels
            step_action_count = int(correction.actions.sum())
            epoch_action_count += step_action_count
            reward_value = float(reward_output.total_reward)
            epoch_rewards.append(reward_value)
            epoch_actor_losses.append(actor_loss)
            current_hard_labels = label_state.argmax(dim=1)
            changed_from_noisy_rate = float(
                current_hard_labels.ne(initial_noisy_labels).float().mean()
            )
            clean_accuracy = float(
                current_hard_labels.eq(clean_labels).float().mean()
            )
            if DIAGNOSTICS_ENABLED:
                probabilities = correction.correction_probabilities.float()
                stable_probabilities = probabilities.clamp(1e-7, 1.0 - 1e-7)
                probability_entropy = -(
                    stable_probabilities * stable_probabilities.log()
                    + (1.0 - stable_probabilities)
                    * torch.log1p(-stable_probabilities)
                )
                soft_labels = correction.corrected_labels.float().clamp_min(
                    1e-12
                )
                label_entropy = -(soft_labels * soft_labels.log()).sum(dim=1)
                reward_scalars = torch.stack(
                    (
                        reward_output.label_consistency,
                        reward_output.noisy_label_alignment,
                        reward_output.total_reward,
                    )
                ).detach().float().cpu().tolist()
                append_check_event(
                    check_log_path,
                    "rl_step",
                    epoch=epoch,
                    trajectory_step=step,
                    global_step=global_step,
                    actor_learning_rate=actor_optimizer.param_groups[0]["lr"],
                    critic_learning_rate=critic_optimizer.param_groups[0]["lr"],
                    reward={
                        "label_consistency": reward_scalars[0],
                        "noisy_label_alignment": reward_scalars[1],
                        "total": reward_scalars[2],
                        "per_sample_consistency": tensor_distribution(
                            reward_output.per_sample_consistency
                        ),
                    },
                    critic={
                        "q_used_by_actor": float(q_value),
                        "state_encoding": tensor_distribution(encoding),
                        "updates": step_critic_diagnostics,
                    },
                    policy={
                        "actor_loss": actor_loss,
                        "action_count": step_action_count,
                        "action_rate": step_action_count / EXPECTED_SAMPLES,
                        "correction_probability": tensor_distribution(
                            probabilities
                        ),
                        "selected_action_probability": tensor_distribution(
                            probabilities[correction.actions]
                        ),
                        "rejected_action_probability": tensor_distribution(
                            probabilities[~correction.actions]
                        ),
                        "bernoulli_entropy": tensor_distribution(
                            probability_entropy
                        ),
                        "corrected_label_entropy": tensor_distribution(
                            label_entropy
                        ),
                    },
                    labels={
                        "changed_from_noisy_rate": changed_from_noisy_rate,
                        "clean_accuracy": clean_accuracy,
                    },
                    knn=step_knn_diagnostics,
                    policy_gradient=policy_gradient_diagnostics,
                    actor_update=actor_update_diagnostics,
                    embedding_drift=step_embedding_drift,
                    timing_seconds={
                        name: timings[name][-1]
                        for name in (
                            "actor_feature_extraction",
                            "exact_policy_knn",
                            "full_correction",
                            "reward_including_clean_knn",
                            "critic_state_encoding",
                            "actor_policy_gradient",
                            "actor_backbone_update",
                            "actor_embedding_drift_diagnostics",
                            "critic_update",
                        )
                        if timings.get(name)
                        and (name != "critic_update" or step_critic_losses)
                    },
                    amp_scale=float(scaler.get_scale()),
                    model_training=model.training,
                    batchnorm_training_modules=sum(
                        int(module.training)
                        for module in model.modules()
                        if isinstance(module, nn.modules.batchnorm._BatchNorm)
                    ),
                    cuda_allocated_gib=torch.cuda.memory_allocated(device)
                    / 1024**3,
                    cuda_reserved_gib=torch.cuda.memory_reserved(device)
                    / 1024**3,
                    cuda_peak_allocated_gib=torch.cuda.max_memory_allocated(device)
                    / 1024**3,
                )
            print(
                f"[RL] epoch={epoch} step={step} "
                f"reward={reward_value:.6f} q={float(q_value):.6f} "
                f"actor={actor_loss:.6f} "
                f"action={step_action_count / EXPECTED_SAMPLES:.4f} "
                f"changed_from_noisy={changed_from_noisy_rate:.4f} "
                f"clean_accuracy={clean_accuracy:.4f}"
            )

            if step < TRAJECTORY_LENGTH:
                previous_encoding = encoding
                previous_reward = reward_output.total_reward.detach()
            del policy_embeddings, policy_neighbors, correction, reward_output

        synchronize(device)
        train_elapsed = time.perf_counter() - epoch_started
        epoch_seconds.append(train_elapsed)
        actor_scheduler.step()
        critic_scheduler.step()
        mean_critic_loss = (
            sum(epoch_critic_losses) / len(epoch_critic_losses)
            if epoch_critic_losses
            else None
        )
        train_summary = correction_summary(
            label_state,
            clean_labels,
            initial_noisy_labels,
            noise_mask,
            epoch=epoch,
            split="train",
            action_count=epoch_action_count,
            action_rate=epoch_action_count
            / (EXPECTED_SAMPLES * TRAJECTORY_LENGTH),
            reward=sum(epoch_rewards) / len(epoch_rewards),
            actor_loss=sum(epoch_actor_losses) / len(epoch_actor_losses),
            critic_loss=mean_critic_loss,
            elapsed_seconds=train_elapsed,
        )
        val_images, val_clean, val_noisy, val_noise_mask = evaluation_data["val"]
        val_summary, _ = evaluate_correction_split(
            split="val",
            epoch=epoch,
            model=model,
            policy=policy,
            raw_images=val_images,
            clean_labels_cpu=val_clean,
            noisy_labels_cpu=val_noisy,
            noise_mask_cpu=val_noise_mask,
            device=device,
            mean=mean,
            std=std,
            timings=timings,
        )
        validation_seconds.append(float(val_summary["elapsed_seconds"]))
        append_csv(
            train_csv_path,
            [train_summary, val_summary],
            SUMMARY_FIELDS,
        )
        if DIAGNOSTICS_ENABLED:
            append_check_event(
                check_log_path,
                "epoch_summary",
                epoch=epoch,
                train=train_summary,
                validation=val_summary,
                actor_learning_rate_next_epoch=actor_optimizer.param_groups[0][
                    "lr"
                ],
                critic_learning_rate_next_epoch=critic_optimizer.param_groups[0][
                    "lr"
                ],
                mean_reward=sum(epoch_rewards) / len(epoch_rewards),
                mean_actor_loss=sum(epoch_actor_losses)
                / len(epoch_actor_losses),
                mean_critic_loss=mean_critic_loss,
                cuda_peak_allocated_gib=torch.cuda.max_memory_allocated(device)
                / 1024**3,
                cuda_peak_reserved_gib=torch.cuda.max_memory_reserved(device)
                / 1024**3,
            )

        def save_epoch_checkpoints(
            actor_path: Path,
            critic_path: Path,
        ) -> None:
            _save_rl_checkpoints(
                actor_path,
                critic_path,
                epoch=epoch,
                model=model,
                critic=critic,
                validation_summary=val_summary,
            )

        measure(
            "last_checkpoints",
            device,
            timings,
            lambda: save_epoch_checkpoints(
                actor_last_checkpoint_path,
                critic_last_checkpoint_path,
            ),
            step=epoch,
        )
        validation_key = _rl_selection_key(val_summary)
        if best_rl_key is None or validation_key > best_rl_key:
            measure(
                "best_checkpoints",
                device,
                timings,
                lambda: save_epoch_checkpoints(
                    actor_best_checkpoint_path,
                    critic_best_checkpoint_path,
                ),
                step=epoch,
            )
            best_rl_key = validation_key
            best_rl_epoch = epoch
            best_validation_summary = dict(val_summary)
            print(
                f"[RL BEST] epoch={best_rl_epoch} "
                f"val_macro_f1={float(val_summary['macro_f1']):.6f} "
                "val_balanced_acc="
                f"{float(val_summary['balanced_accuracy']):.6f} "
                f"val_loss={float(val_summary['loss']):.6f}"
            )
        final_train_summary = train_summary
        print(
            f"[EPOCH] epoch={epoch} "
            f"train_acc={train_summary['accuracy']:.6f} "
            f"train_macro_f1={train_summary['macro_f1']:.6f} "
            f"val_acc={val_summary['accuracy']:.6f} "
            f"val_macro_f1={val_summary['macro_f1']:.6f} "
            f"train_seconds={train_elapsed:.3f}"
        )
        print(f"[CSV] updated={train_csv_path}")

    if final_train_summary is None:
        raise RuntimeError("RL training finished without epoch metrics.")
    if best_validation_summary is None or best_rl_epoch == 0:
        raise RuntimeError("RL training finished without a best checkpoint.")

    deployment_checkpoint_path = actor_last_checkpoint_path
    restored_checkpoint = measure(
        "actor_deployment_restore",
        device,
        timings,
        lambda: _restore_actor(
            model,
            deployment_checkpoint_path,
            device,
        ),
    )
    deployed_epoch = int(restored_checkpoint["epoch"])
    if deployed_epoch != RL_EPOCHS:
        raise RuntimeError(
            "Restored actor checkpoint is not the last RL epoch: "
            f"{deployed_epoch} != {RL_EPOCHS}."
        )
    del restored_checkpoint
    print(
        "[RL RESTORE] mode=last "
        f"checkpoint={deployment_checkpoint_path} epoch={deployed_epoch}"
    )

    cleaning_summary: dict[str, object] | None = None
    if CLEANING_TRAJECTORY_LENGTH > 0:
        cleaning_summary = clean_full_training_labels(
            model=model,
            policy=policy,
            raw_images=raw_images,
            clean_labels_cpu=clean_labels_cpu,
            noisy_labels_cpu=noisy_labels_cpu,
            noise_mask_cpu=noise_mask_cpu,
            device=device,
            mean=mean,
            std=std,
            timings=timings,
            corrected_labels_path=corrected_labels_path,
            cleaning_csv_path=cleaning_csv_path,
            cleaning_summary_path=cleaning_summary_path,
            cleaning_per_class_path=cleaning_per_class_path,
            checkpoint_epoch=deployed_epoch,
        )

    test_images, test_clean, test_noisy, test_noise_mask = evaluation_data["test"]
    test_summary, test_per_class = evaluate_correction_split(
        split="test",
        epoch=deployed_epoch,
        model=model,
        policy=policy,
        raw_images=test_images,
        clean_labels_cpu=test_clean,
        noisy_labels_cpu=test_noisy,
        noise_mask_cpu=test_noise_mask,
        device=device,
        mean=mean,
        std=std,
        timings=timings,
    )
    write_csv(test_csv_path, [test_summary], SUMMARY_FIELDS)
    write_csv(test_per_class_path, test_per_class, PER_CLASS_FIELDS)
    print_timing_summary(timings)

    setup_names = {
        f"{DATASET_STAGE_PREFIX}_load",
        f"{DATASET_STAGE_PREFIX}_eval_load",
        "noise_injection",
        "noise_artifact_load",
        "val_noise_injection",
        "test_noise_injection",
        "model_init",
        "kernel_warmup",
        "supervised_warmup",
        "warmup_checkpoint_load",
        "global_cache_feature_extraction",
        "global_cache_exact_knn",
    }
    setup_total = sum(
        sum(values) for name, values in timings.items() if name in setup_names
    )
    checkpoint_names = {
        "last_checkpoints",
        "best_checkpoints",
        "actor_deployment_restore",
    }
    checkpoint_total = sum(
        sum(values)
        for name, values in timings.items()
        if name in checkpoint_names
    )
    peak_allocated = torch.cuda.max_memory_allocated() / 1024**3
    peak_reserved = torch.cuda.max_memory_reserved() / 1024**3
    measured_total = sum(sum(values) for values in timings.values())
    write_csv(timing_csv_path, build_timing_rows(timings), TIMING_FIELDS)
    write_csv(
        run_summary_path,
        [
            {
                "dataset": DATASET_NAME,
                "model_name": MODEL_NAME,
                "warmup_model_id": WARMUP_MODEL_ID,
                "image_size": IMAGE_SIZE,
                "seed": SEED,
                "noise_rate": NOISE_RATE,
                "train_samples": EXPECTED_SAMPLES,
                "validation_samples": evaluation_data["val"][0].size(0),
                "test_samples": evaluation_data["test"][0].size(0),
                "pretrained": PRETRAINED,
                "warmup_epochs": WARMUP_EPOCHS,
                "warmup_batch_size": WARMUP_BATCH_SIZE,
                "warmup_optimizer": "sgd",
                "warmup_learning_rate": WARMUP_LR,
                "warmup_weight_decay": WARMUP_WEIGHT_DECAY,
                "warmup_best_epoch": warmup_result["best_epoch"],
                "warmup_best_noisy_validation_accuracy": warmup_result[
                    "best_noisy_validation_accuracy"
                ],
                "warmup_best_clean_validation_accuracy": warmup_result[
                    "best_clean_validation_accuracy"
                ],
                "warmup_deployment_mode": warmup_result["deployment_mode"],
                "warmup_deployment_epoch": warmup_result[
                    "deployment_epoch"
                ],
                "warmup_seconds": sum(timings.get("supervised_warmup", ())),
                "rl_epochs": RL_EPOCHS,
                "trajectory_length": TRAJECTORY_LENGTH,
                "initial_state_randomization_rate": (
                    INITIAL_STATE_RANDOMIZATION_RATE
                ),
                "feature_batch_size": FEATURE_BATCH_SIZE,
                "policy_update_mode": ACTOR_UPDATE_MODE,
                "policy_update_samples": POLICY_UPDATE_SAMPLES,
                "policy_update_batch_size": POLICY_UPDATE_BATCH_SIZE,
                "actor_optimizer": "sgd",
                "actor_learning_rate": ACTOR_LR,
                "actor_momentum": ACTOR_MOMENTUM,
                "actor_weight_decay": ACTOR_WEIGHT_DECAY,
                "diagnostics_enabled": DIAGNOSTICS_ENABLED,
                "diagnostic_probe_size": (
                    DIAGNOSTIC_PROBE_SIZE if DIAGNOSTICS_ENABLED else 0
                ),
                "diagnostic_log": (
                    str(check_log_path) if DIAGNOSTICS_ENABLED else ""
                ),
                "rl_epoch0_validation_accuracy": epoch_zero_summary[
                    "accuracy"
                ],
                "rl_epoch0_validation_macro_f1": epoch_zero_summary[
                    "macro_f1"
                ],
                "k": K,
                "temperature": TEMPERATURE,
                "discount_factor": DISCOUNT_FACTOR,
                "reward_nla_weight": NLA_WEIGHT,
                "critic_num_bins": CRITIC_NUM_BINS,
                "lr_decay_fraction": LR_DECAY_FRACTION,
                "lr_decay_factor": LR_DECAY_FACTOR,
                "knn_query_chunk_size": KNN_QUERY_CHUNK_SIZE,
                "knn_reference_chunk_size": KNN_REFERENCE_CHUNK_SIZE,
                "correction_chunk_size": CORRECTION_CHUNK_SIZE,
                "cleaning_trajectory_length": CLEANING_TRAJECTORY_LENGTH,
                "corrected_labels_path": (
                    str(corrected_labels_path)
                    if CLEANING_TRAJECTORY_LENGTH > 0
                    else ""
                ),
                "cleaning_accuracy": (
                    cleaning_summary["accuracy"]
                    if cleaning_summary is not None
                    else ""
                ),
                "cleaning_noisy_recovery_rate": (
                    cleaning_summary["noisy_recovery_rate"]
                    if cleaning_summary is not None
                    else ""
                ),
                "cleaning_false_correction_rate": (
                    cleaning_summary["false_correction_rate"]
                    if cleaning_summary is not None
                    else ""
                ),
                "rl_best_epoch": best_rl_epoch,
                "rl_best_validation_accuracy": best_validation_summary[
                    "accuracy"
                ],
                "rl_best_validation_balanced_accuracy": (
                    best_validation_summary["balanced_accuracy"]
                ),
                "rl_best_validation_macro_f1": best_validation_summary[
                    "macro_f1"
                ],
                "rl_best_validation_loss": best_validation_summary["loss"],
                "actor_best_checkpoint": str(actor_best_checkpoint_path),
                "actor_last_checkpoint": str(actor_last_checkpoint_path),
                "critic_best_checkpoint": str(critic_best_checkpoint_path),
                "critic_last_checkpoint": str(critic_last_checkpoint_path),
                "actor_deployment_checkpoint": str(deployment_checkpoint_path),
                "actor_deployment_epoch": deployed_epoch,
                "setup_seconds": setup_total,
                "train_seconds": sum(epoch_seconds),
                "mean_epoch_seconds": sum(epoch_seconds) / len(epoch_seconds),
                "validation_seconds": sum(validation_seconds),
                "checkpoint_seconds": checkpoint_total,
                "test_seconds": test_summary["elapsed_seconds"],
                "measured_total_seconds": measured_total,
                "peak_cuda_allocated_gib": peak_allocated,
                "peak_cuda_reserved_gib": peak_reserved,
            }
        ],
        RUN_SUMMARY_FIELDS,
    )

    print("\n[RESULT]")
    print(f"dataset={DATASET_NAME}")
    print(
        f"samples={EXPECTED_SAMPLES}, classes={NUM_CLASSES}, "
        f"rl_epochs={RL_EPOCHS}"
    )
    print(
        f"noise_type=stratified_symmetric, noise_rate={NOISE_RATE:.2f}, "
        f"seed={SEED}"
    )
    print(
        f"warmup_best_epoch={warmup_result['best_epoch']}, "
        "warmup_best_noisy_val_acc="
        f"{warmup_result['best_noisy_validation_accuracy']:.6f}, "
        "warmup_best_clean_val_acc="
        f"{warmup_result['best_clean_validation_accuracy']:.6f}"
    )
    print(
        f"final_train_accuracy={final_train_summary['accuracy']:.6f}, "
        f"final_train_macro_f1={final_train_summary['macro_f1']:.6f}"
    )
    print(
        f"rl_best_epoch={best_rl_epoch}, "
        "rl_best_val_macro_f1="
        f"{float(best_validation_summary['macro_f1']):.6f}, "
        "rl_best_val_balanced_acc="
        f"{float(best_validation_summary['balanced_accuracy']):.6f}, "
        f"rl_best_val_loss={float(best_validation_summary['loss']):.6f}"
    )
    print(
        f"test_accuracy={test_summary['accuracy']:.6f}, "
        f"test_balanced_accuracy={test_summary['balanced_accuracy']:.6f}, "
        f"test_macro_f1={test_summary['macro_f1']:.6f}, "
        f"test_noisy_recovery={test_summary['noisy_recovery_rate']:.6f}"
    )
    print(f"setup_seconds={setup_total:.3f}")
    print(f"train_seconds={sum(epoch_seconds):.3f}")
    print(f"checkpoint_seconds={checkpoint_total:.3f}")
    print(
        f"mean_epoch_seconds={sum(epoch_seconds) / len(epoch_seconds):.3f}"
    )
    print(
        f"peak_cuda_allocated_gib={peak_allocated:.3f}, "
        f"peak_cuda_reserved_gib={peak_reserved:.3f}"
    )
    print(f"warmup_checkpoint={warmup_checkpoint_path}")
    print(f"actor_best_checkpoint={actor_best_checkpoint_path}")
    print(f"actor_last_checkpoint={actor_last_checkpoint_path}")
    print(f"critic_best_checkpoint={critic_best_checkpoint_path}")
    print(f"critic_last_checkpoint={critic_last_checkpoint_path}")
    print(
        f"actor_deployment_checkpoint={deployment_checkpoint_path} "
        f"epoch={deployed_epoch}"
    )
    if cleaning_summary is not None:
        print(f"corrected_labels={corrected_labels_path}")
        print(f"cleaning_csv={cleaning_csv_path}")
        print(f"cleaning_summary_csv={cleaning_summary_path}")
        print(f"cleaning_per_class_csv={cleaning_per_class_path}")
    print(f"train_csv={train_csv_path}")
    if DIAGNOSTICS_ENABLED:
        print(f"check_log={check_log_path}")
    print(f"test_csv={test_csv_path}")
    print(f"test_per_class_csv={test_per_class_path}")
    print(f"timing_csv={timing_csv_path}")
    print(f"run_summary_csv={run_summary_path}")


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
