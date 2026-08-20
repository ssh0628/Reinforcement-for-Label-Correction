"""CIFAR-10 RLNLC training and correction primitives.

Dataset loading, preprocessing, model construction, and experiment settings
are supplied by ``cifar_common.py``. This module contains only the reusable
warm-up, KNN, actor-critic, correction, metrics, checkpoint, and timing logic.
"""

from __future__ import annotations

import csv
import math
import random
import sys
import time
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from dataclasses import replace
from pathlib import Path
from typing import Callable, TextIO, TypeVar

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.optim import Optimizer, SGD
from torch.optim.lr_scheduler import MultiStepLR

from rl.actor.policy import CorrectionResult, LabelCorrectionPolicy, PolicyStep
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
CONFIGURED = False
RUN_LOG_FILENAME = "run.log"
ACTOR_BEST_CHECKPOINT_FILENAME = "actor_best.pt"
ACTOR_LAST_CHECKPOINT_FILENAME = "actor_last.pt"
CRITIC_BEST_CHECKPOINT_FILENAME = "critic_best.pt"
CRITIC_LAST_CHECKPOINT_FILENAME = "critic_last.pt"
TRAIN_CSV_FILENAME = "train.csv"
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

ACTOR_UPDATE_MODE = "full"  # "full" or "subset"
if ACTOR_UPDATE_MODE not in {"full", "subset"}:
    raise ValueError("ACTOR_UPDATE_MODE must be 'full' or 'subset'.")
POLICY_UPDATE_SAMPLES = EXPECTED_SAMPLES
OUTPUT_DIR = (
    Path("outputs") / f"cifar10_rl_{ACTOR_UPDATE_MODE}"
)
MODEL_OUTPUT_DIR = OUTPUT_DIR

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
POLICY_UPDATE_BATCH_SIZE = 512

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
CLEANING_SUMMARY_FIELDS = SUMMARY_FIELDS + ("best_step", "best_accuracy")

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
    "warmup_checkpoint_load_seconds",
    "rl_epochs",
    "trajectory_length",
    "initial_state_randomization_rate",
    "feature_batch_size",
    "policy_update_mode",
    "policy_update_samples",
    "policy_update_batch_size",
    "actor_update_implementation",
    "actor_optimizer",
    "actor_learning_rate",
    "actor_momentum",
    "actor_weight_decay",
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
    "rl_best_epoch",
    "rl_best_validation_accuracy",
    "rl_best_validation_balanced_accuracy",
    "rl_best_validation_macro_f1",
    "rl_best_validation_loss",
    "rl_last_epoch",
    "rl_last_validation_accuracy",
    "rl_last_validation_balanced_accuracy",
    "rl_last_validation_macro_f1",
    "rl_last_validation_loss",
    "actor_best_checkpoint",
    "actor_last_checkpoint",
    "critic_best_checkpoint",
    "critic_last_checkpoint",
    "setup_seconds",
    "train_seconds",
    "mean_epoch_seconds",
    "validation_seconds",
    "checkpoint_seconds",
    "measured_total_seconds",
    "total_runtime_seconds",
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
            "TRAIN_DATA_LOADER must be configured by cifar_common.py before "
            "starting the experiment."
        )
    return TRAIN_DATA_LOADER()


def load_evaluation_data() -> dict[str, tuple[Tensor, Tensor]]:
    if EVALUATION_DATA_LOADER is None:
        raise RuntimeError(
            "EVALUATION_DATA_LOADER must be configured by cifar_common.py before "
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
    noisy_array = np.asarray(noisy_array)
    mask_array = np.asarray(mask_array)
    if not np.issubdtype(noisy_array.dtype, np.integer):
        raise TypeError("Noisy-label artifact must use an integer NumPy dtype.")
    if mask_array.dtype != np.bool_:
        raise TypeError("Noise-mask artifact must use the NumPy bool dtype.")
    noisy_labels = torch.from_numpy(noisy_array).to(torch.long)
    noise_mask = torch.from_numpy(mask_array)
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
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    try:
        with temporary_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)


def append_csv(
    path: Path,
    rows: list[dict[str, object]],
    fieldnames: tuple[str, ...],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    needs_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if needs_header:
            writer.writeheader()
        writer.writerows(rows)


def preprocess(
    images: Tensor,
    device: torch.device,
    mean: Tensor,
    std: Tensor,
) -> Tensor:
    if PREPROCESS_FUNCTION is None:
        raise RuntimeError(
            "PREPROCESS_FUNCTION must be configured by cifar_common.py before "
            "starting the experiment."
        )
    return PREPROCESS_FUNCTION(images, device, mean, std)


def create_experiment_model() -> nn.Module:
    if MODEL_FACTORY is None:
        raise RuntimeError(
            "MODEL_FACTORY must be configured by cifar_common.py before starting "
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
                "model_name": MODEL_NAME,
                "num_classes": NUM_CLASSES,
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
    checkpoint_model_name = checkpoint.get("model_name")
    if checkpoint_model_name is not None and checkpoint_model_name != MODEL_NAME:
        raise ValueError("Warmup checkpoint model name does not match this run.")
    checkpoint_num_classes = checkpoint.get("num_classes")
    if (
        checkpoint_num_classes is not None
        and int(checkpoint_num_classes) != NUM_CLASSES
    ):
        raise ValueError("Warmup checkpoint class count does not match this run.")
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


def restore_actor_checkpoint(
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
    *,
    direct_actor: bool = False,
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

    update_image_count = min(
        raw_images.size(0),
        (
            POLICY_UPDATE_BATCH_SIZE * (K + 1)
            if direct_actor
            else WARMUP_BATCH_SIZE
        ),
    )
    update_images = preprocess(
        raw_images[:update_image_count],
        device,
        mean,
        std,
    )
    model.train()
    with preserve_batchnorm_running_stats(model):
        with torch.autocast(
            device_type="cuda",
            dtype=AMP_DTYPE,
            enabled=USE_AMP,
        ):
            encode(model, update_images).mean().backward()
    model.zero_grad(set_to_none=True)
    model.eval()


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
    best_row = max(history, key=lambda row: float(row["clean_accuracy"]))
    summary["best_step"] = int(best_row["step"])
    summary["best_accuracy"] = float(best_row["clean_accuracy"])
    write_csv(cleaning_csv_path, history, CLEANING_FIELDS)
    write_csv(
        cleaning_summary_path,
        [summary],
        CLEANING_SUMMARY_FIELDS,
    )
    write_csv(cleaning_per_class_path, per_class, PER_CLASS_FIELDS)
    print(f"[CLEAN] corrected_labels={corrected_labels_path}")
    del embeddings, neighbors, label_state
    return summary


def select_policy_queries(sample_count: int, step: int) -> Tensor:
    if POLICY_UPDATE_SAMPLES == sample_count:
        return torch.arange(sample_count)
    if step <= 0:
        raise ValueError("step must be positive.")

    # Select a minimal number of complete rollout batches. Replaying those
    # same batch contexts keeps BatchNorm-dependent action probabilities
    # identical between sampling and policy-gradient evaluation. Only the
    # requested number of query log-probabilities contributes to the loss.
    generator = torch.Generator().manual_seed(SEED + step)
    full_batch_count, tail_size = divmod(
        sample_count,
        POLICY_UPDATE_BATCH_SIZE,
    )
    full_batch_capacity = full_batch_count * POLICY_UPDATE_BATCH_SIZE
    selected_chunks: list[Tensor] = []

    if POLICY_UPDATE_SAMPLES <= full_batch_capacity:
        required_batches = math.ceil(
            POLICY_UPDATE_SAMPLES / POLICY_UPDATE_BATCH_SIZE
        )
        selected_batch_ids = torch.randperm(
            full_batch_count,
            generator=generator,
        )[:required_batches]
        full_selected_batches, partial_count = divmod(
            POLICY_UPDATE_SAMPLES,
            POLICY_UPDATE_BATCH_SIZE,
        )
        for batch_id_tensor in selected_batch_ids[:full_selected_batches]:
            start = int(batch_id_tensor) * POLICY_UPDATE_BATCH_SIZE
            selected_chunks.append(
                torch.arange(start, start + POLICY_UPDATE_BATCH_SIZE)
            )
        if partial_count:
            start = int(selected_batch_ids[-1]) * POLICY_UPDATE_BATCH_SIZE
            positions = torch.randperm(
                POLICY_UPDATE_BATCH_SIZE,
                generator=generator,
            )[:partial_count]
            selected_chunks.append(start + positions)
    else:
        # A near-full subset necessarily touches every complete batch.
        selected_chunks.append(torch.arange(full_batch_capacity))
        tail_selected = POLICY_UPDATE_SAMPLES - full_batch_capacity
        if tail_selected > tail_size:
            raise RuntimeError("Failed to select the requested policy subset.")
        positions = torch.randperm(tail_size, generator=generator)[:tail_selected]
        selected_chunks.append(full_batch_capacity + positions)

    return torch.cat(selected_chunks).sort().values


@contextmanager
def preserve_batchnorm_running_stats(model: nn.Module):
    """Use training-mode batch statistics without counting a rollout twice."""
    snapshots: list[tuple[nn.Module, Tensor, Tensor, Tensor]] = []
    for module in model.modules():
        if not isinstance(module, nn.modules.batchnorm._BatchNorm):
            continue
        if (
            module.running_mean is None
            or module.running_var is None
            or module.num_batches_tracked is None
        ):
            continue
        snapshots.append(
            (
                module,
                module.running_mean.detach().clone(),
                module.running_var.detach().clone(),
                module.num_batches_tracked.detach().clone(),
            )
        )
    try:
        yield
    finally:
        with torch.no_grad():
            for module, running_mean, running_var, batches in snapshots:
                module.running_mean.copy_(running_mean)
                module.running_var.copy_(running_var)
                module.num_batches_tracked.copy_(batches)


def direct_policy_batch(
    model: nn.Module,
    policy: LabelCorrectionPolicy,
    raw_images: Tensor,
    label_state: Tensor,
    neighbor_indices_cpu: Tensor,
    query_indices_cpu: Tensor,
    device: torch.device,
    mean: Tensor,
    std: Tensor,
    *,
    actions: Tensor | None = None,
) -> PolicyStep:
    """Run query and KNN-neighbor images through the actor in one graph."""
    sample_count = raw_images.size(0)
    if raw_images.device.type != "cpu":
        raise ValueError("raw_images must remain on CPU.")
    if label_state.ndim != 2 or label_state.size(0) != sample_count:
        raise ValueError("label_state must have shape [N, C].")
    if (
        neighbor_indices_cpu.ndim != 2
        or neighbor_indices_cpu.size(0) != sample_count
    ):
        raise ValueError("neighbor_indices_cpu must have shape [N, K].")
    if neighbor_indices_cpu.device.type != "cpu":
        raise ValueError("neighbor_indices_cpu must remain on CPU.")
    if query_indices_cpu.ndim != 1 or query_indices_cpu.numel() == 0:
        raise ValueError("query_indices_cpu must be a non-empty vector.")
    if query_indices_cpu.device.type != "cpu":
        raise ValueError("query_indices_cpu must remain on CPU.")
    if torch.unique(query_indices_cpu).numel() != query_indices_cpu.numel():
        raise ValueError("query_indices_cpu must not contain duplicates.")
    if torch.any(query_indices_cpu < 0) or torch.any(
        query_indices_cpu >= sample_count
    ):
        raise ValueError("query_indices_cpu contains an out-of-range index.")

    batch_neighbors_cpu = neighbor_indices_cpu[query_indices_cpu]
    combined_indices_cpu = torch.cat(
        (query_indices_cpu, batch_neighbors_cpu.reshape(-1))
    )
    selected_images = pin_for_cuda(
        raw_images[combined_indices_cpu].contiguous()
    )
    images = preprocess(selected_images, device, mean, std)
    embeddings = encode(model, images)
    query_count = query_indices_cpu.numel()
    neighbor_count = batch_neighbors_cpu.size(1)
    query_embeddings = embeddings[:query_count]
    neighbor_embeddings = embeddings[query_count:].reshape(
        query_count,
        neighbor_count,
        -1,
    )
    query_indices = query_indices_cpu.to(device=device, non_blocking=True)
    neighbor_indices = batch_neighbors_cpu.to(
        device=device,
        non_blocking=True,
    )
    batch_actions = None if actions is None else actions[query_indices]
    return policy(
        query_embeddings,
        neighbor_embeddings,
        label_state[query_indices],
        label_state[neighbor_indices],
        actions=batch_actions,
    )


@torch.no_grad()
def direct_policy_correction(
    model: nn.Module,
    policy: LabelCorrectionPolicy,
    raw_images: Tensor,
    label_state: Tensor,
    neighbor_indices_cpu: Tensor,
    device: torch.device,
    mean: Tensor,
    std: Tensor,
) -> CorrectionResult:
    """Sample the full joint action with direct training-mode forwards."""
    sample_count = raw_images.size(0)
    probabilities = torch.empty(
        sample_count,
        dtype=torch.float32,
        device=device,
    )
    actions = torch.empty(sample_count, dtype=torch.bool, device=device)
    corrected_labels = torch.empty_like(label_state, dtype=torch.float32)
    total_batches = math.ceil(sample_count / POLICY_UPDATE_BATCH_SIZE)
    model.train()

    # The rollout determines actions and Q before the actor update. Its forward
    # must use batch statistics, but only the gradient pass should advance the
    # persistent BatchNorm buffers.
    with preserve_batchnorm_running_stats(model):
        for batch_number, start in enumerate(
            range(0, sample_count, POLICY_UPDATE_BATCH_SIZE),
            start=1,
        ):
            end = min(start + POLICY_UPDATE_BATCH_SIZE, sample_count)
            query_indices_cpu = torch.arange(start, end, dtype=torch.long)
            with torch.autocast(
                device_type=device.type,
                dtype=AMP_DTYPE,
                enabled=USE_AMP,
            ):
                policy_step = direct_policy_batch(
                    model,
                    policy,
                    raw_images,
                    label_state,
                    neighbor_indices_cpu,
                    query_indices_cpu,
                    device,
                    mean,
                    std,
                )
            probabilities[start:end] = policy_step.correction_probabilities
            actions[start:end] = policy_step.actions
            corrected_labels[start:end] = policy_step.next_labels
            if batch_number % 100 == 0 or batch_number == total_batches:
                print(
                    f"[DIRECT ACTOR ROLLOUT] batch={batch_number}/"
                    f"{total_batches} samples={end}/{sample_count}"
                )

    if actions.all():
        actions[0] = False
        corrected_labels[0] = label_state[0]
    return CorrectionResult(
        correction_probabilities=probabilities,
        actions=actions,
        corrected_labels=corrected_labels,
    )


def update_actor_direct(
    model: nn.Module,
    policy: LabelCorrectionPolicy,
    optimizer: Optimizer,
    scaler: torch.amp.GradScaler,
    raw_images: Tensor,
    label_state: Tensor,
    neighbor_indices_cpu: Tensor,
    actions: Tensor,
    q_value: Tensor,
    query_indices_cpu: Tensor,
    device: torch.device,
    mean: Tensor,
    std: Tensor,
) -> float:
    """Backpropagate the policy loss directly through the actor backbone.

    The KNN graph and sampled joint action remain fixed for the RL transition.
    ``POLICY_UPDATE_BATCH_SIZE`` is only a memory microbatch. Loss terms are
    normalized by the total selected query count, so full and subset modes use
    the same per-query gradient scale. One optimizer step is taken for the
    complete transition, avoiding stale action/Q targets inside it.
    """
    if query_indices_cpu.ndim != 1 or query_indices_cpu.numel() == 0:
        raise ValueError("query_indices_cpu must be a non-empty vector.")
    if query_indices_cpu.device.type != "cpu":
        raise ValueError("query_indices_cpu must remain on CPU.")

    sample_count = raw_images.size(0)
    if torch.unique(query_indices_cpu).numel() != query_indices_cpu.numel():
        raise ValueError("query_indices_cpu must not contain duplicates.")
    if torch.any(query_indices_cpu < 0) or torch.any(
        query_indices_cpu >= sample_count
    ):
        raise ValueError("query_indices_cpu contains an out-of-range index.")
    model.train()
    optimizer.zero_grad(set_to_none=True)
    query_count = query_indices_cpu.numel()
    selected_mask_cpu = torch.zeros(sample_count, dtype=torch.bool)
    selected_mask_cpu[query_indices_cpu] = True
    selected_batch_ids = torch.unique(
        query_indices_cpu // POLICY_UPDATE_BATCH_SIZE,
        sorted=True,
    )
    total_batches = selected_batch_ids.numel()
    total_loss = torch.zeros((), device=device)
    processed_queries = 0

    # Replay only the rollout batches selected by ``select_policy_queries``.
    # This preserves the action-generating BatchNorm context while restricting
    # subset-mode backward work to roughly ceil(subset_size / batch_size)
    # batches. Unselected queries in the final partial batch provide context
    # only and do not contribute log-probabilities to the policy loss.
    for batch_number, batch_id_tensor in enumerate(
        selected_batch_ids,
        start=1,
    ):
        start = int(batch_id_tensor) * POLICY_UPDATE_BATCH_SIZE
        end = min(start + POLICY_UPDATE_BATCH_SIZE, sample_count)
        batch_indices_cpu = torch.arange(start, end)
        selected_in_batch_cpu = selected_mask_cpu[start:end]
        selected_count = int(selected_in_batch_cpu.sum())
        if selected_count == 0:
            raise RuntimeError("Selected policy batch contains no queries.")
        with torch.autocast(
            device_type=device.type,
            dtype=AMP_DTYPE,
            enabled=USE_AMP,
        ):
            policy_step = direct_policy_batch(
                model,
                policy,
                raw_images,
                label_state,
                neighbor_indices_cpu,
                batch_indices_cpu,
                device,
                mean,
                std,
                actions=actions,
            )
            selected_in_batch = selected_in_batch_cpu.to(
                device=device,
                non_blocking=True,
            )
            loss = -(
                q_value.detach()
                * policy_step.log_probabilities[selected_in_batch].sum()
                / query_count
            )
        scaler.scale(loss).backward()
        total_loss += loss.detach()
        processed_queries += selected_count
        if batch_number % 100 == 0 or batch_number == total_batches:
            print(
                f"[DIRECT ACTOR UPDATE] batch={batch_number}/"
                f"{total_batches} queries={processed_queries}/{query_count}"
            )

    scaler.step(optimizer)
    scaler.update()
    if processed_queries != query_count:
        raise RuntimeError("Direct actor update did not cover every query.")
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


def build_engine_config() -> Config:
    cfg = Config()
    return replace(
        cfg,
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
            critic_lr=CRITIC_LR,
            critic_momentum=CRITIC_MOMENTUM,
            critic_weight_decay=CRITIC_WEIGHT_DECAY,
            critic_num_bins=CRITIC_NUM_BINS,
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
        f"policy_update_batch={POLICY_UPDATE_BATCH_SIZE} "
        "actor_update_implementation=direct_end_to_end"
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
        f"pretrained={PRETRAINED} warmup_epochs={WARMUP_EPOCHS} "
        f"warmup_model_id={WARMUP_MODEL_ID or 'unset'} "
        f"warmup_batch={WARMUP_BATCH_SIZE} "
        "warmup_min_noisy_val_acc="
        f"{WARMUP_MIN_NOISY_VALIDATION_ACCURACY}"
    )
    print(
        "warmup_optimizer=SGD warmup_scheduler=step_halfway "
        "warmup_deploy_checkpoint=best rl_checkpoints=best_and_last"
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
    if not CONFIGURED:
        raise RuntimeError(
            "Configure rl_engine through cifar_common.configure_engine() first."
        )
    run_started = time.perf_counter()
    if RL_EPOCHS <= 0:
        raise ValueError("RL_EPOCHS must be positive.")
    if EXTERNAL_WARMUP_CHECKPOINT_PATH is None:
        raise RuntimeError(
            "CIFAR RL requires the warmup checkpoint configured in "
            "cifar_config.py. Run cifar_warmup.py first."
        )
    if not 0 < POLICY_UPDATE_SAMPLES <= EXPECTED_SAMPLES:
        raise ValueError(
            "POLICY_UPDATE_SAMPLES must be in [1, EXPECTED_SAMPLES]."
        )
    if POLICY_UPDATE_BATCH_SIZE > POLICY_UPDATE_SAMPLES:
        raise ValueError(
            "POLICY_UPDATE_BATCH_SIZE cannot exceed POLICY_UPDATE_SAMPLES."
        )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    actor_best_checkpoint_path = (
        MODEL_OUTPUT_DIR / ACTOR_BEST_CHECKPOINT_FILENAME
    )
    actor_last_checkpoint_path = (
        MODEL_OUTPUT_DIR / ACTOR_LAST_CHECKPOINT_FILENAME
    )
    critic_best_checkpoint_path = (
        MODEL_OUTPUT_DIR / CRITIC_BEST_CHECKPOINT_FILENAME
    )
    critic_last_checkpoint_path = (
        MODEL_OUTPUT_DIR / CRITIC_LAST_CHECKPOINT_FILENAME
    )
    train_csv_path = OUTPUT_DIR / TRAIN_CSV_FILENAME
    timing_csv_path = OUTPUT_DIR / TIMING_CSV_FILENAME
    run_summary_path = OUTPUT_DIR / RUN_SUMMARY_CSV_FILENAME
    write_csv(train_csv_path, [], SUMMARY_FIELDS)

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
    val_images, val_clean_labels = evaluation_splits["val"]
    val_noisy_labels, val_noise_mask = measure(
        "val_noise_injection",
        device,
        timings,
        lambda: inject_stratified_symmetric_noise(val_clean_labels),
    )
    evaluation_data = {
        "val": (
            val_images,
            val_clean_labels,
            val_noisy_labels,
            val_noise_mask,
        )
    }

    print_configuration(
        device,
        clean_labels_cpu,
        noisy_labels_cpu,
        noise_mask_cpu,
    )
    print(f"validation_samples={evaluation_data['val'][0].size(0)}")
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
        lambda: warm_device_kernels(
            model,
            raw_images,
            device,
            mean,
            std,
            direct_actor=True,
        ),
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
            policy_neighbors_cpu = measure(
                "policy_knn_host_cache",
                device,
                timings,
                lambda: pin_for_cuda(policy_neighbors.detach().cpu()),
                step=global_step,
            )
            del policy_embeddings, policy_neighbors
            correction = measure(
                "direct_full_correction",
                device,
                timings,
                lambda: direct_policy_correction(
                    model,
                    policy,
                    raw_images,
                    label_state,
                    policy_neighbors_cpu,
                    device,
                    mean,
                    std,
                ),
                step=global_step,
            )

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
            actor_loss = measure(
                "direct_actor_update",
                device,
                timings,
                lambda: update_actor_direct(
                    model,
                    policy,
                    actor_optimizer,
                    scaler,
                    raw_images,
                    label_state,
                    policy_neighbors_cpu,
                    correction.actions,
                    q_value,
                    query_indices_cpu,
                    device,
                    mean,
                    std,
                ),
                step=global_step,
            )
            del policy_neighbors_cpu

            step_critic_losses: list[float] = []

            def perform_critic_updates() -> tuple[float, ...]:
                if previous_encoding is not None and previous_reward is not None:
                    critic_loss = update_critic(
                        critic,
                        critic_optimizer,
                        previous_encoding,
                        previous_reward,
                        encoding,
                    )
                    step_critic_losses.append(critic_loss)
                if step == TRAJECTORY_LENGTH:
                    critic_loss = update_critic(
                        critic,
                        critic_optimizer,
                        encoding,
                        reward_output.total_reward,
                        None,
                    )
                    step_critic_losses.append(critic_loss)
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
            del correction, reward_output

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

    print_timing_summary(timings)

    setup_names = {
        f"{DATASET_STAGE_PREFIX}_load",
        f"{DATASET_STAGE_PREFIX}_eval_load",
        "noise_injection",
        "noise_artifact_load",
        "val_noise_injection",
        "model_init",
        "kernel_warmup",
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
    }
    checkpoint_total = sum(
        sum(values)
        for name, values in timings.items()
        if name in checkpoint_names
    )
    peak_allocated = torch.cuda.max_memory_allocated() / 1024**3
    peak_reserved = torch.cuda.max_memory_reserved() / 1024**3
    measured_total = sum(sum(values) for values in timings.values())
    total_runtime = time.perf_counter() - run_started
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
                "warmup_checkpoint_load_seconds": sum(
                    timings.get("warmup_checkpoint_load", ())
                ),
                "rl_epochs": RL_EPOCHS,
                "trajectory_length": TRAJECTORY_LENGTH,
                "initial_state_randomization_rate": (
                    INITIAL_STATE_RANDOMIZATION_RATE
                ),
                "feature_batch_size": FEATURE_BATCH_SIZE,
                "policy_update_mode": ACTOR_UPDATE_MODE,
                "policy_update_samples": POLICY_UPDATE_SAMPLES,
                "policy_update_batch_size": POLICY_UPDATE_BATCH_SIZE,
                "actor_update_implementation": "direct_end_to_end",
                "actor_optimizer": "sgd",
                "actor_learning_rate": ACTOR_LR,
                "actor_momentum": ACTOR_MOMENTUM,
                "actor_weight_decay": ACTOR_WEIGHT_DECAY,
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
                "rl_last_epoch": RL_EPOCHS,
                "rl_last_validation_accuracy": val_summary["accuracy"],
                "rl_last_validation_balanced_accuracy": val_summary[
                    "balanced_accuracy"
                ],
                "rl_last_validation_macro_f1": val_summary["macro_f1"],
                "rl_last_validation_loss": val_summary["loss"],
                "actor_best_checkpoint": str(actor_best_checkpoint_path),
                "actor_last_checkpoint": str(actor_last_checkpoint_path),
                "critic_best_checkpoint": str(critic_best_checkpoint_path),
                "critic_last_checkpoint": str(critic_last_checkpoint_path),
                "setup_seconds": setup_total,
                "train_seconds": sum(epoch_seconds),
                "mean_epoch_seconds": sum(epoch_seconds) / len(epoch_seconds),
                "validation_seconds": sum(validation_seconds),
                "checkpoint_seconds": checkpoint_total,
                "measured_total_seconds": measured_total,
                "total_runtime_seconds": total_runtime,
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
        f"rl_last_epoch={RL_EPOCHS}, "
        f"rl_last_val_macro_f1={float(val_summary['macro_f1']):.6f}, "
        "next_stage=cifar_test/cifar_correction.py"
    )
    print(f"setup_seconds={setup_total:.3f}")
    print(f"train_seconds={sum(epoch_seconds):.3f}")
    print(f"checkpoint_seconds={checkpoint_total:.3f}")
    print(f"total_runtime_seconds={total_runtime:.3f}")
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
    print(f"train_csv={train_csv_path}")
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
