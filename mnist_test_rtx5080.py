"""MNIST RLNLC actor-ablation speed profile for an RTX 5080 (16 GiB).

Experiment setup
----------------
Data:
    - Full MNIST train split: all 60,000 images and all ten digits.
    - Validation/test: stratified 50/50 split of the official test set.
    - Label noise: 40% stratified symmetric noise with seed 0.

Backbone and supervised warmup:
    - ConvNeXtV2 Tiny initialized without pretrained weights.
    - 1 warmup epoch using only noisy labels for optimization.
    - The randomly initialized backbone is trainable from epoch 1.
    - Backbone lr=1e-4 and head lr=5e-4.
    - Cosine decay to 1e-6, weight decay=0.05, label smoothing=0.1.
    - Batch size=64 and gradient clipping max norm=1.0.
    - Best warmup checkpoint is selected by noisy-validation accuracy.
    - Clean labels are used only for benchmark reporting, not optimization or
      warmup checkpoint selection.
    - RL does not start if best noisy-validation accuracy is below 45%.

RLNLC:
    - 1 timed outer epoch and trajectory length 10 (10 total RL steps).
    - Exact Euclidean KNN with k=10; cosine attention temperature=0.5.
    - Actor AdamW lr=3e-5, weight decay=0.1.
    - Critic SGD lr=1e-2, momentum=0.9.
    - Discount factor=0.9 and noisy-label-alignment weight=0.5.
    - Three selectable actor profiles isolate subset and gradient-path costs:
      2,048 query-only, 60,000 query-only, and 60,000 query+neighbor.
    - Exact KNN is streamed in 2,048 x 32,768 query/reference chunks.

Runtime:
    - CUDA BF16 autocast and channels-last memory format.
    - Feature extraction batch size=256.
    - Data: K:/rlnlc/data/mnist
    - Outputs are separated automatically by ACTOR_UPDATE_PROFILE.

The constants immediately below are the executable source of truth. Keep this
profile note synchronized whenever those values change.
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
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR
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


# Local RTX 5080 (16 GiB) validation profile.
# Keep data and generated reports inside this workspace so the run is portable.
PROJECT_ROOT = Path(__file__).resolve().parent
MNIST_ROOT = PROJECT_ROOT / "data" / "mnist"
RUN_LOG_FILENAME = "run.log"
WARMUP_CSV_FILENAME = "warmup.csv"
WARMUP_CHECKPOINT_FILENAME = "warmup_best.pt"
RL_BEST_CHECKPOINT_FILENAME = "rl_best.pt"
RL_LAST_CHECKPOINT_FILENAME = "rl_last.pt"
TRAIN_CSV_FILENAME = "train.csv"
TEST_CSV_FILENAME = "test.csv"
TEST_PER_CLASS_CSV_FILENAME = "test_per_class.csv"
TIMING_CSV_FILENAME = "timing.csv"
RUN_SUMMARY_CSV_FILENAME = "run_summary.csv"
DOWNLOAD_MNIST = True
DIGITS = tuple(range(10))
NUM_CLASSES = len(DIGITS)
EXPECTED_SAMPLES = 60_000
NOISE_RATE = 0.40

# Change only this value between actor ablation runs.
# - subset_query_only: optimized baseline already used in the prior run.
# - full_query_only: isolates the cost of expanding 2,048 to all 60,000 queries.
# - full_query_neighbor: additionally propagates gradients through neighbors.
ACTOR_UPDATE_PROFILE = "full_query_neighbor"
ACTOR_UPDATE_PROFILES: dict[str, tuple[int, bool]] = {
    "subset_query_only": (2_048, False),
    "full_query_only": (EXPECTED_SAMPLES, False),
    "full_query_neighbor": (EXPECTED_SAMPLES, True),
}
if ACTOR_UPDATE_PROFILE not in ACTOR_UPDATE_PROFILES:
    raise ValueError(
        "ACTOR_UPDATE_PROFILE must be one of "
        f"{tuple(ACTOR_UPDATE_PROFILES)}."
    )
POLICY_UPDATE_SAMPLES, NEIGHBOR_GRADIENT = ACTOR_UPDATE_PROFILES[
    ACTOR_UPDATE_PROFILE
]
OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / f"mnist_speed_rtx5080_{ACTOR_UPDATE_PROFILE}"
)

MODEL_NAME = "convnextv2_tiny.fcmae_ft_in22k_in1k"
PRETRAINED = False
IMAGE_SIZE = 224
DROP_RATE = 0.1
DROP_PATH_RATE = 0.2

WARMUP_EPOCHS = 1
WARMUP_FREEZE_EPOCHS = 0
WARMUP_BATCH_SIZE = 64
WARMUP_EVAL_BATCH_SIZE = 256
WARMUP_HEAD_LR = 1e-3
WARMUP_BACKBONE_LR = 1e-4
WARMUP_UNFROZEN_HEAD_LR = 5e-4
WARMUP_MIN_LR = 1e-6
WARMUP_WEIGHT_DECAY = 0.05
WARMUP_LABEL_SMOOTHING = 0.1
WARMUP_GRAD_CLIP_NORM = 1.0
WARMUP_MIN_NOISY_VALIDATION_ACCURACY = 0.45

RL_EPOCHS = 1
TRAJECTORY_LENGTH = 10
FEATURE_BATCH_SIZE = 256
POLICY_UPDATE_BATCH_SIZE = 64

K = 10
TEMPERATURE = 0.5
KNN_QUERY_CHUNK_SIZE = 2_048
KNN_REFERENCE_CHUNK_SIZE = 32_768
CORRECTION_CHUNK_SIZE = 16_384

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
LR_DECAY_FRACTION = 0.5

USE_AMP = True
AMP_DTYPE = torch.bfloat16
USE_CHANNELS_LAST = True
CUDNN_BENCHMARK = True
SEED = 0

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
T = TypeVar("T")
Timings = dict[str, list[float]]

WARMUP_FIELDS = (
    "epoch",
    "phase",
    "backbone_lr",
    "head_lr",
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

TIMING_FIELDS = (
    "stage",
    "calls",
    "total_seconds",
    "mean_seconds",
    "percentage",
)

RUN_SUMMARY_FIELDS = (
    "seed",
    "noise_rate",
    "train_samples",
    "validation_samples",
    "test_samples",
    "pretrained",
    "warmup_epochs",
    "warmup_freeze_epochs",
    "warmup_batch_size",
    "warmup_best_epoch",
    "warmup_best_noisy_validation_accuracy",
    "warmup_best_clean_validation_accuracy",
    "warmup_seconds",
    "rl_epochs",
    "trajectory_length",
    "feature_batch_size",
    "policy_update_mode",
    "policy_update_samples",
    "policy_update_batch_size",
    "neighbor_gradient",
    "k",
    "knn_query_chunk_size",
    "knn_reference_chunk_size",
    "correction_chunk_size",
    "rl_best_epoch",
    "rl_best_validation_accuracy",
    "rl_best_validation_balanced_accuracy",
    "rl_best_validation_macro_f1",
    "rl_best_validation_loss",
    "rl_best_checkpoint",
    "rl_last_checkpoint",
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


def resolve_local_device() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "The local MNIST benchmark requires CUDA, but torch.cuda.is_available() "
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


def load_mnist_validation_test() -> dict[str, tuple[Tensor, Tensor]]:
    """Split the official MNIST test set 50/50 within every digit."""
    dataset = MNIST(
        root=MNIST_ROOT,
        train=False,
        download=DOWNLOAD_MNIST,
    )
    labels = dataset.targets.to(torch.long).contiguous()
    generator = torch.Generator().manual_seed(SEED)
    split_indices: dict[str, list[Tensor]] = {"val": [], "test": []}
    for digit in DIGITS:
        indices = labels.eq(digit).nonzero(as_tuple=False).flatten()
        indices = indices[
            torch.randperm(indices.numel(), generator=generator)
        ]
        val_count = indices.numel() // 2
        split_indices["val"].append(indices[:val_count])
        split_indices["test"].append(indices[val_count:])

    result: dict[str, tuple[Tensor, Tensor]] = {}
    for split, chunks in split_indices.items():
        indices = torch.cat(chunks)
        indices = indices[
            torch.randperm(indices.numel(), generator=generator)
        ]
        images = pin_for_cuda(dataset.data[indices].contiguous())
        split_labels = pin_for_cuda(labels[indices].contiguous())
        if set(split_labels.tolist()) != set(DIGITS):
            raise RuntimeError(f"MNIST {split} split is missing a digit class.")
        result[split] = (images, split_labels)

    if result["val"][0].size(0) + result["test"][0].size(0) != len(dataset):
        raise RuntimeError("MNIST validation/test split lost samples.")
    return result


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

    class_sizes = [int(clean_labels.eq(digit).sum()) for digit in DIGITS]
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

    for digit, noise_count in zip(DIGITS, noise_counts):
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
    for class_id in DIGITS:
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


def _warmup_head(model: nn.Module) -> nn.Module:
    head = getattr(model, "head", None)
    if not isinstance(head, nn.Module):
        raise TypeError("The warmup model must expose a .head module.")
    return head


def _set_warmup_phase(model: nn.Module, *, backbone_frozen: bool) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = not backbone_frozen
    if backbone_frozen:
        for parameter in _warmup_head(model).parameters():
            parameter.requires_grad = True
        model.eval()
        _warmup_head(model).train()
    else:
        model.train()


def _build_warmup_optimizer(
    model: nn.Module,
    *,
    backbone_frozen: bool,
) -> AdamW:
    head = _warmup_head(model)
    if backbone_frozen:
        parameter_groups = [
            {
                "params": tuple(head.parameters()),
                "lr": WARMUP_HEAD_LR,
                "name": "head",
            }
        ]
    else:
        head_parameter_ids = {id(parameter) for parameter in head.parameters()}
        backbone_parameters = tuple(
            parameter
            for parameter in model.parameters()
            if id(parameter) not in head_parameter_ids
        )
        parameter_groups = [
            {
                "params": backbone_parameters,
                "lr": WARMUP_BACKBONE_LR,
                "name": "backbone",
            },
            {
                "params": tuple(head.parameters()),
                "lr": WARMUP_UNFROZEN_HEAD_LR,
                "name": "head",
            },
        ]
    return AdamW(
        parameter_groups,
        weight_decay=WARMUP_WEIGHT_DECAY,
        betas=ACTOR_BETAS,
        eps=ACTOR_EPS,
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
            },
            temporary_path,
        )
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)


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


def _save_rl_checkpoint(
    path: Path,
    *,
    epoch: int,
    model: nn.Module,
    critic: nn.Module,
    actor_optimizer: AdamW,
    critic_optimizer: torch.optim.Optimizer,
    actor_scheduler: MultiStepLR,
    critic_scheduler: MultiStepLR,
    scaler: torch.amp.GradScaler,
    label_state: Tensor,
    validation_summary: dict[str, object],
) -> None:
    """Atomically save a complete RL state for testing or continuation."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    validation_metrics = {
        "accuracy": float(validation_summary["accuracy"]),
        "balanced_accuracy": float(validation_summary["balanced_accuracy"]),
        "macro_f1": float(validation_summary["macro_f1"]),
        "loss": float(validation_summary["loss"]),
    }
    try:
        torch.save(
            {
                "version": 1,
                "epoch": epoch,
                "model_name": MODEL_NAME,
                "num_classes": NUM_CLASSES,
                "pretrained": PRETRAINED,
                "selection_order": (
                    "macro_f1",
                    "balanced_accuracy",
                    "loss",
                ),
                "validation": validation_metrics,
                "model": model.state_dict(),
                "critic": critic.state_dict(),
                "actor_optimizer": actor_optimizer.state_dict(),
                "critic_optimizer": critic_optimizer.state_dict(),
                "actor_scheduler": actor_scheduler.state_dict(),
                "critic_scheduler": critic_scheduler.state_dict(),
                "amp_scaler": scaler.state_dict(),
                "label_state": label_state.detach().cpu(),
                "cpu_rng_state": torch.get_rng_state(),
                "cuda_rng_states": torch.cuda.get_rng_state_all(),
            },
            temporary_path,
        )
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _restore_rl_model(
    model: nn.Module,
    checkpoint_path: Path,
    device: torch.device,
) -> dict[str, object]:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"RL checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )
    if not isinstance(checkpoint, dict):
        raise TypeError("RL checkpoint must contain a dictionary.")
    required = {"epoch", "model_name", "num_classes", "validation", "model"}
    missing = required.difference(checkpoint)
    if missing:
        raise KeyError(f"RL checkpoint is missing fields: {sorted(missing)}")
    if checkpoint["model_name"] != MODEL_NAME:
        raise ValueError("RL checkpoint model name does not match this run.")
    if checkpoint["num_classes"] != NUM_CLASSES:
        raise ValueError("RL checkpoint class count does not match this run.")
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
) -> dict[str, float | int]:
    """Warm up semantic features using only the 40%-noisy labels."""
    if not 0 <= WARMUP_FREEZE_EPOCHS < WARMUP_EPOCHS:
        raise ValueError("WARMUP_FREEZE_EPOCHS must be in [0, WARMUP_EPOCHS).")
    criterion = nn.CrossEntropyLoss(
        label_smoothing=WARMUP_LABEL_SMOOTHING,
    )
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=USE_AMP and AMP_DTYPE == torch.float16,
    )
    optimizer: AdamW | None = None
    scheduler: CosineAnnealingLR | None = None
    history: list[dict[str, object]] = []
    best_noisy_accuracy = float("-inf")
    best_clean_accuracy = float("nan")
    best_epoch = 0

    for epoch in range(1, WARMUP_EPOCHS + 1):
        epoch_started = time.perf_counter()
        backbone_frozen = epoch <= WARMUP_FREEZE_EPOCHS
        if epoch == 1 or epoch == WARMUP_FREEZE_EPOCHS + 1:
            _set_warmup_phase(model, backbone_frozen=backbone_frozen)
            optimizer = _build_warmup_optimizer(
                model,
                backbone_frozen=backbone_frozen,
            )
            if not backbone_frozen:
                scheduler = CosineAnnealingLR(
                    optimizer,
                    T_max=WARMUP_EPOCHS - WARMUP_FREEZE_EPOCHS,
                    eta_min=WARMUP_MIN_LR,
                )
        else:
            _set_warmup_phase(model, backbone_frozen=backbone_frozen)
        if optimizer is None:
            raise RuntimeError("Warmup optimizer was not initialized.")

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
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                WARMUP_GRAD_CLIP_NORM,
            )
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
        learning_rates = {
            str(group.get("name")): float(group["lr"])
            for group in optimizer.param_groups
        }
        row: dict[str, object] = {
            "epoch": epoch,
            "phase": "head" if backbone_frozen else "full",
            "backbone_lr": learning_rates.get("backbone", 0.0),
            "head_lr": learning_rates["head"],
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
            f"phase={row['phase']} "
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
        if scheduler is not None:
            scheduler.step()

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device)
    model.eval()
    print(
        f"[WARMUP BEST] epoch={best_epoch} "
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
    neighbor_gradient: bool,
    device: torch.device,
) -> tuple[Tensor, float]:
    """Differentiate the selected policy loss through configured paths.

    KNN indices are discrete and fixed within the current RL step. Query
    embeddings always receive gradients; neighbor embeddings receive them only
    when ``neighbor_gradient`` is enabled.
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
        neighbor_embeddings = embedding_leaf[batch_neighbors]
        if not neighbor_gradient:
            neighbor_embeddings = neighbor_embeddings.detach()
        policy_step = policy(
            embedding_leaf[batch_indices],
            neighbor_embeddings,
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
                f"queries={end}/{query_count} "
                f"neighbor_gradient={neighbor_gradient}"
            )

    embedding_gradient = embedding_leaf.grad
    if embedding_gradient is None:
        raise RuntimeError("Policy loss did not produce an embedding gradient.")
    embedding_gradient = embedding_gradient.detach()
    del embedding_leaf
    return embedding_gradient, float(total_loss)


def update_backbone_from_embedding_gradient(
    model: nn.Module,
    optimizer: AdamW,
    scaler: torch.amp.GradScaler,
    raw_images: Tensor,
    embedding_gradient: Tensor,
    update_indices_cpu: Tensor,
    device: torch.device,
    mean: Tensor,
    std: Tensor,
) -> None:
    """Map the complete embedding gradient through the actor backbone.

    This batched vector-Jacobian product is equivalent to retaining one
    dataset-wide actor autograd graph, without requiring that graph to fit in
    GPU memory.
    """
    model.eval()
    optimizer.zero_grad(set_to_none=True)
    sample_count = raw_images.size(0)
    if (
        embedding_gradient.ndim != 2
        or embedding_gradient.size(0) != sample_count
    ):
        raise ValueError("embedding_gradient must have shape [N, D].")
    if update_indices_cpu.ndim != 1 or update_indices_cpu.numel() == 0:
        raise ValueError("update_indices_cpu must be a non-empty vector.")
    if update_indices_cpu.device.type != "cpu":
        raise ValueError("update_indices_cpu must remain on CPU.")
    update_count = update_indices_cpu.numel()
    use_full_order = update_count == sample_count
    selected_images = (
        raw_images
        if use_full_order
        else pin_for_cuda(raw_images[update_indices_cpu])
    )
    update_indices = update_indices_cpu.to(device=device, non_blocking=True)
    total_batches = math.ceil(update_count / POLICY_UPDATE_BATCH_SIZE)

    for batch_number, start in enumerate(
        range(0, update_count, POLICY_UPDATE_BATCH_SIZE),
        start=1,
    ):
        end = min(start + POLICY_UPDATE_BATCH_SIZE, update_count)
        batch_indices = update_indices[start:end]
        images = preprocess(selected_images[start:end], device, mean, std)
        with torch.autocast(
            device_type=device.type,
            dtype=AMP_DTYPE,
            enabled=USE_AMP,
        ):
            current_embeddings = encode(model, images)
            surrogate = (
                current_embeddings.float() * embedding_gradient[batch_indices]
            ).sum()
        scaler.scale(surrogate).backward()
        if batch_number % 100 == 0 or batch_number == total_batches:
            print(
                f"[ACTOR BACKBONE VJP] batch={batch_number}/{total_batches} "
                f"samples={end}/{update_count}"
            )

    scaler.step(optimizer)
    scaler.update()


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
            discount_factor=DISCOUNT_FACTOR,
            actor_lr=ACTOR_LR,
            actor_weight_decay=ACTOR_WEIGHT_DECAY,
            actor_adamw_betas=ACTOR_BETAS,
            actor_adamw_eps=ACTOR_EPS,
            critic_lr=CRITIC_LR,
            critic_momentum=CRITIC_MOMENTUM,
            critic_weight_decay=CRITIC_WEIGHT_DECAY,
            use_policy_update_subset=(
                POLICY_UPDATE_SAMPLES < EXPECTED_SAMPLES
            ),
            policy_update_subset_size=POLICY_UPDATE_SAMPLES,
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
    print("MNIST local RLNLC validation benchmark")
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
        f"actor_update_profile={ACTOR_UPDATE_PROFILE} "
        f"policy_update_samples={POLICY_UPDATE_SAMPLES} "
        f"policy_update_batch={POLICY_UPDATE_BATCH_SIZE} "
        f"neighbor_gradient={NEIGHBOR_GRADIENT}"
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
        f"actor=AdamW(lr={ACTOR_LR}, wd={ACTOR_WEIGHT_DECAY}) "
        f"critic=SGD(lr={CRITIC_LR}, momentum={CRITIC_MOMENTUM})"
    )
    print(
        f"pretrained={PRETRAINED} warmup_epochs={WARMUP_EPOCHS} "
        f"warmup_freeze_epochs={WARMUP_FREEZE_EPOCHS} "
        f"warmup_batch={WARMUP_BATCH_SIZE} "
        f"warmup_label_smoothing={WARMUP_LABEL_SMOOTHING} "
        "warmup_min_noisy_val_acc="
        f"{WARMUP_MIN_NOISY_VALIDATION_ACCURACY}"
    )
    print(
        "warmup_selection=noisy_validation_accuracy "
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
    if not 0 < POLICY_UPDATE_SAMPLES <= EXPECTED_SAMPLES:
        raise ValueError(
            "POLICY_UPDATE_SAMPLES must be in [1, EXPECTED_SAMPLES]."
        )
    if POLICY_UPDATE_BATCH_SIZE > POLICY_UPDATE_SAMPLES:
        raise ValueError(
            "POLICY_UPDATE_BATCH_SIZE cannot exceed POLICY_UPDATE_SAMPLES."
        )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    warmup_csv_path = OUTPUT_DIR / WARMUP_CSV_FILENAME
    warmup_checkpoint_path = OUTPUT_DIR / WARMUP_CHECKPOINT_FILENAME
    rl_best_checkpoint_path = OUTPUT_DIR / RL_BEST_CHECKPOINT_FILENAME
    rl_last_checkpoint_path = OUTPUT_DIR / RL_LAST_CHECKPOINT_FILENAME
    train_csv_path = OUTPUT_DIR / TRAIN_CSV_FILENAME
    test_csv_path = OUTPUT_DIR / TEST_CSV_FILENAME
    test_per_class_path = OUTPUT_DIR / TEST_PER_CLASS_CSV_FILENAME
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
        "mnist_load",
        device,
        timings,
        load_full_mnist_train,
    )
    evaluation_splits = measure(
        "mnist_eval_load",
        device,
        timings,
        load_mnist_validation_test,
    )
    noisy_labels_cpu, noise_mask_cpu = measure(
        "noise_injection",
        device,
        timings,
        lambda: inject_stratified_symmetric_noise(clean_labels_cpu),
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
    cfg = build_benchmark_config()

    model = measure(
        "model_init",
        device,
        timings,
        lambda: timm.create_model(
            MODEL_NAME,
            pretrained=PRETRAINED,
            num_classes=NUM_CLASSES,
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

    val_images, val_clean_labels, val_noisy_labels, _ = evaluation_data["val"]
    warmup_result = measure(
        "supervised_warmup",
        device,
        timings,
        lambda: train_supervised_warmup(
            model,
            raw_images,
            noisy_labels_cpu,
            val_images,
            val_noisy_labels,
            val_clean_labels,
            device,
            mean,
            std,
            warmup_csv_path,
            warmup_checkpoint_path,
        ),
    )
    reset_classifier = getattr(model, "reset_classifier", None)
    if not callable(reset_classifier):
        raise TypeError("The warmup model cannot remove its classifier.")
    reset_classifier(0)
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
    actor_optimizer = AdamW(
        model.parameters(),
        lr=ACTOR_LR,
        weight_decay=ACTOR_WEIGHT_DECAY,
        betas=ACTOR_BETAS,
        eps=ACTOR_EPS,
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

    for epoch in range(1, RL_EPOCHS + 1):
        synchronize(device)
        epoch_started = time.perf_counter()
        label_state = F.one_hot(
            initial_noisy_labels,
            num_classes=NUM_CLASSES,
        ).to(torch.float32)
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
            embedding_gradient, actor_loss = measure(
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
                    NEIGHBOR_GRADIENT,
                    device,
                ),
                step=global_step,
            )
            backbone_indices_cpu = (
                torch.arange(EXPECTED_SAMPLES)
                if NEIGHBOR_GRADIENT
                else query_indices_cpu
            )
            measure(
                "actor_backbone_update",
                device,
                timings,
                lambda: update_backbone_from_embedding_gradient(
                    model,
                    actor_optimizer,
                    scaler,
                    raw_images,
                    embedding_gradient,
                    backbone_indices_cpu,
                    device,
                    mean,
                    std,
                ),
                step=global_step,
            )
            del embedding_gradient

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

        def save_epoch_checkpoint(path: Path) -> None:
            _save_rl_checkpoint(
                path,
                epoch=epoch,
                model=model,
                critic=critic,
                actor_optimizer=actor_optimizer,
                critic_optimizer=critic_optimizer,
                actor_scheduler=actor_scheduler,
                critic_scheduler=critic_scheduler,
                scaler=scaler,
                label_state=label_state,
                validation_summary=val_summary,
            )

        measure(
            "rl_last_checkpoint",
            device,
            timings,
            lambda: save_epoch_checkpoint(rl_last_checkpoint_path),
            step=epoch,
        )
        validation_key = _rl_selection_key(val_summary)
        if best_rl_key is None or validation_key > best_rl_key:
            measure(
                "rl_best_checkpoint",
                device,
                timings,
                lambda: save_epoch_checkpoint(rl_best_checkpoint_path),
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

    restored_checkpoint = measure(
        "rl_best_restore",
        device,
        timings,
        lambda: _restore_rl_model(model, rl_best_checkpoint_path, device),
    )
    restored_epoch = int(restored_checkpoint["epoch"])
    if restored_epoch != best_rl_epoch:
        raise RuntimeError(
            "Restored RL checkpoint epoch does not match the selected best epoch: "
            f"{restored_epoch} != {best_rl_epoch}."
        )
    del restored_checkpoint
    print(
        f"[RL RESTORE] checkpoint={rl_best_checkpoint_path} "
        f"epoch={restored_epoch}"
    )

    test_images, test_clean, test_noisy, test_noise_mask = evaluation_data["test"]
    test_summary, test_per_class = evaluate_correction_split(
        split="test",
        epoch=best_rl_epoch,
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
        "mnist_load",
        "mnist_eval_load",
        "noise_injection",
        "val_noise_injection",
        "test_noise_injection",
        "model_init",
        "kernel_warmup",
        "supervised_warmup",
        "global_cache_feature_extraction",
        "global_cache_exact_knn",
    }
    setup_total = sum(
        sum(values) for name, values in timings.items() if name in setup_names
    )
    checkpoint_names = {
        "rl_last_checkpoint",
        "rl_best_checkpoint",
        "rl_best_restore",
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
                "seed": SEED,
                "noise_rate": NOISE_RATE,
                "train_samples": EXPECTED_SAMPLES,
                "validation_samples": evaluation_data["val"][0].size(0),
                "test_samples": evaluation_data["test"][0].size(0),
                "pretrained": PRETRAINED,
                "warmup_epochs": WARMUP_EPOCHS,
                "warmup_freeze_epochs": WARMUP_FREEZE_EPOCHS,
                "warmup_batch_size": WARMUP_BATCH_SIZE,
                "warmup_best_epoch": warmup_result["best_epoch"],
                "warmup_best_noisy_validation_accuracy": warmup_result[
                    "best_noisy_validation_accuracy"
                ],
                "warmup_best_clean_validation_accuracy": warmup_result[
                    "best_clean_validation_accuracy"
                ],
                "warmup_seconds": sum(timings["supervised_warmup"]),
                "rl_epochs": RL_EPOCHS,
                "trajectory_length": TRAJECTORY_LENGTH,
                "feature_batch_size": FEATURE_BATCH_SIZE,
                "policy_update_mode": ACTOR_UPDATE_PROFILE,
                "policy_update_samples": POLICY_UPDATE_SAMPLES,
                "policy_update_batch_size": POLICY_UPDATE_BATCH_SIZE,
                "neighbor_gradient": NEIGHBOR_GRADIENT,
                "k": K,
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
                "rl_best_checkpoint": str(rl_best_checkpoint_path),
                "rl_last_checkpoint": str(rl_last_checkpoint_path),
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
    print(f"warmup_csv={warmup_csv_path}")
    print(f"warmup_checkpoint={warmup_checkpoint_path}")
    print(f"rl_best_checkpoint={rl_best_checkpoint_path}")
    print(f"rl_last_checkpoint={rl_last_checkpoint_path}")
    print(f"train_csv={train_csv_path}")
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
