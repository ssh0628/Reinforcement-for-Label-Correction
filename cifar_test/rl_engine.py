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
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import asdict, dataclass, fields, replace
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
from rl.critic.critic import build_critic, build_critic_optimizer, sarsa_td_loss
from rl.reward.reward import RLNLCReward
from setting.config import Config


CONFIGURED = False
RUN_LOG_FILENAME = "run.log"
TRAIN_CSV_FILENAME = "train.csv"
TIMING_CSV_FILENAME = "timing.csv"
RUN_SUMMARY_CSV_FILENAME = "run_summary.csv"
CHANGE_DIAGNOSTICS_CSV_FILENAME = "change_diagnostics.csv"
DATASET_NAME: str
ACTOR_BEST_CHECKPOINT_FILENAME: str
ACTOR_LAST_CHECKPOINT_FILENAME: str
CRITIC_BEST_CHECKPOINT_FILENAME: str
CRITIC_LAST_CHECKPOINT_FILENAME: str
CLASS_IDS: tuple[int, ...]
NUM_CLASSES: int
EXPECTED_SAMPLES: int
TRAIN_SUBSET_SEED: int
NOISE_RATE: float
EXTERNAL_NOISY_LABELS_PATH: Path
EXTERNAL_NOISE_MASK_PATH: Path
EXTERNAL_WARMUP_CHECKPOINT_PATH: Path
ACTOR_UPDATE_MODE: str
POLICY_UPDATE_SAMPLES: int
RECORD_CHANGE_DIAGNOSTICS: bool
CHANGE_DIAGNOSTIC_PROBE_SIZE: int
OUTPUT_DIR: Path
MODEL_OUTPUT_DIR: Path
MODEL_NAME: str
WARMUP_MODEL_ID: str
PRETRAINED: bool
IMAGE_SIZE: int
MODEL_FACTORY: Callable[[bool, int], nn.Module]
TRAIN_DATA_LOADER: Callable[[], tuple[Tensor, Tensor]]
EVALUATION_DATA_LOADER: Callable[[], dict[str, tuple[Tensor, Tensor]]]
PREPROCESS_FUNCTION: Callable[[Tensor, torch.device, Tensor, Tensor], Tensor]
TRAIN_PREPROCESS_FUNCTION: Callable[[Tensor, torch.device, Tensor, Tensor, torch.Generator], Tensor]
TRAIN_AUGMENTATION_ENABLED: bool
TRAIN_RANDOM_CROP_PADDING: int
TRAIN_HORIZONTAL_FLIP_PROBABILITY: float
WARMUP_EPOCHS: int
WARMUP_BATCH_SIZE: int
WARMUP_EVAL_BATCH_SIZE: int
WARMUP_LR: float
WARMUP_WEIGHT_DECAY: float
WARMUP_MIN_NOISY_VALIDATION_ACCURACY: float
WARMUP_MOMENTUM: float
WARMUP_LR_DECAY_FRACTION: float
WARMUP_LR_DECAY_FACTOR: float
RL_EPOCHS: int
TRAJECTORY_LENGTH: int
INITIAL_STATE_RANDOMIZATION_RATE: float
FEATURE_BATCH_SIZE: int
POLICY_UPDATE_BATCH_SIZE: int
K: int
TEMPERATURE: float
KNN_QUERY_CHUNK_SIZE: int
KNN_REFERENCE_CHUNK_SIZE: int
CORRECTION_CHUNK_SIZE: int
ACTOR_LR: float
ACTOR_WEIGHT_DECAY: float
ACTOR_MOMENTUM: float
CRITIC_LR: float
CRITIC_MOMENTUM: float
CRITIC_WEIGHT_DECAY: float
CRITIC_NUM_BINS: int
DISCOUNT_FACTOR: float
NLA_WEIGHT: float
LR_DECAY_FACTOR: float
LR_DECAY_FRACTION: float
USE_AMP: bool
AMP_DTYPE: torch.dtype
USE_CHANNELS_LAST: bool
CUDNN_BENCHMARK: bool
SEED: int
DATA_MEAN: tuple[float, float, float]
DATA_STD: tuple[float, float, float]
T = TypeVar("T")
Timings = dict[str, list[float]]


@dataclass(frozen=True, slots=True)
class ChangeDiagnosticRow:
    epoch: int
    trajectory_step: int
    global_step: int
    reference_global_step: int
    steps_since_reference: int
    learning_rate: float
    probe_samples: int
    reference_parameter_norm: float
    step_gradient_norm: float
    step_lr_gradient_norm: float
    cumulative_lr_gradient_norm_before_update: float
    relative_cumulative_lr_gradient_norm_before_update: float
    cumulative_lr_gradient_norm_after_update: float
    relative_cumulative_lr_gradient_norm_after_update: float
    parameter_drift_norm_before_update: float
    relative_parameter_drift_before_update: float
    parameter_drift_norm_after_update: float
    relative_parameter_drift_after_update: float
    feature_mean_cosine_similarity: float
    feature_cosine_drift: float
    knn_neighbor_overlap: float


class ChangeDiagnosticsRecorder:
    def __init__(self, model: nn.Module, optimizer: Optimizer, probe_indices: Tensor) -> None:
        if probe_indices.ndim != 1 or probe_indices.numel() == 0:
            raise ValueError("probe_indices must be a non-empty vector.")

        head = getattr(model, "head", None)
        head_parameter_ids = (
            {id(parameter) for parameter in head.parameters()} if isinstance(head, nn.Module) else set()
        )
        tracked_parameters = [
            parameter
            for parameter in model.parameters()
            if parameter.requires_grad and id(parameter) not in head_parameter_ids
        ]
        if not tracked_parameters:
            raise RuntimeError("No actor-backbone parameters available to track.")
        if probe_indices.device != tracked_parameters[0].device:
            raise ValueError("probe_indices and actor-backbone parameters must share a device.")

        group_by_parameter_id: dict[int, dict[str, object]] = {}
        for group in optimizer.param_groups:
            for parameter in group["params"]:
                parameter_id = id(parameter)
                if parameter_id in group_by_parameter_id:
                    raise RuntimeError("An actor parameter appears in multiple optimizer groups.")
                group_by_parameter_id[parameter_id] = group
        missing = [
            parameter for parameter in tracked_parameters if id(parameter) not in group_by_parameter_id
        ]
        if missing:
            raise RuntimeError("Tracked actor parameters must belong to the actor optimizer.")

        self.parameters = tracked_parameters
        self.optimizer_groups = [group_by_parameter_id[id(parameter)] for parameter in tracked_parameters]
        self.probe_indices = probe_indices
        self.reference_parameters: list[Tensor] = []
        self.cumulative_updates: list[Tensor] = []
        self.reference_embeddings: Tensor | None = None
        self.reference_neighbors: Tensor | None = None
        self.reference_parameter_norm = 0.0
        self.reference_global_step = 0
        self._cumulative_norm_after = 0.0
        self._parameter_drift_after = 0.0
        self._graph_metrics: dict[str, float] | None = None
        self._update_metrics: dict[str, float] | None = None

    @staticmethod
    def _vector_norm(tensors: list[Tensor]) -> float:
        if not tensors:
            return 0.0
        squared_norm = torch.zeros((), device=tensors[0].device)
        for tensor in tensors:
            squared_norm += tensor.float().square().sum()
        return math.sqrt(float(squared_norm))

    def _parameter_drift_norm(self) -> float:
        squared_norm = torch.zeros((), device=self.parameters[0].device)
        for parameter, reference in zip(self.parameters, self.reference_parameters, strict=True):
            squared_norm += (parameter.detach().float() - reference.float()).square().sum()
        return math.sqrt(float(squared_norm))

    def begin_trajectory(self, reference_global_step: int) -> None:
        if reference_global_step <= 0:
            raise ValueError("reference_global_step must be positive.")
        if not self.reference_parameters:
            self.reference_parameters = [parameter.detach().clone() for parameter in self.parameters]
            self.cumulative_updates = [
                torch.zeros_like(parameter, memory_format=torch.preserve_format)
                for parameter in self.parameters
            ]
        else:
            for parameter, reference, cumulative in zip(
                self.parameters, self.reference_parameters, self.cumulative_updates, strict=True
            ):
                reference.copy_(parameter.detach())
                cumulative.zero_()
        self.reference_parameter_norm = self._vector_norm(self.reference_parameters)
        if self.reference_parameter_norm == 0.0:
            raise RuntimeError("Actor-backbone parameter norm must be non-zero.")
        self.reference_embeddings = None
        self.reference_neighbors = None
        self.reference_global_step = reference_global_step
        self._cumulative_norm_after = 0.0
        self._parameter_drift_after = 0.0
        self._graph_metrics = None
        self._update_metrics = None

    @torch.inference_mode()
    def observe_policy_graph(self, embeddings: Tensor, neighbors: Tensor) -> None:
        if self.probe_indices.numel() == embeddings.size(0):
            current_embeddings = embeddings.float()
            current_neighbors = neighbors
        else:
            current_embeddings = embeddings.index_select(0, self.probe_indices).float()
            current_neighbors = neighbors.index_select(0, self.probe_indices)
        first_observation = self.reference_embeddings is None
        if first_observation:
            self.reference_embeddings = current_embeddings.clone()
            self.reference_neighbors = current_neighbors.clone()

        if self.reference_neighbors is None:
            raise RuntimeError("Reference KNN rows were not initialized.")
        if first_observation:
            cosine_similarity = 1.0
            neighbor_overlap = 1.0
        else:
            cosine_similarity = float(
                F.cosine_similarity(current_embeddings, self.reference_embeddings, dim=1)
                .mean()
                .clamp(-1.0, 1.0)
            )
            neighbor_overlap = float(
                current_neighbors.unsqueeze(2)
                .eq(self.reference_neighbors.unsqueeze(1))
                .any(dim=2)
                .float()
                .mean()
            )
        cumulative_before = self._cumulative_norm_after
        parameter_drift_before = self._parameter_drift_after
        self._graph_metrics = {
            "cumulative_before": cumulative_before,
            "relative_cumulative_before": (cumulative_before / self.reference_parameter_norm),
            "parameter_drift_before": parameter_drift_before,
            "relative_parameter_drift_before": (parameter_drift_before / self.reference_parameter_norm),
            "feature_cosine_similarity": cosine_similarity,
            "knn_neighbor_overlap": neighbor_overlap,
        }

    def capture_unscaled_gradient(self) -> None:
        learning_rates = {float(group["lr"]) for group in self.optimizer_groups}
        if len(learning_rates) != 1:
            raise RuntimeError("Change diagnostics currently require one actor learning rate.")
        learning_rate = learning_rates.pop()
        step_gradient_squared = torch.zeros((), device=self.parameters[0].device)
        for parameter, cumulative in zip(self.parameters, self.cumulative_updates, strict=True):
            gradient = parameter.grad
            if gradient is None:
                continue
            gradient = gradient.detach().float()
            step_gradient_squared += gradient.square().sum()
            cumulative.add_(gradient, alpha=-learning_rate)

        step_gradient_norm = math.sqrt(float(step_gradient_squared))
        step_update_norm = abs(learning_rate) * step_gradient_norm
        cumulative_after = self._vector_norm(self.cumulative_updates)
        self._cumulative_norm_after = cumulative_after
        self._update_metrics = {
            "learning_rate": learning_rate,
            "step_gradient_norm": step_gradient_norm,
            "step_update_norm": step_update_norm,
            "cumulative_after": cumulative_after,
            "relative_cumulative_after": (cumulative_after / self.reference_parameter_norm),
        }

    def finish_step(self, *, epoch: int, trajectory_step: int, global_step: int) -> ChangeDiagnosticRow:
        if self._graph_metrics is None or self._update_metrics is None:
            raise RuntimeError(
                "Policy graph and actor gradient must be observed before finishing a diagnostic step."
            )
        parameter_drift_after = self._parameter_drift_norm()
        self._parameter_drift_after = parameter_drift_after
        graph = self._graph_metrics
        update = self._update_metrics
        row = ChangeDiagnosticRow(
            epoch=epoch,
            trajectory_step=trajectory_step,
            global_step=global_step,
            reference_global_step=self.reference_global_step,
            steps_since_reference=global_step - self.reference_global_step,
            learning_rate=update["learning_rate"],
            probe_samples=self.probe_indices.numel(),
            reference_parameter_norm=self.reference_parameter_norm,
            step_gradient_norm=update["step_gradient_norm"],
            step_lr_gradient_norm=update["step_update_norm"],
            cumulative_lr_gradient_norm_before_update=graph["cumulative_before"],
            relative_cumulative_lr_gradient_norm_before_update=graph["relative_cumulative_before"],
            cumulative_lr_gradient_norm_after_update=update["cumulative_after"],
            relative_cumulative_lr_gradient_norm_after_update=update["relative_cumulative_after"],
            parameter_drift_norm_before_update=graph["parameter_drift_before"],
            relative_parameter_drift_before_update=graph["relative_parameter_drift_before"],
            parameter_drift_norm_after_update=parameter_drift_after,
            relative_parameter_drift_after_update=(parameter_drift_after / self.reference_parameter_norm),
            feature_mean_cosine_similarity=graph["feature_cosine_similarity"],
            feature_cosine_drift=(1.0 - graph["feature_cosine_similarity"]),
            knn_neighbor_overlap=graph["knn_neighbor_overlap"],
        )
        self._graph_metrics = None
        self._update_metrics = None
        return row


WARMUP_FIELDS = (
    "epoch",
    "lr",
    "train_loss",
    "train_accuracy",
    "val_loss",
    "val_noisy_accuracy",
    "val_clean_accuracy",
    "seconds",
)

SUMMARY_FIELDS = (
    "epoch",
    "split",
    "loss",
    "accuracy",
    "correction_rate",
    "correction_precision",
    "false_correction_rate",
    "noisy_recovery_rate",
    "clean_preservation_rate",
    "action_rate",
    "reward",
    "actor_loss",
    "critic_loss",
    "seconds",
)

CHANGE_DIAGNOSTIC_FIELDS = tuple(field.name for field in fields(ChangeDiagnosticRow))

CLEANING_FIELDS = ("step", "action_rate", "changed_rate", "accuracy")
CLEANING_SUMMARY_FIELDS = (
    "best_step",
    "best_accuracy",
    "final_accuracy",
    "correction_rate",
    "correction_precision",
    "noisy_recovery_rate",
    "false_correction_rate",
    "clean_preservation_rate",
    "seconds",
)

TIMING_FIELDS = ("stage", "calls", "total_seconds", "mean_seconds", "percentage")

RUN_SUMMARY_FIELDS = (
    "dataset",
    "model",
    "samples",
    "noise_rate",
    "seed",
    "update_mode",
    "update_samples",
    "warmup_epoch",
    "epochs",
    "steps",
    "k",
    "actor_lr",
    "critic_lr",
    "best_epoch",
    "best_val_accuracy",
    "best_val_loss",
    "last_val_accuracy",
    "last_val_loss",
    "total_seconds",
    "mean_epoch_seconds",
    "gpu_memory_gib",
    "actor_last",
    "critic_last",
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
    noisy_labels: Tensor, *, num_classes: int, randomization_rate: float, epoch: int
) -> tuple[Tensor, int]:
    random_count = min(noisy_labels.numel(), max(1, round(noisy_labels.numel() * randomization_rate)))
    generator = torch.Generator(device=noisy_labels.device).manual_seed(SEED + epoch - 1)
    selected = torch.randperm(noisy_labels.numel(), device=noisy_labels.device, generator=generator)[
        :random_count
    ]
    randomized_labels = noisy_labels.clone()
    original = randomized_labels[selected]
    alternatives = torch.randint(
        num_classes - 1, (random_count,), device=noisy_labels.device, generator=generator
    )
    alternatives += alternatives.ge(original)
    randomized_labels[selected] = alternatives
    label_state = F.one_hot(randomized_labels, num_classes=num_classes).to(torch.float32)
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
    name: str, device: torch.device, timings: Timings, operation: Callable[[], T], *, step: int | None = None
) -> T:
    synchronize(device)
    started = time.perf_counter()
    result = operation()
    synchronize(device)
    elapsed = time.perf_counter() - started
    timings.setdefault(name, []).append(elapsed)
    if step is None:
        print(f"[TIME] {name}={elapsed:.3f}s")
    return result


def load_training_data() -> tuple[Tensor, Tensor]:
    return TRAIN_DATA_LOADER()


def load_evaluation_data() -> dict[str, tuple[Tensor, Tensor]]:
    return EVALUATION_DATA_LOADER()


def inject_stratified_symmetric_noise(clean_labels: Tensor) -> tuple[Tensor, Tensor]:
    generator = torch.Generator().manual_seed(SEED)
    noisy_labels = clean_labels.clone()
    noise_mask = torch.zeros_like(clean_labels, dtype=torch.bool)

    class_sizes = [int(clean_labels.eq(digit).sum()) for digit in CLASS_IDS]
    exact_counts = [size * NOISE_RATE for size in class_sizes]
    noise_counts = [math.floor(value) for value in exact_counts]
    target_noise_count = round(clean_labels.numel() * NOISE_RATE)
    remainder = target_noise_count - sum(noise_counts)
    allocation_order = sorted(
        range(NUM_CLASSES), key=lambda index: exact_counts[index] - noise_counts[index], reverse=True
    )
    for index in allocation_order[:remainder]:
        noise_counts[index] += 1

    for digit, noise_count in zip(CLASS_IDS, noise_counts):
        class_indices = clean_labels.eq(digit).nonzero(as_tuple=False).flatten()
        if noise_count == 0:
            continue
        selected = class_indices[torch.randperm(class_indices.numel(), generator=generator)[:noise_count]]
        alternatives = torch.randint(NUM_CLASSES - 1, (noise_count,), generator=generator)
        original = clean_labels[selected]
        alternatives += alternatives.ge(original)
        noisy_labels[selected] = alternatives
        noise_mask[selected] = True

    return pin_for_cuda(noisy_labels), pin_for_cuda(noise_mask)


def load_noisy_label_artifacts(clean_labels: Tensor) -> tuple[Tensor, Tensor]:
    noisy_path = EXTERNAL_NOISY_LABELS_PATH
    mask_path = EXTERNAL_NOISE_MASK_PATH
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
        raise ValueError(f"Noisy labels must have shape {expected_shape}, got {tuple(noisy_labels.shape)}.")
    if tuple(noise_mask.shape) != expected_shape:
        raise ValueError(f"Noise mask must have shape {expected_shape}, got {tuple(noise_mask.shape)}.")
    if noisy_labels.numel() and (int(noisy_labels.min()) < 0 or int(noisy_labels.max()) >= NUM_CLASSES):
        raise ValueError("Noisy labels contain an out-of-range class ID.")
    if not torch.equal(noisy_labels.ne(clean_labels), noise_mask):
        raise ValueError("Saved noise mask does not match clean/noisy label differences.")
    expected_noise_count = round(clean_labels.numel() * NOISE_RATE)
    if int(noise_mask.sum()) != expected_noise_count:
        raise ValueError(
            f"Saved noise count does not match NOISE_RATE: {int(noise_mask.sum())} != {expected_noise_count}."
        )
    return (pin_for_cuda(noisy_labels.contiguous()), pin_for_cuda(noise_mask.contiguous()))


def safe_ratio(numerator: Tensor, denominator: Tensor) -> Tensor:
    return numerator.float() / denominator.clamp_min(1).float()


def validate_soft_labels(soft_labels: Tensor, sample_count: int) -> None:
    expected_shape = (sample_count, NUM_CLASSES)
    if soft_labels.shape != expected_shape:
        raise ValueError(
            f"Soft labels must have shape [samples, classes]: {tuple(soft_labels.shape)} != {expected_shape}."
        )
    if not soft_labels.is_floating_point():
        raise TypeError("Soft labels must use a floating-point dtype.")
    if not torch.isfinite(soft_labels).all():
        raise ValueError("Soft labels contain non-finite values.")
    if bool(soft_labels.lt(0).any()):
        raise ValueError("Soft labels contain negative values.")
    row_sums = soft_labels.sum(dim=1)
    if not torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5, rtol=1e-5):
        raise ValueError("Each soft-label row must sum to one.")


def correction_summary(
    soft_labels: Tensor,
    clean_labels: Tensor,
    initial_noisy_labels: Tensor,
    noise_mask: Tensor,
    *,
    epoch: int,
    split: str,
    action_rate: float = 0.0,
    reward: float | None = None,
    actor_loss: float | None = None,
    critic_loss: float | None = None,
    seconds: float = 0.0,
) -> dict[str, object]:
    sample_count = clean_labels.numel()
    if soft_labels.shape != (sample_count, NUM_CLASSES):
        raise ValueError("soft_labels must have shape [N, NUM_CLASSES].")
    if initial_noisy_labels.shape != clean_labels.shape or noise_mask.shape != clean_labels.shape:
        raise ValueError("Metric label tensors must have equal [N] shapes.")

    hard_labels = soft_labels.argmax(dim=1)
    clean_probabilities = soft_labels.float().gather(1, clean_labels.unsqueeze(1)).squeeze(1)
    loss = -clean_probabilities.clamp_min(1e-8).log().mean()

    changed = hard_labels.ne(initial_noisy_labels)
    correct_correction = changed & hard_labels.eq(clean_labels)
    clean_mask = ~noise_mask
    false_correction = clean_mask & changed
    recovered = noise_mask & hard_labels.eq(clean_labels)
    clean_preserved = clean_mask & hard_labels.eq(clean_labels)

    correction_count = changed.sum()
    noise_count = noise_mask.sum()
    clean_count = clean_mask.sum()
    values = (
        torch.stack(
            (
                loss,
                hard_labels.eq(clean_labels).float().mean(),
                correction_count.float() / sample_count,
                safe_ratio(correct_correction.sum(), correction_count),
                safe_ratio(false_correction.sum(), clean_count),
                safe_ratio(recovered.sum(), noise_count),
                safe_ratio(clean_preserved.sum(), clean_count),
            )
        )
        .to(torch.float64)
        .cpu()
        .tolist()
    )
    (
        loss_value,
        accuracy,
        correction_rate,
        correction_precision,
        false_correction_rate,
        noisy_recovery_rate,
        clean_preservation_rate,
    ) = values
    return {
        "epoch": epoch,
        "split": split,
        "loss": loss_value,
        "accuracy": accuracy,
        "correction_rate": correction_rate,
        "correction_precision": correction_precision,
        "false_correction_rate": false_correction_rate,
        "noisy_recovery_rate": noisy_recovery_rate,
        "clean_preservation_rate": clean_preservation_rate,
        "action_rate": action_rate,
        "reward": reward,
        "actor_loss": actor_loss,
        "critic_loss": critic_loss,
        "seconds": seconds,
    }


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: tuple[str, ...]) -> None:
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


def append_csv(path: Path, rows: list[dict[str, object]], fieldnames: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    needs_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if needs_header:
            writer.writeheader()
        writer.writerows(rows)


def preprocess(images: Tensor, device: torch.device, mean: Tensor, std: Tensor) -> Tensor:
    return PREPROCESS_FUNCTION(images, device, mean, std)


def preprocess_training(
    images: Tensor, device: torch.device, mean: Tensor, std: Tensor, generator: torch.Generator
) -> Tensor:
    return TRAIN_PREPROCESS_FUNCTION(images, device, mean, std, generator)


def normalization_tensors(device: torch.device) -> tuple[Tensor, Tensor]:
    shape = (1, 3, 1, 1)
    mean = torch.tensor(DATA_MEAN, device=device).reshape(shape)
    std = torch.tensor(DATA_STD, device=device).reshape(shape)
    return mean, std


def build_grad_scaler() -> torch.amp.GradScaler:
    return torch.amp.GradScaler("cuda", enabled=USE_AMP and AMP_DTYPE == torch.float16)


def create_experiment_model() -> nn.Module:
    return MODEL_FACTORY(PRETRAINED, NUM_CLASSES)


def encode(model: nn.Module, images: Tensor) -> Tensor:
    feature_map = model.forward_features(images)
    return model.forward_head(feature_map, pre_logits=True)


def _build_warmup_optimizer(model: nn.Module) -> SGD:
    return SGD(model.parameters(), lr=WARMUP_LR, momentum=WARMUP_MOMENTUM, weight_decay=WARMUP_WEIGHT_DECAY)


def _build_halfway_scheduler(
    optimizer: Optimizer, epochs: int, decay_fraction: float, decay_factor: float
) -> MultiStepLR:
    milestone = max(1, math.ceil(epochs * decay_fraction))
    return MultiStepLR(optimizer, milestones=[milestone], gamma=decay_factor)


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
    noisy_correct = torch.zeros((), dtype=torch.long, device=device)
    clean_correct = torch.zeros((), dtype=torch.long, device=device)
    for start in range(0, sample_count, WARMUP_EVAL_BATCH_SIZE):
        end = min(start + WARMUP_EVAL_BATCH_SIZE, sample_count)
        images = preprocess(raw_images[start:end], device, mean, std)
        noisy_targets = noisy_labels_cpu[start:end].to(device=device, non_blocking=True)
        clean_targets = clean_labels_cpu[start:end].to(device=device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=AMP_DTYPE, enabled=USE_AMP):
            logits = model(images)
            loss = criterion(logits, noisy_targets)
        predictions = logits.argmax(dim=1)
        loss_sum += loss.to(torch.float64) * (end - start)
        noisy_correct += predictions.eq(noisy_targets).sum()
        clean_correct += predictions.eq(clean_targets).sum()

    return {
        "loss": float(loss_sum / sample_count),
        "noisy_accuracy": float(noisy_correct / sample_count),
        "clean_accuracy": float(clean_correct / sample_count),
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
                "training_data": training_data_metadata(),
                "training_augmentation": training_augmentation_metadata(),
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


def training_data_metadata() -> dict[str, object]:
    return {
        "sample_count": EXPECTED_SAMPLES,
        "subset_seed": TRAIN_SUBSET_SEED,
        "selection": "deterministic_stratified_equal_per_class",
    }


def validate_training_data_checkpoint(checkpoint: dict[str, object]) -> None:
    metadata = checkpoint.get("training_data")
    if not isinstance(metadata, dict):
        raise ValueError(
            "Checkpoint is missing the balanced training-data contract. Regenerate it with the current code."
        )
    expected = training_data_metadata()
    actual = {
        "sample_count": int(metadata.get("sample_count", -1)),
        "subset_seed": int(metadata.get("subset_seed", -1)),
        "selection": str(metadata.get("selection", "")),
    }
    if actual != expected:
        raise ValueError(
            f"Checkpoint training data does not match the current config: {actual!r} != {expected!r}."
        )


def training_augmentation_metadata() -> dict[str, object]:
    return {
        "enabled": TRAIN_AUGMENTATION_ENABLED,
        "random_crop_padding": TRAIN_RANDOM_CROP_PADDING,
        "horizontal_flip_probability": TRAIN_HORIZONTAL_FLIP_PROBABILITY,
    }


def validate_training_augmentation_checkpoint(checkpoint: dict[str, object]) -> None:
    metadata = checkpoint.get("training_augmentation")
    if metadata is None:
        if TRAIN_AUGMENTATION_ENABLED:
            raise ValueError(
                "Checkpoint predates enabled training augmentation. Rerun "
                "warm-up and its downstream stages with the current config."
            )
        return
    if not isinstance(metadata, dict):
        raise TypeError("Checkpoint training_augmentation must be a dictionary.")
    expected = training_augmentation_metadata()
    try:
        actual_enabled = bool(metadata["enabled"])
        actual_padding = int(metadata["random_crop_padding"])
        actual_flip = float(metadata["horizontal_flip_probability"])
    except KeyError as error:
        raise KeyError("Checkpoint training_augmentation metadata is incomplete.") from error
    if (
        actual_enabled != expected["enabled"]
        or actual_padding != expected["random_crop_padding"]
        or not math.isclose(actual_flip, float(expected["horizontal_flip_probability"]))
    ):
        raise ValueError(
            "Checkpoint training augmentation does not match the current "
            f"config: {metadata!r} != {expected!r}."
        )


def load_warmup_checkpoint(
    model: nn.Module, checkpoint_path: Path, device: torch.device
) -> dict[str, object]:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Warmup checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
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
        raise KeyError(f"Warmup checkpoint is missing fields: {sorted(missing)}")
    if not math.isclose(float(checkpoint["noise_rate"]), NOISE_RATE):
        raise ValueError("Warmup checkpoint noise rate does not match this run.")
    if bool(checkpoint["pretrained"]) != PRETRAINED:
        raise ValueError("Warmup checkpoint pretrained setting does not match this run.")
    checkpoint_model_name = checkpoint.get("model_name")
    if checkpoint_model_name is not None and checkpoint_model_name != MODEL_NAME:
        raise ValueError("Warmup checkpoint model name does not match this run.")
    checkpoint_num_classes = checkpoint.get("num_classes")
    if checkpoint_num_classes is not None and int(checkpoint_num_classes) != NUM_CLASSES:
        raise ValueError("Warmup checkpoint class count does not match this run.")
    checkpoint_model_id = str(checkpoint.get("warmup_model_id", ""))
    if WARMUP_MODEL_ID and checkpoint_model_id != WARMUP_MODEL_ID:
        raise ValueError(
            f"Warmup model ID does not match this run: {checkpoint_model_id!r} != {WARMUP_MODEL_ID!r}."
        )
    validate_training_augmentation_checkpoint(checkpoint)
    validate_training_data_checkpoint(checkpoint)
    deployment_mode = str(checkpoint.get("selection", "best"))
    if deployment_mode != "best":
        raise ValueError("CIFAR warmup checkpoint must use best selection.")
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(
        device=device, memory_format=(torch.channels_last if USE_CHANNELS_LAST else torch.contiguous_format)
    )
    model.eval()
    return {
        "best_epoch": int(checkpoint.get("best_epoch", checkpoint["epoch"])),
        "best_noisy_validation_accuracy": float(
            checkpoint.get("best_noisy_validation_accuracy", checkpoint["noisy_validation_accuracy"])
        ),
        "best_clean_validation_accuracy": float(
            checkpoint.get("best_clean_validation_accuracy", checkpoint["clean_validation_accuracy"])
        ),
        "deployment_mode": deployment_mode,
        "deployment_epoch": int(checkpoint["epoch"]),
    }


def _rl_selection_key(validation_summary: dict[str, object]) -> tuple[float, float]:
    accuracy = float(validation_summary["accuracy"])
    loss = float(validation_summary["loss"])
    metrics = (accuracy, loss)
    if not all(math.isfinite(value) for value in metrics):
        raise ValueError("RL validation metrics must be finite.")
    return accuracy, -loss


def _save_rl_checkpoints(
    actor_path: Path,
    critic_path: Path,
    *,
    epoch: int,
    model: nn.Module,
    critic: nn.Module,
    validation_summary: dict[str, object],
) -> None:
    actor_path.parent.mkdir(parents=True, exist_ok=True)
    critic_path.parent.mkdir(parents=True, exist_ok=True)
    validation_metrics = {
        "accuracy": float(validation_summary["accuracy"]),
        "loss": float(validation_summary["loss"]),
    }
    actor_payload = {
        "epoch": epoch,
        "model_name": MODEL_NAME,
        "num_classes": NUM_CLASSES,
        "validation": validation_metrics,
        "training_augmentation": training_augmentation_metadata(),
        "training_data": training_data_metadata(),
        "model": model.state_dict(),
    }
    critic_payload = {
        "epoch": epoch,
        "num_bins": CRITIC_NUM_BINS,
        "validation": validation_metrics,
        "critic": critic.state_dict(),
    }
    for path, payload in ((actor_path, actor_payload), (critic_path, critic_payload)):
        temporary_path = path.with_suffix(f"{path.suffix}.tmp")
        try:
            torch.save(payload, temporary_path)
            temporary_path.replace(path)
        finally:
            temporary_path.unlink(missing_ok=True)


def restore_actor_checkpoint(
    model: nn.Module, checkpoint_path: Path, device: torch.device
) -> dict[str, object]:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Actor checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
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
    validate_training_augmentation_checkpoint(checkpoint)
    validate_training_data_checkpoint(checkpoint)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(
        device=device, memory_format=(torch.channels_last if USE_CHANNELS_LAST else torch.contiguous_format)
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
    criterion = nn.CrossEntropyLoss()
    scaler = build_grad_scaler()
    optimizer = _build_warmup_optimizer(model)
    scheduler = _build_halfway_scheduler(
        optimizer, WARMUP_EPOCHS, WARMUP_LR_DECAY_FRACTION, WARMUP_LR_DECAY_FACTOR
    )
    write_csv(warmup_csv_path, [], WARMUP_FIELDS)
    best_noisy_accuracy = float("-inf")
    best_clean_accuracy = float("nan")
    best_epoch = 0

    for epoch in range(1, WARMUP_EPOCHS + 1):
        epoch_started = time.perf_counter()
        model.train()

        generator = torch.Generator().manual_seed(SEED + epoch)
        augmentation_generator = torch.Generator(device=device).manual_seed(SEED + epoch)
        permutation = torch.randperm(train_images.size(0), generator=generator)
        loss_sum = torch.zeros((), device=device, dtype=torch.float64)
        correct_count = torch.zeros((), device=device, dtype=torch.long)
        samples_seen = 0
        for start in range(0, permutation.numel(), WARMUP_BATCH_SIZE):
            end = min(start + WARMUP_BATCH_SIZE, permutation.numel())
            batch_indices = permutation[start:end]
            images = preprocess_training(
                train_images[batch_indices], device, mean, std, augmentation_generator
            )
            targets = train_noisy_labels_cpu[batch_indices].to(device=device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=AMP_DTYPE, enabled=USE_AMP):
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
            "lr": float(optimizer.param_groups[0]["lr"]),
            "train_loss": float(loss_sum / samples_seen),
            "train_accuracy": float(correct_count / samples_seen),
            "val_loss": validation["loss"],
            "val_noisy_accuracy": validation["noisy_accuracy"],
            "val_clean_accuracy": validation["clean_accuracy"],
            "seconds": time.perf_counter() - epoch_started,
        }
        append_csv(warmup_csv_path, [row], WARMUP_FIELDS)
        print(
            f"[WARMUP] epoch={epoch}/{WARMUP_EPOCHS} "
            f"lr={float(row['lr']):.6g} "
            f"train_loss={row['train_loss']:.4f} "
            f"train_acc={row['train_accuracy']:.4f} "
            f"val_noisy_acc={row['val_noisy_accuracy']:.4f} "
            f"val_clean_acc={row['val_clean_accuracy']:.4f}"
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

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
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
    inference_images = preprocess(raw_images[:FEATURE_BATCH_SIZE], device, mean, std)
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=AMP_DTYPE, enabled=USE_AMP):
        encode(model, inference_images)

    update_image_count = min(
        raw_images.size(0), (POLICY_UPDATE_BATCH_SIZE * (K + 1) if direct_actor else WARMUP_BATCH_SIZE)
    )
    update_images = preprocess(raw_images[:update_image_count], device, mean, std)
    model.eval()
    with torch.autocast(device_type="cuda", dtype=AMP_DTYPE, enabled=USE_AMP):
        encode(model, update_images).mean().backward()
    model.zero_grad(set_to_none=True)
    model.eval()


@torch.inference_mode()
def extract_all_embeddings(
    model: nn.Module, raw_images: Tensor, device: torch.device, mean: Tensor, std: Tensor
) -> Tensor:
    model.eval()
    sample_count = raw_images.size(0)
    output: Tensor | None = None
    for start in range(0, sample_count, FEATURE_BATCH_SIZE):
        end = min(start + FEATURE_BATCH_SIZE, sample_count)
        images = preprocess(raw_images[start:end], device, mean, std)
        with torch.autocast(device_type="cuda", dtype=AMP_DTYPE, enabled=USE_AMP):
            embeddings = encode(model, images)
        if output is None:
            output = torch.empty((sample_count, embeddings.size(1)), dtype=torch.float32, device=device)
        output[start:end].copy_(embeddings)
    if output is None:
        raise RuntimeError("Embedding extraction received an empty dataset.")
    return output


@torch.inference_mode()
def build_neighbor_indices(embeddings: Tensor) -> Tensor:
    return build_exact_policy_knn(
        embeddings, k=K, query_chunk_size=KNN_QUERY_CHUNK_SIZE, reference_chunk_size=KNN_REFERENCE_CHUNK_SIZE
    )


@torch.inference_mode()
def build_global_graph(embeddings: Tensor) -> tuple[Tensor, Tensor]:
    neighbor_indices = build_neighbor_indices(embeddings)
    normalized = F.normalize(embeddings.float(), dim=1)
    neighbor_cosines = torch.empty(neighbor_indices.shape, dtype=torch.float32, device=embeddings.device)
    for start in range(0, embeddings.size(0), CORRECTION_CHUNK_SIZE):
        end = min(start + CORRECTION_CHUNK_SIZE, embeddings.size(0))
        indices = neighbor_indices[start:end]
        neighbor_cosines[start:end] = (normalized[start:end].unsqueeze(1) * normalized[indices]).sum(dim=2)
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
) -> dict[str, object]:
    timing_prefix = "val" if split == "val" else split
    synchronize(device)
    started = time.perf_counter()
    device_index = device.index if device.index is not None else 0
    with torch.random.fork_rng(devices=[device_index]):
        torch.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        embeddings = measure(
            f"{timing_prefix}_features",
            device,
            timings,
            lambda: extract_all_embeddings(model, raw_images, device, mean, std),
        )
        neighbors = measure(
            f"{timing_prefix}_knn", device, timings, lambda: build_neighbor_indices(embeddings)
        )
        clean_labels = clean_labels_cpu.to(device, non_blocking=True)
        initial_noisy_labels = noisy_labels_cpu.to(device, non_blocking=True)
        noise_mask = noise_mask_cpu.to(device, non_blocking=True)
        label_state = F.one_hot(initial_noisy_labels, num_classes=NUM_CLASSES).to(torch.float32)
        action_count = torch.zeros((), dtype=torch.long, device=device)
        for evaluation_step in range(1, TRAJECTORY_LENGTH + 1):
            correction = measure(
                f"{timing_prefix}_correction",
                device,
                timings,
                lambda: policy.correct_all(embeddings, label_state, neighbors),
                step=evaluation_step,
            )
            action_count += correction.actions.sum()
            label_state = correction.corrected_labels

    synchronize(device)
    elapsed = time.perf_counter() - started
    action_count_value = int(action_count)
    action_rate = action_count_value / (clean_labels.numel() * TRAJECTORY_LENGTH)
    summary = correction_summary(
        label_state,
        clean_labels,
        initial_noisy_labels,
        noise_mask,
        epoch=epoch,
        split=split,
        action_rate=action_rate,
        seconds=elapsed,
    )
    del embeddings, neighbors, label_state
    return summary


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
    checkpoint_epoch: int,
    trajectory_length: int,
) -> dict[str, object]:
    synchronize(device)
    started = time.perf_counter()
    device_index = device.index if device.index is not None else 0
    with torch.random.fork_rng(devices=[device_index]):
        torch.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        embeddings = measure(
            "final_features",
            device,
            timings,
            lambda: extract_all_embeddings(model, raw_images, device, mean, std),
        )
        neighbors = measure("final_knn", device, timings, lambda: build_neighbor_indices(embeddings))
        clean_labels = clean_labels_cpu.to(device, non_blocking=True)
        initial_noisy_labels = noisy_labels_cpu.to(device, non_blocking=True)
        noise_mask = noise_mask_cpu.to(device, non_blocking=True)
        label_state = F.one_hot(initial_noisy_labels, num_classes=NUM_CLASSES).to(torch.float32)
        history: list[dict[str, object]] = []
        total_action_count = 0

        for cleaning_step in range(1, trajectory_length + 1):
            correction = measure(
                "final_correction",
                device,
                timings,
                lambda: policy.correct_all(embeddings, label_state, neighbors),
                step=cleaning_step,
            )
            hard_labels = correction.corrected_labels.argmax(dim=1)
            step_action_count_value, changed_rate, clean_accuracy = (
                torch.stack(
                    (
                        correction.actions.sum(),
                        hard_labels.ne(initial_noisy_labels).float().mean(),
                        hard_labels.eq(clean_labels).float().mean(),
                    )
                )
                .to(torch.float64)
                .cpu()
                .tolist()
            )
            step_action_count = int(step_action_count_value)
            total_action_count += step_action_count
            label_state = correction.corrected_labels
            row = {
                "step": cleaning_step,
                "action_rate": step_action_count / EXPECTED_SAMPLES,
                "changed_rate": changed_rate,
                "accuracy": clean_accuracy,
            }
            history.append(row)
            print(
                f"[CLEAN] step={cleaning_step}/"
                f"{trajectory_length} "
                f"action={float(row['action_rate']):.4f} "
                f"changed={float(row['changed_rate']):.4f} "
                f"accuracy={float(row['accuracy']):.4f}"
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
        action_rate=(total_action_count / (EXPECTED_SAMPLES * trajectory_length)),
        seconds=elapsed,
    )
    validate_soft_labels(label_state, clean_labels.numel())
    corrected_array = label_state.detach().to(torch.float32).cpu().numpy()
    corrected_labels_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = corrected_labels_path.with_suffix(f"{corrected_labels_path.suffix}.tmp")
    try:
        with temporary_path.open("wb") as handle:
            np.save(handle, corrected_array, allow_pickle=False)
        temporary_path.replace(corrected_labels_path)
    finally:
        temporary_path.unlink(missing_ok=True)
    best_row = max(history, key=lambda row: float(row["accuracy"]))
    summary["best_step"] = int(best_row["step"])
    summary["best_accuracy"] = float(best_row["accuracy"])
    write_csv(cleaning_csv_path, history, CLEANING_FIELDS)
    cleaning_summary = {
        "best_step": summary["best_step"],
        "best_accuracy": summary["best_accuracy"],
        "final_accuracy": summary["accuracy"],
        **{field: summary[field] for field in CLEANING_SUMMARY_FIELDS[3:]},
    }
    write_csv(cleaning_summary_path, [cleaning_summary], CLEANING_SUMMARY_FIELDS)
    print(f"[CLEAN] corrected_labels={corrected_labels_path}")
    del embeddings, neighbors, label_state
    return summary


def select_policy_queries(sample_count: int, step: int) -> Tensor:
    if POLICY_UPDATE_SAMPLES == sample_count:
        return torch.arange(sample_count)

    generator = torch.Generator().manual_seed(SEED + step)
    batch_count = math.ceil(sample_count / POLICY_UPDATE_BATCH_SIZE)
    selected_batch_ids = torch.randperm(batch_count, generator=generator)
    selected_chunks: list[Tensor] = []
    remaining = POLICY_UPDATE_SAMPLES
    for batch_id_tensor in selected_batch_ids:
        start = int(batch_id_tensor) * POLICY_UPDATE_BATCH_SIZE
        end = min(start + POLICY_UPDATE_BATCH_SIZE, sample_count)
        capacity = end - start
        selected_count = min(remaining, capacity)
        if selected_count == capacity:
            selected_chunks.append(torch.arange(start, end))
        else:
            positions = torch.randperm(capacity, generator=generator)[:selected_count]
            selected_chunks.append(start + positions)
        remaining -= selected_count
        if remaining == 0:
            break

    if remaining != 0:
        raise RuntimeError("Failed to select the requested policy subset.")

    return torch.cat(selected_chunks).sort().values


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
    batch_neighbors_cpu = neighbor_indices_cpu[query_indices_cpu]
    combined_indices_cpu = torch.cat((query_indices_cpu, batch_neighbors_cpu.reshape(-1)))
    selected_images = pin_for_cuda(raw_images[combined_indices_cpu].contiguous())
    images = preprocess(selected_images, device, mean, std)
    embeddings = encode(model, images)
    query_count = query_indices_cpu.numel()
    neighbor_count = batch_neighbors_cpu.size(1)
    query_embeddings = embeddings[:query_count]
    neighbor_embeddings = embeddings[query_count:].reshape(query_count, neighbor_count, -1)
    query_indices = query_indices_cpu.to(device=device, non_blocking=True)
    neighbor_indices = batch_neighbors_cpu.to(device=device, non_blocking=True)
    batch_actions = None if actions is None else actions[query_indices]
    return policy(
        query_embeddings,
        neighbor_embeddings,
        label_state[query_indices],
        label_state[neighbor_indices],
        actions=batch_actions,
    )


@torch.inference_mode()
def correct_policy_from_embeddings(
    policy: LabelCorrectionPolicy, policy_embeddings: Tensor, label_state: Tensor, policy_neighbors: Tensor
) -> CorrectionResult:
    correction = policy.correct_all(policy_embeddings, label_state, policy_neighbors)
    if correction.actions.all():
        correction.actions[0] = False
        correction.corrected_labels[0] = label_state[0]
    return correction


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
    change_recorder: ChangeDiagnosticsRecorder | None = None,
) -> float:
    sample_count = raw_images.size(0)
    model.eval()
    optimizer.zero_grad(set_to_none=True)
    query_count = query_indices_cpu.numel()
    selected_mask_cpu = torch.zeros(sample_count, dtype=torch.bool)
    selected_mask_cpu[query_indices_cpu] = True
    selected_batch_ids = torch.unique(query_indices_cpu // POLICY_UPDATE_BATCH_SIZE, sorted=True)
    total_batches = selected_batch_ids.numel()
    total_loss = torch.zeros((), device=device)
    processed_queries = 0

    for batch_number, batch_id_tensor in enumerate(selected_batch_ids, start=1):
        start = int(batch_id_tensor) * POLICY_UPDATE_BATCH_SIZE
        end = min(start + POLICY_UPDATE_BATCH_SIZE, sample_count)
        batch_indices_cpu = torch.arange(start, end)
        selected_in_batch_cpu = selected_mask_cpu[start:end]
        selected_count = int(selected_in_batch_cpu.sum())
        if selected_count == 0:
            raise RuntimeError("Selected policy batch contains no queries.")
        with torch.autocast(device_type=device.type, dtype=AMP_DTYPE, enabled=USE_AMP):
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
            selected_in_batch = selected_in_batch_cpu.to(device=device, non_blocking=True)
            loss = -(q_value.detach() * policy_step.log_probabilities[selected_in_batch].sum() / query_count)
        scaler.scale(loss).backward()
        total_loss += loss.detach()
        processed_queries += selected_count
        if batch_number % 100 == 0 or batch_number == total_batches:
            print(
                f"[DIRECT ACTOR UPDATE] batch={batch_number}/"
                f"{total_batches} queries={processed_queries}/{query_count}"
            )

    if change_recorder is not None:
        scaler.unscale_(optimizer)
        change_recorder.capture_unscaled_gradient()
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
    next_encoding: Tensor,
) -> float:
    optimizer.zero_grad(set_to_none=True)
    current_q = critic.value_from_encoding(encoding)
    next_q = critic.value_from_encoding(next_encoding)
    td = sarsa_td_loss(current_q, reward, next_q, discount_factor=DISCOUNT_FACTOR)
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
        policy=replace(cfg.policy, temperature=TEMPERATURE, correction_chunk_size=CORRECTION_CHUNK_SIZE),
        reward=replace(cfg.reward, nla_weight=NLA_WEIGHT),
        rl_train=replace(
            cfg.rl_train,
            critic_lr=CRITIC_LR,
            critic_momentum=CRITIC_MOMENTUM,
            critic_weight_decay=CRITIC_WEIGHT_DECAY,
            critic_num_bins=CRITIC_NUM_BINS,
        ),
    )


def print_configuration(device: torch.device, sample_count: int) -> None:
    properties = torch.cuda.get_device_properties(device)
    print(
        f"device={properties.name} memory={properties.total_memory / 1024**3:.1f}GiB "
        f"samples={sample_count} noise={NOISE_RATE:.0%} seed={SEED}"
    )
    print(
        f"model={MODEL_NAME} warmup={WARMUP_EPOCHS}epochs "
        f"augmentation={TRAIN_AUGMENTATION_ENABLED} pretrained={PRETRAINED}"
    )
    print(
        f"rl={RL_EPOCHS}x{TRAJECTORY_LENGTH} update={ACTOR_UPDATE_MODE}:{POLICY_UPDATE_SAMPLES} "
        f"batch={POLICY_UPDATE_BATCH_SIZE} actor_lr={ACTOR_LR} critic_lr={CRITIC_LR}"
    )
    print(
        f"knn=k{K} feature_batch={FEATURE_BATCH_SIZE} amp={USE_AMP}:{AMP_DTYPE} "
        f"change_log={RECORD_CHANGE_DIAGNOSTICS}"
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
    rows: list[dict[str, object]] = []
    for name, values in timings.items():
        total = sum(values)
        rows.append(
            {
                "stage": name,
                "calls": len(values),
                "total_seconds": total,
                "mean_seconds": total / len(values),
                "percentage": 100.0 * total / measured_total,
            }
        )
    return rows


def main() -> None:
    if not CONFIGURED:
        raise RuntimeError("Configure rl_engine through cifar_common.configure_engine() first.")
    run_started = time.perf_counter()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    actor_best_checkpoint_path = MODEL_OUTPUT_DIR / ACTOR_BEST_CHECKPOINT_FILENAME
    actor_last_checkpoint_path = MODEL_OUTPUT_DIR / ACTOR_LAST_CHECKPOINT_FILENAME
    critic_best_checkpoint_path = MODEL_OUTPUT_DIR / CRITIC_BEST_CHECKPOINT_FILENAME
    critic_last_checkpoint_path = MODEL_OUTPUT_DIR / CRITIC_LAST_CHECKPOINT_FILENAME
    train_csv_path = OUTPUT_DIR / TRAIN_CSV_FILENAME
    timing_csv_path = OUTPUT_DIR / TIMING_CSV_FILENAME
    run_summary_path = OUTPUT_DIR / RUN_SUMMARY_CSV_FILENAME
    change_diagnostics_path = OUTPUT_DIR / CHANGE_DIAGNOSTICS_CSV_FILENAME
    write_csv(train_csv_path, [], SUMMARY_FIELDS)
    if RECORD_CHANGE_DIAGNOSTICS:
        write_csv(change_diagnostics_path, [], CHANGE_DIAGNOSTIC_FIELDS)

    device = resolve_local_device()
    seed_everything(SEED)
    torch.backends.cudnn.benchmark = CUDNN_BENCHMARK
    torch.cuda.reset_peak_memory_stats()
    timings: Timings = {}
    print(f"output_dir={OUTPUT_DIR}")

    raw_images, clean_labels_cpu = measure("data_load", device, timings, load_training_data)
    evaluation_splits = measure("val_load", device, timings, load_evaluation_data)
    noisy_labels_cpu, noise_mask_cpu = measure(
        "noise_load", device, timings, lambda: load_noisy_label_artifacts(clean_labels_cpu)
    )
    val_images, val_clean_labels = evaluation_splits["val"]
    val_noisy_labels, val_noise_mask = measure(
        "val_noise", device, timings, lambda: inject_stratified_symmetric_noise(val_clean_labels)
    )
    evaluation_data = {"val": (val_images, val_clean_labels, val_noisy_labels, val_noise_mask)}

    print_configuration(device, clean_labels_cpu.numel())
    print(f"validation_samples={evaluation_data['val'][0].size(0)}")
    cfg = build_engine_config()

    model = measure(
        "model_init",
        device,
        timings,
        lambda: create_experiment_model().to(
            device=device,
            memory_format=(torch.channels_last if USE_CHANNELS_LAST else torch.contiguous_format),
        ),
    )
    mean, std = normalization_tensors(device)
    measure(
        "gpu_warmup",
        device,
        timings,
        lambda: warm_device_kernels(model, raw_images, device, mean, std, direct_actor=True),
    )

    warmup_checkpoint_path = EXTERNAL_WARMUP_CHECKPOINT_PATH
    warmup_result = measure(
        "warmup_load", device, timings, lambda: load_warmup_checkpoint(model, warmup_checkpoint_path, device)
    )
    for parameter in model.parameters():
        parameter.requires_grad = True
    model.to(
        device=device, memory_format=(torch.channels_last if USE_CHANNELS_LAST else torch.contiguous_format)
    )
    model.eval()

    policy = LabelCorrectionPolicy(cfg.policy).to(device)
    reward_function = RLNLCReward(cfg).to(device)
    critic = build_critic(cfg).to(device)
    critic_optimizer = build_critic_optimizer(critic, cfg)
    actor_optimizer = SGD(
        model.parameters(), lr=ACTOR_LR, momentum=ACTOR_MOMENTUM, weight_decay=ACTOR_WEIGHT_DECAY
    )
    scheduler_milestone = max(1, math.ceil(RL_EPOCHS * LR_DECAY_FRACTION))
    actor_scheduler = MultiStepLR(actor_optimizer, milestones=[scheduler_milestone], gamma=LR_DECAY_FACTOR)
    critic_scheduler = MultiStepLR(critic_optimizer, milestones=[scheduler_milestone], gamma=LR_DECAY_FACTOR)
    scaler = build_grad_scaler()
    change_recorder: ChangeDiagnosticsRecorder | None = None
    if RECORD_CHANGE_DIAGNOSTICS:
        if CHANGE_DIAGNOSTIC_PROBE_SIZE == EXPECTED_SAMPLES:
            probe_indices = torch.arange(EXPECTED_SAMPLES, device=device)
        else:
            probe_generator = torch.Generator().manual_seed(SEED)
            probe_indices = torch.randperm(EXPECTED_SAMPLES, generator=probe_generator)[
                :CHANGE_DIAGNOSTIC_PROBE_SIZE
            ].to(device)
        change_recorder = ChangeDiagnosticsRecorder(model, actor_optimizer, probe_indices)

    fixed_embeddings = measure(
        "reward_features",
        device,
        timings,
        lambda: extract_all_embeddings(model, raw_images, device, mean, std),
    ).detach()
    global_neighbors, global_cosines = measure(
        "reward_knn", device, timings, lambda: build_global_graph(fixed_embeddings)
    )

    clean_labels = clean_labels_cpu.to(device, non_blocking=True)
    initial_noisy_labels = noisy_labels_cpu.to(device, non_blocking=True)
    noise_mask = noise_mask_cpu.to(device, non_blocking=True)
    epoch_seconds: list[float] = []
    final_train_summary: dict[str, object] | None = None
    best_rl_epoch = 0
    best_rl_key: tuple[float, float] | None = None
    best_validation_summary: dict[str, object] | None = None

    val_images, val_clean, val_noisy, val_noise_mask = evaluation_data["val"]
    epoch_zero_summary = evaluate_correction_split(
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
    append_csv(train_csv_path, [epoch_zero_summary], SUMMARY_FIELDS)
    print(
        "[RL EPOCH 0] reporting_only=True "
        f"val_acc={float(epoch_zero_summary['accuracy']):.6f} "
        f"val_loss={float(epoch_zero_summary['loss']):.6f}"
    )
    seed_everything(SEED)

    for epoch in range(1, RL_EPOCHS + 1):
        synchronize(device)
        epoch_started = time.perf_counter()
        label_state, _ = measure(
            "state_randomize",
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
        previous_encoding: Tensor | None = None
        previous_reward: Tensor | None = None
        epoch_action_count = 0
        epoch_rewards: list[float] = []
        epoch_actor_losses: list[float] = []
        epoch_critic_losses: list[float] = []
        epoch_change_rows: list[dict[str, object]] = []
        if change_recorder is not None:
            change_recorder.begin_trajectory((epoch - 1) * TRAJECTORY_LENGTH + 1)

        for step in range(1, TRAJECTORY_LENGTH + 1):
            global_step = (epoch - 1) * TRAJECTORY_LENGTH + step
            policy_embeddings = measure(
                "policy_features",
                device,
                timings,
                lambda: extract_all_embeddings(model, raw_images, device, mean, std),
                step=global_step,
            )
            policy_neighbors = measure(
                "policy_knn",
                device,
                timings,
                lambda: build_neighbor_indices(policy_embeddings),
                step=global_step,
            )
            if change_recorder is not None:
                measure(
                    "change_knn",
                    device,
                    timings,
                    lambda: change_recorder.observe_policy_graph(policy_embeddings, policy_neighbors),
                    step=global_step,
                )
            correction = measure(
                "label_correction",
                device,
                timings,
                lambda: correct_policy_from_embeddings(
                    policy, policy_embeddings, label_state, policy_neighbors
                ),
                step=global_step,
            )
            policy_neighbors_cpu = measure(
                "knn_to_cpu",
                device,
                timings,
                lambda: pin_for_cuda(policy_neighbors.detach().cpu()),
                step=global_step,
            )
            del policy_embeddings, policy_neighbors

            reward_output = measure(
                "reward",
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
                encoding_value = critic.encode(reward_output.per_sample_consistency).detach()
                with torch.no_grad():
                    q_value_value = critic.value_from_encoding(encoding_value)
                return encoding_value, q_value_value

            encoding, q_value = measure(
                "critic_encode", device, timings, encode_critic_state, step=global_step
            )
            query_indices_cpu = select_policy_queries(EXPECTED_SAMPLES, global_step)
            actor_loss = measure(
                "actor_update",
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
                    change_recorder,
                ),
                step=global_step,
            )
            del policy_neighbors_cpu
            if change_recorder is not None:
                change_row = measure(
                    "change_log",
                    device,
                    timings,
                    lambda: change_recorder.finish_step(
                        epoch=epoch, trajectory_step=step, global_step=global_step
                    ),
                    step=global_step,
                )
                epoch_change_rows.append(asdict(change_row))

            if previous_encoding is not None and previous_reward is not None:
                critic_loss = measure(
                    "critic_update",
                    device,
                    timings,
                    lambda: update_critic(
                        critic, critic_optimizer, previous_encoding, previous_reward, encoding
                    ),
                    step=global_step,
                )
                epoch_critic_losses.append(critic_loss)

            label_state = correction.corrected_labels
            step_action_count_value, reward_value, clean_accuracy = (
                torch.stack(
                    (
                        correction.actions.sum(),
                        reward_output.total_reward.reshape(()),
                        label_state.argmax(dim=1).eq(clean_labels).float().mean(),
                    )
                )
                .to(torch.float64)
                .cpu()
                .tolist()
            )
            step_action_count = int(step_action_count_value)
            epoch_action_count += step_action_count
            epoch_rewards.append(reward_value)
            epoch_actor_losses.append(actor_loss)
            print(
                f"[RL] e={epoch}/{RL_EPOCHS} s={step}/{TRAJECTORY_LENGTH} "
                f"acc={clean_accuracy:.4f} reward={reward_value:.4f} actor={actor_loss:.4f} "
                f"action={step_action_count / EXPECTED_SAMPLES:.4f} "
            )

            if step < TRAJECTORY_LENGTH:
                previous_encoding = encoding
                previous_reward = reward_output.total_reward.detach()
            del correction, reward_output

        synchronize(device)
        train_elapsed = time.perf_counter() - epoch_started
        epoch_seconds.append(train_elapsed)
        if epoch_change_rows:
            append_csv(change_diagnostics_path, epoch_change_rows, CHANGE_DIAGNOSTIC_FIELDS)
        actor_scheduler.step()
        critic_scheduler.step()
        mean_critic_loss = (
            sum(epoch_critic_losses) / len(epoch_critic_losses) if epoch_critic_losses else None
        )
        train_summary = correction_summary(
            label_state,
            clean_labels,
            initial_noisy_labels,
            noise_mask,
            epoch=epoch,
            split="train",
            action_rate=epoch_action_count / (EXPECTED_SAMPLES * TRAJECTORY_LENGTH),
            reward=sum(epoch_rewards) / len(epoch_rewards),
            actor_loss=sum(epoch_actor_losses) / len(epoch_actor_losses),
            critic_loss=mean_critic_loss,
            seconds=train_elapsed,
        )
        val_images, val_clean, val_noisy, val_noise_mask = evaluation_data["val"]
        val_summary = evaluate_correction_split(
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
        append_csv(train_csv_path, [train_summary, val_summary], SUMMARY_FIELDS)

        def save_epoch_checkpoints(actor_path: Path, critic_path: Path) -> None:
            _save_rl_checkpoints(
                actor_path,
                critic_path,
                epoch=epoch,
                model=model,
                critic=critic,
                validation_summary=val_summary,
            )

        measure(
            "last_save",
            device,
            timings,
            lambda: save_epoch_checkpoints(actor_last_checkpoint_path, critic_last_checkpoint_path),
            step=epoch,
        )
        validation_key = _rl_selection_key(val_summary)
        if best_rl_key is None or validation_key > best_rl_key:
            measure(
                "best_save",
                device,
                timings,
                lambda: save_epoch_checkpoints(actor_best_checkpoint_path, critic_best_checkpoint_path),
                step=epoch,
            )
            best_rl_key = validation_key
            best_rl_epoch = epoch
            best_validation_summary = dict(val_summary)
            print(
                f"[RL BEST] epoch={best_rl_epoch} "
                f"val_acc={float(val_summary['accuracy']):.6f} "
                f"val_loss={float(val_summary['loss']):.6f}"
            )
        final_train_summary = train_summary
        print(
            f"[EPOCH] epoch={epoch} "
            f"train_acc={train_summary['accuracy']:.6f} "
            f"val_acc={val_summary['accuracy']:.6f} "
            f"val_loss={val_summary['loss']:.6f} "
            f"train_seconds={train_elapsed:.3f}"
        )

    if final_train_summary is None:
        raise RuntimeError("RL training finished without epoch metrics.")
    if best_validation_summary is None or best_rl_epoch == 0:
        raise RuntimeError("RL training finished without a best checkpoint.")

    print_timing_summary(timings)

    peak_allocated = torch.cuda.max_memory_allocated() / 1024**3
    total_runtime = time.perf_counter() - run_started
    write_csv(timing_csv_path, build_timing_rows(timings), TIMING_FIELDS)
    write_csv(
        run_summary_path,
        [
            {
                "dataset": DATASET_NAME,
                "model": MODEL_NAME,
                "samples": EXPECTED_SAMPLES,
                "noise_rate": NOISE_RATE,
                "seed": SEED,
                "update_mode": ACTOR_UPDATE_MODE,
                "update_samples": POLICY_UPDATE_SAMPLES,
                "warmup_epoch": warmup_result["best_epoch"],
                "epochs": RL_EPOCHS,
                "steps": TRAJECTORY_LENGTH,
                "k": K,
                "actor_lr": ACTOR_LR,
                "critic_lr": CRITIC_LR,
                "best_epoch": best_rl_epoch,
                "best_val_accuracy": best_validation_summary["accuracy"],
                "best_val_loss": best_validation_summary["loss"],
                "last_val_accuracy": val_summary["accuracy"],
                "last_val_loss": val_summary["loss"],
                "total_seconds": total_runtime,
                "mean_epoch_seconds": sum(epoch_seconds) / len(epoch_seconds),
                "gpu_memory_gib": peak_allocated,
                "actor_last": str(actor_last_checkpoint_path),
                "critic_last": str(critic_last_checkpoint_path),
            }
        ],
        RUN_SUMMARY_FIELDS,
    )

    print("\n[RESULT]")
    print(
        f"best_epoch={best_rl_epoch} val_acc={float(best_validation_summary['accuracy']):.4f} "
        f"val_loss={float(best_validation_summary['loss']):.4f}"
    )
    print(
        f"last_epoch={RL_EPOCHS} val_acc={float(val_summary['accuracy']):.4f} "
        f"val_loss={float(val_summary['loss']):.4f}"
    )
    print(
        f"total={total_runtime:.1f}s mean_epoch={sum(epoch_seconds) / len(epoch_seconds):.1f}s "
        f"gpu={peak_allocated:.2f}GiB"
    )
    print(f"output={OUTPUT_DIR}")
    print("next=cifar_test/cifar_correction.py")


def run_with_file_logging() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    log_path = OUTPUT_DIR / RUN_LOG_FILENAME
    with log_path.open("w", encoding="utf-8", buffering=1) as log_handle:
        stdout = TeeStream(sys.stdout, log_handle)
        stderr = TeeStream(sys.stderr, log_handle)
        with redirect_stdout(stdout), redirect_stderr(stderr):
            print(f"run_log={log_path}")
            main()
