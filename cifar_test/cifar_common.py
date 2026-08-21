"""Shared CIFAR-10 data, model, preprocessing, and engine configuration.

This module has no executable stage of its own.  Keeping these primitives here
prevents warm-up, RL, correction, fine-tuning, and evaluation entry points from
importing one another.
"""

from __future__ import annotations

import sys
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from torch import Tensor, nn
from torchvision.datasets import CIFAR10

from cifar_test import rl_engine as engine
from cifar_test.cifar_config import CONFIG
from cifar_test.resnet import build_cifar_resnet18


CIFAR10_ROOT = CONFIG.data_root
DOWNLOAD_CIFAR10 = CONFIG.data.download
TRAIN_INDICES_PATH = CONFIG.noise_output_dir / "train_indices.npy"
NOISY_LABELS_PATH = CONFIG.noise_output_dir / "train_noisy_labels.npy"
NOISE_MASK_PATH = CONFIG.noise_output_dir / "train_noise_mask.npy"
NOISE_ARTIFACT_PATHS = (TRAIN_INDICES_PATH, NOISY_LABELS_PATH, NOISE_MASK_PATH)
WARMUP_CHECKPOINT_PATH = CONFIG.warmup_checkpoint_path

CLASSES = CONFIG.data.classes
NUM_CLASSES = len(CLASSES)
EXPECTED_SAMPLES = CONFIG.data.train_samples
NATIVE_TRAIN_SAMPLES = 50_000
SUBSET_SEED = CONFIG.data.subset_seed
SEED = CONFIG.data.seed

MODEL_NAME = CONFIG.model.name
PRETRAINED = CONFIG.model.pretrained
IMAGE_SIZE = CONFIG.data.image_size
CIFAR10_MEAN = CONFIG.data.mean
CIFAR10_STD = CONFIG.data.std


def require_files(paths: tuple[Path, ...], *, stage: str) -> None:
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"{stage} inputs not found: {missing}")


def require_available_outputs(paths: list[Path] | tuple[Path, ...], *, overwrite: bool, stage: str) -> None:
    existing = [path for path in paths if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            f"{stage} outputs already exist: {existing}. Set the matching "
            "runtime.overwrite_* option to True only when replacement is "
            "intentional."
        )


def run_with_log(log_path: Path, operation: Callable[[], None]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8", buffering=1) as handle:
        stdout = engine.TeeStream(sys.stdout, handle)
        stderr = engine.TeeStream(sys.stderr, handle)
        with redirect_stdout(stdout), redirect_stderr(stderr):
            print(f"run_log={log_path}")
            operation()


def build_balanced_training_indices(labels: Tensor) -> Tensor:
    samples_per_class = EXPECTED_SAMPLES // NUM_CLASSES
    if EXPECTED_SAMPLES == NATIVE_TRAIN_SAMPLES:
        return torch.arange(NATIVE_TRAIN_SAMPLES, dtype=torch.long)

    generator = torch.Generator().manual_seed(SUBSET_SEED)
    selected = []
    for class_id in CLASSES:
        class_indices = torch.where(labels == class_id)[0]
        order = torch.randperm(class_indices.numel(), generator=generator)
        selected.append(class_indices[order[:samples_per_class]])
    return torch.cat(selected).sort().values.to(torch.long).contiguous()


def load_training_indices(source_labels: Tensor) -> Tensor:
    if not TRAIN_INDICES_PATH.is_file():
        raise FileNotFoundError(
            f"Training-index artifact not found: {TRAIN_INDICES_PATH}. Run cifar_noise.py first."
        )
    indices = torch.from_numpy(np.load(TRAIN_INDICES_PATH, allow_pickle=False)).to(torch.long).contiguous()
    if indices.shape != (EXPECTED_SAMPLES,):
        raise ValueError(
            f"Training indices must have shape ({EXPECTED_SAMPLES},), got {tuple(indices.shape)}."
        )
    if not torch.equal(indices, build_balanced_training_indices(source_labels)):
        raise ValueError(
            "Training-index artifact does not match train_samples/subset_seed. "
            "Regenerate the noise artifacts with cifar_noise.py."
        )
    return indices


def load_selected_cifar10_train() -> tuple[Tensor, Tensor]:
    dataset = CIFAR10(root=CIFAR10_ROOT, train=True, download=DOWNLOAD_CIFAR10)
    source_labels = torch.tensor(dataset.targets, dtype=torch.long)
    indices = load_training_indices(source_labels)
    source_images = torch.from_numpy(dataset.data).permute(0, 3, 1, 2)
    images = engine.pin_for_cuda(source_images[indices].contiguous())
    labels = engine.pin_for_cuda(source_labels[indices].contiguous())
    return images, labels


def _evaluation_indices(labels: Tensor) -> dict[str, Tensor]:
    generator = torch.Generator().manual_seed(SEED)
    chunks: dict[str, list[Tensor]] = {"val": [], "test": []}
    for class_id in CLASSES:
        class_indices = labels.eq(class_id).nonzero(as_tuple=False).flatten()
        class_indices = class_indices[torch.randperm(class_indices.numel(), generator=generator)]
        midpoint = class_indices.numel() // 2
        chunks["val"].append(class_indices[:midpoint])
        chunks["test"].append(class_indices[midpoint:])

    result: dict[str, Tensor] = {}
    for split in ("val", "test"):
        indices = torch.cat(chunks[split])
        result[split] = indices[torch.randperm(indices.numel(), generator=generator)]
    return result


def load_cifar10_evaluation_split(split: str) -> tuple[Tensor, Tensor]:
    if split not in {"val", "test"}:
        raise ValueError("split must be 'val' or 'test'.")
    dataset = CIFAR10(root=CIFAR10_ROOT, train=False, download=DOWNLOAD_CIFAR10)
    all_images = torch.from_numpy(dataset.data).permute(0, 3, 1, 2)
    all_labels = torch.tensor(dataset.targets, dtype=torch.long)
    indices = _evaluation_indices(all_labels)[split]
    images = engine.pin_for_cuda(all_images[indices].contiguous())
    labels = engine.pin_for_cuda(all_labels[indices].contiguous())
    return images, labels


def load_cifar10_validation() -> dict[str, tuple[Tensor, Tensor]]:
    return {"val": load_cifar10_evaluation_split("val")}


def build_model(pretrained: bool = PRETRAINED, num_classes: int = NUM_CLASSES) -> nn.Module:
    return build_cifar_resnet18(pretrained, num_classes)


def preprocess_cifar10(images: Tensor, device: torch.device, mean: Tensor, std: Tensor) -> Tensor:
    images = images.to(
        device=device,
        dtype=torch.float32,
        non_blocking=True,
        memory_format=(torch.channels_last if engine.USE_CHANNELS_LAST else torch.contiguous_format),
    )
    return images.div_(255.0).sub_(mean).div_(std)


def preprocess_cifar10_training(
    images: Tensor, device: torch.device, mean: Tensor, std: Tensor, generator: torch.Generator
) -> Tensor:
    if not CONFIG.augmentation.enabled:
        return preprocess_cifar10(images, device, mean, std)
    batch = images.to(device=device, dtype=torch.float32, non_blocking=True)
    padding = CONFIG.augmentation.random_crop_padding
    if padding > 0:
        padded = torch.nn.functional.pad(
            batch, (padding, padding, padding, padding), mode="constant", value=0.0
        )
        batch_size, channels, height, width = batch.shape
        padded_width = padded.size(3)
        offset_limit = 2 * padding + 1
        top = torch.randint(offset_limit, (batch_size,), device=device, generator=generator)
        left = torch.randint(offset_limit, (batch_size,), device=device, generator=generator)
        rows = top[:, None, None] + torch.arange(height, device=device)[None, :, None]
        columns = left[:, None, None] + torch.arange(width, device=device)[None, None, :]
        flat_indices = (rows * padded_width + columns).flatten(1)
        batch = (
            padded.flatten(2)
            .gather(2, flat_indices[:, None].expand(-1, channels, -1))
            .reshape(batch_size, channels, height, width)
        )

    flip_probability = CONFIG.augmentation.horizontal_flip_probability
    if flip_probability > 0:
        flip_mask = torch.rand(batch.size(0), device=device, generator=generator) < flip_probability
        batch = torch.where(flip_mask[:, None, None, None], batch.flip(-1), batch)

    batch = batch.contiguous(
        memory_format=(torch.channels_last if engine.USE_CHANNELS_LAST else torch.contiguous_format)
    )
    return batch.div_(255.0).sub_(mean).div_(std)


def configure_engine() -> None:
    engine.DATASET_NAME = "CIFAR-10"
    engine.CLASS_IDS = CLASSES
    engine.NUM_CLASSES = NUM_CLASSES
    engine.EXPECTED_SAMPLES = EXPECTED_SAMPLES
    engine.TRAIN_SUBSET_SEED = SUBSET_SEED
    engine.NOISE_RATE = CONFIG.data.noise_rate
    engine.SEED = SEED
    engine.ACTOR_UPDATE_MODE = CONFIG.rl.update_mode
    engine.POLICY_UPDATE_SAMPLES = CONFIG.actor_update_samples
    engine.RECORD_CHANGE_DIAGNOSTICS = CONFIG.rl.record_change_diagnostics
    engine.CHANGE_DIAGNOSTIC_PROBE_SIZE = CONFIG.change_diagnostic_probe_size
    engine.OUTPUT_DIR = CONFIG.rl_output_dir
    engine.MODEL_OUTPUT_DIR = CONFIG.rl_model_dir
    engine.ACTOR_BEST_CHECKPOINT_FILENAME = CONFIG.output.actor_best_checkpoint_name
    engine.ACTOR_LAST_CHECKPOINT_FILENAME = CONFIG.output.actor_last_checkpoint_name
    engine.CRITIC_BEST_CHECKPOINT_FILENAME = CONFIG.output.critic_best_checkpoint_name
    engine.CRITIC_LAST_CHECKPOINT_FILENAME = CONFIG.output.critic_last_checkpoint_name
    engine.EXTERNAL_NOISY_LABELS_PATH = NOISY_LABELS_PATH
    engine.EXTERNAL_NOISE_MASK_PATH = NOISE_MASK_PATH
    engine.EXTERNAL_WARMUP_CHECKPOINT_PATH = WARMUP_CHECKPOINT_PATH
    engine.MODEL_NAME = MODEL_NAME
    engine.WARMUP_MODEL_ID = CONFIG.warmup.model_id
    engine.MODEL_FACTORY = build_cifar_resnet18
    engine.PRETRAINED = PRETRAINED
    engine.IMAGE_SIZE = IMAGE_SIZE
    engine.DATA_MEAN = CIFAR10_MEAN
    engine.DATA_STD = CIFAR10_STD
    engine.TRAIN_AUGMENTATION_ENABLED = CONFIG.augmentation.enabled
    engine.TRAIN_RANDOM_CROP_PADDING = CONFIG.augmentation.random_crop_padding
    engine.TRAIN_HORIZONTAL_FLIP_PROBABILITY = CONFIG.augmentation.horizontal_flip_probability
    engine.WARMUP_EPOCHS = CONFIG.warmup.epochs
    engine.WARMUP_BATCH_SIZE = CONFIG.warmup.batch_size
    engine.WARMUP_EVAL_BATCH_SIZE = CONFIG.warmup.eval_batch_size
    engine.WARMUP_LR = CONFIG.warmup.learning_rate
    engine.WARMUP_WEIGHT_DECAY = CONFIG.warmup.weight_decay
    engine.WARMUP_MOMENTUM = CONFIG.warmup.momentum
    engine.WARMUP_LR_DECAY_FRACTION = CONFIG.warmup.lr_decay_fraction
    engine.WARMUP_LR_DECAY_FACTOR = CONFIG.warmup.lr_decay_factor
    engine.WARMUP_MIN_NOISY_VALIDATION_ACCURACY = CONFIG.warmup.min_noisy_validation_accuracy
    engine.RL_EPOCHS = CONFIG.rl.epochs
    engine.TRAJECTORY_LENGTH = CONFIG.rl.trajectory_length
    engine.INITIAL_STATE_RANDOMIZATION_RATE = CONFIG.rl.initial_state_randomization_rate
    engine.FEATURE_BATCH_SIZE = CONFIG.rl.feature_batch_size
    engine.POLICY_UPDATE_BATCH_SIZE = CONFIG.actor_update_batch_size
    engine.K = CONFIG.knn.k
    engine.TEMPERATURE = CONFIG.knn.temperature
    engine.KNN_QUERY_CHUNK_SIZE = CONFIG.knn.query_chunk_size
    engine.KNN_REFERENCE_CHUNK_SIZE = CONFIG.knn.reference_chunk_size
    engine.CORRECTION_CHUNK_SIZE = CONFIG.knn.correction_chunk_size
    engine.ACTOR_LR = CONFIG.rl.actor_learning_rate
    engine.ACTOR_MOMENTUM = CONFIG.rl.actor_momentum
    engine.ACTOR_WEIGHT_DECAY = CONFIG.rl.actor_weight_decay
    engine.CRITIC_LR = CONFIG.rl.critic_learning_rate
    engine.CRITIC_MOMENTUM = CONFIG.rl.critic_momentum
    engine.CRITIC_WEIGHT_DECAY = CONFIG.rl.critic_weight_decay
    engine.CRITIC_NUM_BINS = CONFIG.rl.critic_num_bins
    engine.DISCOUNT_FACTOR = CONFIG.rl.discount_factor
    engine.NLA_WEIGHT = CONFIG.rl.reward_nla_weight
    engine.LR_DECAY_FACTOR = CONFIG.rl.lr_decay_factor
    engine.LR_DECAY_FRACTION = CONFIG.rl.lr_decay_fraction
    engine.USE_AMP = CONFIG.runtime.use_amp
    engine.AMP_DTYPE = getattr(torch, CONFIG.runtime.amp_dtype)
    engine.USE_CHANNELS_LAST = CONFIG.runtime.use_channels_last
    engine.CUDNN_BENCHMARK = CONFIG.runtime.cudnn_benchmark

    engine.TRAIN_DATA_LOADER = load_selected_cifar10_train
    engine.EVALUATION_DATA_LOADER = load_cifar10_validation
    engine.PREPROCESS_FUNCTION = preprocess_cifar10
    engine.TRAIN_PREPROCESS_FUNCTION = preprocess_cifar10_training
    engine.CONFIGURED = True
