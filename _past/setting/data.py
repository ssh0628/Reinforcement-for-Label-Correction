"""Shared CIFAR-10 data loading, preprocessing, and model construction.

This module has no executable stage of its own.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn
from torchvision.datasets import CIFAR10

from cifar_test.setting.config import CONFIG
from cifar_test.setting.model import build_cifar_resnet18


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
SUBSET_SEED = CONFIG.data.subset_seed
SEED = CONFIG.data.seed

MODEL_NAME = CONFIG.model.name
PRETRAINED = CONFIG.model.pretrained
CIFAR10_MEAN = CONFIG.data.mean
CIFAR10_STD = CONFIG.data.std


def pin_for_cuda(tensor: Tensor) -> Tensor:
    return tensor.pin_memory() if torch.cuda.is_available() else tensor


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


def build_balanced_training_indices(labels: Tensor) -> Tensor:
    samples_per_class = EXPECTED_SAMPLES // NUM_CLASSES
    if EXPECTED_SAMPLES == 50_000:
        return torch.arange(50_000, dtype=torch.long)

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
    images = pin_for_cuda(source_images[indices].contiguous())
    labels = pin_for_cuda(source_labels[indices].contiguous())
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
    images = pin_for_cuda(all_images[indices].contiguous())
    labels = pin_for_cuda(all_labels[indices].contiguous())
    return images, labels


def build_model(pretrained: bool = PRETRAINED, num_classes: int = NUM_CLASSES) -> nn.Module:
    return build_cifar_resnet18(pretrained, num_classes)


def inject_stratified_symmetric_noise(clean_labels: Tensor) -> tuple[Tensor, Tensor]:
    generator = torch.Generator().manual_seed(SEED)
    noisy_labels = clean_labels.clone()
    noise_mask = torch.zeros_like(clean_labels, dtype=torch.bool)
    class_sizes = [int(clean_labels.eq(class_id).sum()) for class_id in CLASSES]
    exact_counts = [size * CONFIG.data.noise_rate for size in class_sizes]
    noise_counts = [math.floor(count) for count in exact_counts]
    remainder = round(clean_labels.numel() * CONFIG.data.noise_rate) - sum(noise_counts)
    allocation_order = sorted(
        range(NUM_CLASSES), key=lambda index: exact_counts[index] - noise_counts[index], reverse=True
    )
    for index in allocation_order[:remainder]:
        noise_counts[index] += 1

    for class_id, noise_count in zip(CLASSES, noise_counts):
        if noise_count == 0:
            continue
        class_indices = clean_labels.eq(class_id).nonzero(as_tuple=False).flatten()
        selected = class_indices[torch.randperm(class_indices.numel(), generator=generator)[:noise_count]]
        alternatives = torch.randint(NUM_CLASSES - 1, (noise_count,), generator=generator)
        alternatives += alternatives.ge(clean_labels[selected])
        noisy_labels[selected] = alternatives
        noise_mask[selected] = True

    return pin_for_cuda(noisy_labels), pin_for_cuda(noise_mask)


def preprocess_cifar10(images: Tensor, device: torch.device, mean: Tensor, std: Tensor) -> Tensor:
    images = images.to(
        device=device,
        dtype=torch.float32,
        non_blocking=True,
        memory_format=(
            torch.channels_last if CONFIG.runtime.use_channels_last else torch.contiguous_format
        ),
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
        memory_format=(
            torch.channels_last if CONFIG.runtime.use_channels_last else torch.contiguous_format
        )
    )
    return batch.div_(255.0).sub_(mean).div_(std)
