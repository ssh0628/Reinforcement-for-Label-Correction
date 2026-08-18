"""Create one reusable 40% symmetric-noise artifact for CIFAR-10.

All paths and generation settings come from ``config_resent18.py``. Every
baseline loads the same saved labels and mask instead of generating noise
independently.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from torchvision.datasets import CIFAR10


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cifar_test.config_resent18 import CONFIG


CIFAR10_ROOT = CONFIG.data.root
OUTPUT_DIR = CONFIG.noise_output_dir
NOISE_RATE = CONFIG.data.noise_rate
SEED = CONFIG.data.seed
NUM_CLASSES = len(CONFIG.data.classes)
EXPECTED_SAMPLES = CONFIG.data.train_samples
DOWNLOAD_CIFAR10 = CONFIG.data.download
OVERWRITE = CONFIG.runtime.overwrite_noise

CLEAN_LABELS_FILENAME = "train_clean_labels.npy"
NOISY_LABELS_FILENAME = "train_noisy_labels.npy"
NOISE_MASK_FILENAME = "train_noise_mask.npy"
METADATA_FILENAME = "noise_metadata.json"


def _array_sha256(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    return hashlib.sha256(contiguous.view(np.uint8)).hexdigest()


def _save_npy_atomic(path: Path, array: np.ndarray) -> None:
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    try:
        with temporary_path.open("wb") as handle:
            np.save(handle, array, allow_pickle=False)
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _save_json_atomic(path: Path, payload: dict[str, object]) -> None:
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    try:
        temporary_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)


def inject_stratified_symmetric_noise(
    clean_labels: Tensor,
) -> tuple[Tensor, Tensor]:
    if clean_labels.shape != (EXPECTED_SAMPLES,):
        raise ValueError(
            f"Expected {EXPECTED_SAMPLES} labels, got {tuple(clean_labels.shape)}."
        )
    if clean_labels.dtype != torch.long:
        raise TypeError("CIFAR-10 labels must use torch.long.")

    generator = torch.Generator().manual_seed(SEED)
    noisy_labels = clean_labels.clone()
    noise_mask = torch.zeros(EXPECTED_SAMPLES, dtype=torch.bool)
    class_sizes = [
        int(clean_labels.eq(class_id).sum())
        for class_id in range(NUM_CLASSES)
    ]
    exact_counts = [size * NOISE_RATE for size in class_sizes]
    noise_counts = [math.floor(value) for value in exact_counts]
    target_count = round(EXPECTED_SAMPLES * NOISE_RATE)
    remainder = target_count - sum(noise_counts)
    allocation_order = sorted(
        range(NUM_CLASSES),
        key=lambda index: exact_counts[index] - noise_counts[index],
        reverse=True,
    )
    for class_id in allocation_order[:remainder]:
        noise_counts[class_id] += 1

    for class_id, noise_count in enumerate(noise_counts):
        class_indices = clean_labels.eq(class_id).nonzero(as_tuple=False).flatten()
        selected = class_indices[
            torch.randperm(class_indices.numel(), generator=generator)[:noise_count]
        ]
        alternatives = torch.randint(
            NUM_CLASSES - 1,
            (noise_count,),
            generator=generator,
        )
        alternatives += alternatives.ge(clean_labels[selected])
        noisy_labels[selected] = alternatives
        noise_mask[selected] = True

    if not torch.equal(noisy_labels.ne(clean_labels), noise_mask):
        raise RuntimeError("Noise mask does not match the changed labels.")
    if int(noise_mask.sum()) != target_count:
        raise RuntimeError("Noise injection did not reach the requested rate.")
    return noisy_labels, noise_mask


def main() -> None:
    if not 0.0 <= NOISE_RATE < 1.0:
        raise ValueError("NOISE_RATE must be in [0, 1).")
    output_paths = (
        OUTPUT_DIR / CLEAN_LABELS_FILENAME,
        OUTPUT_DIR / NOISY_LABELS_FILENAME,
        OUTPUT_DIR / NOISE_MASK_FILENAME,
        OUTPUT_DIR / METADATA_FILENAME,
    )
    existing = [path for path in output_paths if path.exists()]
    if existing and not OVERWRITE:
        raise FileExistsError(
            f"Noise artifacts already exist: {existing}. Set OVERWRITE=True "
            "only when replacement is intentional."
        )

    dataset = CIFAR10(
        root=CIFAR10_ROOT,
        train=True,
        download=DOWNLOAD_CIFAR10,
    )
    clean_labels = torch.tensor(dataset.targets, dtype=torch.long)
    if clean_labels.numel() != EXPECTED_SAMPLES:
        raise RuntimeError(
            f"Unexpected CIFAR-10 train size: {clean_labels.numel()}."
        )
    if set(clean_labels.tolist()) != set(range(NUM_CLASSES)):
        raise RuntimeError("CIFAR-10 train is missing one or more classes.")

    noisy_labels, noise_mask = inject_stratified_symmetric_noise(clean_labels)
    clean_array = clean_labels.numpy().astype(np.int64, copy=False)
    noisy_array = noisy_labels.numpy().astype(np.int64, copy=False)
    mask_array = noise_mask.numpy().astype(np.bool_, copy=False)
    class_counts = np.bincount(clean_array, minlength=NUM_CLASSES)
    noisy_class_counts = np.bincount(noisy_array, minlength=NUM_CLASSES)
    per_class_noise_counts = np.bincount(
        clean_array[mask_array],
        minlength=NUM_CLASSES,
    )

    metadata: dict[str, object] = {
        "dataset": "CIFAR-10",
        "split": "train",
        "sample_count": EXPECTED_SAMPLES,
        "num_classes": NUM_CLASSES,
        "noise_type": "stratified_symmetric",
        "noise_rate": NOISE_RATE,
        "noise_count": int(mask_array.sum()),
        "seed": SEED,
        "class_counts": class_counts.tolist(),
        "noisy_class_counts": noisy_class_counts.tolist(),
        "per_class_noise_counts": per_class_noise_counts.tolist(),
        "clean_labels_sha256": _array_sha256(clean_array),
        "noisy_labels_sha256": _array_sha256(noisy_array),
        "noise_mask_sha256": _array_sha256(mask_array),
        "files": {
            "clean_labels": CLEAN_LABELS_FILENAME,
            "noisy_labels": NOISY_LABELS_FILENAME,
            "noise_mask": NOISE_MASK_FILENAME,
        },
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _save_npy_atomic(output_paths[0], clean_array)
    _save_npy_atomic(output_paths[1], noisy_array)
    _save_npy_atomic(output_paths[2], mask_array)
    _save_json_atomic(output_paths[3], metadata)

    print(f"output_dir={OUTPUT_DIR}")
    print(f"samples={EXPECTED_SAMPLES} classes={NUM_CLASSES}")
    print(
        f"noise_rate={NOISE_RATE:.2f} noise_count={int(mask_array.sum())} "
        f"seed={SEED}"
    )
    print(f"per_class_noise_counts={tuple(per_class_noise_counts.tolist())}")
    for path in output_paths:
        print(f"saved={path}")


if __name__ == "__main__":
    main()
