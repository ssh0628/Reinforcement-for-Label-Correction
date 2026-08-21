"""Create one reusable 40% symmetric-noise artifact for CIFAR-10.

All paths and generation settings come from ``cifar_config.py``. Every
baseline loads the same saved labels and mask instead of generating noise
independently.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from torchvision.datasets import CIFAR10


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cifar_test import cifar_common as cifar


CONFIG = cifar.CONFIG
CIFAR10_ROOT = CONFIG.data_root
OUTPUT_DIR = CONFIG.noise_output_dir
NOISE_RATE = CONFIG.data.noise_rate
SEED = CONFIG.data.seed
SUBSET_SEED = CONFIG.data.subset_seed
NUM_CLASSES = len(CONFIG.data.classes)
EXPECTED_SAMPLES = CONFIG.data.train_samples
DOWNLOAD_CIFAR10 = CONFIG.data.download
OVERWRITE = CONFIG.runtime.overwrite_noise

TRAIN_INDICES_FILENAME = cifar.TRAIN_INDICES_PATH.name
NOISY_LABELS_FILENAME = cifar.NOISY_LABELS_PATH.name
NOISE_MASK_FILENAME = cifar.NOISE_MASK_PATH.name


def main() -> None:
    output_paths = (
        OUTPUT_DIR / TRAIN_INDICES_FILENAME,
        OUTPUT_DIR / NOISY_LABELS_FILENAME,
        OUTPUT_DIR / NOISE_MASK_FILENAME,
    )
    existing = [path for path in output_paths if path.exists()]
    if existing and not OVERWRITE:
        raise FileExistsError(
            f"Noise artifacts already exist: {existing}. Set OVERWRITE=True "
            "only when replacement is intentional."
        )

    dataset = CIFAR10(root=CIFAR10_ROOT, train=True, download=DOWNLOAD_CIFAR10)
    source_labels = torch.tensor(dataset.targets, dtype=torch.long)
    training_indices = cifar.build_balanced_training_indices(source_labels)
    clean_labels = source_labels[training_indices].contiguous()

    cifar.configure_engine()
    noisy_labels, noise_mask = cifar.engine.inject_stratified_symmetric_noise(clean_labels)
    indices_array = training_indices.numpy().astype(np.int64, copy=False)
    noisy_array = noisy_labels.numpy().astype(np.int64, copy=False)
    mask_array = noise_mask.numpy().astype(np.bool_, copy=False)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(output_paths[0], indices_array, allow_pickle=False)
    np.save(output_paths[1], noisy_array, allow_pickle=False)
    np.save(output_paths[2], mask_array, allow_pickle=False)

    print(f"output_dir={OUTPUT_DIR}")
    print(
        f"samples={EXPECTED_SAMPLES} samples_per_class="
        f"{EXPECTED_SAMPLES // NUM_CLASSES} classes={NUM_CLASSES} "
        f"subset_seed={SUBSET_SEED}"
    )
    print(f"noise_rate={NOISE_RATE:.2f} noise_count={int(mask_array.sum())} seed={SEED}")
    for path in output_paths:
        print(f"saved={path}")


if __name__ == "__main__":
    main()
