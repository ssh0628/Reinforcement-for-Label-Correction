"""Create one reusable symmetric-noise artifact for CIFAR-10.

All paths and generation settings come from ``cifar_test.setting.config``. Every
baseline loads the same saved labels and mask instead of generating noise
independently.
"""

from __future__ import annotations

import numpy as np
import torch
from torchvision.datasets import CIFAR10

from cifar_test.setting import data as cifar


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


def main() -> None:
    cifar.require_available_outputs(cifar.NOISE_ARTIFACT_PATHS, overwrite=OVERWRITE, stage="Noise")

    dataset = CIFAR10(root=CIFAR10_ROOT, train=True, download=DOWNLOAD_CIFAR10)
    source_labels = torch.tensor(dataset.targets, dtype=torch.long)
    training_indices = cifar.build_balanced_training_indices(source_labels)
    clean_labels = source_labels[training_indices].contiguous()

    noisy_labels, noise_mask = cifar.inject_stratified_symmetric_noise(clean_labels)
    indices_array = training_indices.numpy().astype(np.int64, copy=False)
    noisy_array = noisy_labels.numpy().astype(np.int64, copy=False)
    mask_array = noise_mask.numpy().astype(np.bool_, copy=False)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(cifar.TRAIN_INDICES_PATH, indices_array, allow_pickle=False)
    np.save(cifar.NOISY_LABELS_PATH, noisy_array, allow_pickle=False)
    np.save(cifar.NOISE_MASK_PATH, mask_array, allow_pickle=False)

    print(f"output_dir={OUTPUT_DIR}")
    print(
        f"samples={EXPECTED_SAMPLES} samples_per_class="
        f"{EXPECTED_SAMPLES // NUM_CLASSES} classes={NUM_CLASSES} "
        f"subset_seed={SUBSET_SEED}"
    )
    print(f"noise_rate={NOISE_RATE:.2f} noise_count={int(mask_array.sum())} seed={SEED}")
    for path in cifar.NOISE_ARTIFACT_PATHS:
        print(f"saved={path}")


if __name__ == "__main__":
    main()
