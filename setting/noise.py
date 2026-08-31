"""Create one reusable synthetic-noise artifact for CIFAR-10.

All paths and generation settings come from ``setting.config``. Every
baseline loads the same saved labels and mask instead of generating noise
independently.
"""

from __future__ import annotations

import torch
from torchvision.datasets import CIFAR10

from log.common import save_numpy
from setting import data as cifar


CONFIG = cifar.CONFIG
CIFAR10_ROOT = CONFIG.data_root
OUTPUT_DIR = CONFIG.noise_output_dir
NOISE_RATE = CONFIG.data.noise_rate
NOISE_TYPE = CONFIG.data.noise_type
SEED = CONFIG.data.seed
SUBSET_SEED = CONFIG.data.subset_seed
NUM_CLASSES = len(CONFIG.data.classes)
EXPECTED_SAMPLES = CONFIG.data.train_samples
DOWNLOAD_CIFAR10 = CONFIG.data.download
OVERWRITE = CONFIG.runtime.overwrite_noise


def main() -> None:
    cifar.require_available_outputs(cifar.NOISE_ARTIFACT_PATHS, overwrite=OVERWRITE, stage="Noise")

    dataset = CIFAR10(root=CIFAR10_ROOT, train=True, download=DOWNLOAD_CIFAR10)
    source_images = torch.from_numpy(dataset.data).permute(0, 3, 1, 2).contiguous()
    source_labels = torch.tensor(dataset.targets, dtype=torch.long)
    training_indices = cifar.build_balanced_training_indices(source_labels)
    clean_labels = source_labels[training_indices].contiguous()

    if NOISE_TYPE == "idn":
        full_noisy_labels, full_noise_mask = cifar.inject_instance_dependent_noise(
            source_images, source_labels, seed=SEED
        )
        noisy_labels = full_noisy_labels[training_indices].contiguous()
        noise_mask = full_noise_mask[training_indices].contiguous()
    else:
        noisy_labels, noise_mask = cifar.inject_stratified_symmetric_noise(clean_labels, seed=SEED)
    save_numpy(cifar.TRAIN_INDICES_PATH, training_indices.numpy())
    save_numpy(cifar.NOISY_LABELS_PATH, noisy_labels.numpy())
    save_numpy(cifar.NOISE_MASK_PATH, noise_mask.numpy())

    print(f"output_dir={OUTPUT_DIR}")
    print(
        f"samples={EXPECTED_SAMPLES} samples_per_class="
        f"{EXPECTED_SAMPLES // NUM_CLASSES} classes={NUM_CLASSES} "
        f"subset_seed={SUBSET_SEED}"
    )
    print(
        f"noise_type={NOISE_TYPE} target_noise_rate={NOISE_RATE:.4f} "
        f"actual_noise_rate={float(noise_mask.float().mean()):.6f} "
        f"noise_count={int(noise_mask.sum())} seed={SEED}"
    )
    if NOISE_TYPE == "idn":
        print(f"idn_flip_rate_std={CONFIG.data.idn_flip_rate_std}")
    for path in cifar.NOISE_ARTIFACT_PATHS:
        print(f"saved={path}")
