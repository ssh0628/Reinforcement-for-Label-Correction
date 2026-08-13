"""CIFAR-10 RLNLC full-vs-subset experiment for an RTX 5080.

Set the shared noisy-label and warmup-checkpoint paths below, then run this file
twice with ``ACTOR_UPDATE_MODE`` set to ``"full"`` and ``"subset"``.  After
restoring the best actor, each run performs 25-step cleaning over all 50,000
training examples and saves the resulting hard labels for fine-tuning.

Experiment profile
------------------
- CIFAR-10 train: all 50,000 images and all ten classes.
- Validation/test: stratified 50/50 split of the official 10,000 test images.
- 40% stratified symmetric label noise, seed 0.
- ConvNeXtV2 Tiny without pretrained weights.
- Three supervised warmup epochs.
- Three RL epochs x five trajectory steps (15 RL steps).
- Full actor update: 50,000 samples per RL step.
- Subset actor update: 5,000 samples per RL step (10%).
- Validation selects the best checkpoint; the held-out test half is evaluated
  only after restoring the best actor.

The common engine is imported rather than copied so fixes to RL, KNN, metrics,
checkpointing, and timing cannot silently diverge between MNIST and CIFAR-10.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import Tensor
from torch.nn import functional as F
from torchvision.datasets import CIFAR10


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import mnist_test_rtx5080 as benchmark


CIFAR10_ROOT = PROJECT_ROOT / "data" / "cifar10"
DOWNLOAD_CIFAR10 = True

# User-editable shared input paths.
NOISY_LABELS_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "cifar10_shared"
    / "noise_40_seed0"
    / "train_noisy_labels.npy"
)
NOISE_MASK_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "cifar10_shared"
    / "noise_40_seed0"
    / "train_noise_mask.npy"
)
WARMUP_CHECKPOINT_PATH = (
    PROJECT_ROOT / "outputs" / "cifar10_shared" / "warmup" / "warmup_best.pt"
)

CLASSES = tuple(range(10))
NUM_CLASSES = len(CLASSES)
EXPECTED_SAMPLES = 50_000
NOISE_RATE = 0.40
SEED = 0

# Change only this value between the two comparison runs.
ACTOR_UPDATE_MODE = "full"  # "full" or "subset"
POLICY_UPDATE_SUBSET_SIZE = 5_000
OUTPUT_DIR = (
    PROJECT_ROOT / "outputs" / f"cifar10_test_rtx5080_{ACTOR_UPDATE_MODE}"
)
CLEANING_TRAJECTORY_LENGTH = 25
CORRECTED_LABELS_PATH = OUTPUT_DIR / "train_corrected_labels.npy"
OVERWRITE = False

MODEL_NAME = "convnextv2_tiny.fcmae_ft_in22k_in1k"
PRETRAINED = False
IMAGE_SIZE = 224
DROP_RATE = 0.1
DROP_PATH_RATE = 0.2

WARMUP_EPOCHS = 3
WARMUP_FREEZE_EPOCHS = 0
WARMUP_BATCH_SIZE = 64
WARMUP_EVAL_BATCH_SIZE = 256
WARMUP_HEAD_LR = 1e-3
WARMUP_BACKBONE_LR = 1e-4
WARMUP_UNFROZEN_HEAD_LR = 5e-4
WARMUP_MIN_LR = 1e-6
WARMUP_WEIGHT_DECAY = 0.05
# Plain cross entropy keeps this baseline free of label-smoothing heuristics.
WARMUP_LABEL_SMOOTHING = 0.0
WARMUP_GRAD_CLIP_NORM = 1.0
# A sanity guard against a completely failed warmup. Clean labels are not used
# for this gate or for checkpoint selection.
WARMUP_MIN_NOISY_VALIDATION_ACCURACY = 0.20

RL_EPOCHS = 3
TRAJECTORY_LENGTH = 5
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
ACTOR_BETAS = (0.9, 0.999)
ACTOR_EPS = 1e-8
CRITIC_LR = 1e-2
CRITIC_MOMENTUM = 0.9
CRITIC_WEIGHT_DECAY = 5e-4
DISCOUNT_FACTOR = 0.9
NLA_WEIGHT = 0.5
LR_DECAY_FACTOR = 0.1
LR_DECAY_FRACTION = 0.5

RL_OUTPUT_FILENAMES = (
    benchmark.RUN_LOG_FILENAME,
    benchmark.RL_BEST_CHECKPOINT_FILENAME,
    benchmark.RL_LAST_CHECKPOINT_FILENAME,
    benchmark.TRAIN_CSV_FILENAME,
    benchmark.TEST_CSV_FILENAME,
    benchmark.TEST_PER_CLASS_CSV_FILENAME,
    benchmark.TIMING_CSV_FILENAME,
    benchmark.RUN_SUMMARY_CSV_FILENAME,
    benchmark.CLEANING_CSV_FILENAME,
    benchmark.CLEANING_SUMMARY_FILENAME,
    benchmark.CLEANING_PER_CLASS_FILENAME,
)


def validate_output_destination() -> None:
    paths = [OUTPUT_DIR / filename for filename in RL_OUTPUT_FILENAMES]
    paths.append(CORRECTED_LABELS_PATH)
    existing = [path for path in paths if path.exists()]
    if existing and not OVERWRITE:
        raise FileExistsError(
            f"RL outputs already exist: {existing}. Set OVERWRITE=True only "
            "when replacement is intentional."
        )


def validate_input_artifacts() -> None:
    paths = (
        NOISY_LABELS_PATH,
        NOISE_MASK_PATH,
        WARMUP_CHECKPOINT_PATH,
    )
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"RL inputs not found: {missing}")


def _as_image_tensor(dataset: CIFAR10) -> Tensor:
    """Convert torchvision's NHWC uint8 array to pinned NCHW storage."""
    return benchmark.pin_for_cuda(
        torch.from_numpy(dataset.data).permute(0, 3, 1, 2).contiguous()
    )


def load_full_cifar10_train() -> tuple[Tensor, Tensor]:
    dataset = CIFAR10(
        root=CIFAR10_ROOT,
        train=True,
        download=DOWNLOAD_CIFAR10,
    )
    images = _as_image_tensor(dataset)
    labels = benchmark.pin_for_cuda(
        torch.tensor(dataset.targets, dtype=torch.long).contiguous()
    )
    if images.size(0) != EXPECTED_SAMPLES or labels.size(0) != EXPECTED_SAMPLES:
        raise RuntimeError(
            "Unexpected CIFAR-10 train size: "
            f"images={images.size(0)}, labels={labels.size(0)}."
        )
    if set(labels.tolist()) != set(CLASSES):
        raise RuntimeError("CIFAR-10 train must contain every class from 0 to 9.")
    return images, labels


def load_cifar10_validation_test() -> dict[str, tuple[Tensor, Tensor]]:
    """Split the official CIFAR-10 test set 50/50 within every class."""
    dataset = CIFAR10(
        root=CIFAR10_ROOT,
        train=False,
        download=DOWNLOAD_CIFAR10,
    )
    all_images = torch.from_numpy(dataset.data).permute(0, 3, 1, 2)
    labels = torch.tensor(dataset.targets, dtype=torch.long)
    generator = torch.Generator().manual_seed(SEED)
    split_indices: dict[str, list[Tensor]] = {"val": [], "test": []}

    for class_id in CLASSES:
        indices = labels.eq(class_id).nonzero(as_tuple=False).flatten()
        indices = indices[torch.randperm(indices.numel(), generator=generator)]
        val_count = indices.numel() // 2
        split_indices["val"].append(indices[:val_count])
        split_indices["test"].append(indices[val_count:])

    result: dict[str, tuple[Tensor, Tensor]] = {}
    for split, chunks in split_indices.items():
        indices = torch.cat(chunks)
        indices = indices[torch.randperm(indices.numel(), generator=generator)]
        images = benchmark.pin_for_cuda(all_images[indices].contiguous())
        split_labels = benchmark.pin_for_cuda(labels[indices].contiguous())
        if set(split_labels.tolist()) != set(CLASSES):
            raise RuntimeError(f"CIFAR-10 {split} split is missing a class.")
        result[split] = (images, split_labels)

    if result["val"][0].size(0) + result["test"][0].size(0) != len(dataset):
        raise RuntimeError("CIFAR-10 validation/test split lost samples.")
    return result


def preprocess_cifar10(
    images: Tensor,
    device: torch.device,
    mean: Tensor,
    std: Tensor,
) -> Tensor:
    if images.ndim != 4 or images.size(1) != 3:
        raise ValueError(
            "CIFAR-10 images must have shape [N, 3, H, W], got "
            f"{tuple(images.shape)}."
        )
    images = images.to(device=device, dtype=torch.float32, non_blocking=True)
    images = images.div_(255.0)
    images = F.interpolate(
        images,
        size=(IMAGE_SIZE, IMAGE_SIZE),
        mode="bicubic",
        align_corners=False,
        antialias=True,
    )
    images = images.contiguous(
        memory_format=(
            torch.channels_last
            if benchmark.USE_CHANNELS_LAST
            else torch.contiguous_format
        )
    )
    return (images - mean) / std


def configure_benchmark() -> None:
    if ACTOR_UPDATE_MODE not in {"full", "subset"}:
        raise ValueError("ACTOR_UPDATE_MODE must be 'full' or 'subset'.")

    policy_update_samples = (
        EXPECTED_SAMPLES
        if ACTOR_UPDATE_MODE == "full"
        else min(EXPECTED_SAMPLES, POLICY_UPDATE_SUBSET_SIZE)
    )

    benchmark.DATASET_NAME = "CIFAR-10"
    benchmark.DATASET_STAGE_PREFIX = "cifar10"
    benchmark.DIGITS = CLASSES
    benchmark.NUM_CLASSES = NUM_CLASSES
    benchmark.EXPECTED_SAMPLES = EXPECTED_SAMPLES
    benchmark.NOISE_RATE = NOISE_RATE
    benchmark.SEED = SEED
    benchmark.ACTOR_UPDATE_MODE = ACTOR_UPDATE_MODE
    benchmark.POLICY_UPDATE_SUBSET_SIZE = POLICY_UPDATE_SUBSET_SIZE
    benchmark.POLICY_UPDATE_SAMPLES = policy_update_samples
    benchmark.OUTPUT_DIR = OUTPUT_DIR
    benchmark.EXTERNAL_NOISY_LABELS_PATH = NOISY_LABELS_PATH
    benchmark.EXTERNAL_NOISE_MASK_PATH = NOISE_MASK_PATH
    benchmark.EXTERNAL_WARMUP_CHECKPOINT_PATH = WARMUP_CHECKPOINT_PATH
    # Preserve the warmup classifier head so the RL checkpoint can be
    # fine-tuned directly on the cleaned labels.
    benchmark.REMOVE_CLASSIFIER_FOR_RL = False
    benchmark.CLEANING_TRAJECTORY_LENGTH = CLEANING_TRAJECTORY_LENGTH
    benchmark.CORRECTED_LABELS_OUTPUT_PATH = CORRECTED_LABELS_PATH

    benchmark.MODEL_NAME = MODEL_NAME
    benchmark.PRETRAINED = PRETRAINED
    benchmark.IMAGE_SIZE = IMAGE_SIZE
    benchmark.DROP_RATE = DROP_RATE
    benchmark.DROP_PATH_RATE = DROP_PATH_RATE
    benchmark.WARMUP_EPOCHS = WARMUP_EPOCHS
    benchmark.WARMUP_FREEZE_EPOCHS = WARMUP_FREEZE_EPOCHS
    benchmark.WARMUP_BATCH_SIZE = WARMUP_BATCH_SIZE
    benchmark.WARMUP_EVAL_BATCH_SIZE = WARMUP_EVAL_BATCH_SIZE
    benchmark.WARMUP_HEAD_LR = WARMUP_HEAD_LR
    benchmark.WARMUP_BACKBONE_LR = WARMUP_BACKBONE_LR
    benchmark.WARMUP_UNFROZEN_HEAD_LR = WARMUP_UNFROZEN_HEAD_LR
    benchmark.WARMUP_MIN_LR = WARMUP_MIN_LR
    benchmark.WARMUP_WEIGHT_DECAY = WARMUP_WEIGHT_DECAY
    benchmark.WARMUP_LABEL_SMOOTHING = WARMUP_LABEL_SMOOTHING
    benchmark.WARMUP_GRAD_CLIP_NORM = WARMUP_GRAD_CLIP_NORM
    benchmark.WARMUP_MIN_NOISY_VALIDATION_ACCURACY = (
        WARMUP_MIN_NOISY_VALIDATION_ACCURACY
    )
    benchmark.RL_EPOCHS = RL_EPOCHS
    benchmark.TRAJECTORY_LENGTH = TRAJECTORY_LENGTH
    benchmark.INITIAL_STATE_RANDOMIZATION_RATE = (
        INITIAL_STATE_RANDOMIZATION_RATE
    )
    benchmark.FEATURE_BATCH_SIZE = FEATURE_BATCH_SIZE
    benchmark.POLICY_UPDATE_BATCH_SIZE = POLICY_UPDATE_BATCH_SIZE
    benchmark.K = K
    benchmark.TEMPERATURE = TEMPERATURE
    benchmark.KNN_QUERY_CHUNK_SIZE = KNN_QUERY_CHUNK_SIZE
    benchmark.KNN_REFERENCE_CHUNK_SIZE = KNN_REFERENCE_CHUNK_SIZE
    benchmark.CORRECTION_CHUNK_SIZE = CORRECTION_CHUNK_SIZE
    benchmark.ACTOR_LR = ACTOR_LR
    benchmark.ACTOR_WEIGHT_DECAY = ACTOR_WEIGHT_DECAY
    benchmark.ACTOR_BETAS = ACTOR_BETAS
    benchmark.ACTOR_EPS = ACTOR_EPS
    benchmark.CRITIC_LR = CRITIC_LR
    benchmark.CRITIC_MOMENTUM = CRITIC_MOMENTUM
    benchmark.CRITIC_WEIGHT_DECAY = CRITIC_WEIGHT_DECAY
    benchmark.DISCOUNT_FACTOR = DISCOUNT_FACTOR
    benchmark.NLA_WEIGHT = NLA_WEIGHT
    benchmark.LR_DECAY_FACTOR = LR_DECAY_FACTOR
    benchmark.LR_DECAY_FRACTION = LR_DECAY_FRACTION

    # main() resolves these names in the imported engine module at runtime.
    benchmark.load_full_mnist_train = load_full_cifar10_train
    benchmark.load_mnist_validation_test = load_cifar10_validation_test
    benchmark.preprocess = preprocess_cifar10


if __name__ == "__main__":
    configure_benchmark()
    validate_input_artifacts()
    validate_output_destination()
    benchmark.run_with_file_logging()
