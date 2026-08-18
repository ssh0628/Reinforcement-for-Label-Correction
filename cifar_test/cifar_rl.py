"""Paper-aligned CIFAR-10 symmetric-noise RLNLC baseline.

The actor is a CIFAR-stem ResNet-18 trained with the optimizer and schedule
reported by the paper.  Every policy update uses all 50,000 training samples.
After 500 RL epochs, the last actor performs 25-step label cleaning.

Experiment profile
------------------
- CIFAR-10 train: all 50,000 images and all ten classes.
- Validation/test: stratified 50/50 split of the official 10,000 test images.
- 40% stratified symmetric label noise, seed 0.
- CIFAR-stem ResNet-18 without pretrained weights; native 32x32 inputs.
- 50 supervised warmup epochs with SGD.
- 500 RL epochs x 10 trajectory steps.
- Full actor update: all 50,000 samples per RL step.
- The best checkpoint is retained for diagnostics, but the last actor is used
  for cleaning and held-out evaluation to avoid clean-validation selection.

The common engine is imported rather than copied so fixes to RL, KNN, metrics,
checkpointing, and timing cannot silently diverge between MNIST and CIFAR-10.
All experiment values and output paths come from ``config_resent18.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from torch import Tensor, nn
from torchvision.datasets import CIFAR10


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cifar_test.config_resent18 import CONFIG
from cifar_test.resnet import build_cifar_resnet18

if TYPE_CHECKING:
    import mnist_test_rtx5080.mnist_test_rtx5080 as benchmark
else:
    import mnist_test_rtx5080 as benchmark


CIFAR10_ROOT = CONFIG.data.root
DOWNLOAD_CIFAR10 = CONFIG.data.download
NOISY_LABELS_PATH = CONFIG.noise_output_dir / "train_noisy_labels.npy"
NOISE_MASK_PATH = CONFIG.noise_output_dir / "train_noise_mask.npy"
WARMUP_CHECKPOINT_PATH = CONFIG.warmup_checkpoint_path

CLASSES = CONFIG.data.classes
NUM_CLASSES = len(CLASSES)
EXPECTED_SAMPLES = CONFIG.data.train_samples
NOISE_RATE = CONFIG.data.noise_rate
SEED = CONFIG.data.seed

ACTOR_UPDATE_MODE = CONFIG.rl.update_mode
OUTPUT_DIR = CONFIG.rl_output_dir
CLEANING_TRAJECTORY_LENGTH = CONFIG.rl.cleaning_trajectory_length
CORRECTED_LABELS_PATH = OUTPUT_DIR / "train_corrected_labels.npy"
OVERWRITE = CONFIG.runtime.overwrite_rl

MODEL_NAME = CONFIG.model.name
PRETRAINED = CONFIG.model.pretrained
IMAGE_SIZE = CONFIG.data.image_size
CIFAR10_MEAN = CONFIG.data.mean
CIFAR10_STD = CONFIG.data.std

WARMUP_EPOCHS = CONFIG.warmup.epochs
WARMUP_FREEZE_EPOCHS = CONFIG.warmup.freeze_epochs
WARMUP_BATCH_SIZE = CONFIG.warmup.batch_size
WARMUP_EVAL_BATCH_SIZE = CONFIG.warmup.eval_batch_size
WARMUP_LR = CONFIG.warmup.learning_rate
WARMUP_WEIGHT_DECAY = CONFIG.warmup.weight_decay
WARMUP_LABEL_SMOOTHING = CONFIG.warmup.label_smoothing
WARMUP_GRAD_CLIP_NORM = CONFIG.warmup.grad_clip_norm
WARMUP_MIN_NOISY_VALIDATION_ACCURACY = (
    CONFIG.warmup.min_noisy_validation_accuracy
)

RL_EPOCHS = CONFIG.rl.epochs
TRAJECTORY_LENGTH = CONFIG.rl.trajectory_length
INITIAL_STATE_RANDOMIZATION_RATE = (
    CONFIG.rl.initial_state_randomization_rate
)
FEATURE_BATCH_SIZE = CONFIG.rl.feature_batch_size
POLICY_UPDATE_BATCH_SIZE = CONFIG.rl.update_batch_size

K = CONFIG.knn.k
TEMPERATURE = CONFIG.knn.temperature
KNN_QUERY_CHUNK_SIZE = CONFIG.knn.query_chunk_size
KNN_REFERENCE_CHUNK_SIZE = CONFIG.knn.reference_chunk_size
CORRECTION_CHUNK_SIZE = CONFIG.knn.correction_chunk_size

ACTOR_LR = CONFIG.rl.actor_learning_rate
ACTOR_MOMENTUM = CONFIG.rl.actor_momentum
ACTOR_WEIGHT_DECAY = CONFIG.rl.actor_weight_decay
CRITIC_LR = CONFIG.rl.critic_learning_rate
CRITIC_MOMENTUM = CONFIG.rl.critic_momentum
CRITIC_WEIGHT_DECAY = CONFIG.rl.critic_weight_decay
CRITIC_NUM_BINS = CONFIG.rl.critic_num_bins
DISCOUNT_FACTOR = CONFIG.rl.discount_factor
NLA_WEIGHT = CONFIG.rl.reward_nla_weight
LR_DECAY_FACTOR = CONFIG.rl.lr_decay_factor
LR_DECAY_FRACTION = CONFIG.rl.lr_decay_fraction

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


def load_full_cifar10_test() -> tuple[Tensor, Tensor]:
    """Load the official 10,000-image clean test split for final reporting."""
    dataset = CIFAR10(
        root=CIFAR10_ROOT,
        train=False,
        download=DOWNLOAD_CIFAR10,
    )
    images = _as_image_tensor(dataset)
    labels = benchmark.pin_for_cuda(
        torch.tensor(dataset.targets, dtype=torch.long).contiguous()
    )
    if images.size(0) != 10_000 or labels.size(0) != 10_000:
        raise RuntimeError("Unexpected CIFAR-10 official test size.")
    return images, labels


def build_model(
    pretrained: bool = PRETRAINED,
    num_classes: int = NUM_CLASSES,
) -> nn.Module:
    return build_cifar_resnet18(pretrained, num_classes)


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
    if images.shape[-2:] != (IMAGE_SIZE, IMAGE_SIZE):
        raise ValueError(
            f"Expected native {IMAGE_SIZE}x{IMAGE_SIZE} CIFAR images, got "
            f"{tuple(images.shape[-2:])}."
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
    benchmark.DATASET_NAME = "CIFAR-10"
    benchmark.DATASET_STAGE_PREFIX = "cifar10"
    benchmark.DIGITS = CLASSES
    benchmark.NUM_CLASSES = NUM_CLASSES
    benchmark.EXPECTED_SAMPLES = EXPECTED_SAMPLES
    benchmark.NOISE_RATE = NOISE_RATE
    benchmark.SEED = SEED
    benchmark.ACTOR_UPDATE_MODE = ACTOR_UPDATE_MODE
    benchmark.POLICY_UPDATE_SUBSET_SIZE = EXPECTED_SAMPLES
    benchmark.POLICY_UPDATE_SAMPLES = EXPECTED_SAMPLES
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
    benchmark.WARMUP_MODEL_ID = CONFIG.warmup.model_id
    benchmark.MODEL_FACTORY = build_cifar_resnet18
    benchmark.PRETRAINED = PRETRAINED
    benchmark.IMAGE_SIZE = IMAGE_SIZE
    benchmark.DROP_RATE = CONFIG.model.drop_rate
    benchmark.DROP_PATH_RATE = CONFIG.model.drop_path_rate
    benchmark.IMAGENET_MEAN = CIFAR10_MEAN
    benchmark.IMAGENET_STD = CIFAR10_STD
    benchmark.WARMUP_EPOCHS = WARMUP_EPOCHS
    benchmark.WARMUP_FREEZE_EPOCHS = WARMUP_FREEZE_EPOCHS
    benchmark.WARMUP_BATCH_SIZE = WARMUP_BATCH_SIZE
    benchmark.WARMUP_EVAL_BATCH_SIZE = WARMUP_EVAL_BATCH_SIZE
    benchmark.WARMUP_HEAD_LR = WARMUP_LR
    benchmark.WARMUP_BACKBONE_LR = WARMUP_LR
    benchmark.WARMUP_UNFROZEN_HEAD_LR = WARMUP_LR
    benchmark.WARMUP_WEIGHT_DECAY = WARMUP_WEIGHT_DECAY
    benchmark.WARMUP_LABEL_SMOOTHING = WARMUP_LABEL_SMOOTHING
    benchmark.WARMUP_GRAD_CLIP_NORM = WARMUP_GRAD_CLIP_NORM
    benchmark.WARMUP_OPTIMIZER_NAME = CONFIG.warmup.optimizer
    benchmark.WARMUP_MOMENTUM = CONFIG.warmup.momentum
    benchmark.WARMUP_SCHEDULER_NAME = CONFIG.warmup.scheduler
    benchmark.WARMUP_LR_DECAY_FRACTION = CONFIG.warmup.lr_decay_fraction
    benchmark.WARMUP_LR_DECAY_FACTOR = CONFIG.warmup.lr_decay_factor
    benchmark.WARMUP_DEPLOY_CHECKPOINT = (
        CONFIG.warmup.deployment_checkpoint
    )
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
    benchmark.ACTOR_OPTIMIZER_NAME = CONFIG.rl.actor_optimizer
    benchmark.ACTOR_LR = ACTOR_LR
    benchmark.ACTOR_MOMENTUM = ACTOR_MOMENTUM
    benchmark.ACTOR_WEIGHT_DECAY = ACTOR_WEIGHT_DECAY
    benchmark.CRITIC_LR = CRITIC_LR
    benchmark.CRITIC_MOMENTUM = CRITIC_MOMENTUM
    benchmark.CRITIC_WEIGHT_DECAY = CRITIC_WEIGHT_DECAY
    benchmark.CRITIC_NUM_BINS = CRITIC_NUM_BINS
    benchmark.DISCOUNT_FACTOR = DISCOUNT_FACTOR
    benchmark.NLA_WEIGHT = NLA_WEIGHT
    benchmark.LR_DECAY_FACTOR = LR_DECAY_FACTOR
    benchmark.LR_DECAY_FRACTION = LR_DECAY_FRACTION
    benchmark.RL_DEPLOY_CHECKPOINT = CONFIG.rl.deployment_checkpoint
    benchmark.USE_AMP = CONFIG.runtime.use_amp
    benchmark.AMP_DTYPE = getattr(torch, CONFIG.runtime.amp_dtype)
    benchmark.USE_CHANNELS_LAST = CONFIG.runtime.use_channels_last
    benchmark.CUDNN_BENCHMARK = CONFIG.runtime.cudnn_benchmark

    # main() resolves these names in the imported engine module at runtime.
    benchmark.load_full_mnist_train = load_full_cifar10_train
    benchmark.load_mnist_validation_test = load_cifar10_validation_test
    benchmark.preprocess = preprocess_cifar10


if __name__ == "__main__":
    configure_benchmark()
    validate_input_artifacts()
    validate_output_destination()
    benchmark.run_with_file_logging()
