"""CIFAR-10 RLNLC training orchestration.

Dataset loading, preprocessing, model construction, and experiment settings
are supplied by ``cifar_test.setting.data``. Warm-up, final correction, metrics,
logging, and optional change diagnostics live in their stage-specific modules.
"""

from __future__ import annotations

import math
import random
import time
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.optim import SGD
from torch.optim.lr_scheduler import MultiStepLR

from cifar_test.rl.change import (
    CHANGE_DIAGNOSTIC_FIELDS,
    ChangeDiagnosticsRecorder,
    serialize_change_row,
)
from cifar_test.evaluate.metrics import correction_summary
from cifar_test.log.common import (
    TIMING_FIELDS,
    Timings,
    append_csv,
    build_timing_rows,
    measure,
    print_timing_summary,
    run_with_log,
    write_csv,
)
from cifar_test.rl.actor import correct_from_embeddings, select_queries, update_actor
from cifar_test.rl.critic import build_critic, build_critic_optimizer, encode_state_action, update_critic
from cifar_test.rl.diagnostics import (
    build_reference_states,
    evaluate_reference_states,
    print_reference_ranking,
    print_reward_diagnostics,
    reward_diagnostic_row,
)
from cifar_test.log.rl import (
    CHANGE_DIAGNOSTICS_CSV_FILENAME,
    RUN_LOG_FILENAME,
    REWARD_DIAGNOSTICS_CSV_FILENAME,
    REWARD_DIAGNOSTIC_FIELDS,
    RUN_SUMMARY_CSV_FILENAME,
    RUN_SUMMARY_FIELDS,
    SUMMARY_FIELDS,
    TIMING_CSV_FILENAME,
    TRAIN_CSV_FILENAME,
)
from cifar_test.rl.knn import build_exact_policy_knn
from cifar_test.rl.policy import LabelCorrectionPolicy
from cifar_test.rl.reward import RLNLCReward


CONFIGURED = False
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
RECORD_REWARD_DIAGNOSTICS: bool
OUTPUT_DIR: Path
MODEL_OUTPUT_DIR: Path
MODEL_NAME: str
WARMUP_MODEL_ID: str
PRETRAINED: bool
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
CRITIC_HIDDEN_DIMS: tuple[int, ...]
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
        "hidden_dims": CRITIC_HIDDEN_DIMS,
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
        raw_images.size(0), POLICY_UPDATE_BATCH_SIZE if direct_actor else WARMUP_BATCH_SIZE
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
        num_classes=NUM_CLASSES,
        epoch=epoch,
        split=split,
        action_rate=action_rate,
        seconds=elapsed,
    )
    del embeddings, neighbors, label_state
    return summary


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
    print(f"critic=mlp:{CRITIC_NUM_BINS}->{ '->'.join(map(str, CRITIC_HIDDEN_DIMS)) }->1")
    print(
        f"knn=k{K} feature_batch={FEATURE_BATCH_SIZE} amp={USE_AMP}:{AMP_DTYPE} "
        f"change_log={RECORD_CHANGE_DIAGNOSTICS} reward_log={RECORD_REWARD_DIAGNOSTICS}"
    )


def main() -> None:
    if not CONFIGURED:
        raise RuntimeError("Configure the RL engine through cifar_test.setting.data.configure_engine() first.")
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
    reward_diagnostics_path = OUTPUT_DIR / REWARD_DIAGNOSTICS_CSV_FILENAME
    write_csv(train_csv_path, [], SUMMARY_FIELDS)
    if RECORD_CHANGE_DIAGNOSTICS:
        write_csv(change_diagnostics_path, [], CHANGE_DIAGNOSTIC_FIELDS)
    if RECORD_REWARD_DIAGNOSTICS:
        write_csv(reward_diagnostics_path, [], REWARD_DIAGNOSTIC_FIELDS)

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

    policy = LabelCorrectionPolicy(TEMPERATURE, CORRECTION_CHUNK_SIZE).to(device)
    reward_function = RLNLCReward(
        nla_weight=NLA_WEIGHT,
        temperature=TEMPERATURE,
        k=K,
        query_chunk_size=KNN_QUERY_CHUNK_SIZE,
        reference_chunk_size=KNN_REFERENCE_CHUNK_SIZE,
        state_chunk_size=CORRECTION_CHUNK_SIZE,
    ).to(device)
    critic = build_critic(CRITIC_NUM_BINS, CRITIC_HIDDEN_DIMS).to(device)
    critic_optimizer = build_critic_optimizer(
        critic,
        learning_rate=CRITIC_LR,
        momentum=CRITIC_MOMENTUM,
        weight_decay=CRITIC_WEIGHT_DECAY,
    )
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
    if RECORD_REWARD_DIAGNOSTICS:
        reference_states = build_reference_states(
            initial_noisy_labels, clean_labels, noise_mask, global_neighbors, global_cosines,
            num_classes=NUM_CLASSES, temperature=TEMPERATURE,
        )
        reference_rows = measure(
            "reward_check",
            device,
            timings,
            lambda: evaluate_reference_states(
                reward_function, reference_states, fixed_embeddings, global_neighbors, global_cosines,
                clean_labels, initial_noisy_labels, noise_mask,
                num_classes=NUM_CLASSES, nla_weight=NLA_WEIGHT,
            ),
        )
        append_csv(reward_diagnostics_path, reference_rows, REWARD_DIAGNOSTIC_FIELDS)
        print_reward_diagnostics(reference_rows)
        print_reference_ranking(reference_rows)
        del reference_states
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
        learned_reward_row: dict[str, object] | None = None
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
                lambda: correct_from_embeddings(policy, policy_embeddings, label_state, policy_neighbors),
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

            encoding, q_value = measure(
                "critic_encode",
                device,
                timings,
                lambda: encode_state_action(critic, reward_output.per_sample_consistency),
                step=global_step,
            )
            query_indices_cpu = select_queries(
                EXPECTED_SAMPLES,
                POLICY_UPDATE_SAMPLES,
                SEED,
                global_step,
            )
            actor_loss = measure(
                "actor_update",
                device,
                timings,
                lambda: update_actor(
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
                    batch_size=POLICY_UPDATE_BATCH_SIZE,
                    use_amp=USE_AMP,
                    amp_dtype=AMP_DTYPE,
                    preprocess=preprocess,
                    encode=encode,
                    change_recorder=change_recorder,
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
                epoch_change_rows.append(serialize_change_row(change_row))

            critic_update = None
            if previous_encoding is not None and previous_reward is not None:
                critic_update = measure(
                    "critic_update",
                    device,
                    timings,
                    lambda: update_critic(
                        critic,
                        critic_optimizer,
                        previous_encoding,
                        previous_reward,
                        encoding,
                        discount_factor=DISCOUNT_FACTOR,
                    ),
                    step=global_step,
                )
                epoch_critic_losses.append(critic_update.loss)

            label_state = correction.corrected_labels
            (
                step_action_count_value,
                reward_value,
                clean_accuracy,
                label_consistency_value,
                noisy_alignment_value,
                q_value_value,
            ) = (
                torch.stack(
                    (
                        correction.actions.sum(),
                        reward_output.total_reward.reshape(()),
                        label_state.argmax(dim=1).eq(clean_labels).float().mean(),
                        reward_output.label_consistency.reshape(()),
                        reward_output.noisy_label_alignment.reshape(()),
                        q_value.reshape(()),
                    )
                )
                .to(torch.float64)
                .cpu()
                .tolist()
            )
            log_reward_value = label_consistency_value + NLA_WEIGHT * noisy_alignment_value
            step_action_count = int(step_action_count_value)
            epoch_action_count += step_action_count
            epoch_rewards.append(reward_value)
            epoch_actor_losses.append(actor_loss)
            td_log = "td=none" if critic_update is None else (
                f"prev_q={critic_update.current_q:.6f} next_q={critic_update.next_q:.6f} "
                f"td_target={critic_update.target:.6f} td_error={critic_update.error:.6f}"
            )
            print(
                f"[RL] e={epoch}/{RL_EPOCHS} s={step}/{TRAJECTORY_LENGTH} "
                f"acc={clean_accuracy:.4f} reward={reward_value:.8e} "
                f"lcr={label_consistency_value:.6f} nla={noisy_alignment_value:.6f} "
                f"log_reward={log_reward_value:.6f} reward_zero={reward_value == 0.0} "
                f"q={q_value_value:.6f} {td_log} actor={actor_loss:.4f} "
                f"action={step_action_count / EXPECTED_SAMPLES:.4f} "
            )

            if RECORD_REWARD_DIAGNOSTICS and step == TRAJECTORY_LENGTH:
                learned_reward_row = reward_diagnostic_row(
                    epoch=epoch, state_name="learned", labels=label_state,
                    clean_labels=clean_labels, noisy_labels=initial_noisy_labels,
                    noise_mask=noise_mask, reward_output=reward_output,
                    num_classes=NUM_CLASSES, nla_weight=NLA_WEIGHT,
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
        if learned_reward_row is not None:
            append_csv(reward_diagnostics_path, [learned_reward_row], REWARD_DIAGNOSTIC_FIELDS)
            print_reward_diagnostics([learned_reward_row])
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
            num_classes=NUM_CLASSES,
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
                "critic_hidden_dims": "x".join(map(str, CRITIC_HIDDEN_DIMS)),
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
    run_with_log(OUTPUT_DIR / RUN_LOG_FILENAME, main)
