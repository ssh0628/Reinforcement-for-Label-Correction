"""CIFAR-10 RLNLC training orchestration.

Dataset loading, preprocessing, model construction, and experiment settings
are supplied by ``setting.data``. Warm-up, final correction, metrics,
and logging live in their stage-specific modules.
"""

from __future__ import annotations

import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.optim import SGD
from torch.optim.lr_scheduler import MultiStepLR

from evaluate.metrics import correction_summary
from log.common import (
    TIMING_FIELDS,
    Timings,
    append_csv,
    build_timing_rows,
    measure,
    print_timing_summary,
    run_with_log,
    save_torch,
    write_csv,
)
from rl.actor import correct_from_embeddings, update_actor
from rl.critic import StateActionCritic, build_critic_optimizer, encode_state_action, update_critic
from log.rl import (
    RUN_LOG_FILENAME,
    RUN_SUMMARY_CSV_FILENAME,
    RUN_SUMMARY_FIELDS,
    SUMMARY_FIELDS,
    TIMING_CSV_FILENAME,
    TRAIN_CSV_FILENAME,
)
from rl.knn import build_exact_policy_knn
from rl.policy import LabelCorrectionPolicy
from rl.reward import RLNLCReward
from setting.config import CONFIG
from setting.data import (
    CIFAR10_MEAN,
    CIFAR10_STD,
    EXPECTED_SAMPLES,
    MODEL_NAME,
    NOISE_MASK_PATH,
    NOISY_LABELS_PATH,
    NUM_CLASSES,
    PRETRAINED,
    SEED,
    SUBSET_SEED,
    WARMUP_CHECKPOINT_PATH,
    build_model,
    inject_configured_noise,
    load_cifar10_evaluation_split,
    load_selected_cifar10_train,
    move_model_to_device,
    pin_for_cuda,
    preprocess_cifar10 as preprocess,
)


NOISE_RATE = CONFIG.data.noise_rate
NOISE_TYPE = CONFIG.data.noise_type
USE_REMAINING_HORIZON = CONFIG.rl.use_remaining_horizon
USE_TERMINAL_CRITIC_UPDATE = CONFIG.rl.use_terminal_critic_update
ACTOR_MICROBATCH_SIZE = CONFIG.rl.actor_microbatch_size
OUTPUT_DIR = CONFIG.rl_output_dir
TRAIN_AUGMENTATION_ENABLED = CONFIG.augmentation.enabled
RL_EPOCHS = CONFIG.rl.epochs
TRAJECTORY_LENGTH = CONFIG.rl.trajectory_length
FEATURE_BATCH_SIZE = CONFIG.rl.feature_batch_size
K = CONFIG.knn.k
TEMPERATURE = CONFIG.knn.temperature
KNN_QUERY_CHUNK_SIZE = CONFIG.knn.query_chunk_size
KNN_REFERENCE_CHUNK_SIZE = CONFIG.knn.reference_chunk_size
CORRECTION_CHUNK_SIZE = CONFIG.knn.correction_chunk_size
ACTOR_LR = CONFIG.rl.actor_learning_rate
CRITIC_OPTIMIZER = CONFIG.rl.critic_optimizer
CRITIC_LR, CRITIC_MOMENTUM, CRITIC_WEIGHT_DECAY, CRITIC_LR_DECAY = (
    CONFIG.rl.effective_critic_options
)
CRITIC_NUM_BINS = CONFIG.rl.critic_num_bins
CRITIC_HIDDEN_DIMS = CONFIG.rl.critic_hidden_dims
DISCOUNT_FACTOR = CONFIG.rl.discount_factor
NLA_WEIGHT = CONFIG.rl.reward_nla_weight
LR_DECAY_FACTOR = CONFIG.rl.lr_decay_factor
LR_DECAY_EPOCH = CONFIG.rl.lr_decay_epoch
USE_AMP = CONFIG.runtime.use_amp
AMP_DTYPE = getattr(torch, CONFIG.runtime.amp_dtype)
USE_CHANNELS_LAST = CONFIG.runtime.use_channels_last
CUDNN_BENCHMARK = CONFIG.runtime.cudnn_benchmark


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
            "The CIFAR-10 experiment requires CUDA, but "
            "torch.cuda.is_available() "
            "is False. Check the NVIDIA driver and CUDA-enabled PyTorch build."
        )
    return torch.device("cuda", 0)


def synchronize(device: torch.device) -> None:
    torch.cuda.synchronize(device)


def load_noisy_label_artifacts(clean_labels: Tensor) -> tuple[Tensor, Tensor]:
    noisy_path = NOISY_LABELS_PATH
    mask_path = NOISE_MASK_PATH
    if not noisy_path.is_file():
        raise FileNotFoundError(f"Noisy-label artifact not found: {noisy_path}")
    if not mask_path.is_file():
        raise FileNotFoundError(f"Noise-mask artifact not found: {mask_path}")

    noisy_array = np.load(noisy_path, allow_pickle=False)
    mask_array = np.load(mask_path, allow_pickle=False)
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
    if NOISE_TYPE == "symmetric" and int(noise_mask.sum()) != expected_noise_count:
        raise ValueError(
            f"Saved noise count does not match NOISE_RATE: {int(noise_mask.sum())} != {expected_noise_count}."
        )
    return (pin_for_cuda(noisy_labels.contiguous()), pin_for_cuda(noise_mask.contiguous()))


def normalization_tensors(device: torch.device) -> tuple[Tensor, Tensor]:
    shape = (1, 3, 1, 1)
    mean = torch.tensor(CIFAR10_MEAN, device=device).reshape(shape)
    std = torch.tensor(CIFAR10_STD, device=device).reshape(shape)
    return mean, std


def build_grad_scaler() -> torch.amp.GradScaler:
    return torch.amp.GradScaler("cuda", enabled=USE_AMP and AMP_DTYPE == torch.float16)


def encode(model: nn.Module, images: Tensor) -> Tensor:
    feature_map = model.forward_features(images)
    return model.forward_head(feature_map, pre_logits=True)


def training_data_metadata() -> dict[str, object]:
    metadata: dict[str, object] = {
        "sample_count": EXPECTED_SAMPLES,
        "subset_seed": SUBSET_SEED,
        "selection": "deterministic_stratified_equal_per_class",
        "noise_type": NOISE_TYPE,
    }
    if NOISE_TYPE == "idn":
        metadata["idn_flip_rate_std"] = CONFIG.data.idn_flip_rate_std
    return metadata


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
        "noise_type": str(metadata.get("noise_type", "symmetric")),
    }
    if NOISE_TYPE == "idn":
        actual["idn_flip_rate_std"] = float(metadata.get("idn_flip_rate_std", -1.0))
    if actual != expected:
        raise ValueError(
            f"Checkpoint training data does not match the current config: {actual!r} != {expected!r}."
        )


def training_augmentation_metadata() -> dict[str, object]:
    return {
        "enabled": TRAIN_AUGMENTATION_ENABLED,
        "random_crop_padding": CONFIG.augmentation.random_crop_padding,
        "horizontal_flip_probability": CONFIG.augmentation.horizontal_flip_probability,
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
    if CONFIG.warmup.model_id and checkpoint_model_id != CONFIG.warmup.model_id:
        raise ValueError(
            "Warmup model ID does not match this run: "
            f"{checkpoint_model_id!r} != {CONFIG.warmup.model_id!r}."
        )
    validate_training_augmentation_checkpoint(checkpoint)
    validate_training_data_checkpoint(checkpoint)
    deployment_mode = str(checkpoint.get("selection", "best"))
    if deployment_mode != "best":
        raise ValueError("CIFAR warmup checkpoint must use best selection.")
    model.load_state_dict(checkpoint["model"], strict=True)
    move_model_to_device(model, device)
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
        "actor_update_samples": EXPECTED_SAMPLES,
        "actor_microbatch_size": ACTOR_MICROBATCH_SIZE,
        "actor_microbatches_per_rl_step": math.ceil(
            EXPECTED_SAMPLES / ACTOR_MICROBATCH_SIZE
        ),
        "actor_optimizer_steps_per_rl_step": 1,
        "remaining_horizon": USE_REMAINING_HORIZON,
        "terminal_update": USE_TERMINAL_CRITIC_UPDATE,
        "model": model.state_dict(),
    }
    critic_payload = {
        "epoch": epoch,
        "num_bins": CRITIC_NUM_BINS,
        "hidden_dims": CRITIC_HIDDEN_DIMS,
        "optimizer": CRITIC_OPTIMIZER,
        "learning_rate": CRITIC_LR,
        "momentum": CRITIC_MOMENTUM,
        "weight_decay": CRITIC_WEIGHT_DECAY,
        "lr_decay": CRITIC_LR_DECAY,
        "validation": validation_metrics,
        "remaining_horizon": USE_REMAINING_HORIZON,
        "terminal_update": USE_TERMINAL_CRITIC_UPDATE,
        "critic": critic.state_dict(),
    }
    save_torch(actor_path, actor_payload)
    save_torch(critic_path, critic_payload)


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
    move_model_to_device(model, device)
    model.eval()
    return checkpoint


def warm_device_kernels(
    model: nn.Module,
    raw_images: Tensor,
    device: torch.device,
    mean: Tensor,
    std: Tensor,
    *,
    update_batch_size: int,
) -> None:
    model.eval()
    model.zero_grad(set_to_none=True)
    inference_images = preprocess(raw_images[:FEATURE_BATCH_SIZE], device, mean, std)
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=AMP_DTYPE, enabled=USE_AMP):
        encode(model, inference_images)

    update_image_count = min(raw_images.size(0), update_batch_size)
    update_images = preprocess(raw_images[:update_image_count], device, mean, std)
    with torch.autocast(device_type="cuda", dtype=AMP_DTYPE, enabled=USE_AMP):
        encode(model, update_images).mean().backward()
    model.zero_grad(set_to_none=True)


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


def print_configuration(device: torch.device, sample_count: int, actual_noise_rate: float) -> None:
    properties = torch.cuda.get_device_properties(device)
    print(
        f"device={properties.name} memory={properties.total_memory / 1024**3:.1f}GiB "
        f"samples={sample_count} noise={NOISE_TYPE}:{NOISE_RATE:.0%} "
        f"actual_noise={actual_noise_rate:.2%} seed={SEED}"
    )
    print(
        f"model={MODEL_NAME} warmup={CONFIG.warmup.epochs}epochs "
        f"augmentation={TRAIN_AUGMENTATION_ENABLED} pretrained={PRETRAINED}"
    )
    print(
        f"rl={RL_EPOCHS}x{TRAJECTORY_LENGTH} actor_update={sample_count} "
        f"microbatch={ACTOR_MICROBATCH_SIZE} "
        f"microbatches={math.ceil(sample_count / ACTOR_MICROBATCH_SIZE)} "
        f"optimizer_steps_per_rl_step=1 actor_lr={ACTOR_LR} "
        f"lr_decay={LR_DECAY_EPOCH}:{LR_DECAY_FACTOR}"
    )
    critic_input = CRITIC_NUM_BINS + int(USE_REMAINING_HORIZON)
    print(
        f"critic=mlp:{critic_input}->{ '->'.join(map(str, CRITIC_HIDDEN_DIMS)) }->1 "
        f"optimizer={CRITIC_OPTIMIZER} lr={CRITIC_LR} momentum={CRITIC_MOMENTUM} "
        f"weight_decay={CRITIC_WEIGHT_DECAY} lr_decay={CRITIC_LR_DECAY} "
        f"horizon={USE_REMAINING_HORIZON} terminal={USE_TERMINAL_CRITIC_UPDATE}"
    )
    print(
        f"knn=k{K} feature_batch={FEATURE_BATCH_SIZE} amp={USE_AMP}:{AMP_DTYPE}"
    )


def main() -> None:
    run_started = time.perf_counter()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG.rl_model_dir.mkdir(parents=True, exist_ok=True)
    actor_best_checkpoint_path = CONFIG.actor_best_checkpoint_path
    actor_last_checkpoint_path = CONFIG.actor_last_checkpoint_path
    critic_best_checkpoint_path = CONFIG.critic_best_checkpoint_path
    critic_last_checkpoint_path = CONFIG.critic_last_checkpoint_path
    train_csv_path = OUTPUT_DIR / TRAIN_CSV_FILENAME
    timing_csv_path = OUTPUT_DIR / TIMING_CSV_FILENAME
    run_summary_path = OUTPUT_DIR / RUN_SUMMARY_CSV_FILENAME
    write_csv(train_csv_path, [], SUMMARY_FIELDS)

    device = resolve_local_device()
    seed_everything(SEED)
    torch.backends.cudnn.benchmark = CONFIG.runtime.cudnn_benchmark
    torch.cuda.reset_peak_memory_stats()
    timings: Timings = {}
    print(f"output_dir={OUTPUT_DIR}")

    raw_images, clean_labels_cpu = measure(
        "data_load", device, timings, load_selected_cifar10_train
    )
    val_images, val_clean_labels = measure(
        "val_load", device, timings, lambda: load_cifar10_evaluation_split("val")
    )
    noisy_labels_cpu, noise_mask_cpu = measure(
        "noise_load", device, timings, lambda: load_noisy_label_artifacts(clean_labels_cpu)
    )
    val_noisy_labels, val_noise_mask = measure(
        "val_noise",
        device,
        timings,
        lambda: inject_configured_noise(val_images, val_clean_labels, seed=SEED),
    )
    actual_noise_rate = float(noise_mask_cpu.float().mean())
    print_configuration(device, clean_labels_cpu.numel(), actual_noise_rate)
    print(f"validation_samples={val_images.size(0)}")
    model = measure(
        "model_init",
        device,
        timings,
        lambda: build_model(device=device),
    )
    mean, std = normalization_tensors(device)
    measure(
        "gpu_warmup",
        device,
        timings,
        lambda: warm_device_kernels(
            model, raw_images, device, mean, std, update_batch_size=ACTOR_MICROBATCH_SIZE
        ),
    )

    warmup_result = measure(
        "warmup_load",
        device,
        timings,
        lambda: load_warmup_checkpoint(model, WARMUP_CHECKPOINT_PATH, device),
    )
    policy = LabelCorrectionPolicy(TEMPERATURE, CORRECTION_CHUNK_SIZE).to(device)
    reward_function = RLNLCReward(
        nla_weight=NLA_WEIGHT,
        temperature=TEMPERATURE,
        k=K,
        query_chunk_size=KNN_QUERY_CHUNK_SIZE,
        reference_chunk_size=KNN_REFERENCE_CHUNK_SIZE,
        state_chunk_size=CORRECTION_CHUNK_SIZE,
    ).to(device)
    critic = StateActionCritic(
        CRITIC_NUM_BINS,
        CRITIC_HIDDEN_DIMS,
        use_remaining_horizon=USE_REMAINING_HORIZON,
    ).to(device)
    critic_optimizer = build_critic_optimizer(
        critic,
        name=CRITIC_OPTIMIZER,
        learning_rate=CRITIC_LR,
        momentum=CRITIC_MOMENTUM,
        weight_decay=CRITIC_WEIGHT_DECAY,
    )
    actor_optimizer = SGD(
        model.parameters(),
        lr=ACTOR_LR,
        momentum=CONFIG.rl.actor_momentum,
        weight_decay=CONFIG.rl.actor_weight_decay,
    )
    actor_scheduler = MultiStepLR(
        actor_optimizer, milestones=[LR_DECAY_EPOCH], gamma=LR_DECAY_FACTOR
    )
    critic_scheduler = (
        MultiStepLR(critic_optimizer, milestones=[LR_DECAY_EPOCH], gamma=LR_DECAY_FACTOR)
        if CRITIC_LR_DECAY
        else None
    )
    scaler = build_grad_scaler()
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
    best_rl_epoch = 0
    best_rl_key: tuple[float, float] | None = None
    best_validation_summary: dict[str, object] | None = None
    epoch_zero_summary = evaluate_correction_split(
        split="val",
        epoch=0,
        model=model,
        policy=policy,
        raw_images=val_images,
        clean_labels_cpu=val_clean_labels,
        noisy_labels_cpu=val_noisy_labels,
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
                randomization_rate=CONFIG.rl.initial_state_randomization_rate,
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
            correction = measure(
                "label_correction",
                device,
                timings,
                lambda: correct_from_embeddings(policy, policy_embeddings, label_state, policy_neighbors),
                step=global_step,
            )
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
                lambda: encode_state_action(
                    critic,
                    reward_output.per_sample_consistency,
                    (TRAJECTORY_LENGTH - step) / TRAJECTORY_LENGTH,
                ),
                step=global_step,
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
                    policy_embeddings,
                    policy_neighbors,
                    correction.actions,
                    q_value,
                    device,
                    mean,
                    std,
                    microbatch_size=ACTOR_MICROBATCH_SIZE,
                    use_amp=USE_AMP,
                    amp_dtype=AMP_DTYPE,
                    preprocess=preprocess,
                    encode=encode,
                ),
                step=global_step,
            )
            del policy_embeddings, policy_neighbors

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

            terminal_update = None
            if USE_TERMINAL_CRITIC_UPDATE and step == TRAJECTORY_LENGTH:
                terminal_update = measure(
                    "critic_update",
                    device,
                    timings,
                    lambda: update_critic(
                        critic,
                        critic_optimizer,
                        encoding,
                        reward_output.total_reward.detach(),
                        None,
                        discount_factor=DISCOUNT_FACTOR,
                        terminal=True,
                    ),
                    step=global_step,
                )
                epoch_critic_losses.append(terminal_update.loss)

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
            terminal_log = "" if terminal_update is None else (
                f" terminal_q={terminal_update.current_q:.6f} "
                f"terminal_target={terminal_update.target:.6f} "
                f"terminal_error={terminal_update.error:.6f}"
            )
            print(
                f"[RL] e={epoch}/{RL_EPOCHS} s={step}/{TRAJECTORY_LENGTH} "
                f"acc={clean_accuracy:.4f} reward={reward_value:.8e} "
                f"lcr={label_consistency_value:.6f} nla={noisy_alignment_value:.6f} "
                f"log_reward={log_reward_value:.6f} reward_zero={reward_value == 0.0} "
                f"q={q_value_value:.6f} {td_log}{terminal_log} actor={actor_loss:.4f} "
                f"action={step_action_count / EXPECTED_SAMPLES:.4f} "
            )

            if step < TRAJECTORY_LENGTH:
                previous_encoding = encoding
                previous_reward = reward_output.total_reward.detach()
            del correction, reward_output

        synchronize(device)
        train_elapsed = time.perf_counter() - epoch_started
        epoch_seconds.append(train_elapsed)
        actor_scheduler.step()
        if critic_scheduler is not None:
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
        val_summary = evaluate_correction_split(
            split="val",
            epoch=epoch,
            model=model,
            policy=policy,
            raw_images=val_images,
            clean_labels_cpu=val_clean_labels,
            noisy_labels_cpu=val_noisy_labels,
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
        print(
            f"[EPOCH] epoch={epoch} "
            f"train_acc={train_summary['accuracy']:.6f} "
            f"val_acc={val_summary['accuracy']:.6f} "
            f"val_loss={val_summary['loss']:.6f} "
            f"train_seconds={train_elapsed:.3f}"
        )

    if best_validation_summary is None:
        raise RuntimeError("RL training finished without a best checkpoint.")

    print_timing_summary(timings)

    peak_allocated = torch.cuda.max_memory_allocated() / 1024**3
    total_runtime = time.perf_counter() - run_started
    write_csv(timing_csv_path, build_timing_rows(timings), TIMING_FIELDS)
    write_csv(
        run_summary_path,
        [
            {
                "dataset": "CIFAR-10",
                "model": MODEL_NAME,
                "samples": EXPECTED_SAMPLES,
                "noise_type": NOISE_TYPE,
                "noise_rate": NOISE_RATE,
                "actual_noise_rate": actual_noise_rate,
                "idn_flip_rate_std": (
                    CONFIG.data.idn_flip_rate_std if NOISE_TYPE == "idn" else ""
                ),
                "seed": SEED,
                "actor_update_samples": EXPECTED_SAMPLES,
                "actor_microbatch_size": ACTOR_MICROBATCH_SIZE,
                "actor_microbatches_per_rl_step": math.ceil(
                    EXPECTED_SAMPLES / ACTOR_MICROBATCH_SIZE
                ),
                "actor_optimizer_steps_per_rl_step": 1,
                "remaining_horizon": USE_REMAINING_HORIZON,
                "terminal_update": USE_TERMINAL_CRITIC_UPDATE,
                "warmup_epoch": warmup_result["best_epoch"],
                "epochs": RL_EPOCHS,
                "steps": TRAJECTORY_LENGTH,
                "k": K,
                "actor_lr": ACTOR_LR,
                "critic_optimizer": CRITIC_OPTIMIZER,
                "critic_lr": CRITIC_LR,
                "critic_momentum": CRITIC_MOMENTUM,
                "critic_weight_decay": CRITIC_WEIGHT_DECAY,
                "critic_lr_decay": CRITIC_LR_DECAY,
                "lr_decay_epoch": LR_DECAY_EPOCH,
                "lr_decay_factor": LR_DECAY_FACTOR,
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
    print("next=cifar_correction.py")


def run_with_file_logging() -> None:
    run_with_log(OUTPUT_DIR / RUN_LOG_FILENAME, main)
