"""Fine-tune an RLNLC CIFAR-10 model on saved corrected labels.

The last RL actor always supplies the corrected labels. The initial model is
selected in ``cifar_config.py`` as warmup, best_actor, or last_actor. Run
``cifar_evaluate.py`` separately to evaluate the saved checkpoint.
"""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn
from torch.optim import SGD
from torch.optim.lr_scheduler import MultiStepLR

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cifar_test import cifar_common as cifar


CONFIG = cifar.CONFIG
INITIALIZATION = CONFIG.finetune.initialization
INITIAL_CHECKPOINT_PATH = CONFIG.finetune_initial_checkpoint_path
CORRECTED_LABELS_PATH = CONFIG.corrected_labels_path
OUTPUT_DIR = CONFIG.finetune_output_dir

FINETUNE_EPOCHS = CONFIG.finetune.epochs
TRAIN_BATCH_SIZE = CONFIG.finetune.batch_size
LEARNING_RATE = CONFIG.finetune.learning_rate
MOMENTUM = CONFIG.finetune.momentum
WEIGHT_DECAY = CONFIG.finetune.weight_decay
LR_DECAY_EPOCH = math.ceil(
    FINETUNE_EPOCHS * CONFIG.finetune.lr_decay_fraction
)
LR_DECAY_FACTOR = CONFIG.finetune.lr_decay_factor
SEED = CONFIG.data.seed
OVERWRITE = CONFIG.runtime.overwrite_finetune

RUN_LOG_PATH = OUTPUT_DIR / "run.log"
TRAIN_CSV_PATH = OUTPUT_DIR / "train.csv"
FINAL_CHECKPOINT_PATH = CONFIG.finetune_checkpoint_path

TRAIN_FIELDS = (
    "epoch",
    "learning_rate",
    "loss",
    "corrected_label_accuracy",
    "clean_label_accuracy",
    "elapsed_seconds",
)


def _validate_output_destination(*, include_log: bool) -> None:
    paths = [
        TRAIN_CSV_PATH,
        FINAL_CHECKPOINT_PATH,
    ]
    if include_log:
        paths.append(RUN_LOG_PATH)
    cifar.require_available_outputs(
        paths,
        overwrite=OVERWRITE,
        stage="Fine-tuning",
    )


def _validate_input_artifacts() -> None:
    cifar.require_files(
        (INITIAL_CHECKPOINT_PATH, CORRECTED_LABELS_PATH),
        stage="Fine-tuning",
    )


def _load_corrected_labels(clean_labels: Tensor) -> Tensor:
    if not CORRECTED_LABELS_PATH.is_file():
        raise FileNotFoundError(
            f"Corrected-label artifact not found: {CORRECTED_LABELS_PATH}"
        )
    array = np.load(CORRECTED_LABELS_PATH, allow_pickle=False)
    array = np.asarray(array)
    if not np.issubdtype(array.dtype, np.integer):
        raise TypeError("Corrected labels must use an integer NumPy dtype.")
    labels = torch.from_numpy(array).to(torch.long).contiguous()
    if labels.shape != clean_labels.shape:
        raise ValueError(
            "Corrected labels do not match CIFAR-10 train size: "
            f"{tuple(labels.shape)} != {tuple(clean_labels.shape)}."
        )
    if labels.numel() and (
        int(labels.min()) < 0 or int(labels.max()) >= cifar.NUM_CLASSES
    ):
        raise ValueError("Corrected labels contain an out-of-range class ID.")
    return cifar.engine.pin_for_cuda(labels)


def _load_initial_model(device: torch.device) -> nn.Module:
    if not INITIAL_CHECKPOINT_PATH.is_file():
        raise FileNotFoundError(
            f"Initial checkpoint not found: {INITIAL_CHECKPOINT_PATH}"
        )
    checkpoint = torch.load(
        INITIAL_CHECKPOINT_PATH,
        map_location="cpu",
        weights_only=True,
    )
    if not isinstance(checkpoint, dict):
        raise TypeError("Initial checkpoint must contain a dictionary.")
    required = {"model", "epoch"}
    missing = required.difference(checkpoint)
    if missing:
        raise KeyError(
            f"Initial checkpoint is missing fields: {sorted(missing)}"
        )
    if (
        "model_name" in checkpoint
        and checkpoint["model_name"] != cifar.MODEL_NAME
    ):
        raise ValueError("Initial checkpoint model name does not match config.")
    if (
        "num_classes" in checkpoint
        and int(checkpoint["num_classes"]) != cifar.NUM_CLASSES
    ):
        raise ValueError("Initial checkpoint class count does not match CIFAR-10.")
    if INITIALIZATION == "warmup" and (
        checkpoint.get("warmup_model_id") != CONFIG.warmup.model_id
    ):
        raise ValueError("Warmup checkpoint model ID does not match config.")
    if INITIALIZATION == "last_actor" and int(checkpoint["epoch"]) != CONFIG.rl.epochs:
        raise ValueError(
            "last_actor initialization must use the configured final RL epoch."
        )

    model = cifar.build_model()
    try:
        model.load_state_dict(checkpoint["model"], strict=True)
    except RuntimeError as error:
        raise RuntimeError(
            "The initial checkpoint does not contain the classifier head."
        ) from error
    model.to(
        device=device,
        memory_format=(
            torch.channels_last
            if cifar.engine.USE_CHANNELS_LAST
            else torch.contiguous_format
        ),
    )
    return model


def _save_checkpoint(model: nn.Module) -> None:
    temporary_path = FINAL_CHECKPOINT_PATH.with_suffix(
        f"{FINAL_CHECKPOINT_PATH.suffix}.tmp"
    )
    payload = {
        "epoch": FINETUNE_EPOCHS,
        "model_name": cifar.MODEL_NAME,
        "num_classes": cifar.NUM_CLASSES,
        "initialization": INITIALIZATION,
        "source_checkpoint": str(INITIAL_CHECKPOINT_PATH),
        "corrected_labels": str(CORRECTED_LABELS_PATH),
        "optimizer": CONFIG.finetune.optimizer,
        "learning_rate": LEARNING_RATE,
        "momentum": MOMENTUM,
        "weight_decay": WEIGHT_DECAY,
        "model": model.state_dict(),
    }
    try:
        torch.save(payload, temporary_path)
        temporary_path.replace(FINAL_CHECKPOINT_PATH)
    finally:
        temporary_path.unlink(missing_ok=True)


def main() -> None:
    if FINETUNE_EPOCHS <= 0:
        raise ValueError("FINETUNE_EPOCHS must be positive.")
    _validate_input_artifacts()
    _validate_output_destination(include_log=False)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FINAL_CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)

    cifar.configure_engine()
    engine = cifar.engine
    device = engine.resolve_local_device()
    engine.seed_everything(SEED)
    torch.backends.cudnn.benchmark = engine.CUDNN_BENCHMARK
    torch.cuda.reset_peak_memory_stats()

    train_images, clean_train_labels = cifar.load_full_cifar10_train()
    corrected_labels = _load_corrected_labels(clean_train_labels)
    model = _load_initial_model(device)
    mean = torch.tensor(cifar.CIFAR10_MEAN, device=device).reshape(
        1, 3, 1, 1
    )
    std = torch.tensor(cifar.CIFAR10_STD, device=device).reshape(
        1, 3, 1, 1
    )
    optimizer = SGD(
        model.parameters(),
        lr=LEARNING_RATE,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = MultiStepLR(
        optimizer,
        milestones=[LR_DECAY_EPOCH],
        gamma=LR_DECAY_FACTOR,
    )
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=(
            engine.USE_AMP and engine.AMP_DTYPE == torch.float16
        ),
    )
    history: list[dict[str, object]] = []

    print(f"device={device} ({torch.cuda.get_device_name(device)})")
    print(f"initialization={INITIALIZATION}")
    print(f"initial_checkpoint={INITIAL_CHECKPOINT_PATH}")
    print(f"corrected_labels={CORRECTED_LABELS_PATH}")
    print(f"epochs={FINETUNE_EPOCHS} batch_size={TRAIN_BATCH_SIZE}")
    print(
        f"optimizer={CONFIG.finetune.optimizer.upper()} "
        f"lr={LEARNING_RATE} momentum={MOMENTUM} "
        f"weight_decay={WEIGHT_DECAY} lr_decay_epoch={LR_DECAY_EPOCH}"
    )
    print(
        "initial_corrected_label_accuracy="
        f"{float(corrected_labels.eq(clean_train_labels).float().mean()):.6f}"
    )

    for epoch in range(1, FINETUNE_EPOCHS + 1):
        engine.synchronize(device)
        started = time.perf_counter()
        model.train()
        learning_rate = float(optimizer.param_groups[0]["lr"])
        generator = torch.Generator().manual_seed(SEED + epoch)
        permutation = torch.randperm(
            corrected_labels.numel(),
            generator=generator,
        )
        loss_sum = torch.zeros((), dtype=torch.float64, device=device)
        corrected_count = torch.zeros((), dtype=torch.long, device=device)
        clean_count = torch.zeros((), dtype=torch.long, device=device)

        for start in range(0, permutation.numel(), TRAIN_BATCH_SIZE):
            end = min(start + TRAIN_BATCH_SIZE, permutation.numel())
            indices = permutation[start:end]
            images = cifar.preprocess_cifar10(
                train_images[indices],
                device,
                mean,
                std,
            )
            targets = corrected_labels[indices].to(device, non_blocking=True)
            clean_targets = clean_train_labels[indices].to(
                device,
                non_blocking=True,
            )
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type="cuda",
                dtype=engine.AMP_DTYPE,
                enabled=engine.USE_AMP,
            ):
                logits = model(images)
                loss = criterion(logits, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            batch_count = end - start
            predictions = logits.argmax(dim=1)
            loss_sum += loss.detach().to(torch.float64) * batch_count
            corrected_count += predictions.eq(targets).sum()
            clean_count += predictions.eq(clean_targets).sum()

        scheduler.step()
        engine.synchronize(device)
        elapsed = time.perf_counter() - started
        row: dict[str, object] = {
            "epoch": epoch,
            "learning_rate": learning_rate,
            "loss": float(loss_sum / corrected_labels.numel()),
            "corrected_label_accuracy": float(
                corrected_count / corrected_labels.numel()
            ),
            "clean_label_accuracy": float(
                clean_count / clean_train_labels.numel()
            ),
            "elapsed_seconds": elapsed,
        }
        history.append(row)
        engine.write_csv(TRAIN_CSV_PATH, history, TRAIN_FIELDS)
        print(
            f"[FINETUNE] epoch={epoch}/{FINETUNE_EPOCHS} "
            f"loss={float(row['loss']):.6f} "
            f"corrected_acc={float(row['corrected_label_accuracy']):.4f} "
            f"clean_acc={float(row['clean_label_accuracy']):.4f} "
            f"seconds={elapsed:.3f}"
        )

    _save_checkpoint(model)
    print(f"checkpoint={FINAL_CHECKPOINT_PATH}")
    print(f"train_csv={TRAIN_CSV_PATH}")
    print("next_stage=cifar_test/cifar_evaluate.py")


def run_with_file_logging() -> None:
    _validate_input_artifacts()
    _validate_output_destination(include_log=True)
    cifar.run_with_log(RUN_LOG_PATH, main)


if __name__ == "__main__":
    run_with_file_logging()
