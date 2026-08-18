"""Fine-tune an RLNLC CIFAR-10 model on saved corrected labels.

This stage only trains and saves ``finetune_last.pt``. Run
``cifar_final_test.py`` separately to evaluate the saved checkpoint.
"""

from __future__ import annotations

import csv
import sys
import time
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
from torch import Tensor, nn
from torch.optim import SGD
from torch.optim.lr_scheduler import MultiStepLR

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if TYPE_CHECKING:
    from cifar_test import cifar_rl as cifar
else:
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from cifar_test import cifar_rl as cifar


CONFIG = cifar.CONFIG
ACTOR_CHECKPOINT_PATH = CONFIG.actor_last_checkpoint_path
CORRECTED_LABELS_PATH = CONFIG.corrected_labels_path
OUTPUT_DIR = CONFIG.finetune_output_dir

FINETUNE_EPOCHS = CONFIG.finetune.epochs
TRAIN_BATCH_SIZE = CONFIG.finetune.batch_size
LEARNING_RATE = CONFIG.finetune.learning_rate
MOMENTUM = CONFIG.finetune.momentum
WEIGHT_DECAY = CONFIG.finetune.weight_decay
LR_DECAY_EPOCH = round(
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
    existing = [path for path in paths if path.exists()]
    if existing and not OVERWRITE:
        raise FileExistsError(
            f"Fine-tuning outputs already exist: {existing}. Set "
            "OVERWRITE=True only when replacement is intentional."
        )


def _validate_input_artifacts() -> None:
    missing = [
        path
        for path in (ACTOR_CHECKPOINT_PATH, CORRECTED_LABELS_PATH)
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(f"Fine-tuning inputs not found: {missing}")
    if (
        ACTOR_CHECKPOINT_PATH.parent.resolve()
        != CORRECTED_LABELS_PATH.parent.resolve()
    ):
        raise ValueError(
            "ACTOR_CHECKPOINT_PATH and CORRECTED_LABELS_PATH must come from "
            "the same RL output directory."
        )


def _write_csv(
    path: Path,
    rows: list[dict[str, object]],
    fieldnames: tuple[str, ...],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _load_corrected_labels(clean_labels: Tensor) -> Tensor:
    if not CORRECTED_LABELS_PATH.is_file():
        raise FileNotFoundError(
            f"Corrected-label artifact not found: {CORRECTED_LABELS_PATH}"
        )
    array = np.load(CORRECTED_LABELS_PATH, allow_pickle=False)
    labels = torch.from_numpy(np.asarray(array)).to(torch.long).contiguous()
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


def _load_actor(device: torch.device) -> nn.Module:
    if not ACTOR_CHECKPOINT_PATH.is_file():
        raise FileNotFoundError(
            f"Actor checkpoint not found: {ACTOR_CHECKPOINT_PATH}"
        )
    checkpoint = torch.load(
        ACTOR_CHECKPOINT_PATH,
        map_location="cpu",
        weights_only=True,
    )
    if not isinstance(checkpoint, dict):
        raise TypeError("Actor checkpoint must contain a dictionary.")
    required = {"model", "model_name", "num_classes", "epoch"}
    missing = required.difference(checkpoint)
    if missing:
        raise KeyError(f"Actor checkpoint is missing fields: {sorted(missing)}")
    if checkpoint["model_name"] != cifar.MODEL_NAME:
        raise ValueError("Actor checkpoint model name does not match CIFAR config.")
    if int(checkpoint["num_classes"]) != cifar.NUM_CLASSES:
        raise ValueError("Actor checkpoint class count does not match CIFAR-10.")

    model = cifar.build_model()
    try:
        model.load_state_dict(checkpoint["model"], strict=True)
    except RuntimeError as error:
        raise RuntimeError(
            "The actor checkpoint does not contain the classifier head."
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
        "source_actor_checkpoint": str(ACTOR_CHECKPOINT_PATH),
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

    cifar.configure_engine()
    engine = cifar.engine
    device = engine.resolve_local_device()
    engine.seed_everything(SEED)
    torch.backends.cudnn.benchmark = engine.CUDNN_BENCHMARK
    torch.cuda.reset_peak_memory_stats()

    train_images, clean_train_labels = cifar.load_full_cifar10_train()
    corrected_labels = _load_corrected_labels(clean_train_labels)
    model = _load_actor(device)
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
    print(f"actor_checkpoint={ACTOR_CHECKPOINT_PATH}")
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
        _write_csv(TRAIN_CSV_PATH, history, TRAIN_FIELDS)
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
    print("next_stage=cifar_test/cifar_final_test.py")


def run_with_file_logging() -> None:
    _validate_input_artifacts()
    _validate_output_destination(include_log=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with RUN_LOG_PATH.open("w", encoding="utf-8", buffering=1) as handle:
        stdout = cifar.engine.TeeStream(sys.stdout, handle)
        stderr = cifar.engine.TeeStream(sys.stderr, handle)
        with redirect_stdout(stdout), redirect_stderr(stderr):
            print(f"run_log={RUN_LOG_PATH}")
            main()


if __name__ == "__main__":
    run_with_file_logging()
