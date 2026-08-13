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
import timm
import torch
from torch import Tensor, nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if TYPE_CHECKING:
    from cifar_test import cifar_test_rtx5080 as cifar
else:
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from cifar_test import cifar_test_rtx5080 as cifar


# User-editable input. Both artifacts are derived from one full/subset run
# directory so a checkpoint cannot be accidentally paired with other labels.
SOURCE_RL_OUTPUT_DIR = (
    PROJECT_ROOT / "outputs" / "cifar10_test_rtx5080_full"
)
RL_CHECKPOINT_PATH = SOURCE_RL_OUTPUT_DIR / "rl_best.pt"
CORRECTED_LABELS_PATH = SOURCE_RL_OUTPUT_DIR / "train_corrected_labels.npy"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "cifar10_finetuning_full"

FINETUNE_EPOCHS = 100
TRAIN_BATCH_SIZE = 64
LEARNING_RATE = 3e-5
MIN_LR = 1e-6
WEIGHT_DECAY = 0.1
ADAMW_BETAS = (0.9, 0.999)
ADAMW_EPS = 1e-8
LABEL_SMOOTHING = 0.0
SEED = 0
OVERWRITE = False

RUN_LOG_PATH = OUTPUT_DIR / "run.log"
TRAIN_CSV_PATH = OUTPUT_DIR / "train.csv"
FINAL_CHECKPOINT_PATH = OUTPUT_DIR / "finetune_last.pt"

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
        for path in (RL_CHECKPOINT_PATH, CORRECTED_LABELS_PATH)
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(f"Fine-tuning inputs not found: {missing}")
    if (
        RL_CHECKPOINT_PATH.parent.resolve()
        != CORRECTED_LABELS_PATH.parent.resolve()
    ):
        raise ValueError(
            "RL_CHECKPOINT_PATH and CORRECTED_LABELS_PATH must come from "
            "the same Full/Subset RL output directory."
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
    return cifar.benchmark.pin_for_cuda(labels)


def _load_rl_model(device: torch.device) -> nn.Module:
    if not RL_CHECKPOINT_PATH.is_file():
        raise FileNotFoundError(f"RL checkpoint not found: {RL_CHECKPOINT_PATH}")
    checkpoint = torch.load(
        RL_CHECKPOINT_PATH,
        map_location="cpu",
        weights_only=True,
    )
    if not isinstance(checkpoint, dict):
        raise TypeError("RL checkpoint must contain a dictionary.")
    required = {"model", "model_name", "num_classes", "epoch"}
    missing = required.difference(checkpoint)
    if missing:
        raise KeyError(f"RL checkpoint is missing fields: {sorted(missing)}")
    if checkpoint["model_name"] != cifar.MODEL_NAME:
        raise ValueError("RL checkpoint model name does not match CIFAR config.")
    if int(checkpoint["num_classes"]) != cifar.NUM_CLASSES:
        raise ValueError("RL checkpoint class count does not match CIFAR-10.")

    model = timm.create_model(
        cifar.MODEL_NAME,
        pretrained=False,
        num_classes=cifar.NUM_CLASSES,
        drop_rate=cifar.DROP_RATE,
        drop_path_rate=cifar.DROP_PATH_RATE,
    )
    try:
        model.load_state_dict(checkpoint["model"], strict=True)
    except RuntimeError as error:
        raise RuntimeError(
            "The RL checkpoint does not contain the warmup classifier head. "
            "Run CIFAR RL with REMOVE_CLASSIFIER_FOR_RL=False."
        ) from error
    model.to(
        device=device,
        memory_format=(
            torch.channels_last
            if cifar.benchmark.USE_CHANNELS_LAST
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
        "source_rl_checkpoint": str(RL_CHECKPOINT_PATH),
        "corrected_labels": str(CORRECTED_LABELS_PATH),
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

    cifar.configure_benchmark()
    benchmark = cifar.benchmark
    device = benchmark.resolve_local_device()
    benchmark.seed_everything(SEED)
    torch.backends.cudnn.benchmark = benchmark.CUDNN_BENCHMARK
    torch.cuda.reset_peak_memory_stats()

    train_images, clean_train_labels = cifar.load_full_cifar10_train()
    corrected_labels = _load_corrected_labels(clean_train_labels)
    model = _load_rl_model(device)
    mean = torch.tensor(benchmark.IMAGENET_MEAN, device=device).reshape(
        1, 3, 1, 1
    )
    std = torch.tensor(benchmark.IMAGENET_STD, device=device).reshape(
        1, 3, 1, 1
    )
    optimizer = AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        betas=ADAMW_BETAS,
        eps=ADAMW_EPS,
    )
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=FINETUNE_EPOCHS,
        eta_min=MIN_LR,
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=(
            benchmark.USE_AMP and benchmark.AMP_DTYPE == torch.float16
        ),
    )
    history: list[dict[str, object]] = []

    print(f"device={device} ({torch.cuda.get_device_name(device)})")
    print(f"rl_checkpoint={RL_CHECKPOINT_PATH}")
    print(f"corrected_labels={CORRECTED_LABELS_PATH}")
    print(f"epochs={FINETUNE_EPOCHS} batch_size={TRAIN_BATCH_SIZE}")
    print(
        "initial_corrected_label_accuracy="
        f"{float(corrected_labels.eq(clean_train_labels).float().mean()):.6f}"
    )

    for epoch in range(1, FINETUNE_EPOCHS + 1):
        benchmark.synchronize(device)
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
                dtype=benchmark.AMP_DTYPE,
                enabled=benchmark.USE_AMP,
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
        benchmark.synchronize(device)
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
        stdout = cifar.benchmark.TeeStream(sys.stdout, handle)
        stderr = cifar.benchmark.TeeStream(sys.stderr, handle)
        with redirect_stdout(stdout), redirect_stderr(stderr):
            print(f"run_log={RUN_LOG_PATH}")
            main()


if __name__ == "__main__":
    run_with_file_logging()
