"""Fine-tune an RLNLC CIFAR-10 model on saved corrected labels.

Set ``RL_CHECKPOINT_PATH``, ``CORRECTED_LABELS_PATH``, and ``OUTPUT_DIR`` to
the outputs from either the full or subset RL run.  The clean held-out half of
the official CIFAR-10 test set is evaluated once after fine-tuning.  Its other
half is reserved for warmup/RL validation and is never included here.
"""

from __future__ import annotations

import csv
import math
import sys
import time
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import numpy as np
import timm
import torch
from torch import Tensor, nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

PROJECT_ROOT = Path(__file__).resolve().parents[1]
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
TEST_BATCH_SIZE = 256
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
TEST_CSV_PATH = OUTPUT_DIR / "test.csv"
TEST_PER_CLASS_CSV_PATH = OUTPUT_DIR / "test_per_class.csv"
FINAL_CHECKPOINT_PATH = OUTPUT_DIR / "finetune_last.pt"

TRAIN_FIELDS = (
    "epoch",
    "learning_rate",
    "loss",
    "corrected_label_accuracy",
    "clean_label_accuracy",
    "elapsed_seconds",
)
TEST_FIELDS = (
    "samples",
    "loss",
    "accuracy",
    "balanced_accuracy",
    "macro_recall",
    "macro_precision",
    "macro_f1",
)
PER_CLASS_FIELDS = (
    "class_id",
    "support",
    "precision",
    "recall",
    "f1",
)


def _validate_output_destination(*, include_log: bool) -> None:
    paths = [
        TRAIN_CSV_PATH,
        TEST_CSV_PATH,
        TEST_PER_CLASS_CSV_PATH,
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


def _load_heldout_test() -> tuple[Tensor, Tensor]:
    images, labels = cifar.load_cifar10_validation_test()["test"]
    if images.size(0) != 5_000 or labels.size(0) != 5_000:
        raise RuntimeError("Unexpected held-out CIFAR-10 test size.")
    return images, labels


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


def _classification_rows(
    confusion: Tensor,
) -> tuple[dict[str, float], list[dict[str, object]]]:
    matrix = confusion.to(torch.float64)
    true_positive = matrix.diag()
    support = matrix.sum(dim=1)
    predicted = matrix.sum(dim=0)
    recall = true_positive / support.clamp_min(1)
    precision = true_positive / predicted.clamp_min(1)
    f1 = torch.where(
        precision + recall > 0,
        2 * precision * recall / (precision + recall),
        torch.zeros_like(precision),
    )
    summary = {
        "accuracy": float(true_positive.sum() / matrix.sum().clamp_min(1)),
        "balanced_accuracy": float(recall.mean()),
        "macro_recall": float(recall.mean()),
        "macro_precision": float(precision.mean()),
        "macro_f1": float(f1.mean()),
    }
    per_class = [
        {
            "class_id": class_id,
            "support": int(support[class_id]),
            "precision": float(precision[class_id]),
            "recall": float(recall[class_id]),
            "f1": float(f1[class_id]),
        }
        for class_id in range(cifar.NUM_CLASSES)
    ]
    return summary, per_class


@torch.inference_mode()
def evaluate_clean_test(
    model: nn.Module,
    images: Tensor,
    labels: Tensor,
    device: torch.device,
    mean: Tensor,
    std: Tensor,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="sum")
    confusion = torch.zeros(
        (cifar.NUM_CLASSES, cifar.NUM_CLASSES),
        dtype=torch.long,
        device=device,
    )
    loss_sum = torch.zeros((), dtype=torch.float64, device=device)
    for start in range(0, labels.numel(), TEST_BATCH_SIZE):
        end = min(start + TEST_BATCH_SIZE, labels.numel())
        batch_images = cifar.preprocess_cifar10(
            images[start:end],
            device,
            mean,
            std,
        )
        targets = labels[start:end].to(device, non_blocking=True)
        with torch.autocast(
            device_type="cuda",
            dtype=cifar.benchmark.AMP_DTYPE,
            enabled=cifar.benchmark.USE_AMP,
        ):
            logits = model(batch_images)
            loss = criterion(logits, targets)
        predictions = logits.argmax(dim=1)
        flat = targets * cifar.NUM_CLASSES + predictions
        confusion += torch.bincount(
            flat,
            minlength=cifar.NUM_CLASSES * cifar.NUM_CLASSES,
        ).reshape(cifar.NUM_CLASSES, cifar.NUM_CLASSES)
        loss_sum += loss.to(torch.float64)
    summary, per_class = _classification_rows(confusion)
    return (
        {
            "samples": labels.numel(),
            "loss": float(loss_sum / labels.numel()),
            **summary,
        },
        per_class,
    )


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
    test_images, test_labels = _load_heldout_test()
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
    benchmark.synchronize(device)
    test_started = time.perf_counter()
    test_summary, test_per_class = evaluate_clean_test(
        model,
        test_images,
        test_labels,
        device,
        mean,
        std,
    )
    benchmark.synchronize(device)
    test_seconds = time.perf_counter() - test_started
    if not math.isfinite(float(test_summary["loss"])):
        raise RuntimeError("Final test loss is not finite.")
    _write_csv(TEST_CSV_PATH, [test_summary], TEST_FIELDS)
    _write_csv(
        TEST_PER_CLASS_CSV_PATH,
        test_per_class,
        PER_CLASS_FIELDS,
    )
    print(
        f"[TEST] samples={test_summary['samples']} "
        f"loss={float(test_summary['loss']):.6f} "
        f"accuracy={float(test_summary['accuracy']):.6f} "
        f"balanced_accuracy="
        f"{float(test_summary['balanced_accuracy']):.6f} "
        f"macro_f1={float(test_summary['macro_f1']):.6f} "
        f"seconds={test_seconds:.3f}"
    )
    print(f"checkpoint={FINAL_CHECKPOINT_PATH}")
    print(f"train_csv={TRAIN_CSV_PATH}")
    print(f"test_csv={TEST_CSV_PATH}")
    print(f"test_per_class_csv={TEST_PER_CLASS_CSV_PATH}")


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
