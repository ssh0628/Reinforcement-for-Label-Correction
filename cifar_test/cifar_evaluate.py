"""Evaluate the fine-tuned model on the held-out clean CIFAR-10 test split.

The other half of the official CIFAR-10 test set is used for validation, so
only the untouched 5,000-image half is used here to avoid validation leakage.
"""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import torch
from torch import Tensor, nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cifar_test import cifar_common as cifar


CONFIG = cifar.CONFIG
FINETUNE_CHECKPOINT_PATH = CONFIG.finetune_checkpoint_path
INITIALIZATION = CONFIG.finetune.initialization
OUTPUT_DIR = CONFIG.evaluate_output_dir

TEST_BATCH_SIZE = CONFIG.runtime.evaluate_batch_size
SEED = CONFIG.data.seed
OVERWRITE = CONFIG.runtime.overwrite_evaluate

RUN_LOG_PATH = OUTPUT_DIR / "run.log"
TEST_CSV_PATH = OUTPUT_DIR / "test.csv"
TEST_PER_CLASS_CSV_PATH = OUTPUT_DIR / "test_per_class.csv"

TEST_FIELDS = (
    "samples",
    "loss",
    "accuracy",
    "balanced_accuracy",
    "macro_recall",
    "macro_precision",
    "macro_f1",
    "elapsed_seconds",
)
PER_CLASS_FIELDS = (
    "class_id",
    "support",
    "precision",
    "recall",
    "f1",
)


def _validate_paths(*, include_log: bool) -> None:
    cifar.require_files((FINETUNE_CHECKPOINT_PATH,), stage="Evaluation")
    outputs = [TEST_CSV_PATH, TEST_PER_CLASS_CSV_PATH]
    if include_log:
        outputs.append(RUN_LOG_PATH)
    cifar.require_available_outputs(
        outputs,
        overwrite=OVERWRITE,
        stage="Evaluation",
    )


def _load_model(device: torch.device) -> nn.Module:
    checkpoint = torch.load(
        FINETUNE_CHECKPOINT_PATH,
        map_location="cpu",
        weights_only=True,
    )
    if not isinstance(checkpoint, dict):
        raise TypeError("Fine-tuned checkpoint must contain a dictionary.")
    required = {"model", "model_name", "num_classes", "epoch"}
    missing = required.difference(checkpoint)
    if missing:
        raise KeyError(
            f"Fine-tuned checkpoint is missing fields: {sorted(missing)}"
        )
    if checkpoint["model_name"] != cifar.MODEL_NAME:
        raise ValueError("Checkpoint model name does not match CIFAR config.")
    if int(checkpoint["num_classes"]) != cifar.NUM_CLASSES:
        raise ValueError("Checkpoint class count does not match CIFAR-10.")
    if checkpoint.get("initialization") != INITIALIZATION:
        raise ValueError(
            "Checkpoint initialization does not match the configured ablation."
        )

    model = cifar.build_model()
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(
        device=device,
        memory_format=(
            torch.channels_last
            if cifar.engine.USE_CHANNELS_LAST
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
def _evaluate(
    model: nn.Module,
    images: Tensor,
    labels: Tensor,
    device: torch.device,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    model.eval()
    engine = cifar.engine
    mean = torch.tensor(cifar.CIFAR10_MEAN, device=device).reshape(
        1, 3, 1, 1
    )
    std = torch.tensor(cifar.CIFAR10_STD, device=device).reshape(
        1, 3, 1, 1
    )
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
            dtype=engine.AMP_DTYPE,
            enabled=engine.USE_AMP,
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


def main() -> None:
    if TEST_BATCH_SIZE <= 0:
        raise ValueError("TEST_BATCH_SIZE must be positive.")
    _validate_paths(include_log=False)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    cifar.configure_engine()
    engine = cifar.engine
    device = engine.resolve_local_device()
    engine.seed_everything(SEED)
    torch.backends.cudnn.benchmark = engine.CUDNN_BENCHMARK

    test_images, test_labels = cifar.load_cifar10_evaluation_split("test")
    if test_images.size(0) != 5_000 or test_labels.size(0) != 5_000:
        raise RuntimeError("Unexpected held-out CIFAR-10 test size.")
    model = _load_model(device)

    print(f"device={device} ({torch.cuda.get_device_name(device)})")
    print(f"initialization={INITIALIZATION}")
    print(f"checkpoint={FINETUNE_CHECKPOINT_PATH}")
    print(f"held_out_test_samples={test_labels.numel()}")
    engine.synchronize(device)
    started = time.perf_counter()
    summary, per_class = _evaluate(
        model,
        test_images,
        test_labels,
        device,
    )
    engine.synchronize(device)
    elapsed = time.perf_counter() - started
    summary["elapsed_seconds"] = elapsed
    if not math.isfinite(float(summary["loss"])):
        raise RuntimeError("Final test loss is not finite.")

    engine.write_csv(TEST_CSV_PATH, [summary], TEST_FIELDS)
    engine.write_csv(
        TEST_PER_CLASS_CSV_PATH,
        per_class,
        PER_CLASS_FIELDS,
    )
    print(
        f"[FINAL TEST] samples={summary['samples']} "
        f"loss={float(summary['loss']):.6f} "
        f"accuracy={float(summary['accuracy']):.6f} "
        f"balanced_accuracy={float(summary['balanced_accuracy']):.6f} "
        f"macro_f1={float(summary['macro_f1']):.6f} "
        f"seconds={elapsed:.3f}"
    )
    print(f"test_csv={TEST_CSV_PATH}")
    print(f"test_per_class_csv={TEST_PER_CLASS_CSV_PATH}")


def run_with_file_logging() -> None:
    _validate_paths(include_log=True)
    cifar.run_with_log(RUN_LOG_PATH, main)


if __name__ == "__main__":
    run_with_file_logging()
