"""Evaluate the fine-tuned model on the held-out clean CIFAR-10 test split.

The other half of the official CIFAR-10 test set is used for validation, so
only the untouched 5,000-image half is used here to avoid validation leakage.
"""

from __future__ import annotations

import time

import torch
from torch import Tensor, nn

from log.common import run_with_log, write_csv
from rl import engine
from setting import data as cifar


CONFIG = cifar.CONFIG
FINETUNE_SELECTION_METRIC = CONFIG.finetune.evaluation_checkpoint
FINETUNE_CHECKPOINT_PATH = CONFIG.finetune_evaluation_checkpoint_path
INITIALIZATION = CONFIG.finetune.initialization
OUTPUT_DIR = CONFIG.evaluate_output_dir

TEST_BATCH_SIZE = CONFIG.runtime.evaluate_batch_size
SEED = CONFIG.data.seed
OVERWRITE = CONFIG.runtime.overwrite_evaluate

RUN_LOG_PATH = OUTPUT_DIR / "run.log"
TEST_CSV_PATH = OUTPUT_DIR / "test.csv"

TEST_FIELDS = ("epoch", "initialization", "selection", "loss", "accuracy", "seconds")


def _load_model(device: torch.device) -> tuple[nn.Module, dict[str, object]]:
    checkpoint = torch.load(FINETUNE_CHECKPOINT_PATH, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise TypeError("Fine-tuned checkpoint must contain a dictionary.")
    required = {"model", "model_name", "num_classes", "epoch"}
    missing = required.difference(checkpoint)
    if missing:
        raise KeyError(f"Fine-tuned checkpoint is missing fields: {sorted(missing)}")
    if checkpoint["model_name"] != cifar.MODEL_NAME:
        raise ValueError("Checkpoint model name does not match CIFAR config.")
    if int(checkpoint["num_classes"]) != cifar.NUM_CLASSES:
        raise ValueError("Checkpoint class count does not match CIFAR-10.")
    if checkpoint.get("initialization") != INITIALIZATION:
        raise ValueError("Checkpoint initialization does not match the configured initialization.")
    engine.validate_training_data_checkpoint(checkpoint)
    if checkpoint.get("checkpoint_kind") != "best":
        raise ValueError("Final evaluation must use the fine-tuning best model.")
    if checkpoint.get("selection_metric") != FINETUNE_SELECTION_METRIC:
        raise ValueError("Fine-tuning checkpoint selection metric does not match config.")

    model = cifar.build_model()
    model.load_state_dict(checkpoint["model"], strict=True)
    cifar.move_model_to_device(model, device)
    metadata = {
        "checkpoint_kind": checkpoint["checkpoint_kind"],
        "selection_metric": checkpoint["selection_metric"],
        "epoch": int(checkpoint["epoch"]),
    }
    del checkpoint
    return model, metadata


@torch.inference_mode()
def _evaluate(model: nn.Module, images: Tensor, labels: Tensor, device: torch.device) -> dict[str, object]:
    model.eval()
    mean, std = engine.normalization_tensors(device)
    criterion = nn.CrossEntropyLoss(reduction="sum")
    loss_sum = torch.zeros((), dtype=torch.float64, device=device)
    correct_count = torch.zeros((), dtype=torch.long, device=device)

    for start in range(0, labels.numel(), TEST_BATCH_SIZE):
        end = min(start + TEST_BATCH_SIZE, labels.numel())
        batch_images = cifar.preprocess_cifar10(images[start:end], device, mean, std)
        targets = labels[start:end].to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=engine.AMP_DTYPE, enabled=engine.USE_AMP):
            logits = model(batch_images)
            loss = criterion(logits, targets)
        correct_count += logits.argmax(dim=1).eq(targets).sum()
        loss_sum += loss.to(torch.float64)

    return {"loss": float(loss_sum / labels.numel()), "accuracy": float(correct_count / labels.numel())}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    device = engine.resolve_local_device()
    engine.seed_everything(SEED)
    torch.backends.cudnn.benchmark = engine.CUDNN_BENCHMARK

    test_images, test_labels = cifar.load_cifar10_evaluation_split("test")
    model, checkpoint = _load_model(device)

    print(
        f"device={torch.cuda.get_device_name(device)} initialization={INITIALIZATION} "
        f"epoch={int(checkpoint['epoch'])} samples={test_labels.numel()}"
    )
    engine.synchronize(device)
    started = time.perf_counter()
    summary = _evaluate(model, test_images, test_labels, device)
    engine.synchronize(device)
    elapsed = time.perf_counter() - started
    summary = {
        "epoch": int(checkpoint["epoch"]),
        "initialization": INITIALIZATION,
        "selection": checkpoint["selection_metric"],
        "loss": summary["loss"],
        "accuracy": summary["accuracy"],
        "seconds": elapsed,
    }
    write_csv(TEST_CSV_PATH, [summary], TEST_FIELDS)
    print(
        f"[TEST] loss={float(summary['loss']):.4f} "
        f"accuracy={float(summary['accuracy']):.4f} "
        f"seconds={elapsed:.3f}"
    )
    print(f"output={OUTPUT_DIR}")


def run_with_file_logging() -> None:
    cifar.require_files((FINETUNE_CHECKPOINT_PATH,), stage="Evaluation")
    cifar.require_available_outputs([TEST_CSV_PATH, RUN_LOG_PATH], overwrite=OVERWRITE, stage="Evaluation")
    run_with_log(RUN_LOG_PATH, main)


if __name__ == "__main__":
    run_with_file_logging()
