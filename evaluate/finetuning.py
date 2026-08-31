"""Fine-tune a CIFAR-10 model on RL or weighted-KNN soft labels.

The label source and initial model are selected in ``setting.config``.
Separate best checkpoints are saved for clean-validation accuracy and loss,
along with the final-epoch model.
"""

from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn
from torch.optim import SGD
from torch.optim.lr_scheduler import MultiStepLR

from evaluate.metrics import evaluate_classifier, validate_soft_labels
from log.common import append_csv, run_with_log, save_torch, write_csv
from rl import engine
from setting import data as cifar


CONFIG = cifar.CONFIG
INITIALIZATION = CONFIG.finetune.initialization
INITIAL_CHECKPOINT_PATH = CONFIG.finetune_initial_checkpoint_path
LABEL_SOURCE = CONFIG.finetune.corrected_label_source
CORRECTED_LABELS_PATH = CONFIG.finetune_corrected_labels_path
OUTPUT_DIR = CONFIG.finetune_output_dir

FINETUNE_EPOCHS = CONFIG.finetune.epochs
TRAIN_BATCH_SIZE = CONFIG.finetune.batch_size
VALIDATION_BATCH_SIZE = CONFIG.runtime.evaluate_batch_size
LEARNING_RATE = CONFIG.finetune.learning_rate
MOMENTUM = CONFIG.finetune.momentum
WEIGHT_DECAY = CONFIG.finetune.weight_decay
LR_DECAY_EPOCH = math.ceil(FINETUNE_EPOCHS * CONFIG.finetune.lr_decay_fraction)
LR_DECAY_FACTOR = CONFIG.finetune.lr_decay_factor
SEED = CONFIG.data.seed
OVERWRITE = CONFIG.runtime.overwrite_finetune

RUN_LOG_PATH = OUTPUT_DIR / "run.log"
TRAIN_CSV_PATH = OUTPUT_DIR / "train.csv"
BEST_ACCURACY_CHECKPOINT_PATH = CONFIG.finetune_best_accuracy_checkpoint_path
BEST_LOSS_CHECKPOINT_PATH = CONFIG.finetune_best_loss_checkpoint_path
LAST_CHECKPOINT_PATH = CONFIG.finetune_last_checkpoint_path

TRAIN_FIELDS = (
    "epoch",
    "lr",
    "train_loss",
    "corrected_accuracy",
    "clean_accuracy",
    "val_loss",
    "val_accuracy",
    "seconds",
)


def _load_corrected_soft_labels(clean_labels: Tensor) -> Tensor:
    array = np.load(CORRECTED_LABELS_PATH, allow_pickle=False)
    if not np.issubdtype(array.dtype, np.floating):
        raise TypeError(
            "Corrected labels must be a floating-point soft-label array. "
            "Run the selected RL or KNN correction stage again."
        )
    labels = torch.from_numpy(array).to(torch.float32).contiguous()
    validate_soft_labels(labels, clean_labels.numel(), cifar.NUM_CLASSES)
    return cifar.pin_for_cuda(labels)


def _load_initial_model(device: torch.device) -> nn.Module:
    checkpoint = torch.load(INITIAL_CHECKPOINT_PATH, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise TypeError("Initial checkpoint must contain a dictionary.")
    required = {"model", "epoch"}
    missing = required.difference(checkpoint)
    if missing:
        raise KeyError(f"Initial checkpoint is missing fields: {sorted(missing)}")
    if "model_name" in checkpoint and checkpoint["model_name"] != cifar.MODEL_NAME:
        raise ValueError("Initial checkpoint model name does not match config.")
    if "num_classes" in checkpoint and int(checkpoint["num_classes"]) != cifar.NUM_CLASSES:
        raise ValueError("Initial checkpoint class count does not match CIFAR-10.")
    if INITIALIZATION == "warmup":
        if checkpoint.get("warmup_model_id") != CONFIG.warmup.model_id:
            raise ValueError("Warmup checkpoint model ID does not match config.")
        if checkpoint.get("selection", "best") != CONFIG.warmup.checkpoint_selection:
            raise ValueError("Warmup checkpoint selection does not match config.")
    engine.validate_training_augmentation_checkpoint(checkpoint)
    engine.validate_training_data_checkpoint(checkpoint)
    if INITIALIZATION == "last_actor" and int(checkpoint["epoch"]) != CONFIG.rl.epochs:
        raise ValueError("last_actor initialization must use the configured final RL epoch.")

    model = cifar.build_model()
    try:
        model.load_state_dict(checkpoint["model"], strict=True)
    except RuntimeError as error:
        raise RuntimeError("The initial checkpoint does not contain the classifier head.") from error
    return cifar.move_model_to_device(model, device)


def _save_checkpoint(
    model: nn.Module,
    path: Path,
    *,
    epoch: int,
    checkpoint_kind: str,
    selection_metric: str | None,
    validation: dict[str, float],
) -> None:
    payload = {
        "epoch": epoch,
        "checkpoint_kind": checkpoint_kind,
        "selection_metric": selection_metric,
        "selection_mode": ("min" if selection_metric == "loss" else "max")
        if selection_metric is not None
        else None,
        "selection_value": (validation[selection_metric] if selection_metric is not None else None),
        "validation": validation,
        "model_name": cifar.MODEL_NAME,
        "num_classes": cifar.NUM_CLASSES,
        "initialization": INITIALIZATION,
        "corrected_label_source": LABEL_SOURCE,
        "source_checkpoint": str(INITIAL_CHECKPOINT_PATH),
        "corrected_labels": str(CORRECTED_LABELS_PATH),
        "optimizer": CONFIG.finetune.optimizer,
        "learning_rate": LEARNING_RATE,
        "momentum": MOMENTUM,
        "weight_decay": WEIGHT_DECAY,
        "training_augmentation": engine.training_augmentation_metadata(),
        "training_data": engine.training_data_metadata(),
        "model": model.state_dict(),
    }
    save_torch(path, payload)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    BEST_ACCURACY_CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)

    device = engine.initialize_cuda_runtime(SEED, reset_peak_memory=True)

    train_images, clean_train_labels = cifar.load_selected_cifar10_train()
    validation_images, validation_labels = cifar.load_cifar10_evaluation_split("val")
    corrected_labels = _load_corrected_soft_labels(clean_train_labels)
    training_sample_count = clean_train_labels.numel()
    model = _load_initial_model(device)
    mean, std = engine.normalization_tensors(device)
    optimizer = SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    scheduler = MultiStepLR(optimizer, milestones=[LR_DECAY_EPOCH], gamma=LR_DECAY_FACTOR)
    criterion = nn.CrossEntropyLoss()
    scaler = engine.build_grad_scaler()
    write_csv(TRAIN_CSV_PATH, [], TRAIN_FIELDS)

    print(
        f"device={torch.cuda.get_device_name(device)} initialization={INITIALIZATION} "
        f"label_source={LABEL_SOURCE}"
    )
    print(
        f"epochs={FINETUNE_EPOCHS} batch={TRAIN_BATCH_SIZE} lr={LEARNING_RATE} decay_epoch={LR_DECAY_EPOCH}"
    )
    initial_argmax_accuracy = float(corrected_labels.argmax(dim=1).eq(clean_train_labels).float().mean())
    print(f"corrected_label_accuracy={initial_argmax_accuracy:.4f}")

    best_values = {"accuracy": float("-inf"), "loss": float("inf")}
    best_epochs = {metric: 0 for metric in best_values}
    best_validations: dict[str, dict[str, float]] = {}
    best_paths = {"accuracy": BEST_ACCURACY_CHECKPOINT_PATH, "loss": BEST_LOSS_CHECKPOINT_PATH}
    last_validation: dict[str, float] | None = None
    for epoch in range(1, FINETUNE_EPOCHS + 1):
        engine.synchronize(device)
        epoch_started = time.perf_counter()
        model.train()
        learning_rate = float(optimizer.param_groups[0]["lr"])
        generator = torch.Generator().manual_seed(SEED + epoch)
        augmentation_generator = torch.Generator(device=device).manual_seed(SEED + epoch)
        permutation = torch.randperm(training_sample_count, generator=generator)
        loss_sum = torch.zeros((), dtype=torch.float64, device=device)
        corrected_count = torch.zeros((), dtype=torch.long, device=device)
        clean_count = torch.zeros((), dtype=torch.long, device=device)

        for start in range(0, permutation.numel(), TRAIN_BATCH_SIZE):
            end = min(start + TRAIN_BATCH_SIZE, permutation.numel())
            indices = permutation[start:end]
            images = cifar.preprocess_cifar10_training(
                train_images[indices], device, mean, std, augmentation_generator
            )
            targets = corrected_labels[indices].to(device, non_blocking=True)
            clean_targets = clean_train_labels[indices].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=engine.AMP_DTYPE, enabled=engine.USE_AMP):
                logits = model(images)
                loss = criterion(logits, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            batch_count = end - start
            predictions = logits.argmax(dim=1)
            loss_sum += loss.detach().to(torch.float64) * batch_count
            corrected_count += predictions.eq(targets.argmax(dim=1)).sum()
            clean_count += predictions.eq(clean_targets).sum()

        engine.synchronize(device)
        validation = evaluate_classifier(
            model,
            validation_images,
            validation_labels,
            device,
            mean,
            std,
            batch_size=VALIDATION_BATCH_SIZE,
            use_amp=engine.USE_AMP,
            amp_dtype=engine.AMP_DTYPE,
            preprocess=cifar.preprocess_cifar10,
        )
        engine.synchronize(device)
        improved = {
            "accuracy": validation["accuracy"] > best_values["accuracy"],
            "loss": validation["loss"] < best_values["loss"],
        }
        for metric, is_improved in improved.items():
            if not is_improved:
                continue
            best_values[metric] = validation[metric]
            best_epochs[metric] = epoch
            best_validations[metric] = validation.copy()
            _save_checkpoint(
                model,
                best_paths[metric],
                epoch=epoch,
                checkpoint_kind="best",
                selection_metric=metric,
                validation=validation,
            )
        last_validation = validation.copy()
        scheduler.step()
        elapsed = time.perf_counter() - epoch_started
        row: dict[str, object] = {
            "epoch": epoch,
            "lr": learning_rate,
            "train_loss": float(loss_sum / training_sample_count),
            "corrected_accuracy": float(corrected_count / training_sample_count),
            "clean_accuracy": float(clean_count / clean_train_labels.numel()),
            "val_loss": validation["loss"],
            "val_accuracy": validation["accuracy"],
            "seconds": elapsed,
        }
        append_csv(TRAIN_CSV_PATH, [row], TRAIN_FIELDS)
        print(
            f"[FINETUNE] epoch={epoch}/{FINETUNE_EPOCHS} "
            f"loss={float(row['train_loss']):.4f} clean_acc={float(row['clean_accuracy']):.4f} "
            f"val_loss={validation['loss']:.4f} val_acc={validation['accuracy']:.4f} "
            f"seconds={elapsed:.3f}"
        )

    if len(best_validations) != 2 or last_validation is None:
        raise RuntimeError("Fine-tuning did not produce validation metrics.")
    _save_checkpoint(
        model,
        LAST_CHECKPOINT_PATH,
        epoch=FINETUNE_EPOCHS,
        checkpoint_kind="last",
        selection_metric=None,
        validation=last_validation,
    )
    for metric in ("accuracy", "loss"):
        validation = best_validations[metric]
        print(
            f"[FINETUNE BEST {metric.upper()}] "
            f"epoch={best_epochs[metric]} "
            f"val_loss={validation['loss']:.6f} "
            f"val_accuracy={validation['accuracy']:.6f} "
            f"checkpoint={best_paths[metric]}"
        )
    print(f"output={OUTPUT_DIR}")
    print("next=cifar_evaluate.py")


def run_with_file_logging() -> None:
    cifar.require_files(
        (cifar.TRAIN_INDICES_PATH, INITIAL_CHECKPOINT_PATH, CORRECTED_LABELS_PATH), stage="Fine-tuning"
    )
    cifar.require_available_outputs(
        [
            TRAIN_CSV_PATH,
            BEST_ACCURACY_CHECKPOINT_PATH,
            BEST_LOSS_CHECKPOINT_PATH,
            LAST_CHECKPOINT_PATH,
            RUN_LOG_PATH,
        ],
        overwrite=OVERWRITE,
        stage="Fine-tuning",
    )
    run_with_log(RUN_LOG_PATH, main)
