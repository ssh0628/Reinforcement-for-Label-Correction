"""Create the fixed CIFAR-10 ResNet-18 warm-up artifact.

Both full and subset RL runs load the checkpoint produced here.
"""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.optim import SGD
from torch.optim.lr_scheduler import MultiStepLR

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cifar_test.log.common import (
    TIMING_FIELDS,
    Timings,
    append_csv,
    build_timing_rows,
    measure,
    print_timing_summary,
    write_csv,
)
from cifar_test.setting import data as cifar


WARMUP_CHECKPOINT_PATH = cifar.WARMUP_CHECKPOINT_PATH
OUTPUT_DIR = cifar.CONFIG.warmup_log_dir

RUN_LOG_PATH = OUTPUT_DIR / "run.log"
WARMUP_CSV_PATH = OUTPUT_DIR / "warmup.csv"
TIMING_CSV_PATH = OUTPUT_DIR / "timing.csv"
OVERWRITE = cifar.CONFIG.runtime.overwrite_warmup

WARMUP_FIELDS = (
    "epoch",
    "lr",
    "train_loss",
    "train_accuracy",
    "val_loss",
    "val_noisy_accuracy",
    "val_clean_accuracy",
    "seconds",
)


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    images: Tensor,
    noisy_labels: Tensor,
    clean_labels: Tensor,
    device: torch.device,
    mean: Tensor,
    std: Tensor,
    criterion: nn.CrossEntropyLoss,
) -> dict[str, float]:
    engine = cifar.engine
    model.eval()
    loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    noisy_correct = torch.zeros((), device=device, dtype=torch.long)
    clean_correct = torch.zeros((), device=device, dtype=torch.long)
    for start in range(0, images.size(0), engine.WARMUP_EVAL_BATCH_SIZE):
        end = min(start + engine.WARMUP_EVAL_BATCH_SIZE, images.size(0))
        batch = engine.preprocess(images[start:end], device, mean, std)
        noisy_targets = noisy_labels[start:end].to(device, non_blocking=True)
        clean_targets = clean_labels[start:end].to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=engine.AMP_DTYPE, enabled=engine.USE_AMP):
            logits = model(batch)
            loss = criterion(logits, noisy_targets)
        predictions = logits.argmax(dim=1)
        loss_sum += loss.to(torch.float64) * (end - start)
        noisy_correct += predictions.eq(noisy_targets).sum()
        clean_correct += predictions.eq(clean_targets).sum()
    sample_count = images.size(0)
    return {
        "loss": float(loss_sum / sample_count),
        "noisy_accuracy": float(noisy_correct / sample_count),
        "clean_accuracy": float(clean_correct / sample_count),
    }


def save_checkpoint(
    model: nn.Module,
    path: Path,
    epoch: int,
    noisy_accuracy: float,
    clean_accuracy: float,
) -> None:
    engine = cifar.engine
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    try:
        torch.save(
            {
                "epoch": epoch,
                "model_name": engine.MODEL_NAME,
                "num_classes": engine.NUM_CLASSES,
                "model": model.state_dict(),
                "noisy_validation_accuracy": noisy_accuracy,
                "clean_validation_accuracy": clean_accuracy,
                "noise_rate": engine.NOISE_RATE,
                "pretrained": engine.PRETRAINED,
                "warmup_model_id": engine.WARMUP_MODEL_ID,
                "training_data": engine.training_data_metadata(),
                "training_augmentation": engine.training_augmentation_metadata(),
                "selection": "best",
                "best_epoch": epoch,
                "best_noisy_validation_accuracy": noisy_accuracy,
                "best_clean_validation_accuracy": clean_accuracy,
            },
            temporary_path,
        )
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)


def train_warmup(
    model: nn.Module,
    train_images: Tensor,
    train_noisy_labels: Tensor,
    val_images: Tensor,
    val_noisy_labels: Tensor,
    val_clean_labels: Tensor,
    device: torch.device,
    mean: Tensor,
    std: Tensor,
) -> dict[str, object]:
    engine = cifar.engine
    criterion = nn.CrossEntropyLoss()
    scaler = engine.build_grad_scaler()
    optimizer = SGD(
        model.parameters(),
        lr=engine.WARMUP_LR,
        momentum=engine.WARMUP_MOMENTUM,
        weight_decay=engine.WARMUP_WEIGHT_DECAY,
    )
    milestone = max(1, math.ceil(engine.WARMUP_EPOCHS * engine.WARMUP_LR_DECAY_FRACTION))
    scheduler = MultiStepLR(optimizer, milestones=[milestone], gamma=engine.WARMUP_LR_DECAY_FACTOR)
    write_csv(WARMUP_CSV_PATH, [], WARMUP_FIELDS)
    best_noisy_accuracy = float("-inf")
    best_clean_accuracy = float("nan")
    best_epoch = 0

    for epoch in range(1, engine.WARMUP_EPOCHS + 1):
        started = time.perf_counter()
        model.train()
        generator = torch.Generator().manual_seed(cifar.SEED + epoch)
        augmentation_generator = torch.Generator(device=device).manual_seed(cifar.SEED + epoch)
        permutation = torch.randperm(train_images.size(0), generator=generator)
        loss_sum = torch.zeros((), device=device, dtype=torch.float64)
        correct_count = torch.zeros((), device=device, dtype=torch.long)

        for start in range(0, permutation.numel(), engine.WARMUP_BATCH_SIZE):
            indices = permutation[start : start + engine.WARMUP_BATCH_SIZE]
            images = engine.preprocess_training(
                train_images[indices], device, mean, std, augmentation_generator
            )
            targets = train_noisy_labels[indices].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=engine.AMP_DTYPE, enabled=engine.USE_AMP):
                logits = model(images)
                loss = criterion(logits, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            loss_sum += loss.detach().to(torch.float64) * indices.numel()
            correct_count += logits.argmax(dim=1).eq(targets).sum()

        validation = evaluate(
            model, val_images, val_noisy_labels, val_clean_labels, device, mean, std, criterion
        )
        row: dict[str, object] = {
            "epoch": epoch,
            "lr": float(optimizer.param_groups[0]["lr"]),
            "train_loss": float(loss_sum / train_images.size(0)),
            "train_accuracy": float(correct_count / train_images.size(0)),
            "val_loss": validation["loss"],
            "val_noisy_accuracy": validation["noisy_accuracy"],
            "val_clean_accuracy": validation["clean_accuracy"],
            "seconds": time.perf_counter() - started,
        }
        append_csv(WARMUP_CSV_PATH, [row], WARMUP_FIELDS)
        print(
            f"[WARMUP] epoch={epoch}/{engine.WARMUP_EPOCHS} lr={float(row['lr']):.6g} "
            f"train_loss={float(row['train_loss']):.4f} "
            f"train_acc={float(row['train_accuracy']):.4f} "
            f"val_noisy_acc={float(row['val_noisy_accuracy']):.4f} "
            f"val_clean_acc={float(row['val_clean_accuracy']):.4f}"
        )

        noisy_accuracy = float(validation["noisy_accuracy"])
        if noisy_accuracy > best_noisy_accuracy:
            best_noisy_accuracy = noisy_accuracy
            best_clean_accuracy = float(validation["clean_accuracy"])
            best_epoch = epoch
            save_checkpoint(model, WARMUP_CHECKPOINT_PATH, epoch, best_noisy_accuracy, best_clean_accuracy)
        scheduler.step()

    checkpoint = torch.load(WARMUP_CHECKPOINT_PATH, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device).eval()
    deployed_epoch = int(checkpoint["epoch"])
    print(
        f"[WARMUP] deployment=best deployment_epoch={deployed_epoch} best_epoch={best_epoch} "
        f"val_noisy_acc={best_noisy_accuracy:.4f} val_clean_acc={best_clean_accuracy:.4f}"
    )
    if best_noisy_accuracy < engine.WARMUP_MIN_NOISY_VALIDATION_ACCURACY:
        raise RuntimeError(
            "Warmup quality is too low to build a semantic RL reward graph: "
            f"{best_noisy_accuracy:.4f} < {engine.WARMUP_MIN_NOISY_VALIDATION_ACCURACY:.4f}."
        )
    return {
        "best_epoch": best_epoch,
        "best_noisy_validation_accuracy": best_noisy_accuracy,
        "best_clean_validation_accuracy": best_clean_accuracy,
        "deployment_mode": "best",
        "deployment_epoch": deployed_epoch,
    }


def main() -> None:
    cifar.configure_engine()
    engine = cifar.engine

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    WARMUP_CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)

    device = engine.resolve_local_device()
    engine.seed_everything(cifar.SEED)
    torch.backends.cudnn.benchmark = engine.CUDNN_BENCHMARK
    torch.cuda.reset_peak_memory_stats()
    timings: Timings = {}

    raw_images, clean_labels = measure("data_load", device, timings, cifar.load_selected_cifar10_train)
    noisy_labels, noise_mask = measure(
        "noise_load", device, timings, lambda: engine.load_noisy_label_artifacts(clean_labels)
    )
    evaluation_splits = measure("val_load", device, timings, cifar.load_cifar10_validation)
    val_images, val_clean_labels = evaluation_splits["val"]
    val_noisy_labels, _ = measure(
        "val_noise", device, timings, lambda: engine.inject_stratified_symmetric_noise(val_clean_labels)
    )

    engine.print_configuration(device, clean_labels.numel())
    print(f"output={OUTPUT_DIR}")

    model = measure(
        "model_init",
        device,
        timings,
        lambda: cifar.build_model().to(
            device=device,
            memory_format=(torch.channels_last if engine.USE_CHANNELS_LAST else torch.contiguous_format),
        ),
    )
    mean, std = engine.normalization_tensors(device)
    measure(
        "gpu_warmup",
        device,
        timings,
        lambda: engine.warm_device_kernels(model, raw_images, device, mean, std),
    )
    result = measure(
        "warmup_train",
        device,
        timings,
        lambda: train_warmup(
            model,
            raw_images,
            noisy_labels,
            val_images,
            val_noisy_labels,
            val_clean_labels,
            device,
            mean,
            std,
        ),
    )
    write_csv(TIMING_CSV_PATH, build_timing_rows(timings), TIMING_FIELDS)
    print_timing_summary(timings)
    print(
        f"[RESULT] best_epoch={result['best_epoch']} "
        f"val_acc={float(result['best_noisy_validation_accuracy']):.4f}"
    )


def run_with_file_logging() -> None:
    cifar.require_files(cifar.NOISE_ARTIFACT_PATHS, stage="Warmup")
    cifar.require_available_outputs(
        [WARMUP_CHECKPOINT_PATH, WARMUP_CSV_PATH, TIMING_CSV_PATH, RUN_LOG_PATH],
        overwrite=OVERWRITE,
        stage="Warmup",
    )
    cifar.run_with_log(RUN_LOG_PATH, main)


if __name__ == "__main__":
    run_with_file_logging()
