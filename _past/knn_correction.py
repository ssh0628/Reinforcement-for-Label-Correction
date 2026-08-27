"""Create a weighted exact-KNN soft-label baseline from the best warm-up model."""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from cifar_test.evaluate.metrics import correction_summary, validate_soft_labels
from cifar_test.log.common import (
    TIMING_FIELDS,
    Timings,
    build_timing_rows,
    measure,
    print_timing_summary,
    run_with_log,
    write_csv,
)
from cifar_test.rl import engine
from cifar_test.setting import data as cifar


CONFIG = cifar.CONFIG
OUTPUT_DIR = CONFIG.knn_correction_output_dir
LABELS_PATH = CONFIG.knn_corrected_labels_path
OVERWRITE = CONFIG.runtime.overwrite_knn_correction

RUN_LOG_PATH = OUTPUT_DIR / "run.log"
SUMMARY_CSV_PATH = OUTPUT_DIR / "summary.csv"
TIMING_CSV_PATH = OUTPUT_DIR / "timing.csv"
OUTPUT_PATHS = (LABELS_PATH, SUMMARY_CSV_PATH, TIMING_CSV_PATH, RUN_LOG_PATH)
SUMMARY_FIELDS = (
    "samples",
    "warmup_epoch",
    "k",
    "temperature",
    "noisy_accuracy",
    "weighted_knn_accuracy",
    "correction_rate",
    "correction_precision",
    "noisy_recovery_rate",
    "false_correction_rate",
    "clean_preservation_rate",
    "seconds",
    "labels",
)


@torch.inference_mode()
def weighted_knn_labels(
    noisy_labels: Tensor,
    neighbors: Tensor,
    neighbor_cosines: Tensor,
    *,
    num_classes: int,
    temperature: float,
    chunk_size: int,
) -> Tensor:
    """Return each query's cosine-weighted neighbor-label distribution."""
    if temperature <= 0 or chunk_size <= 0:
        raise ValueError("temperature and chunk_size must be positive.")
    one_hot = F.one_hot(noisy_labels, num_classes=num_classes).float()
    labels = torch.empty((noisy_labels.numel(), num_classes), device=noisy_labels.device)
    for start in range(0, noisy_labels.numel(), chunk_size):
        end = min(start + chunk_size, noisy_labels.numel())
        indices = neighbors[start:end]
        weights = torch.softmax(neighbor_cosines[start:end] / temperature, dim=1)
        labels[start:end] = torch.einsum("nk,nkc->nc", weights, one_hot[indices])
    return labels


def save_numpy(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    try:
        with temporary_path.open("wb") as handle:
            np.save(handle, array, allow_pickle=False)
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = engine.resolve_local_device()
    engine.seed_everything(cifar.SEED)
    torch.backends.cudnn.benchmark = engine.CUDNN_BENCHMARK
    torch.cuda.reset_peak_memory_stats()
    timings: Timings = {}
    started = time.perf_counter()

    raw_images, clean_labels_cpu = measure(
        "data_load", device, timings, cifar.load_selected_cifar10_train
    )
    noisy_labels_cpu, noise_mask_cpu = measure(
        "noise_load", device, timings, lambda: engine.load_noisy_label_artifacts(clean_labels_cpu)
    )
    model: nn.Module = measure(
        "model_init",
        device,
        timings,
        lambda: cifar.build_model().to(
            device=device,
            memory_format=(torch.channels_last if engine.USE_CHANNELS_LAST else torch.contiguous_format),
        ),
    )
    checkpoint = measure(
        "warmup_load",
        device,
        timings,
        lambda: engine.load_warmup_checkpoint(model, cifar.WARMUP_CHECKPOINT_PATH, device),
    )
    mean, std = engine.normalization_tensors(device)
    embeddings = measure(
        "feature_extraction",
        device,
        timings,
        lambda: engine.extract_all_embeddings(model, raw_images, device, mean, std),
    )
    neighbors, cosines = measure(
        "exact_knn", device, timings, lambda: engine.build_global_graph(embeddings)
    )
    clean_labels = clean_labels_cpu.to(device, non_blocking=True)
    noisy_labels = noisy_labels_cpu.to(device, non_blocking=True)
    noise_mask = noise_mask_cpu.to(device, non_blocking=True)
    soft_labels = measure(
        "weighted_label_sum",
        device,
        timings,
        lambda: weighted_knn_labels(
            noisy_labels,
            neighbors,
            cosines,
            num_classes=cifar.NUM_CLASSES,
            temperature=CONFIG.knn.temperature,
            chunk_size=CONFIG.knn.correction_chunk_size,
        ),
    )
    validate_soft_labels(soft_labels, clean_labels.numel(), cifar.NUM_CLASSES)
    elapsed = time.perf_counter() - started
    summary = correction_summary(
        soft_labels,
        clean_labels,
        noisy_labels,
        noise_mask,
        num_classes=cifar.NUM_CLASSES,
        epoch=int(checkpoint["deployment_epoch"]),
        split="weighted_knn",
        seconds=elapsed,
    )
    save_numpy(LABELS_PATH, soft_labels.float().cpu().numpy())
    row = {
        "samples": clean_labels.numel(),
        "warmup_epoch": checkpoint["deployment_epoch"],
        "k": CONFIG.knn.k,
        "temperature": CONFIG.knn.temperature,
        "noisy_accuracy": float(noisy_labels.eq(clean_labels).float().mean()),
        "weighted_knn_accuracy": summary["accuracy"],
        "correction_rate": summary["correction_rate"],
        "correction_precision": summary["correction_precision"],
        "noisy_recovery_rate": summary["noisy_recovery_rate"],
        "false_correction_rate": summary["false_correction_rate"],
        "clean_preservation_rate": summary["clean_preservation_rate"],
        "seconds": elapsed,
        "labels": str(LABELS_PATH),
    }
    write_csv(SUMMARY_CSV_PATH, [row], SUMMARY_FIELDS)
    write_csv(TIMING_CSV_PATH, build_timing_rows(timings), TIMING_FIELDS)
    print_timing_summary(timings)
    print(
        f"[RESULT] noisy_accuracy={float(row['noisy_accuracy']):.6f} "
        f"weighted_knn_accuracy={float(row['weighted_knn_accuracy']):.6f} "
        f"correction_precision={float(row['correction_precision']):.6f} "
        f"recovery={float(row['noisy_recovery_rate']):.6f} "
        f"preservation={float(row['clean_preservation_rate']):.6f}"
    )
    print(f"labels={LABELS_PATH}")
    print("next=set FineTuneConfig.corrected_label_source='knn', then run cifar_finetuning.py")


def run_with_file_logging() -> None:
    cifar.require_files(
        (cifar.WARMUP_CHECKPOINT_PATH, *cifar.NOISE_ARTIFACT_PATHS), stage="KNN correction"
    )
    cifar.require_available_outputs(OUTPUT_PATHS, overwrite=OVERWRITE, stage="KNN correction")
    run_with_log(RUN_LOG_PATH, main)


if __name__ == "__main__":
    run_with_file_logging()
