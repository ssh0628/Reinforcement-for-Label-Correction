"""Create the fixed CIFAR-10 ResNet-18 warm-up artifact.

Both full and subset RL runs load the checkpoint produced here.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cifar_test import cifar_common as cifar


WARMUP_CHECKPOINT_PATH = cifar.WARMUP_CHECKPOINT_PATH
OUTPUT_DIR = cifar.CONFIG.warmup_log_dir

RUN_LOG_PATH = OUTPUT_DIR / "run.log"
WARMUP_CSV_PATH = OUTPUT_DIR / "warmup.csv"
TIMING_CSV_PATH = OUTPUT_DIR / "timing.csv"
OVERWRITE = cifar.CONFIG.runtime.overwrite_warmup


def main() -> None:
    cifar.configure_engine()
    engine = cifar.engine

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    WARMUP_CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)

    device = engine.resolve_local_device()
    engine.seed_everything(cifar.SEED)
    torch.backends.cudnn.benchmark = engine.CUDNN_BENCHMARK
    torch.cuda.reset_peak_memory_stats()
    timings: dict[str, list[float]] = {}

    raw_images, clean_labels = engine.measure("data_load", device, timings, cifar.load_selected_cifar10_train)
    noisy_labels, noise_mask = engine.measure(
        "noise_load", device, timings, lambda: engine.load_noisy_label_artifacts(clean_labels)
    )
    evaluation_splits = engine.measure("val_load", device, timings, cifar.load_cifar10_validation)
    val_images, val_clean_labels = evaluation_splits["val"]
    val_noisy_labels, _ = engine.measure(
        "val_noise", device, timings, lambda: engine.inject_stratified_symmetric_noise(val_clean_labels)
    )

    engine.print_configuration(device, clean_labels.numel())
    print(f"output={OUTPUT_DIR}")

    model = engine.measure(
        "model_init",
        device,
        timings,
        lambda: cifar.build_model().to(
            device=device,
            memory_format=(torch.channels_last if engine.USE_CHANNELS_LAST else torch.contiguous_format),
        ),
    )
    mean, std = engine.normalization_tensors(device)
    engine.measure(
        "gpu_warmup",
        device,
        timings,
        lambda: engine.warm_device_kernels(model, raw_images, device, mean, std),
    )
    result = engine.measure(
        "warmup_train",
        device,
        timings,
        lambda: engine.train_supervised_warmup(
            model,
            raw_images,
            noisy_labels,
            val_images,
            val_noisy_labels,
            val_clean_labels,
            device,
            mean,
            std,
            WARMUP_CSV_PATH,
            WARMUP_CHECKPOINT_PATH,
        ),
    )
    engine.write_csv(TIMING_CSV_PATH, engine.build_timing_rows(timings), engine.TIMING_FIELDS)
    engine.print_timing_summary(timings)
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
