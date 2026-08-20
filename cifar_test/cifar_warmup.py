"""Create the fixed ``resnet18_cifar10_sn40_warmup50`` warm-up artifact.

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


NOISY_LABELS_PATH = cifar.NOISY_LABELS_PATH
NOISE_MASK_PATH = cifar.NOISE_MASK_PATH
WARMUP_CHECKPOINT_PATH = cifar.WARMUP_CHECKPOINT_PATH
OUTPUT_DIR = cifar.CONFIG.warmup_log_dir

RUN_LOG_PATH = OUTPUT_DIR / "run.log"
WARMUP_CSV_PATH = OUTPUT_DIR / "warmup.csv"
TIMING_CSV_PATH = OUTPUT_DIR / "timing.csv"
OVERWRITE = cifar.CONFIG.runtime.overwrite_warmup


def _validate_input_artifacts() -> None:
    cifar.require_files(
        (NOISY_LABELS_PATH, NOISE_MASK_PATH),
        stage="Warmup",
    )


def _validate_output_destination(*, include_log: bool) -> None:
    paths = [WARMUP_CHECKPOINT_PATH, WARMUP_CSV_PATH, TIMING_CSV_PATH]
    if include_log:
        paths.append(RUN_LOG_PATH)
    cifar.require_available_outputs(
        paths,
        overwrite=OVERWRITE,
        stage="Warmup",
    )


def main() -> None:
    _validate_input_artifacts()
    _validate_output_destination(include_log=False)
    cifar.configure_engine()
    engine = cifar.engine
    engine.EXTERNAL_NOISY_LABELS_PATH = NOISY_LABELS_PATH
    engine.EXTERNAL_NOISE_MASK_PATH = NOISE_MASK_PATH

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    WARMUP_CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)

    device = engine.resolve_local_device()
    engine.seed_everything(cifar.SEED)
    torch.backends.cudnn.benchmark = engine.CUDNN_BENCHMARK
    torch.cuda.reset_peak_memory_stats()
    timings: dict[str, list[float]] = {}

    raw_images, clean_labels = engine.measure(
        "cifar10_load",
        device,
        timings,
        cifar.load_full_cifar10_train,
    )
    noisy_labels, noise_mask = engine.measure(
        "noise_artifact_load",
        device,
        timings,
        lambda: engine.load_noisy_label_artifacts(clean_labels),
    )
    evaluation_splits = engine.measure(
        "cifar10_eval_load",
        device,
        timings,
        cifar.load_cifar10_validation,
    )
    val_images, val_clean_labels = evaluation_splits["val"]
    val_noisy_labels, _ = engine.measure(
        "val_noise_injection",
        device,
        timings,
        lambda: engine.inject_stratified_symmetric_noise(
            val_clean_labels
        ),
    )

    engine.print_configuration(
        device,
        clean_labels,
        noisy_labels,
        noise_mask,
    )
    print(f"noisy_labels_path={NOISY_LABELS_PATH}")
    print(f"noise_mask_path={NOISE_MASK_PATH}")
    print(f"warmup_output={WARMUP_CHECKPOINT_PATH}")

    model = engine.measure(
        "model_init",
        device,
        timings,
        lambda: cifar.build_model().to(
            device=device,
            memory_format=(
                torch.channels_last
                if engine.USE_CHANNELS_LAST
                else torch.contiguous_format
            ),
        ),
    )
    mean = torch.tensor(
        cifar.CIFAR10_MEAN,
        device=device,
    ).reshape(1, 3, 1, 1)
    std = torch.tensor(
        cifar.CIFAR10_STD,
        device=device,
    ).reshape(1, 3, 1, 1)
    engine.measure(
        "kernel_warmup",
        device,
        timings,
        lambda: engine.warm_device_kernels(
            model,
            raw_images,
            device,
            mean,
            std,
        ),
    )
    result = engine.measure(
        "supervised_warmup",
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
    engine.write_csv(
        TIMING_CSV_PATH,
        engine.build_timing_rows(timings),
        engine.TIMING_FIELDS,
    )
    engine.print_timing_summary(timings)
    print(
        f"[OK] deployment={result['deployment_mode']} "
        f"deployment_epoch={result['deployment_epoch']} "
        f"best_epoch={result['best_epoch']} "
        "best_noisy_val_acc="
        f"{float(result['best_noisy_validation_accuracy']):.6f}"
    )
    print(f"[OK] checkpoint={WARMUP_CHECKPOINT_PATH}")


def run_with_file_logging() -> None:
    _validate_input_artifacts()
    _validate_output_destination(include_log=True)
    cifar.run_with_log(RUN_LOG_PATH, main)


if __name__ == "__main__":
    run_with_file_logging()
