"""Warm up one shared CIFAR-10 model from the saved 40% noisy labels.

Edit the three paths below when artifacts live outside the default workspace.
Both full and subset RL runs must load the checkpoint produced here.
"""

from __future__ import annotations

import sys
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import TYPE_CHECKING

import timm
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if TYPE_CHECKING:
    from cifar_test import cifar_test_rtx5080 as cifar
else:
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from cifar_test import cifar_test_rtx5080 as cifar


NOISY_LABELS_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "cifar10_shared"
    / "noise_40_seed0"
    / "train_noisy_labels.npy"
)
NOISE_MASK_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "cifar10_shared"
    / "noise_40_seed0"
    / "train_noise_mask.npy"
)
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "cifar10_shared" / "warmup"
WARMUP_CHECKPOINT_PATH = OUTPUT_DIR / "warmup_best.pt"

RUN_LOG_PATH = OUTPUT_DIR / "run.log"
WARMUP_CSV_PATH = OUTPUT_DIR / "warmup.csv"
TIMING_CSV_PATH = OUTPUT_DIR / "timing.csv"
OVERWRITE = False


def _validate_input_artifacts() -> None:
    missing = [
        path
        for path in (NOISY_LABELS_PATH, NOISE_MASK_PATH)
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(f"Warmup inputs not found: {missing}")


def _validate_output_destination(*, include_log: bool) -> None:
    paths = [WARMUP_CHECKPOINT_PATH, WARMUP_CSV_PATH, TIMING_CSV_PATH]
    if include_log:
        paths.append(RUN_LOG_PATH)
    existing = [path for path in paths if path.exists()]
    if existing and not OVERWRITE:
        raise FileExistsError(
            f"Warmup outputs already exist: {existing}. Set OVERWRITE=True "
            "only when replacement is intentional."
        )


def main() -> None:
    _validate_input_artifacts()
    _validate_output_destination(include_log=False)
    cifar.configure_benchmark()
    benchmark = cifar.benchmark
    benchmark.EXTERNAL_NOISY_LABELS_PATH = NOISY_LABELS_PATH
    benchmark.EXTERNAL_NOISE_MASK_PATH = NOISE_MASK_PATH

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    device = benchmark.resolve_local_device()
    benchmark.seed_everything(cifar.SEED)
    torch.backends.cudnn.benchmark = benchmark.CUDNN_BENCHMARK
    torch.cuda.reset_peak_memory_stats()
    timings: dict[str, list[float]] = {}

    raw_images, clean_labels = benchmark.measure(
        "cifar10_load",
        device,
        timings,
        cifar.load_full_cifar10_train,
    )
    noisy_labels, noise_mask = benchmark.measure(
        "noise_artifact_load",
        device,
        timings,
        lambda: benchmark.load_noisy_label_artifacts(clean_labels),
    )
    evaluation_splits = benchmark.measure(
        "cifar10_eval_load",
        device,
        timings,
        cifar.load_cifar10_validation_test,
    )
    val_images, val_clean_labels = evaluation_splits["val"]
    val_noisy_labels, _ = benchmark.measure(
        "val_noise_injection",
        device,
        timings,
        lambda: benchmark.inject_stratified_symmetric_noise(
            val_clean_labels
        ),
    )

    benchmark.print_configuration(
        device,
        clean_labels,
        noisy_labels,
        noise_mask,
    )
    print(f"noisy_labels_path={NOISY_LABELS_PATH}")
    print(f"noise_mask_path={NOISE_MASK_PATH}")
    print(f"warmup_output={WARMUP_CHECKPOINT_PATH}")

    model = benchmark.measure(
        "model_init",
        device,
        timings,
        lambda: timm.create_model(
            cifar.MODEL_NAME,
            pretrained=cifar.PRETRAINED,
            num_classes=cifar.NUM_CLASSES,
            drop_rate=cifar.DROP_RATE,
            drop_path_rate=cifar.DROP_PATH_RATE,
        ).to(
            device=device,
            memory_format=(
                torch.channels_last
                if benchmark.USE_CHANNELS_LAST
                else torch.contiguous_format
            ),
        ),
    )
    mean = torch.tensor(
        benchmark.IMAGENET_MEAN,
        device=device,
    ).reshape(1, 3, 1, 1)
    std = torch.tensor(
        benchmark.IMAGENET_STD,
        device=device,
    ).reshape(1, 3, 1, 1)
    benchmark.measure(
        "kernel_warmup",
        device,
        timings,
        lambda: benchmark.warm_device_kernels(
            model,
            raw_images,
            device,
            mean,
            std,
        ),
    )
    result = benchmark.measure(
        "supervised_warmup",
        device,
        timings,
        lambda: benchmark.train_supervised_warmup(
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
    benchmark.write_csv(
        TIMING_CSV_PATH,
        benchmark.build_timing_rows(timings),
        benchmark.TIMING_FIELDS,
    )
    benchmark.print_timing_summary(timings)
    print(
        f"[OK] best_epoch={result['best_epoch']} "
        "best_noisy_val_acc="
        f"{float(result['best_noisy_validation_accuracy']):.6f}"
    )
    print(f"[OK] checkpoint={WARMUP_CHECKPOINT_PATH}")


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
