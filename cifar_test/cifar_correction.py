"""Create the paper-style 25-step cleaned-label artifact.

This stage always deploys the final RL actor. It is intentionally independent
from RL training so the same corrected labels can be reused by every
fine-tuning initialization ablation.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cifar_test import cifar_common as cifar
from rl.actor.policy import LabelCorrectionPolicy


CONFIG = cifar.CONFIG
ACTOR_CHECKPOINT_PATH = CONFIG.correction_actor_checkpoint_path
OUTPUT_DIR = CONFIG.correction_output_dir
CORRECTED_LABELS_PATH = CONFIG.corrected_labels_path
TRAJECTORY_LENGTH = CONFIG.correction.trajectory_length
OVERWRITE = CONFIG.runtime.overwrite_correction

RUN_LOG_PATH = OUTPUT_DIR / "run.log"
CLEANING_CSV_PATH = OUTPUT_DIR / "cleaning.csv"
CLEANING_SUMMARY_PATH = OUTPUT_DIR / "cleaning_summary.csv"
CLEANING_PER_CLASS_PATH = OUTPUT_DIR / "cleaning_per_class.csv"
TIMING_CSV_PATH = OUTPUT_DIR / "timing.csv"
RUN_SUMMARY_PATH = OUTPUT_DIR / "run_summary.csv"

RUN_SUMMARY_FIELDS = (
    "dataset",
    "model_name",
    "seed",
    "noise_rate",
    "samples",
    "actor_checkpoint",
    "actor_epoch",
    "trajectory_length",
    "initial_noisy_label_accuracy",
    "best_cleaning_step",
    "best_cleaning_accuracy",
    "final_cleaning_accuracy",
    "final_cleaning_macro_f1",
    "correction_count",
    "correction_rate",
    "correction_precision",
    "noisy_recovery_rate",
    "false_correction_rate",
    "clean_preservation_rate",
    "elapsed_seconds",
    "peak_cuda_allocated_gib",
    "peak_cuda_reserved_gib",
    "corrected_labels_path",
)


def _output_paths(*, include_log: bool) -> list[Path]:
    paths = [
        CORRECTED_LABELS_PATH,
        CLEANING_CSV_PATH,
        CLEANING_SUMMARY_PATH,
        CLEANING_PER_CLASS_PATH,
        TIMING_CSV_PATH,
        RUN_SUMMARY_PATH,
    ]
    if include_log:
        paths.append(RUN_LOG_PATH)
    return paths


def _validate_input_artifacts() -> None:
    cifar.require_files(
        (
            ACTOR_CHECKPOINT_PATH,
            cifar.NOISY_LABELS_PATH,
            cifar.NOISE_MASK_PATH,
        ),
        stage="Correction",
    )


def _validate_output_destination(*, include_log: bool) -> None:
    cifar.require_available_outputs(
        _output_paths(include_log=include_log),
        overwrite=OVERWRITE,
        stage="Correction",
    )


def main() -> None:
    if TRAJECTORY_LENGTH <= 0:
        raise ValueError("Correction trajectory length must be positive.")
    _validate_input_artifacts()
    _validate_output_destination(include_log=False)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    cifar.configure_engine()
    engine = cifar.engine
    engine.CLEANING_TRAJECTORY_LENGTH = TRAJECTORY_LENGTH
    device = engine.resolve_local_device()
    engine.seed_everything(cifar.SEED)
    torch.backends.cudnn.benchmark = engine.CUDNN_BENCHMARK
    torch.cuda.reset_peak_memory_stats()
    timings: engine.Timings = {}

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
    checkpoint = engine.measure(
        "actor_checkpoint_load",
        device,
        timings,
        lambda: engine.restore_actor_checkpoint(
            model,
            ACTOR_CHECKPOINT_PATH,
            device,
        ),
    )
    actor_epoch = int(checkpoint["epoch"])
    if actor_epoch != CONFIG.rl.epochs:
        raise ValueError(
            "Correction must use the final RL actor: "
            f"checkpoint epoch {actor_epoch} != configured epoch "
            f"{CONFIG.rl.epochs}."
        )
    del checkpoint

    cfg = engine.build_engine_config()
    policy = LabelCorrectionPolicy(cfg.policy).to(device)
    mean = torch.tensor(
        cifar.CIFAR10_MEAN,
        device=device,
    ).reshape(1, 3, 1, 1)
    std = torch.tensor(
        cifar.CIFAR10_STD,
        device=device,
    ).reshape(1, 3, 1, 1)

    print(f"device={device} ({torch.cuda.get_device_name(device)})")
    print(f"actor_checkpoint={ACTOR_CHECKPOINT_PATH}")
    print(f"actor_epoch={actor_epoch}")
    print(f"trajectory_length={TRAJECTORY_LENGTH}")
    summary = engine.clean_full_training_labels(
        model=model,
        policy=policy,
        raw_images=raw_images,
        clean_labels_cpu=clean_labels,
        noisy_labels_cpu=noisy_labels,
        noise_mask_cpu=noise_mask,
        device=device,
        mean=mean,
        std=std,
        timings=timings,
        corrected_labels_path=CORRECTED_LABELS_PATH,
        cleaning_csv_path=CLEANING_CSV_PATH,
        cleaning_summary_path=CLEANING_SUMMARY_PATH,
        cleaning_per_class_path=CLEANING_PER_CLASS_PATH,
        checkpoint_epoch=actor_epoch,
    )
    initial_accuracy = float(
        noisy_labels.eq(clean_labels).float().mean()
    )
    engine.write_csv(
        TIMING_CSV_PATH,
        engine.build_timing_rows(timings),
        engine.TIMING_FIELDS,
    )
    engine.write_csv(
        RUN_SUMMARY_PATH,
        [
            {
                "dataset": engine.DATASET_NAME,
                "model_name": cifar.MODEL_NAME,
                "seed": cifar.SEED,
                "noise_rate": CONFIG.data.noise_rate,
                "samples": clean_labels.numel(),
                "actor_checkpoint": str(ACTOR_CHECKPOINT_PATH),
                "actor_epoch": actor_epoch,
                "trajectory_length": TRAJECTORY_LENGTH,
                "initial_noisy_label_accuracy": initial_accuracy,
                "best_cleaning_step": summary["best_step"],
                "best_cleaning_accuracy": summary["best_accuracy"],
                "final_cleaning_accuracy": summary["accuracy"],
                "final_cleaning_macro_f1": summary["macro_f1"],
                "correction_count": summary["correction_count"],
                "correction_rate": summary["correction_rate"],
                "correction_precision": summary["correction_precision"],
                "noisy_recovery_rate": summary["noisy_recovery_rate"],
                "false_correction_rate": summary["false_correction_rate"],
                "clean_preservation_rate": summary[
                    "clean_preservation_rate"
                ],
                "elapsed_seconds": summary["elapsed_seconds"],
                "peak_cuda_allocated_gib": (
                    torch.cuda.max_memory_allocated() / 1024**3
                ),
                "peak_cuda_reserved_gib": (
                    torch.cuda.max_memory_reserved() / 1024**3
                ),
                "corrected_labels_path": str(CORRECTED_LABELS_PATH),
            }
        ],
        RUN_SUMMARY_FIELDS,
    )
    engine.print_timing_summary(timings)
    print(
        f"[RESULT] best_step={summary['best_step']} "
        f"best_accuracy={float(summary['best_accuracy']):.6f} "
        f"final_accuracy={float(summary['accuracy']):.6f} "
        f"final_macro_f1={float(summary['macro_f1']):.6f}"
    )
    print(f"corrected_labels={CORRECTED_LABELS_PATH}")
    print("next_stage=cifar_test/cifar_finetuning.py")


def run_with_file_logging() -> None:
    _validate_input_artifacts()
    _validate_output_destination(include_log=True)
    cifar.run_with_log(RUN_LOG_PATH, main)


if __name__ == "__main__":
    run_with_file_logging()
