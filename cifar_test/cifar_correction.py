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
TIMING_CSV_PATH = OUTPUT_DIR / "timing.csv"
RUN_SUMMARY_PATH = OUTPUT_DIR / "run_summary.csv"

RUN_SUMMARY_FIELDS = (
    "samples",
    "actor_epoch",
    "steps",
    "noisy_accuracy",
    "best_step",
    "best_accuracy",
    "final_accuracy",
    "correction_rate",
    "correction_precision",
    "recovery_rate",
    "false_rate",
    "preservation_rate",
    "seconds",
    "labels",
)


def _output_paths(*, include_log: bool) -> list[Path]:
    paths = [
        CORRECTED_LABELS_PATH,
        CLEANING_CSV_PATH,
        CLEANING_SUMMARY_PATH,
        TIMING_CSV_PATH,
        RUN_SUMMARY_PATH,
    ]
    if include_log:
        paths.append(RUN_LOG_PATH)
    return paths


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    cifar.configure_engine()
    engine = cifar.engine
    device = engine.resolve_local_device()
    engine.seed_everything(cifar.SEED)
    torch.backends.cudnn.benchmark = engine.CUDNN_BENCHMARK
    torch.cuda.reset_peak_memory_stats()
    timings: engine.Timings = {}

    raw_images, clean_labels = engine.measure("data_load", device, timings, cifar.load_selected_cifar10_train)
    noisy_labels, noise_mask = engine.measure(
        "noise_load", device, timings, lambda: engine.load_noisy_label_artifacts(clean_labels)
    )
    model = engine.measure(
        "model_init",
        device,
        timings,
        lambda: cifar.build_model().to(
            device=device,
            memory_format=(torch.channels_last if engine.USE_CHANNELS_LAST else torch.contiguous_format),
        ),
    )
    checkpoint = engine.measure(
        "actor_load",
        device,
        timings,
        lambda: engine.restore_actor_checkpoint(model, ACTOR_CHECKPOINT_PATH, device),
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
    mean, std = engine.normalization_tensors(device)

    print(f"device={torch.cuda.get_device_name(device)} actor_epoch={actor_epoch} steps={TRAJECTORY_LENGTH}")
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
        checkpoint_epoch=actor_epoch,
        trajectory_length=TRAJECTORY_LENGTH,
    )
    initial_accuracy = float(noisy_labels.eq(clean_labels).float().mean())
    engine.write_csv(TIMING_CSV_PATH, engine.build_timing_rows(timings), engine.TIMING_FIELDS)
    engine.write_csv(
        RUN_SUMMARY_PATH,
        [
            {
                "samples": clean_labels.numel(),
                "actor_epoch": actor_epoch,
                "steps": TRAJECTORY_LENGTH,
                "noisy_accuracy": initial_accuracy,
                "best_step": summary["best_step"],
                "best_accuracy": summary["best_accuracy"],
                "final_accuracy": summary["accuracy"],
                "correction_rate": summary["correction_rate"],
                "correction_precision": summary["correction_precision"],
                "recovery_rate": summary["noisy_recovery_rate"],
                "false_rate": summary["false_correction_rate"],
                "preservation_rate": summary["clean_preservation_rate"],
                "seconds": summary["seconds"],
                "labels": str(CORRECTED_LABELS_PATH),
            }
        ],
        RUN_SUMMARY_FIELDS,
    )
    engine.print_timing_summary(timings)
    print(
        f"[RESULT] best_step={summary['best_step']} "
        f"best_accuracy={float(summary['best_accuracy']):.6f} "
        f"final_accuracy={float(summary['accuracy']):.6f}"
    )
    print(f"output={OUTPUT_DIR}")
    print("next=cifar_test/cifar_finetuning.py")


def run_with_file_logging() -> None:
    cifar.require_files((ACTOR_CHECKPOINT_PATH, *cifar.NOISE_ARTIFACT_PATHS), stage="Correction")
    cifar.require_available_outputs(_output_paths(include_log=True), overwrite=OVERWRITE, stage="Correction")
    cifar.run_with_log(RUN_LOG_PATH, main)


if __name__ == "__main__":
    run_with_file_logging()
