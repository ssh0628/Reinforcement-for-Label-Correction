"""Create the paper-style cleaned-label artifact.

This stage always deploys the final RL actor. It is intentionally independent
from RL training so the same corrected labels can be reused.
"""

from __future__ import annotations

import time

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
from cifar_test.rl.policy import LabelCorrectionPolicy
from cifar_test.setting import data as cifar


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

CLEANING_FIELDS = ("step", "action_rate", "changed_rate", "accuracy")
CLEANING_SUMMARY_FIELDS = (
    "best_step",
    "best_accuracy",
    "final_accuracy",
    "correction_rate",
    "correction_precision",
    "noisy_recovery_rate",
    "false_correction_rate",
    "clean_preservation_rate",
    "seconds",
)
OUTPUT_PATHS = (
    CORRECTED_LABELS_PATH,
    CLEANING_CSV_PATH,
    CLEANING_SUMMARY_PATH,
    TIMING_CSV_PATH,
    RUN_SUMMARY_PATH,
    RUN_LOG_PATH,
)


def run_correction(
    model: nn.Module,
    policy: LabelCorrectionPolicy,
    raw_images: Tensor,
    clean_labels_cpu: Tensor,
    noisy_labels_cpu: Tensor,
    noise_mask_cpu: Tensor,
    device: torch.device,
    mean: Tensor,
    std: Tensor,
    timings: Timings,
    checkpoint_epoch: int,
) -> dict[str, object]:
    engine.synchronize(device)
    started = time.perf_counter()
    device_index = device.index if device.index is not None else 0
    with torch.random.fork_rng(devices=[device_index]):
        torch.manual_seed(cifar.SEED)
        torch.cuda.manual_seed_all(cifar.SEED)
        embeddings = measure(
            "final_features",
            device,
            timings,
            lambda: engine.extract_all_embeddings(model, raw_images, device, mean, std),
        )
        neighbors = measure(
            "final_knn", device, timings, lambda: engine.build_neighbor_indices(embeddings)
        )
        clean = clean_labels_cpu.to(device, non_blocking=True)
        noisy = noisy_labels_cpu.to(device, non_blocking=True)
        noise_mask = noise_mask_cpu.to(device, non_blocking=True)
        label_state = F.one_hot(noisy, num_classes=engine.NUM_CLASSES).to(torch.float32)
        history: list[dict[str, object]] = []
        action_count = 0

        for step in range(1, TRAJECTORY_LENGTH + 1):
            correction = measure(
                "final_correction",
                device,
                timings,
                lambda: policy.correct_all(embeddings, label_state, neighbors),
                step=step,
            )
            hard_labels = correction.corrected_labels.argmax(dim=1)
            actions, changed_rate, accuracy = (
                torch.stack(
                    (
                        correction.actions.sum(),
                        hard_labels.ne(noisy).float().mean(),
                        hard_labels.eq(clean).float().mean(),
                    )
                )
                .to(torch.float64)
                .cpu()
                .tolist()
            )
            action_count += int(actions)
            label_state = correction.corrected_labels
            row = {
                "step": step,
                "action_rate": int(actions) / clean.numel(),
                "changed_rate": changed_rate,
                "accuracy": accuracy,
            }
            history.append(row)
            print(
                f"[CLEAN] step={step}/{TRAJECTORY_LENGTH} "
                f"action={float(row['action_rate']):.4f} "
                f"changed={float(row['changed_rate']):.4f} "
                f"accuracy={float(row['accuracy']):.4f}"
            )

    engine.synchronize(device)
    summary = correction_summary(
        label_state,
        clean,
        noisy,
        noise_mask,
        num_classes=engine.NUM_CLASSES,
        epoch=checkpoint_epoch,
        split="train_cleaning",
        action_rate=action_count / (clean.numel() * TRAJECTORY_LENGTH),
        seconds=time.perf_counter() - started,
    )
    validate_soft_labels(label_state, clean.numel(), engine.NUM_CLASSES)
    CORRECTED_LABELS_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = CORRECTED_LABELS_PATH.with_suffix(f"{CORRECTED_LABELS_PATH.suffix}.tmp")
    try:
        with temporary_path.open("wb") as handle:
            np.save(handle, label_state.detach().float().cpu().numpy(), allow_pickle=False)
        temporary_path.replace(CORRECTED_LABELS_PATH)
    finally:
        temporary_path.unlink(missing_ok=True)

    best = max(history, key=lambda row: float(row["accuracy"]))
    summary.update(best_step=int(best["step"]), best_accuracy=float(best["accuracy"]))
    write_csv(CLEANING_CSV_PATH, history, CLEANING_FIELDS)
    write_csv(
        CLEANING_SUMMARY_PATH,
        [
            {
                "best_step": summary["best_step"],
                "best_accuracy": summary["best_accuracy"],
                "final_accuracy": summary["accuracy"],
                **{field: summary[field] for field in CLEANING_SUMMARY_FIELDS[3:]},
            }
        ],
        CLEANING_SUMMARY_FIELDS,
    )
    print(f"[CLEAN] corrected_labels={CORRECTED_LABELS_PATH}")
    return summary


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    device = engine.resolve_local_device()
    engine.seed_everything(cifar.SEED)
    torch.backends.cudnn.benchmark = engine.CUDNN_BENCHMARK
    torch.cuda.reset_peak_memory_stats()
    timings: Timings = {}

    raw_images, clean_labels = measure("data_load", device, timings, cifar.load_selected_cifar10_train)
    noisy_labels, noise_mask = measure(
        "noise_load", device, timings, lambda: engine.load_noisy_label_artifacts(clean_labels)
    )
    model = measure(
        "model_init",
        device,
        timings,
        lambda: cifar.build_model().to(
            device=device,
            memory_format=(torch.channels_last if engine.USE_CHANNELS_LAST else torch.contiguous_format),
        ),
    )
    checkpoint = measure(
        "actor_load",
        device,
        timings,
        lambda: engine.restore_actor_checkpoint(model, ACTOR_CHECKPOINT_PATH, device),
    )
    actor_epoch = int(checkpoint["epoch"])
    if not 1 <= actor_epoch <= CONFIG.rl.epochs:
        raise ValueError(
            f"Actor checkpoint epoch must be between 1 and {CONFIG.rl.epochs}, "
            f"got {actor_epoch}."
        )
    if actor_epoch < CONFIG.rl.epochs:
        print(
            "[CLEAN] using intermediate actor checkpoint: "
            f"epoch={actor_epoch}/{CONFIG.rl.epochs}"
        )
    del checkpoint

    policy = LabelCorrectionPolicy(engine.TEMPERATURE, engine.CORRECTION_CHUNK_SIZE).to(device)
    mean, std = engine.normalization_tensors(device)

    print(f"device={torch.cuda.get_device_name(device)} actor_epoch={actor_epoch} steps={TRAJECTORY_LENGTH}")
    summary = run_correction(
        model,
        policy,
        raw_images,
        clean_labels,
        noisy_labels,
        noise_mask,
        device,
        mean,
        std,
        timings,
        actor_epoch,
    )
    initial_accuracy = float(noisy_labels.eq(clean_labels).float().mean())
    write_csv(TIMING_CSV_PATH, build_timing_rows(timings), TIMING_FIELDS)
    write_csv(
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
    print_timing_summary(timings)
    print(
        f"[RESULT] best_step={summary['best_step']} "
        f"best_accuracy={float(summary['best_accuracy']):.6f} "
        f"final_accuracy={float(summary['accuracy']):.6f}"
    )
    print(f"output={OUTPUT_DIR}")
    print("next=cifar_test/cifar_finetuning.py")


def run_with_file_logging() -> None:
    cifar.require_files((ACTOR_CHECKPOINT_PATH, *cifar.NOISE_ARTIFACT_PATHS), stage="Correction")
    cifar.require_available_outputs(OUTPUT_PATHS, overwrite=OVERWRITE, stage="Correction")
    run_with_log(RUN_LOG_PATH, main)


if __name__ == "__main__":
    run_with_file_logging()
