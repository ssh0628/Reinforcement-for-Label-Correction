"""Run the configured pipeline from warm-up through final evaluation."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from setting.config import CONFIG


ROOT = Path(__file__).resolve().parent
STAGES = (
    "cifar_warmup.py",
    "cifar_knn_quality.py",
    "cifar_rl.py",
    "cifar_correction.py",
    "cifar_finetuning.py",
    "cifar_evaluate.py",
)


def run_pipeline(start_from: str, dry_run: bool) -> None:
    start_index = STAGES.index(start_from)
    print(
        f"[EXPERIMENT] {CONFIG.output.experiment_name} "
        f"warmup={CONFIG.warmup.checkpoint_selection} "
        f"augmentation={CONFIG.augmentation.enabled} "
        f"rl_epochs={CONFIG.rl.epochs} terminal={CONFIG.rl.use_terminal_critic_update} "
        f"horizon={CONFIG.rl.use_remaining_horizon} "
        f"correction=last_actor finetune={CONFIG.finetune.initialization} "
        f"evaluate={CONFIG.finetune.evaluation_checkpoint}",
        flush=True,
    )
    for stage in STAGES[start_index:]:
        print(f"[STAGE] {stage}", flush=True)
        if not dry_run:
            subprocess.run([sys.executable, str(ROOT / stage)], cwd=ROOT, check=True)
    print(f"[COMPLETE] {CONFIG.output.experiment_name}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-from", choices=STAGES, default=STAGES[0])
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_pipeline(args.start_from, args.dry_run)


if __name__ == "__main__":
    main()
