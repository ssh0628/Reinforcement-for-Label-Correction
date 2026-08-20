"""Train the paper-style CIFAR-10 RLNLC actor and critic.

This stage consumes the shared noisy-label and warm-up artifacts, then saves
matching best/last actor and critic checkpoints.  The 25-step deployment pass
is intentionally handled by ``cifar_correction.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cifar_test import cifar_common as cifar


CONFIG = cifar.CONFIG
OUTPUT_DIR = CONFIG.rl_output_dir

RL_OUTPUT_PATHS = (
    OUTPUT_DIR / cifar.engine.RUN_LOG_FILENAME,
    OUTPUT_DIR / cifar.engine.TRAIN_CSV_FILENAME,
    OUTPUT_DIR / cifar.engine.TIMING_CSV_FILENAME,
    OUTPUT_DIR / cifar.engine.RUN_SUMMARY_CSV_FILENAME,
    CONFIG.actor_best_checkpoint_path,
    CONFIG.actor_last_checkpoint_path,
    CONFIG.critic_best_checkpoint_path,
    CONFIG.critic_last_checkpoint_path,
)


def _validate_input_artifacts() -> None:
    cifar.require_files(
        (
            cifar.NOISY_LABELS_PATH,
            cifar.NOISE_MASK_PATH,
            cifar.WARMUP_CHECKPOINT_PATH,
        ),
        stage="RL",
    )


def _validate_output_destination() -> None:
    cifar.require_available_outputs(
        RL_OUTPUT_PATHS,
        overwrite=CONFIG.runtime.overwrite_rl,
        stage="RL",
    )


def main() -> None:
    _validate_input_artifacts()
    _validate_output_destination()
    cifar.configure_engine()
    cifar.engine.run_with_file_logging()


if __name__ == "__main__":
    main()
