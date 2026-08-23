"""Train the paper-style CIFAR-10 RLNLC actor and critic.

This stage consumes the shared noisy-label and warm-up artifacts, then saves
matching best/last actor and critic checkpoints.  The 25-step deployment pass
is intentionally handled by ``cifar_correction.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cifar_test.log import rl as rl_log
from cifar_test.setting import data as cifar


CONFIG = cifar.CONFIG
OUTPUT_DIR = CONFIG.rl_output_dir

RL_OUTPUT_PATHS = (
    OUTPUT_DIR / rl_log.RUN_LOG_FILENAME,
    OUTPUT_DIR / rl_log.TRAIN_CSV_FILENAME,
    OUTPUT_DIR / rl_log.TIMING_CSV_FILENAME,
    OUTPUT_DIR / rl_log.RUN_SUMMARY_CSV_FILENAME,
    *(
        (OUTPUT_DIR / rl_log.CHANGE_DIAGNOSTICS_CSV_FILENAME,)
        if CONFIG.rl.record_change_diagnostics
        else ()
    ),
    *(
        (OUTPUT_DIR / rl_log.REWARD_DIAGNOSTICS_CSV_FILENAME,)
        if CONFIG.rl.record_reward_diagnostics
        else ()
    ),
    CONFIG.actor_best_checkpoint_path,
    CONFIG.actor_last_checkpoint_path,
    CONFIG.critic_best_checkpoint_path,
    CONFIG.critic_last_checkpoint_path,
)


def main() -> None:
    cifar.require_files((*cifar.NOISE_ARTIFACT_PATHS, cifar.WARMUP_CHECKPOINT_PATH), stage="RL")
    cifar.require_available_outputs(RL_OUTPUT_PATHS, overwrite=CONFIG.runtime.overwrite_rl, stage="RL")
    cifar.configure_engine()
    cifar.engine.run_with_file_logging()


if __name__ == "__main__":
    main()
