from __future__ import annotations

from rl.trainer import RLTrainingResult, build_rl_trainer
from setting.config import CFG, Config


def main(cfg: Config = CFG) -> RLTrainingResult:
    cfg.validate()
    trainer = build_rl_trainer(cfg)
    resume_path = cfg.rl_train.resume_checkpoint_path
    mode = "resume" if resume_path is not None else "new"
    print(f"device={trainer.device}")
    print(f"mode={mode} completed_epochs={trainer.completed_epochs}")
    if resume_path is not None:
        print(f"resume_checkpoint={resume_path}")
    print(f"global_knn_cache={trainer.cache.cache_path}")
    print(
        f"samples={len(trainer.dataset)} "
        f"epochs={cfg.rl_train.epochs} "
        f"trajectory_length={cfg.rl_train.trajectory_length}"
    )

    result = trainer.fit()
    print(f"[OK] RL checkpoint saved: {result.checkpoint_path}")
    return result


if __name__ == "__main__":
    main()
