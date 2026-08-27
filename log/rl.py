"""RL-stage filenames and compact CSV schemas."""

RUN_LOG_FILENAME = "run.log"
TRAIN_CSV_FILENAME = "train.csv"
TIMING_CSV_FILENAME = "timing.csv"
RUN_SUMMARY_CSV_FILENAME = "run_summary.csv"

SUMMARY_FIELDS = (
    "epoch", "split", "loss", "accuracy", "correction_rate", "correction_precision",
    "false_correction_rate", "noisy_recovery_rate", "clean_preservation_rate", "action_rate",
    "reward", "actor_loss", "critic_loss", "seconds",
)

RUN_SUMMARY_FIELDS = (
    "dataset", "model", "samples", "noise_rate", "seed", "actor_batch_size",
    "actor_optimizer_steps_per_rl_step", "remaining_horizon", "terminal_update",
    "warmup_epoch", "epochs", "steps", "k",
    "actor_lr", "critic_optimizer", "critic_lr", "critic_momentum", "critic_weight_decay",
    "critic_lr_decay", "lr_decay_epoch", "lr_decay_factor", "critic_hidden_dims", "best_epoch",
    "best_val_accuracy", "best_val_loss", "last_val_accuracy", "last_val_loss", "total_seconds",
    "mean_epoch_seconds", "gpu_memory_gib", "actor_last", "critic_last",
)
