"""Diagnose reward and critic ranking on repeated actions from one fixed label state."""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F

from log.common import run_with_log, write_csv
from rl import engine
from rl.critic import StateActionCritic, encode_state_action
from rl.policy import LabelCorrectionPolicy
from rl.reward import RLNLCReward
from setting import data as cifar


CONFIG = cifar.CONFIG
SAMPLE_FIELDS = (
    "sample",
    "action_rate",
    "changed_rate",
    "accuracy",
    "correction_precision",
    "noisy_recovery_rate",
    "clean_preservation_rate",
    "lcr",
    "nla",
    "log_reward",
    "reward",
    "q_value",
    "discounted_return",
    "final_reward",
    "final_accuracy",
)
SUMMARY_FIELDS = (
    "checkpoint",
    "actor_epoch",
    "critic_epoch",
    "state",
    "state_epoch",
    "trajectory_step",
    "samples",
    "initial_accuracy",
    "mean_accuracy",
    "std_accuracy",
    "mean_final_accuracy",
    "std_final_accuracy",
    "mean_reward",
    "std_reward",
    "mean_q",
    "std_q",
    "pearson_reward_accuracy",
    "spearman_reward_accuracy",
    "pearson_q_reward",
    "spearman_q_reward",
    "pearson_q_return",
    "spearman_q_return",
    "pearson_return_final_accuracy",
    "spearman_return_final_accuracy",
    "pearson_q_accuracy",
    "spearman_q_accuracy",
    "best_accuracy",
    "best_accuracy_sample",
    "best_reward_accuracy",
    "best_reward_sample",
    "best_q_accuracy",
    "best_q_sample",
    "best_return_final_accuracy",
    "best_return_sample",
    "seconds",
)


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1)
        start = end
    return ranks


def _pearson(left: np.ndarray, right: np.ndarray) -> float:
    if left.size != right.size or left.size < 2:
        raise ValueError("Correlation inputs must have the same length of at least two.")
    centered_left = left.astype(np.float64) - left.mean()
    centered_right = right.astype(np.float64) - right.mean()
    denominator = math.sqrt(float(centered_left @ centered_left) * float(centered_right @ centered_right))
    return float(centered_left @ centered_right / denominator) if denominator else float("nan")


def _spearman(left: np.ndarray, right: np.ndarray) -> float:
    return _pearson(_average_ranks(left), _average_ranks(right))


def _checkpoint_paths(kind: str) -> tuple[Path, Path]:
    if kind == "last":
        return CONFIG.actor_last_checkpoint_path, CONFIG.critic_last_checkpoint_path
    if kind == "best":
        return CONFIG.actor_best_checkpoint_path, CONFIG.critic_best_checkpoint_path
    raise ValueError(f"Unsupported checkpoint kind: {kind!r}")


def _load_critic(path: Path, device: torch.device) -> tuple[StateActionCritic, dict[str, object]]:
    if not path.is_file():
        raise FileNotFoundError(f"Critic checkpoint not found: {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise TypeError("Critic checkpoint must contain a dictionary.")
    required = {"epoch", "num_bins", "hidden_dims", "remaining_horizon", "critic"}
    missing = required.difference(checkpoint)
    if missing:
        raise KeyError(f"Critic checkpoint is missing fields: {sorted(missing)}")
    critic = StateActionCritic(
        int(checkpoint["num_bins"]),
        tuple(int(width) for width in checkpoint["hidden_dims"]),
        use_remaining_horizon=bool(checkpoint["remaining_horizon"]),
    ).to(device)
    critic.load_state_dict(checkpoint["critic"], strict=True)
    critic.eval()
    return critic, checkpoint


def _sample_metrics(
    corrected_labels: Tensor,
    actions: Tensor,
    clean_labels: Tensor,
    noisy_labels: Tensor,
    noise_mask: Tensor,
) -> dict[str, float]:
    hard_labels = corrected_labels.argmax(dim=1)
    changed = hard_labels.ne(noisy_labels)
    correct_changes = changed & hard_labels.eq(clean_labels)
    clean_mask = ~noise_mask
    changed_count = changed.sum().clamp_min(1)
    noise_count = noise_mask.sum().clamp_min(1)
    clean_count = clean_mask.sum().clamp_min(1)
    values = torch.stack(
        (
            actions.float().mean(),
            changed.float().mean(),
            hard_labels.eq(clean_labels).float().mean(),
            correct_changes.sum() / changed_count,
            (noise_mask & hard_labels.eq(clean_labels)).sum() / noise_count,
            (clean_mask & hard_labels.eq(clean_labels)).sum() / clean_count,
        )
    ).to(torch.float64).cpu().tolist()
    names = (
        "action_rate",
        "changed_rate",
        "accuracy",
        "correction_precision",
        "noisy_recovery_rate",
        "clean_preservation_rate",
    )
    return dict(zip(names, values, strict=True))


def _build_summary(
    rows: list[dict[str, object]],
    *,
    args: argparse.Namespace,
    actor_epoch: int,
    critic_epoch: int,
    initial_accuracy: float,
    seconds: float,
) -> dict[str, object]:
    arrays = {
        name: np.asarray([float(row[name]) for row in rows], dtype=np.float64)
        for name in ("accuracy", "reward", "q_value", "discounted_return", "final_accuracy")
    }
    accuracy, reward, q_value = arrays["accuracy"], arrays["reward"], arrays["q_value"]
    discounted_return, final_accuracy = arrays["discounted_return"], arrays["final_accuracy"]
    best_accuracy = int(accuracy.argmax())
    best_reward = int(reward.argmax())
    best_q = int(q_value.argmax())
    best_return = int(discounted_return.argmax())
    return {
        "checkpoint": args.checkpoint,
        "actor_epoch": actor_epoch,
        "critic_epoch": critic_epoch,
        "state": args.state,
        "state_epoch": args.state_epoch if args.state == "randomized" else "",
        "trajectory_step": args.trajectory_step,
        "samples": args.samples,
        "initial_accuracy": initial_accuracy,
        "mean_accuracy": float(accuracy.mean()),
        "std_accuracy": float(accuracy.std()),
        "mean_final_accuracy": float(final_accuracy.mean()),
        "std_final_accuracy": float(final_accuracy.std()),
        "mean_reward": float(reward.mean()),
        "std_reward": float(reward.std()),
        "mean_q": float(q_value.mean()),
        "std_q": float(q_value.std()),
        "pearson_reward_accuracy": _pearson(reward, accuracy),
        "spearman_reward_accuracy": _spearman(reward, accuracy),
        "pearson_q_reward": _pearson(q_value, reward),
        "spearman_q_reward": _spearman(q_value, reward),
        "pearson_q_return": _pearson(q_value, discounted_return),
        "spearman_q_return": _spearman(q_value, discounted_return),
        "pearson_return_final_accuracy": _pearson(discounted_return, final_accuracy),
        "spearman_return_final_accuracy": _spearman(discounted_return, final_accuracy),
        "pearson_q_accuracy": _pearson(q_value, final_accuracy),
        "spearman_q_accuracy": _spearman(q_value, final_accuracy),
        "best_accuracy": float(accuracy[best_accuracy]),
        "best_accuracy_sample": int(rows[best_accuracy]["sample"]),
        "best_reward_accuracy": float(accuracy[best_reward]),
        "best_reward_sample": int(rows[best_reward]["sample"]),
        "best_q_accuracy": float(final_accuracy[best_q]),
        "best_q_sample": int(rows[best_q]["sample"]),
        "best_return_final_accuracy": float(final_accuracy[best_return]),
        "best_return_sample": int(rows[best_return]["sample"]),
        "seconds": seconds,
    }


def _print_summary(summary: dict[str, object]) -> None:
    print("\n[CORRELATION]")
    print(
        "reward_accuracy "
        f"pearson={float(summary['pearson_reward_accuracy']):.6f} "
        f"spearman={float(summary['spearman_reward_accuracy']):.6f}"
    )
    print(
        "q_reward "
        f"pearson={float(summary['pearson_q_reward']):.6f} "
        f"spearman={float(summary['spearman_q_reward']):.6f}"
    )
    print(
        "q_return "
        f"pearson={float(summary['pearson_q_return']):.6f} "
        f"spearman={float(summary['spearman_q_return']):.6f}"
    )
    print(
        "return_final_accuracy "
        f"pearson={float(summary['pearson_return_final_accuracy']):.6f} "
        f"spearman={float(summary['spearman_return_final_accuracy']):.6f}"
    )
    print(
        "q_final_accuracy "
        f"pearson={float(summary['pearson_q_accuracy']):.6f} "
        f"spearman={float(summary['spearman_q_accuracy']):.6f}"
    )
    print("\n[RESULT]")
    print(
        f"initial_accuracy={float(summary['initial_accuracy']):.6f} "
        f"sampled_accuracy={float(summary['mean_accuracy']):.6f}"
        f"±{float(summary['std_accuracy']):.6f}"
    )
    print(
        f"final_accuracy={float(summary['mean_final_accuracy']):.6f}"
        f"±{float(summary['std_final_accuracy']):.6f}"
    )
    print(
        f"best_accuracy={float(summary['best_accuracy']):.6f} "
        f"best_reward_accuracy={float(summary['best_reward_accuracy']):.6f} "
        f"best_q_accuracy={float(summary['best_q_accuracy']):.6f}"
    )
    print(
        f"best_return_final_accuracy={float(summary['best_return_final_accuracy']):.6f}"
    )


def run(args: argparse.Namespace, output_dir: Path) -> None:
    started = time.perf_counter()
    device = engine.initialize_cuda_runtime(args.seed)
    actor_path, critic_path = _checkpoint_paths(args.checkpoint)
    print(f"device={torch.cuda.get_device_name(device)} samples={args.samples} seed={args.seed}")
    print(f"actor={actor_path}")
    print(f"critic={critic_path}")
    print(f"state={args.state} trajectory_step={args.trajectory_step}")

    raw_images, clean_labels_cpu = cifar.load_selected_cifar10_train()
    noisy_labels_cpu, noise_mask_cpu = engine.load_noisy_label_artifacts(clean_labels_cpu)
    mean, std = engine.normalization_tensors(device)

    reward_model = cifar.build_model(device=device)
    engine.load_warmup_checkpoint(reward_model, cifar.WARMUP_CHECKPOINT_PATH, device)
    fixed_embeddings = engine.extract_all_embeddings(reward_model, raw_images, device, mean, std)
    global_neighbors, global_cosines = engine.build_global_graph(fixed_embeddings)
    del reward_model

    actor = cifar.build_model(device=device)
    actor_checkpoint = engine.restore_actor_checkpoint(actor, actor_path, device)
    critic, critic_checkpoint = _load_critic(critic_path, device)
    actor_epoch = int(actor_checkpoint["epoch"])
    critic_epoch = int(critic_checkpoint["epoch"])
    if actor_epoch != critic_epoch:
        raise ValueError(f"Actor/Critic epochs do not match: {actor_epoch} != {critic_epoch}")

    policy_embeddings = engine.extract_all_embeddings(actor, raw_images, device, mean, std)
    policy_neighbors = engine.build_neighbor_indices(policy_embeddings)
    policy = LabelCorrectionPolicy(CONFIG.knn.temperature, CONFIG.knn.correction_chunk_size).to(device)
    reward_function = RLNLCReward(
        nla_weight=CONFIG.rl.reward_nla_weight,
        temperature=CONFIG.knn.temperature,
        k=CONFIG.knn.k,
        query_chunk_size=CONFIG.knn.query_chunk_size,
        reference_chunk_size=CONFIG.knn.reference_chunk_size,
        state_chunk_size=CONFIG.knn.correction_chunk_size,
    ).to(device)

    clean_labels = clean_labels_cpu.to(device, non_blocking=True)
    noisy_labels = noisy_labels_cpu.to(device, non_blocking=True)
    noise_mask = noise_mask_cpu.to(device, non_blocking=True)
    if args.state == "randomized":
        label_state, _ = engine.initialize_randomized_label_state(
            noisy_labels,
            num_classes=engine.NUM_CLASSES,
            randomization_rate=CONFIG.rl.initial_state_randomization_rate,
            epoch=args.state_epoch,
        )
    else:
        label_state = F.one_hot(noisy_labels, num_classes=engine.NUM_CLASSES).to(torch.float32)
    initial_accuracy = float(label_state.argmax(dim=1).eq(clean_labels).float().mean())
    remaining_horizon = (
        CONFIG.rl.trajectory_length - args.trajectory_step
    ) / CONFIG.rl.trajectory_length

    rows: list[dict[str, object]] = []
    device_index = device.index if device.index is not None else 0
    with torch.random.fork_rng(devices=[device_index]), torch.inference_mode():
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        for sample in range(1, args.samples + 1):
            rollout_state = label_state
            discounted_return = 0.0
            first_metrics: dict[str, float] | None = None
            first_lcr = first_nla = first_reward = first_q = 0.0
            final_reward = 0.0
            rollout_steps = CONFIG.rl.trajectory_length - args.trajectory_step + 1
            for offset in range(rollout_steps):
                correction = engine.correct_from_embeddings(
                    policy, policy_embeddings, rollout_state, policy_neighbors
                )
                reward_output = reward_function(
                    correction.corrected_labels,
                    correction.actions,
                    fixed_embeddings,
                    global_neighbors,
                    global_cosines,
                )
                reward = float(reward_output.total_reward)
                discounted_return += CONFIG.rl.discount_factor**offset * reward
                final_reward = reward
                if offset == 0:
                    _, q_value = encode_state_action(
                        critic, reward_output.per_sample_consistency, remaining_horizon
                    )
                    first_metrics = _sample_metrics(
                        correction.corrected_labels,
                        correction.actions,
                        clean_labels,
                        noisy_labels,
                        noise_mask,
                    )
                    first_lcr = float(reward_output.label_consistency)
                    first_nla = float(reward_output.noisy_label_alignment)
                    first_reward = reward
                    first_q = float(q_value)
                rollout_state = correction.corrected_labels

            if first_metrics is None:
                raise RuntimeError("Q/reward diagnostic produced an empty rollout.")
            final_accuracy = float(rollout_state.argmax(dim=1).eq(clean_labels).float().mean())
            row = {
                "sample": sample,
                **first_metrics,
                "lcr": first_lcr,
                "nla": first_nla,
                "log_reward": first_lcr + CONFIG.rl.reward_nla_weight * first_nla,
                "reward": first_reward,
                "q_value": first_q,
                "discounted_return": discounted_return,
                "final_reward": final_reward,
                "final_accuracy": final_accuracy,
            }
            rows.append(row)
            print(
                f"[SAMPLE] {sample:03d}/{args.samples} "
                f"action={float(row['action_rate']):.4f} "
                f"accuracy={float(row['accuracy']):.4f} "
                f"reward={float(row['reward']):.8e} q={float(row['q_value']):.6f} "
                f"return={float(row['discounted_return']):.6f} "
                f"final_acc={float(row['final_accuracy']):.4f}"
            )

    engine.synchronize(device)
    elapsed = time.perf_counter() - started
    summary = _build_summary(
        rows,
        args=args,
        actor_epoch=actor_epoch,
        critic_epoch=critic_epoch,
        initial_accuracy=initial_accuracy,
        seconds=elapsed,
    )
    write_csv(output_dir / "samples.csv", rows, SAMPLE_FIELDS)
    write_csv(output_dir / "summary.csv", [summary], SUMMARY_FIELDS)
    _print_summary(summary)
    print(f"output={output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample repeated joint actions from one state and compare reward, Q, and GT accuracy."
    )
    parser.add_argument("--samples", type=int, default=64, help="Number of joint actions to sample.")
    parser.add_argument("--checkpoint", choices=("last", "best"), default="last")
    parser.add_argument("--state", choices=("noisy", "randomized"), default="noisy")
    parser.add_argument("--state-epoch", type=int, default=1)
    parser.add_argument("--trajectory-step", type=int, default=1)
    parser.add_argument("--seed", type=int, default=CONFIG.data.seed)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.samples < 2:
        parser.error("--samples must be at least 2.")
    if not 1 <= args.state_epoch <= CONFIG.rl.epochs:
        parser.error(f"--state-epoch must be in [1, {CONFIG.rl.epochs}].")
    if not 1 <= args.trajectory_step <= CONFIG.rl.trajectory_length:
        parser.error(f"--trajectory-step must be in [1, {CONFIG.rl.trajectory_length}].")
    return args


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or CONFIG.log_output_dir / "q_reward_test" / args.checkpoint
    run_with_log(output_dir / "run.log", lambda: run(args, output_dir))


if __name__ == "__main__":
    main()
