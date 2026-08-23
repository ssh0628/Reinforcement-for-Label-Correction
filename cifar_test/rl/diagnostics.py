"""Reward sanity checks for fixed reference states and learned label states."""

from __future__ import annotations

import math

import torch
from torch import Tensor
from torch.nn import functional as F

from cifar_test.evaluate.metrics import correction_summary
from cifar_test.rl.reward import RLNLCReward, RewardOutput


@torch.inference_mode()
def build_reference_states(
    noisy_labels: Tensor,
    clean_labels: Tensor,
    noise_mask: Tensor,
    global_neighbors: Tensor,
    global_cosines: Tensor,
    *,
    num_classes: int,
    temperature: float,
) -> list[tuple[str, Tensor, Tensor]]:
    noisy_state = F.one_hot(noisy_labels, num_classes=num_classes).float()
    attention = torch.softmax(global_cosines.float() / temperature, dim=1)
    knn_prediction = torch.einsum("nk,nkc->nc", attention, noisy_state[global_neighbors])
    knn_actions = knn_prediction.argmax(dim=1).ne(noisy_labels)
    knn_state = torch.where(knn_actions[:, None], knn_prediction, noisy_state)
    clean_state = F.one_hot(clean_labels, num_classes=num_classes).float()

    collapsed_class = int(torch.bincount(noisy_labels, minlength=num_classes).argmax())
    collapsed_labels = torch.full_like(noisy_labels, collapsed_class)
    collapsed_state = F.one_hot(collapsed_labels, num_classes=num_classes).float()
    return [
        ("noisy", noisy_state, torch.zeros_like(noisy_labels, dtype=torch.bool)),
        ("knn_one_step", knn_state, knn_actions),
        ("clean_oracle", clean_state, noise_mask.bool()),
        ("single_class", collapsed_state, collapsed_labels.ne(noisy_labels)),
    ]


@torch.inference_mode()
def reward_diagnostic_row(
    *,
    epoch: int,
    state_name: str,
    labels: Tensor,
    clean_labels: Tensor,
    noisy_labels: Tensor,
    noise_mask: Tensor,
    reward_output: RewardOutput,
    num_classes: int,
    nla_weight: float,
) -> dict[str, object]:
    metrics = correction_summary(
        labels, clean_labels, noisy_labels, noise_mask,
        num_classes=num_classes, epoch=epoch, split=state_name,
    )
    hard_labels = labels.argmax(dim=1)
    distribution = labels.float().mean(dim=0).clamp_min(torch.finfo(torch.float32).tiny)
    class_entropy = -(distribution * distribution.log()).sum() / math.log(num_classes)
    lcr = float(reward_output.label_consistency)
    nla = float(reward_output.noisy_label_alignment)
    return {
        "epoch": epoch,
        "state": state_name,
        "accuracy": metrics["accuracy"],
        "correction_rate": metrics["correction_rate"],
        "correction_precision": metrics["correction_precision"],
        "false_correction_rate": metrics["false_correction_rate"],
        "noisy_recovery_rate": metrics["noisy_recovery_rate"],
        "clean_preservation_rate": metrics["clean_preservation_rate"],
        "active_classes": int(hard_labels.unique().numel()),
        "class_entropy": float(class_entropy),
        "lcr": lcr,
        "nla": nla,
        "log_reward": lcr + nla_weight * nla,
        "reward": float(reward_output.total_reward),
    }


@torch.inference_mode()
def evaluate_reference_states(
    reward_function: RLNLCReward,
    states: list[tuple[str, Tensor, Tensor]],
    fixed_embeddings: Tensor,
    global_neighbors: Tensor,
    global_cosines: Tensor,
    clean_labels: Tensor,
    noisy_labels: Tensor,
    noise_mask: Tensor,
    *,
    num_classes: int,
    nla_weight: float,
) -> list[dict[str, object]]:
    rows = []
    for state_name, labels, actions in states:
        output = reward_function(labels, actions, fixed_embeddings, global_neighbors, global_cosines)
        rows.append(
            reward_diagnostic_row(
                epoch=0, state_name=state_name, labels=labels, clean_labels=clean_labels,
                noisy_labels=noisy_labels, noise_mask=noise_mask, reward_output=output,
                num_classes=num_classes, nla_weight=nla_weight,
            )
        )
    return rows


def print_reward_diagnostics(rows: list[dict[str, object]]) -> None:
    for row in rows:
        print(
            f"[REWARD CHECK] epoch={row['epoch']} state={row['state']} "
            f"acc={float(row['accuracy']):.4f} reward={float(row['reward']):.8e} "
            f"lcr={float(row['lcr']):.6f} nla={float(row['nla']):.6f} "
            f"changed={float(row['correction_rate']):.4f} "
            f"recovery={float(row['noisy_recovery_rate']):.4f} "
            f"false={float(row['false_correction_rate']):.4f} "
            f"classes={row['active_classes']} entropy={float(row['class_entropy']):.4f}"
        )


def print_reference_ranking(rows: list[dict[str, object]]) -> None:
    ranking = sorted(rows, key=lambda row: float(row["reward"]), reverse=True)
    print("[REWARD RANK] " + " > ".join(f"{row['state']}({float(row['reward']):.6g})" for row in ranking))
