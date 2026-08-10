from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.optim import SGD

from setting.config import Config


CONSISTENCY_TOLERANCE = 1e-6


def encode_consistency_histogram(
    consistency_scores: Tensor,
    num_bins: int,
) -> Tensor:
    """Encode Eq. (12) consistency scores using the bins in Eqs. (13)-(14)."""
    if num_bins < 2:
        raise ValueError("num_bins must be at least two.")
    if consistency_scores.ndim not in (1, 2):
        raise ValueError("consistency_scores must be [N] or [B, N].")
    if consistency_scores.numel() == 0 or consistency_scores.shape[-1] == 0:
        raise ValueError("consistency_scores must not be empty.")
    if not consistency_scores.is_floating_point():
        raise TypeError("consistency_scores must be floating-point tensors.")
    if not torch.isfinite(consistency_scores).all():
        raise ValueError("consistency_scores contains NaN or infinity.")
    if torch.any(consistency_scores < 0) or torch.any(
        consistency_scores > 1.0 + CONSISTENCY_TOLERANCE
    ):
        raise ValueError("consistency_scores must be in [0, 1].")

    scores = (
        consistency_scores.detach()
        .to(dtype=torch.float32)
        .clone()
        .clamp_(0.0, 1.0)
    )
    bin_indices = (
        torch.ceil(scores * num_bins)
        .to(dtype=torch.long)
        .sub_(1)
        .clamp_(0, num_bins - 1)
    )
    sample_count = scores.shape[-1]

    if scores.ndim == 1:
        counts = torch.bincount(bin_indices, minlength=num_bins)
        return counts.to(torch.float32) / sample_count

    batch_size = scores.size(0)
    offsets = (
        torch.arange(batch_size, device=scores.device).unsqueeze(1) * num_bins
    )
    counts = torch.bincount(
        (bin_indices + offsets).flatten(),
        minlength=batch_size * num_bins,
    ).reshape(batch_size, num_bins)
    return counts.to(torch.float32) / sample_count


class StateActionCritic(nn.Module):
    """Linear Q_phi over the deterministic next-state histogram encoding."""

    def __init__(self, num_bins: int) -> None:
        super().__init__()
        if num_bins < 2:
            raise ValueError("num_bins must be at least two.")
        self.num_bins = num_bins
        self.value_head = nn.Linear(num_bins, 1)

    def encode(self, consistency_scores: Tensor) -> Tensor:
        return encode_consistency_histogram(
            consistency_scores,
            self.num_bins,
        )

    def value_from_encoding(self, state_encoding: Tensor) -> Tensor:
        if state_encoding.ndim not in (1, 2):
            raise ValueError("state_encoding must be [H] or [B, H].")
        if state_encoding.shape[-1] != self.num_bins:
            raise ValueError(
                "state_encoding last dimension must equal num_bins."
            )
        if not state_encoding.is_floating_point():
            raise TypeError("state_encoding must be floating-point.")
        if not torch.isfinite(state_encoding).all():
            raise ValueError("state_encoding contains NaN or infinity.")
        return self.value_head(state_encoding).squeeze(-1)

    def forward(self, consistency_scores: Tensor) -> Tensor:
        return self.value_from_encoding(self.encode(consistency_scores))


def build_critic(cfg: Config) -> StateActionCritic:
    return StateActionCritic(cfg.rl_train.critic_num_bins)


def build_critic_optimizer(
    critic: StateActionCritic,
    cfg: Config,
) -> SGD:
    if cfg.rl_train.critic_optimizer_name.lower() != "sgd":
        raise ValueError("The RL critic optimizer must be SGD.")
    return SGD(
        critic.parameters(),
        lr=cfg.rl_train.critic_lr,
        momentum=cfg.rl_train.critic_momentum,
        weight_decay=cfg.rl_train.critic_weight_decay,
    )


@dataclass(frozen=True, slots=True)
class TDOutput:
    target: Tensor
    error: Tensor
    loss: Tensor


def sarsa_td_loss(
    current_q: Tensor,
    reward: Tensor,
    next_q: Tensor,
    *,
    discount_factor: float,
    terminal: bool = False,
) -> TDOutput:
    """Compute the semi-gradient SARSA update in Eqs. (10)-(11)."""
    if current_q.shape != reward.shape or current_q.shape != next_q.shape:
        raise ValueError("current_q, reward, and next_q must have equal shapes.")
    if not all(tensor.is_floating_point() for tensor in (current_q, reward, next_q)):
        raise TypeError("SARSA tensors must be floating-point tensors.")
    devices = {current_q.device, reward.device, next_q.device}
    if len(devices) != 1:
        raise ValueError("SARSA tensors must share the same device.")
    if not 0 <= discount_factor <= 1:
        raise ValueError("discount_factor must be in [0, 1].")
    if not isinstance(terminal, bool):
        raise TypeError("terminal must be a boolean.")
    if not all(
        torch.isfinite(tensor).all()
        for tensor in (current_q, reward, next_q)
    ):
        raise ValueError("SARSA tensors contain NaN or infinity.")

    target = reward.detach()
    if not terminal:
        target = target + discount_factor * next_q.detach()
    error = target - current_q
    loss = 0.5 * error.square().mean()
    return TDOutput(
        target=target,
        error=error,
        loss=loss,
    )
