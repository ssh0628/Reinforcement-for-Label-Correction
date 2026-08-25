"""Histogram critic and SARSA update used by CIFAR RLNLC."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.optim import SGD


def encode_consistency_histogram(scores: Tensor, num_bins: int) -> Tensor:
    """Encode equations (12)-(14) as a normalized histogram."""
    if num_bins < 2 or scores.ndim not in (1, 2) or scores.shape[-1] == 0:
        raise ValueError("Critic encoding requires non-empty [N] or [B,N] scores and at least two bins.")
    scores = scores.detach().float().clamp(0, 1)
    indices = torch.ceil(scores * num_bins).long().sub_(1).clamp_(0, num_bins - 1)
    sample_count = scores.shape[-1]
    if scores.ndim == 1:
        return torch.bincount(indices, minlength=num_bins).float() / sample_count
    offsets = torch.arange(scores.size(0), device=scores.device).unsqueeze(1) * num_bins
    counts = torch.bincount((indices + offsets).flatten(), minlength=scores.size(0) * num_bins)
    return counts.reshape(scores.size(0), num_bins).float() / sample_count


class StateActionCritic(nn.Module):
    """MLP Q-value estimator over the configured state encoding."""

    def __init__(
        self,
        num_bins: int,
        hidden_dims: tuple[int, ...],
        *,
        use_remaining_horizon: bool = True,
    ) -> None:
        super().__init__()
        self.num_bins = num_bins
        self.use_remaining_horizon = use_remaining_horizon
        layers: list[nn.Module] = []
        input_width = num_bins + int(use_remaining_horizon)
        for output_width in hidden_dims:
            layers.extend((nn.Linear(input_width, output_width), nn.ReLU()))
            input_width = output_width
        layers.append(nn.Linear(input_width, 1))
        self.value_head = nn.Sequential(*layers)

    def encode(self, consistency_scores: Tensor, remaining_horizon: float) -> Tensor:
        histogram = encode_consistency_histogram(consistency_scores, self.num_bins)
        if not self.use_remaining_horizon:
            return histogram
        horizon = histogram.new_full((*histogram.shape[:-1], 1), remaining_horizon)
        return torch.cat((histogram, horizon), dim=-1)

    def value_from_encoding(self, encoding: Tensor) -> Tensor:
        expected_width = self.num_bins + int(self.use_remaining_horizon)
        if encoding.shape[-1] != expected_width:
            raise ValueError(f"Critic encoding width must be {expected_width}.")
        return self.value_head(encoding).squeeze(-1)

    def forward(self, consistency_scores: Tensor, remaining_horizon: float) -> Tensor:
        return self.value_from_encoding(self.encode(consistency_scores, remaining_horizon))


def build_critic(
    num_bins: int,
    hidden_dims: tuple[int, ...],
    *,
    use_remaining_horizon: bool = True,
) -> StateActionCritic:
    return StateActionCritic(
        num_bins,
        hidden_dims,
        use_remaining_horizon=use_remaining_horizon,
    )


def build_critic_optimizer(
    critic: StateActionCritic, *, learning_rate: float, momentum: float, weight_decay: float
) -> SGD:
    return SGD(critic.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)


@dataclass(frozen=True, slots=True)
class TDOutput:
    target: Tensor
    error: Tensor
    loss: Tensor


@dataclass(frozen=True, slots=True)
class CriticUpdateOutput:
    loss: float
    current_q: float
    next_q: float
    target: float
    error: float


def sarsa_td_loss(
    current_q: Tensor, reward: Tensor, next_q: Tensor, *, discount_factor: float, terminal: bool = False
) -> TDOutput:
    if current_q.shape != reward.shape or current_q.shape != next_q.shape:
        raise ValueError("SARSA tensors must have equal shapes.")
    target = reward.detach() if terminal else reward.detach() + discount_factor * next_q.detach()
    error = target - current_q
    return TDOutput(target, error, 0.5 * error.square().mean())


@torch.no_grad()
def encode_state_action(
    critic: StateActionCritic, consistency: Tensor, remaining_horizon: float
) -> tuple[Tensor, Tensor]:
    if not 0.0 <= remaining_horizon <= 1.0:
        raise ValueError("remaining_horizon must be normalized to [0, 1].")
    encoding = critic.encode(consistency, remaining_horizon).detach()
    return encoding, critic.value_from_encoding(encoding)


def update_critic(
    critic: StateActionCritic,
    optimizer: torch.optim.Optimizer,
    encoding: Tensor,
    reward: Tensor,
    next_encoding: Tensor | None,
    *,
    discount_factor: float,
    terminal: bool = False,
) -> CriticUpdateOutput:
    optimizer.zero_grad(set_to_none=True)
    current_q = critic.value_from_encoding(encoding)
    if terminal:
        next_q = torch.zeros_like(current_q)
    else:
        if next_encoding is None:
            raise ValueError("Non-terminal critic updates require the next encoding.")
        next_q = critic.value_from_encoding(next_encoding)
    td = sarsa_td_loss(
        current_q, reward, next_q, discount_factor=discount_factor, terminal=terminal
    )
    td.loss.backward()
    optimizer.step()
    loss, current, following, target, error = torch.stack(
        (td.loss.detach(), current_q.detach(), next_q.detach(), td.target.detach(), td.error.detach())
    ).to(torch.float64).cpu().tolist()
    return CriticUpdateOutput(loss, current, following, target, error)
