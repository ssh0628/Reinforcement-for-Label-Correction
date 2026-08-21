from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import torch
from torch import Tensor

from rl.actor.policy import initialize_label_state


STATE_VERSION = 1
PROBABILITY_TOLERANCE = 1e-5
REQUIRED_STATE_KEYS = frozenset({"version", "step", "noisy_labels", "current_labels"})


def _validate_state(noisy_labels: Tensor, current_labels: Tensor, step: int) -> None:
    if not isinstance(step, int) or isinstance(step, bool) or step < 0:
        raise ValueError("state step must be a non-negative integer.")
    if noisy_labels.ndim != 1 or noisy_labels.numel() == 0:
        raise ValueError("noisy_labels must be a non-empty [N] tensor.")
    if noisy_labels.dtype != torch.long:
        raise TypeError("noisy_labels must use torch.long.")
    if current_labels.ndim != 2:
        raise ValueError("current_labels must be a [N, C] tensor.")
    if current_labels.shape[0] != noisy_labels.numel():
        raise ValueError("noisy_labels and current_labels must have equal N.")
    if current_labels.shape[1] <= 1:
        raise ValueError("current_labels must contain at least two classes.")
    if current_labels.dtype != torch.float32:
        raise TypeError("current_labels must use torch.float32.")
    if current_labels.device != noisy_labels.device:
        raise ValueError("State labels must share the same device.")
    if torch.any(noisy_labels < 0) or torch.any(noisy_labels >= current_labels.shape[1]):
        raise ValueError("noisy_labels contains an out-of-range class index.")
    if not torch.isfinite(current_labels).all():
        raise ValueError("current_labels contains NaN or infinity.")
    if torch.any(current_labels < 0.0) or torch.any(current_labels > 1.0):
        raise ValueError("current_labels contains an invalid probability.")
    row_sums = current_labels.sum(dim=1)
    if not torch.allclose(row_sums, torch.ones_like(row_sums), rtol=0.0, atol=PROBABILITY_TOLERANCE):
        raise ValueError("Every current_labels row must sum to one.")


@dataclass(frozen=True, slots=True)
class LabelState:
    """Committed noisy-label state for one RL correction step."""

    noisy_labels: Tensor
    current_labels: Tensor
    step: int = 0

    def __post_init__(self) -> None:
        _validate_state(self.noisy_labels, self.current_labels, self.step)

    @classmethod
    def from_noisy_labels(
        cls, noisy_labels: Tensor, num_classes: int, *, device: torch.device | str | None = None
    ) -> LabelState:
        if not isinstance(noisy_labels, Tensor):
            raise TypeError("noisy_labels must be a tensor.")
        labels = noisy_labels
        if device is not None:
            labels = labels.to(device=torch.device(device))
        current_labels = initialize_label_state(labels, num_classes)
        return cls(
            noisy_labels=labels.to(dtype=torch.long).contiguous().clone(),
            current_labels=current_labels.contiguous().clone(),
            step=0,
        )

    @property
    def sample_count(self) -> int:
        return self.current_labels.size(0)

    @property
    def num_classes(self) -> int:
        return self.current_labels.size(1)

    @property
    def device(self) -> torch.device:
        return self.current_labels.device

    @property
    def hard_labels(self) -> Tensor:
        return self.current_labels.argmax(dim=1)

    @property
    def changed_mask(self) -> Tensor:
        return self.hard_labels.ne(self.noisy_labels)

    @property
    def correction_rate(self) -> float:
        return self.changed_mask.float().mean().item()

    def transition(self, next_labels: Tensor) -> LabelState:
        if not isinstance(next_labels, Tensor):
            raise TypeError("next_labels must be a tensor.")
        if next_labels.shape != self.current_labels.shape:
            raise ValueError(
                "next_labels shape mismatch: "
                f"{tuple(next_labels.shape)} != {tuple(self.current_labels.shape)}."
            )
        if next_labels.device != self.device:
            raise ValueError("next_labels must be on the state device.")
        committed_labels = next_labels.detach().to(dtype=torch.float32).contiguous().clone()
        return LabelState(noisy_labels=self.noisy_labels, current_labels=committed_labels, step=self.step + 1)

    def to(self, device: torch.device | str, *, non_blocking: bool = False) -> LabelState:
        target_device = torch.device(device)
        if target_device == self.device:
            return self
        return LabelState(
            noisy_labels=self.noisy_labels.to(device=target_device, non_blocking=non_blocking),
            current_labels=self.current_labels.to(device=target_device, non_blocking=non_blocking),
            step=self.step,
        )

    def state_dict(self) -> dict[str, object]:
        return {
            "version": STATE_VERSION,
            "step": self.step,
            "noisy_labels": self.noisy_labels.detach().cpu().clone(),
            "current_labels": self.current_labels.detach().cpu().clone(),
        }

    @classmethod
    def from_state_dict(
        cls, state_dict: Mapping[str, object], *, device: torch.device | str = "cpu"
    ) -> LabelState:
        missing = REQUIRED_STATE_KEYS.difference(state_dict)
        if missing:
            raise ValueError(f"Label state is missing fields: {sorted(missing)}.")
        if state_dict["version"] != STATE_VERSION:
            raise ValueError(f"Unsupported label state version: {state_dict['version']} != {STATE_VERSION}.")
        noisy_labels = state_dict["noisy_labels"]
        current_labels = state_dict["current_labels"]
        step = state_dict["step"]
        if not isinstance(noisy_labels, Tensor) or not isinstance(current_labels, Tensor):
            raise TypeError("Saved label state values must be tensors.")
        if not isinstance(step, int) or isinstance(step, bool):
            raise TypeError("Saved label state step must be an integer.")

        target_device = torch.device(device)
        return cls(
            noisy_labels=noisy_labels.to(target_device).contiguous().clone(),
            current_labels=current_labels.to(target_device).contiguous().clone(),
            step=step,
        )
