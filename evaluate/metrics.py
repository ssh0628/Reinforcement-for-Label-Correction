"""Metrics for soft-label correction states."""

from __future__ import annotations

from typing import Callable

import torch
from torch import Tensor, nn


Preprocess = Callable[[Tensor, torch.device, Tensor, Tensor], Tensor]


def evaluate_classifier(
    model: nn.Module,
    images: Tensor,
    labels: Tensor,
    device: torch.device,
    mean: Tensor,
    std: Tensor,
    *,
    batch_size: int,
    use_amp: bool,
    amp_dtype: torch.dtype,
    preprocess: Preprocess,
) -> dict[str, float]:
    if batch_size <= 0 or labels.numel() == 0 or images.size(0) != labels.numel():
        raise ValueError("Classifier evaluation requires aligned non-empty inputs and a positive batch size.")
    model.eval()
    loss_sum = torch.zeros((), dtype=torch.float64, device=device)
    correct_count = torch.zeros((), dtype=torch.long, device=device)
    with torch.inference_mode():
        for start in range(0, labels.numel(), batch_size):
            end = min(start + batch_size, labels.numel())
            batch_images = preprocess(images[start:end], device, mean, std)
            targets = labels[start:end].to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                logits = model(batch_images)
                loss = nn.functional.cross_entropy(logits, targets, reduction="sum")
            correct_count += logits.argmax(dim=1).eq(targets).sum()
            loss_sum += loss.to(torch.float64)
    sample_count = labels.numel()
    return {
        "loss": float(loss_sum / sample_count),
        "accuracy": float(correct_count / sample_count),
    }


def _safe_ratio(numerator: Tensor, denominator: Tensor) -> Tensor:
    return numerator.float() / denominator.clamp_min(1).float()


def validate_soft_labels(soft_labels: Tensor, sample_count: int, num_classes: int) -> None:
    if soft_labels.shape != (sample_count, num_classes):
        raise ValueError(
            f"Soft labels must have shape ({sample_count}, {num_classes}), got {tuple(soft_labels.shape)}."
        )
    if not soft_labels.is_floating_point():
        raise TypeError("Soft labels must use a floating-point dtype.")
    if not torch.isfinite(soft_labels).all() or bool(soft_labels.lt(0).any()):
        raise ValueError("Soft labels must be finite and non-negative.")
    row_sums = soft_labels.sum(dim=1)
    if not torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5, rtol=1e-5):
        raise ValueError("Each soft-label row must sum to one.")


def correction_summary(
    soft_labels: Tensor,
    clean_labels: Tensor,
    initial_noisy_labels: Tensor,
    noise_mask: Tensor,
    *,
    num_classes: int,
    epoch: int,
    split: str,
    action_rate: float = 0.0,
    reward: float | None = None,
    actor_loss: float | None = None,
    critic_loss: float | None = None,
    seconds: float = 0.0,
) -> dict[str, object]:
    sample_count = clean_labels.numel()
    if soft_labels.shape != (sample_count, num_classes):
        raise ValueError("soft_labels must have shape [N, num_classes].")
    if initial_noisy_labels.shape != clean_labels.shape or noise_mask.shape != clean_labels.shape:
        raise ValueError("Metric label tensors must have equal [N] shapes.")

    hard_labels = soft_labels.argmax(dim=1)
    clean_probabilities = soft_labels.float().gather(1, clean_labels.unsqueeze(1)).squeeze(1)
    changed = hard_labels.ne(initial_noisy_labels)
    correct_correction = changed & hard_labels.eq(clean_labels)
    clean_mask = ~noise_mask
    correction_count = changed.sum()
    noise_count = noise_mask.sum()
    clean_count = clean_mask.sum()
    values = (
        torch.stack(
            (
                -clean_probabilities.clamp_min(1e-8).log().mean(),
                hard_labels.eq(clean_labels).float().mean(),
                correction_count.float() / sample_count,
                _safe_ratio(correct_correction.sum(), correction_count),
                _safe_ratio((clean_mask & changed).sum(), clean_count),
                _safe_ratio((noise_mask & hard_labels.eq(clean_labels)).sum(), noise_count),
                _safe_ratio((clean_mask & hard_labels.eq(clean_labels)).sum(), clean_count),
            )
        )
        .to(torch.float64)
        .cpu()
        .tolist()
    )
    loss, accuracy, correction_rate, precision, false_rate, recovery_rate, preservation_rate = values
    return {
        "epoch": epoch,
        "split": split,
        "loss": loss,
        "accuracy": accuracy,
        "correction_rate": correction_rate,
        "correction_precision": precision,
        "false_correction_rate": false_rate,
        "noisy_recovery_rate": recovery_rate,
        "clean_preservation_rate": preservation_rate,
        "action_rate": action_rate,
        "reward": reward,
        "actor_loss": actor_loss,
        "critic_loss": critic_loss,
        "seconds": seconds,
    }
