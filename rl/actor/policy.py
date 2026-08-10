from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from setting.config import PolicyConfig


@dataclass(frozen=True, slots=True)
class PolicyStep:
    correction_probabilities: Tensor
    actions: Tensor
    log_probabilities: Tensor
    next_labels: Tensor


@dataclass(frozen=True, slots=True)
class CorrectionResult:
    correction_probabilities: Tensor
    actions: Tensor
    corrected_labels: Tensor


def initialize_label_state(labels: Tensor, num_classes: int) -> Tensor:
    """Convert integer class indices into the paper's soft-label state."""
    if labels.ndim != 1:
        raise ValueError(f"labels must be one-dimensional, got {labels.shape}.")
    if labels.numel() == 0:
        raise ValueError("labels must not be empty.")
    if num_classes <= 1:
        raise ValueError("num_classes must be greater than one.")
    if labels.is_floating_point() or labels.is_complex():
        raise TypeError("labels must contain integer class indices.")

    labels = labels.to(dtype=torch.long)
    if torch.any(labels < 0) or torch.any(labels >= num_classes):
        raise ValueError(f"labels must be in [0, {num_classes}).")
    return F.one_hot(labels, num_classes=num_classes).to(torch.float32)


def _validate_policy_batch(
    query_embeddings: Tensor,
    neighbor_embeddings: Tensor,
    current_labels: Tensor,
    neighbor_labels: Tensor,
) -> None:
    if query_embeddings.ndim != 2:
        raise ValueError(
            f"query_embeddings must be [B, D], got {query_embeddings.shape}."
        )
    if neighbor_embeddings.ndim != 3:
        raise ValueError(
            "neighbor_embeddings must be [B, K, D], got "
            f"{neighbor_embeddings.shape}."
        )
    if current_labels.ndim != 2:
        raise ValueError(
            f"current_labels must be [B, C], got {current_labels.shape}."
        )
    if neighbor_labels.ndim != 3:
        raise ValueError(
            f"neighbor_labels must be [B, K, C], got {neighbor_labels.shape}."
        )

    batch_size, feature_dim = query_embeddings.shape
    neighbor_batch, neighbor_count, neighbor_dim = neighbor_embeddings.shape
    label_batch, num_classes = current_labels.shape
    neighbor_label_batch, label_neighbor_count, label_classes = (
        neighbor_labels.shape
    )
    if batch_size == 0 or neighbor_count == 0:
        raise ValueError("Policy batches and neighbor sets must not be empty.")
    if num_classes <= 1:
        raise ValueError("Policy labels must contain at least two classes.")
    if (
        neighbor_batch != batch_size
        or label_batch != batch_size
        or neighbor_label_batch != batch_size
        or neighbor_dim != feature_dim
        or label_neighbor_count != neighbor_count
        or label_classes != num_classes
    ):
        raise ValueError("Policy input dimensions do not align.")
    devices = {
        query_embeddings.device,
        neighbor_embeddings.device,
        current_labels.device,
        neighbor_labels.device,
    }
    if len(devices) != 1:
        raise ValueError("All policy inputs must be on the same device.")
    if not query_embeddings.is_floating_point():
        raise TypeError("Policy embeddings must be floating-point tensors.")
    if (
        not current_labels.is_floating_point()
        or not neighbor_labels.is_floating_point()
    ):
        raise TypeError("Policy labels must be floating-point tensors.")


class LabelCorrectionPolicy(nn.Module):
    """Vectorized RLNLC policy equations (1)-(4).

    Trainable actor parameters live in the feature extractor that produces the
    embeddings. KNN indices are discrete, while cosine attention remains
    differentiable with respect to the selected embeddings.
    """

    def __init__(self, config: PolicyConfig) -> None:
        super().__init__()
        if config.temperature <= 0:
            raise ValueError("Policy temperature must be positive.")
        if config.correction_chunk_size <= 0:
            raise ValueError("correction_chunk_size must be positive.")
        self.temperature = config.temperature
        self.correction_chunk_size = config.correction_chunk_size

    def compute_attention(
        self,
        query_embeddings: Tensor,
        neighbor_embeddings: Tensor,
    ) -> Tensor:
        similarities = F.cosine_similarity(
            query_embeddings.float().unsqueeze(1),
            neighbor_embeddings.float(),
            dim=-1,
        )
        return torch.softmax(similarities / self.temperature, dim=1)

    @staticmethod
    def predict_neighbor_labels(
        attention_weights: Tensor,
        neighbor_labels: Tensor,
    ) -> Tensor:
        return torch.einsum(
            "bk,bkc->bc",
            attention_weights,
            neighbor_labels.float(),
        )

    @staticmethod
    def compute_correction_probabilities(
        predicted_labels: Tensor,
        current_labels: Tensor,
    ) -> Tensor:
        current_classes = current_labels.argmax(dim=1, keepdim=True)
        current_scores = predicted_labels.gather(1, current_classes)
        more_likely = predicted_labels > current_scores
        no_less_likely = predicted_labels >= current_scores
        numerator = (predicted_labels * more_likely).sum(dim=1)
        denominator = (predicted_labels * no_less_likely).sum(dim=1)
        probabilities = numerator / denominator.clamp_min(
            torch.finfo(predicted_labels.dtype).tiny
        )
        return probabilities.clamp(0.0, 1.0)

    @staticmethod
    def _sample_or_validate_actions(
        probabilities: Tensor,
        actions: Tensor | None,
    ) -> Tensor:
        if actions is None:
            return torch.bernoulli(probabilities).to(torch.bool)
        if actions.shape != probabilities.shape:
            raise ValueError(
                f"actions must have shape {probabilities.shape}, got {actions.shape}."
            )
        actions = actions.to(device=probabilities.device)
        if torch.any((actions != 0) & (actions != 1)):
            raise ValueError("actions must contain only zero or one.")
        return actions.to(torch.bool)

    def forward(
        self,
        query_embeddings: Tensor,
        neighbor_embeddings: Tensor,
        current_labels: Tensor,
        neighbor_labels: Tensor,
        *,
        actions: Tensor | None = None,
    ) -> PolicyStep:
        _validate_policy_batch(
            query_embeddings,
            neighbor_embeddings,
            current_labels,
            neighbor_labels,
        )
        current_labels = current_labels.float()
        attention_weights = self.compute_attention(
            query_embeddings,
            neighbor_embeddings,
        )
        predicted_labels = self.predict_neighbor_labels(
            attention_weights,
            neighbor_labels,
        )
        probabilities = self.compute_correction_probabilities(
            predicted_labels,
            current_labels,
        )
        sampled_actions = self._sample_or_validate_actions(
            probabilities,
            actions,
        )

        probability_epsilon = torch.finfo(probabilities.dtype).eps
        stable_probabilities = probabilities.clamp(
            min=probability_epsilon,
            max=1.0 - probability_epsilon,
        )
        log_probabilities = torch.where(
            sampled_actions,
            stable_probabilities.log(),
            torch.log1p(-stable_probabilities),
        )
        next_labels = torch.where(
            sampled_actions.unsqueeze(1),
            predicted_labels,
            current_labels,
        )
        return PolicyStep(
            correction_probabilities=probabilities,
            actions=sampled_actions,
            log_probabilities=log_probabilities,
            next_labels=next_labels,
        )

    @torch.inference_mode()
    def correct_all(
        self,
        embeddings: Tensor,
        labels: Tensor,
        neighbor_indices: Tensor,
    ) -> CorrectionResult:
        """Apply one simultaneous stochastic transition to the full state."""
        if embeddings.ndim != 2:
            raise ValueError(f"embeddings must be [N, D], got {embeddings.shape}.")
        if labels.ndim != 2:
            raise ValueError(f"labels must be [N, C], got {labels.shape}.")
        if neighbor_indices.ndim != 2:
            raise ValueError(
                f"neighbor_indices must be [N, K], got {neighbor_indices.shape}."
            )
        sample_count = embeddings.size(0)
        if sample_count == 0:
            raise ValueError("The policy state must not be empty.")
        if labels.size(0) != sample_count or neighbor_indices.size(0) != sample_count:
            raise ValueError("Embeddings, labels, and KNN graph must have equal N.")
        if embeddings.device != labels.device:
            raise ValueError("Embeddings and labels must be on the same device.")
        if neighbor_indices.dtype != torch.long:
            raise TypeError("neighbor_indices must use torch.long.")
        if torch.any(neighbor_indices < 0) or torch.any(
            neighbor_indices >= sample_count
        ):
            raise ValueError("neighbor_indices contains an out-of-range index.")

        device = embeddings.device
        probabilities = torch.empty(
            sample_count,
            dtype=torch.float32,
            device=device,
        )
        actions = torch.empty(sample_count, dtype=torch.bool, device=device)
        corrected_labels = torch.empty_like(labels, dtype=torch.float32)

        for start in range(0, sample_count, self.correction_chunk_size):
            end = min(start + self.correction_chunk_size, sample_count)
            indices = neighbor_indices[start:end].to(device=device)
            step = self(
                embeddings[start:end],
                embeddings[indices],
                labels[start:end],
                labels[indices],
            )
            probabilities[start:end] = step.correction_probabilities
            actions[start:end] = step.actions
            corrected_labels[start:end] = step.next_labels

        return CorrectionResult(
            correction_probabilities=probabilities,
            actions=actions,
            corrected_labels=corrected_labels,
        )
