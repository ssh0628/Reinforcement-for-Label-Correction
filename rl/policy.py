"""KNN label-correction policy used by the CIFAR RL experiment."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F


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


def _validate_batch(
    query_embeddings: Tensor, neighbor_embeddings: Tensor, current_labels: Tensor, neighbor_labels: Tensor
) -> None:
    if query_embeddings.ndim != 2 or neighbor_embeddings.ndim != 3:
        raise ValueError("Policy embeddings must be [B, D] and [B, K, D].")
    if current_labels.ndim != 2 or neighbor_labels.ndim != 3:
        raise ValueError("Policy labels must be [B, C] and [B, K, C].")
    batch_size, feature_dim = query_embeddings.shape
    if batch_size == 0 or neighbor_embeddings.size(1) == 0:
        raise ValueError("Policy inputs must not be empty.")
    expected = (
        neighbor_embeddings.size(0) == batch_size,
        neighbor_embeddings.size(2) == feature_dim,
        current_labels.size(0) == batch_size,
        neighbor_labels.size(0) == batch_size,
        neighbor_labels.size(1) == neighbor_embeddings.size(1),
        neighbor_labels.size(2) == current_labels.size(1),
    )
    if not all(expected):
        raise ValueError("Policy input dimensions do not align.")


class LabelCorrectionPolicy(nn.Module):
    """Vectorized implementation of RLNLC equations (1)-(4)."""

    def __init__(self, temperature: float, correction_chunk_size: int) -> None:
        super().__init__()
        if temperature <= 0 or correction_chunk_size <= 0:
            raise ValueError("Policy temperature and chunk size must be positive.")
        self.temperature = temperature
        self.correction_chunk_size = correction_chunk_size

    def forward(
        self,
        query_embeddings: Tensor,
        neighbor_embeddings: Tensor,
        current_labels: Tensor,
        neighbor_labels: Tensor,
        *,
        actions: Tensor | None = None,
    ) -> PolicyStep:
        _validate_batch(query_embeddings, neighbor_embeddings, current_labels, neighbor_labels)
        similarities = F.cosine_similarity(
            query_embeddings.float().unsqueeze(1), neighbor_embeddings.float(), dim=-1
        )
        attention = torch.softmax(similarities / self.temperature, dim=1)
        predicted_labels = torch.einsum("bk,bkc->bc", attention, neighbor_labels.float())
        current_labels = current_labels.float()
        current_scores = predicted_labels.gather(1, current_labels.argmax(dim=1, keepdim=True))
        numerator = (predicted_labels * predicted_labels.gt(current_scores)).sum(dim=1)
        denominator = (predicted_labels * predicted_labels.ge(current_scores)).sum(dim=1)
        probabilities = (numerator / denominator.clamp_min(torch.finfo(torch.float32).tiny)).clamp(0, 1)

        if actions is None:
            sampled_actions = torch.bernoulli(probabilities).to(torch.bool)
        else:
            if actions.shape != probabilities.shape:
                raise ValueError("Provided actions do not match the policy batch.")
            sampled_actions = actions.to(device=probabilities.device, dtype=torch.bool)
        epsilon = torch.finfo(probabilities.dtype).eps
        stable = probabilities.clamp(epsilon, 1.0 - epsilon)
        log_probabilities = torch.where(sampled_actions, stable.log(), torch.log1p(-stable))
        next_labels = torch.where(sampled_actions.unsqueeze(1), predicted_labels, current_labels)
        return PolicyStep(probabilities, sampled_actions, log_probabilities, next_labels)

    @torch.inference_mode()
    def correct_all(self, embeddings: Tensor, labels: Tensor, neighbors: Tensor) -> CorrectionResult:
        sample_count = embeddings.size(0)
        if embeddings.ndim != 2 or labels.ndim != 2 or neighbors.ndim != 2:
            raise ValueError("Correction inputs must be [N,D], [N,C], and [N,K].")
        if sample_count == 0 or labels.size(0) != sample_count or neighbors.size(0) != sample_count:
            raise ValueError("Correction inputs must share a non-empty sample dimension.")
        probabilities = torch.empty(sample_count, dtype=torch.float32, device=embeddings.device)
        actions = torch.empty(sample_count, dtype=torch.bool, device=embeddings.device)
        corrected_labels = torch.empty_like(labels, dtype=torch.float32)
        for start in range(0, sample_count, self.correction_chunk_size):
            end = min(start + self.correction_chunk_size, sample_count)
            indices = neighbors[start:end].to(device=embeddings.device)
            step = self(embeddings[start:end], embeddings[indices], labels[start:end], labels[indices])
            probabilities[start:end] = step.correction_probabilities
            actions[start:end] = step.actions
            corrected_labels[start:end] = step.next_labels
        return CorrectionResult(probabilities, actions, corrected_labels)

