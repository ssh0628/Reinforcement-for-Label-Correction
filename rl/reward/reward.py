from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from rl.reward.clean_knn import build_exact_clean_knn
from setting.config import Config


@dataclass(frozen=True, slots=True)
class RewardOutput:
    label_consistency: Tensor
    noisy_label_alignment: Tensor
    total_reward: Tensor
    per_sample_consistency: Tensor


def soft_kl_divergence(target: Tensor, prediction: Tensor) -> Tensor:
    """Compute KL(target || prediction) for each soft-label row."""
    if target.shape != prediction.shape or target.ndim != 2:
        raise ValueError("target and prediction must have the same [B, C] shape.")
    target = target.float()
    prediction = prediction.float().clamp_min(torch.finfo(torch.float32).tiny)
    return F.kl_div(prediction.log(), target, reduction="none").sum(dim=1).clamp_min_(0)


class RLNLCReward(nn.Module):
    """Fixed-backbone LCR and NLA reward from RLNLC equations (5)-(7)."""

    def __init__(self, cfg: Config) -> None:
        super().__init__()
        if cfg.reward.nla_weight < 0:
            raise ValueError("reward.nla_weight must be non-negative.")
        self.nla_weight = cfg.reward.nla_weight
        self.temperature = cfg.policy.temperature
        self.k = cfg.global_knn.k
        self.query_chunk_size = cfg.global_knn.query_chunk_size
        self.reference_chunk_size = cfg.global_knn.reference_chunk_size
        self.state_chunk_size = cfg.policy.correction_chunk_size

    def _attention(self, cosine_similarities: Tensor) -> Tensor:
        return torch.softmax(cosine_similarities.float() / self.temperature, dim=1)

    @staticmethod
    def _neighbor_prediction(labels: Tensor, neighbor_indices: Tensor, attention_weights: Tensor) -> Tensor:
        return torch.einsum("bk,bkc->bc", attention_weights, labels[neighbor_indices].float())

    def _global_consistency(
        self, labels: Tensor, neighbor_indices: Tensor, cosine_similarities: Tensor
    ) -> tuple[Tensor, Tensor]:
        sample_count = labels.size(0)
        device = labels.device
        per_sample_kl = torch.empty(sample_count, dtype=torch.float32, device=device)
        for start in range(0, sample_count, self.state_chunk_size):
            end = min(start + self.state_chunk_size, sample_count)
            indices = neighbor_indices[start:end].to(device=device)
            attention = self._attention(cosine_similarities[start:end].to(device=device))
            prediction = self._neighbor_prediction(labels, indices, attention)
            per_sample_kl[start:end] = soft_kl_divergence(labels[start:end], prediction)
        return -per_sample_kl.mean(), torch.exp(-per_sample_kl)

    def _noisy_alignment(self, labels: Tensor, actions: Tensor, fixed_embeddings: Tensor) -> Tensor:
        clean_knn = build_exact_clean_knn(
            fixed_embeddings,
            actions,
            k=self.k,
            query_chunk_size=self.query_chunk_size,
            reference_chunk_size=self.reference_chunk_size,
        )
        noisy_count = clean_knn.noisy_indices.numel()
        if noisy_count == 0:
            return labels.new_zeros((), dtype=torch.float32)

        total_kl = labels.new_zeros((), dtype=torch.float32)
        for start in range(0, noisy_count, self.state_chunk_size):
            end = min(start + self.state_chunk_size, noisy_count)
            noisy_indices = clean_knn.noisy_indices[start:end]
            neighbor_indices = clean_knn.neighbor_indices[start:end]
            attention = self._attention(clean_knn.neighbor_cosine_similarities[start:end])
            prediction = self._neighbor_prediction(labels, neighbor_indices, attention)
            total_kl += soft_kl_divergence(labels[noisy_indices], prediction).sum()
        return -(total_kl / noisy_count)

    @torch.inference_mode()
    def forward(
        self,
        next_labels: Tensor,
        actions: Tensor,
        fixed_embeddings: Tensor,
        global_neighbor_indices: Tensor,
        global_neighbor_cosine_similarities: Tensor,
    ) -> RewardOutput:
        if next_labels.ndim != 2 or not next_labels.is_floating_point():
            raise ValueError("next_labels must be a floating-point [N, C] tensor.")
        sample_count = next_labels.size(0)
        if sample_count == 0:
            raise ValueError("The reward state must not be empty.")
        if fixed_embeddings.ndim != 2 or fixed_embeddings.size(0) != sample_count:
            raise ValueError("fixed_embeddings must be [N, D].")
        if fixed_embeddings.device != next_labels.device:
            raise ValueError("Labels and fixed embeddings must share a device.")
        if actions.ndim != 1 or actions.size(0) != sample_count:
            raise ValueError("actions must be [N].")
        expected_graph_shape = (sample_count, self.k)
        if global_neighbor_indices.shape != expected_graph_shape:
            raise ValueError(f"global_neighbor_indices must have shape {expected_graph_shape}.")
        if global_neighbor_cosine_similarities.shape != expected_graph_shape:
            raise ValueError(f"global_neighbor_cosine_similarities must have shape {expected_graph_shape}.")
        if global_neighbor_indices.dtype != torch.long:
            raise TypeError("global_neighbor_indices must use torch.long.")
        if torch.any(global_neighbor_indices < 0) or torch.any(global_neighbor_indices >= sample_count):
            raise ValueError("Global KNN graph contains an out-of-range index.")

        labels = next_labels.float()
        label_consistency, per_sample_consistency = self._global_consistency(
            labels, global_neighbor_indices, global_neighbor_cosine_similarities
        )
        noisy_label_alignment = self._noisy_alignment(labels, actions, fixed_embeddings)
        total_reward = torch.exp(label_consistency + self.nla_weight * noisy_label_alignment).clamp(max=1.0)
        return RewardOutput(
            label_consistency=label_consistency,
            noisy_label_alignment=noisy_label_alignment,
            total_reward=total_reward,
            per_sample_consistency=per_sample_consistency,
        )
