"""Fixed-backbone LCR and NLA reward for the CIFAR experiment."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from rl.knn import build_exact_clean_knn


@dataclass(frozen=True, slots=True)
class RewardOutput:
    label_consistency: Tensor
    noisy_label_alignment: Tensor
    total_reward: Tensor
    per_sample_consistency: Tensor


def soft_kl_divergence(target: Tensor, prediction: Tensor) -> Tensor:
    prediction = prediction.float().clamp_min(torch.finfo(torch.float32).tiny)
    return F.kl_div(prediction.log(), target.float(), reduction="none").sum(dim=1).clamp_min_(0)


class RLNLCReward(nn.Module):
    """RLNLC equations (5)-(7) evaluated with a fixed feature graph."""

    def __init__(
        self,
        *,
        nla_weight: float,
        temperature: float,
        k: int,
        query_chunk_size: int,
        reference_chunk_size: int,
        state_chunk_size: int,
    ) -> None:
        super().__init__()
        self.nla_weight = nla_weight
        self.temperature = temperature
        self.k = k
        self.query_chunk_size = query_chunk_size
        self.reference_chunk_size = reference_chunk_size
        self.state_chunk_size = state_chunk_size

    def _attention(self, cosine: Tensor) -> Tensor:
        return torch.softmax(cosine.float() / self.temperature, dim=1)

    @staticmethod
    def _prediction(labels: Tensor, neighbors: Tensor, attention: Tensor) -> Tensor:
        return torch.einsum("bk,bkc->bc", attention, labels[neighbors].float())

    def _global_consistency(
        self, labels: Tensor, neighbors: Tensor, cosine: Tensor
    ) -> tuple[Tensor, Tensor]:
        kl = torch.empty(labels.size(0), dtype=torch.float32, device=labels.device)
        for start in range(0, labels.size(0), self.state_chunk_size):
            end = min(start + self.state_chunk_size, labels.size(0))
            indices = neighbors[start:end].to(labels.device)
            prediction = self._prediction(labels, indices, self._attention(cosine[start:end]))
            kl[start:end] = soft_kl_divergence(labels[start:end], prediction)
        return -kl.mean(), torch.exp(-kl)

    def _noisy_alignment(self, labels: Tensor, actions: Tensor, embeddings: Tensor) -> Tensor:
        graph = build_exact_clean_knn(
            embeddings, actions, k=self.k, query_chunk_size=self.query_chunk_size,
            reference_chunk_size=self.reference_chunk_size,
        )
        if graph.noisy_indices.numel() == 0:
            return labels.new_zeros((), dtype=torch.float32)
        total_kl = labels.new_zeros((), dtype=torch.float32)
        for start in range(0, graph.noisy_indices.numel(), self.state_chunk_size):
            end = min(start + self.state_chunk_size, graph.noisy_indices.numel())
            noisy = graph.noisy_indices[start:end]
            neighbors = graph.neighbor_indices[start:end]
            prediction = self._prediction(
                labels, neighbors, self._attention(graph.neighbor_cosine_similarities[start:end])
            )
            total_kl += soft_kl_divergence(labels[noisy], prediction).sum()
        return -total_kl / graph.noisy_indices.numel()

    @torch.inference_mode()
    def forward(
        self,
        next_labels: Tensor,
        actions: Tensor,
        fixed_embeddings: Tensor,
        global_neighbors: Tensor,
        global_cosines: Tensor,
    ) -> RewardOutput:
        labels = next_labels.float()
        label_consistency, per_sample = self._global_consistency(labels, global_neighbors, global_cosines)
        noisy_alignment = self._noisy_alignment(labels, actions, fixed_embeddings)
        total = torch.exp(label_consistency + self.nla_weight * noisy_alignment).clamp(max=1.0)
        return RewardOutput(label_consistency, noisy_alignment, total, per_sample)

