"""Exact chunked KNN operations used by CIFAR policy and reward."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional as F


def _validate(embeddings: Tensor, k: int, query_chunk_size: int, reference_chunk_size: int) -> None:
    if embeddings.ndim != 2 or not embeddings.is_floating_point() or embeddings.size(0) <= k:
        raise ValueError("KNN requires floating [N,D] embeddings with N > k.")
    if min(k, query_chunk_size, reference_chunk_size) <= 0:
        raise ValueError("KNN sizes must be positive.")


def _squared_distances(
    queries: Tensor,
    references: Tensor,
    query_squared_norms: Tensor,
    reference_squared_norms: Tensor,
) -> Tensor:
    return (
        query_squared_norms.unsqueeze(1)
        + reference_squared_norms.unsqueeze(0)
        - 2.0 * queries @ references.T
    ).clamp_min_(0)


def _merge_nearest(
    best_distances: Tensor,
    best_indices: Tensor,
    local_distances: Tensor,
    local_indices: Tensor,
    k: int,
) -> tuple[Tensor, Tensor]:
    candidate_distances = torch.cat((best_distances, local_distances), dim=1)
    candidate_indices = torch.cat((best_indices, local_indices), dim=1)
    next_distances, keep = candidate_distances.topk(k, dim=1, largest=False, sorted=True)
    return next_distances, candidate_indices.gather(1, keep)


@torch.inference_mode()
def build_exact_policy_knn(
    embeddings: Tensor, *, k: int, query_chunk_size: int, reference_chunk_size: int
) -> Tensor:
    """Build the Euclidean KNN graph and exclude each query itself."""
    _validate(embeddings, k, query_chunk_size, reference_chunk_size)
    sample_count, device = embeddings.size(0), embeddings.device
    features = embeddings.float()
    squared_norms = features.square().sum(dim=1)
    neighbors = torch.empty((sample_count, k), dtype=torch.long, device=device)

    for query_start in range(0, sample_count, query_chunk_size):
        query_end = min(query_start + query_chunk_size, sample_count)
        queries = features[query_start:query_end]
        query_norms = squared_norms[query_start:query_end]
        best_distances = torch.full((queries.size(0), k), float("inf"), device=device)
        best_indices = torch.full((queries.size(0), k), -1, dtype=torch.long, device=device)
        for reference_start in range(0, sample_count, reference_chunk_size):
            reference_end = min(reference_start + reference_chunk_size, sample_count)
            references = features[reference_start:reference_end]
            distances = _squared_distances(
                queries,
                references,
                query_norms,
                squared_norms[reference_start:reference_end],
            )
            overlap_start, overlap_end = max(query_start, reference_start), min(query_end, reference_end)
            if overlap_start < overlap_end:
                rows = torch.arange(overlap_start - query_start, overlap_end - query_start, device=device)
                columns = torch.arange(
                    overlap_start - reference_start, overlap_end - reference_start, device=device
                )
                distances[rows, columns] = float("inf")
            local_k = min(k, references.size(0))
            local_distances, local_positions = distances.topk(local_k, dim=1, largest=False, sorted=False)
            best_distances, best_indices = _merge_nearest(
                best_distances,
                best_indices,
                local_distances,
                local_positions + reference_start,
                k,
            )
        neighbors[query_start:query_end] = best_indices
    return neighbors


@dataclass(frozen=True, slots=True)
class CleanKNNResult:
    noisy_indices: Tensor
    neighbor_indices: Tensor
    neighbor_cosine_similarities: Tensor


@torch.inference_mode()
def build_exact_clean_knn(
    embeddings: Tensor, actions: Tensor, *, k: int, query_chunk_size: int, reference_chunk_size: int
) -> CleanKNNResult:
    """Find each corrected sample's exact neighbors among uncorrected samples."""
    if embeddings.ndim != 2 or actions.shape != (embeddings.size(0),):
        raise ValueError("Clean KNN inputs must be [N,D] embeddings and [N] actions.")
    action_mask = actions.to(device=embeddings.device, dtype=torch.bool)
    noisy_indices = action_mask.nonzero().flatten()
    clean_indices = (~action_mask).nonzero().flatten()
    if clean_indices.numel() == 0:
        raise ValueError("NLA reward requires at least one uncorrected sample.")
    effective_k = min(k, clean_indices.numel())
    if noisy_indices.numel() == 0:
        empty = (0, effective_k)
        return CleanKNNResult(
            noisy_indices,
            torch.empty(empty, dtype=torch.long, device=embeddings.device),
            torch.empty(empty, dtype=torch.float32, device=embeddings.device),
        )

    features = embeddings.float()
    squared_norms = features.square().sum(dim=1)
    normalized = F.normalize(features, dim=1)
    neighbor_indices = torch.empty(
        (noisy_indices.numel(), effective_k), dtype=torch.long, device=embeddings.device
    )
    neighbor_cosines = torch.empty_like(neighbor_indices, dtype=torch.float32)
    for query_start in range(0, noisy_indices.numel(), query_chunk_size):
        query_end = min(query_start + query_chunk_size, noisy_indices.numel())
        query_ids = noisy_indices[query_start:query_end]
        queries = features[query_ids]
        best_distances = torch.full((queries.size(0), effective_k), float("inf"), device=embeddings.device)
        best_indices = torch.full_like(best_distances, -1, dtype=torch.long)
        for reference_start in range(0, clean_indices.numel(), reference_chunk_size):
            reference_ids = clean_indices[reference_start : reference_start + reference_chunk_size]
            distances = _squared_distances(
                queries,
                features[reference_ids],
                squared_norms[query_ids],
                squared_norms[reference_ids],
            )
            local_k = min(effective_k, reference_ids.numel())
            local_distances, positions = distances.topk(local_k, dim=1, largest=False, sorted=False)
            best_distances, best_indices = _merge_nearest(
                best_distances,
                best_indices,
                local_distances,
                reference_ids[positions],
                effective_k,
            )
        neighbor_indices[query_start:query_end] = best_indices
        neighbor_cosines[query_start:query_end] = (
            normalized[query_ids].unsqueeze(1) * normalized[best_indices]
        ).sum(dim=2)
    return CleanKNNResult(noisy_indices, neighbor_indices, neighbor_cosines)
