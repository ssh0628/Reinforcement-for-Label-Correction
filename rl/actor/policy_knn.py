from __future__ import annotations

import torch
from torch import Tensor

from setting.config import Config


def _validate_inputs(
    embeddings: Tensor,
    k: int,
    query_chunk_size: int,
    reference_chunk_size: int,
) -> None:
    if embeddings.ndim != 2:
        raise ValueError(f"embeddings must be [N, D], got {embeddings.shape}.")
    if not embeddings.is_floating_point():
        raise TypeError("embeddings must be floating-point tensors.")
    if k <= 0:
        raise ValueError("k must be positive.")
    if embeddings.size(0) <= k:
        raise ValueError(
            f"Policy KNN requires more than k={k} samples; "
            f"found {embeddings.size(0)}."
        )
    if query_chunk_size <= 0 or reference_chunk_size <= 0:
        raise ValueError("KNN chunk sizes must be positive.")


@torch.inference_mode()
def build_exact_policy_knn(
    embeddings: Tensor,
    *,
    k: int,
    query_chunk_size: int,
    reference_chunk_size: int,
) -> Tensor:
    """Build the current f_theta Euclidean KNN graph, excluding self."""
    _validate_inputs(
        embeddings,
        k,
        query_chunk_size,
        reference_chunk_size,
    )
    sample_count = embeddings.size(0)
    device = embeddings.device
    neighbor_indices = torch.empty(
        (sample_count, k),
        dtype=torch.long,
        device=device,
    )

    for query_start in range(0, sample_count, query_chunk_size):
        query_end = min(query_start + query_chunk_size, sample_count)
        queries = embeddings[query_start:query_end].float()
        query_norms = queries.square().sum(dim=1, keepdim=True)
        query_count = query_end - query_start
        best_distances = torch.full(
            (query_count, k),
            float("inf"),
            device=device,
        )
        best_indices = torch.full(
            (query_count, k),
            -1,
            dtype=torch.long,
            device=device,
        )

        for reference_start in range(0, sample_count, reference_chunk_size):
            reference_end = min(
                reference_start + reference_chunk_size,
                sample_count,
            )
            references = embeddings[reference_start:reference_end].float()
            distances = (
                query_norms
                + references.square().sum(dim=1).unsqueeze(0)
                - 2.0 * queries @ references.transpose(0, 1)
            ).clamp_min_(0)

            overlap_start = max(query_start, reference_start)
            overlap_end = min(query_end, reference_end)
            if overlap_start < overlap_end:
                rows = torch.arange(
                    overlap_start - query_start,
                    overlap_end - query_start,
                    device=device,
                )
                columns = torch.arange(
                    overlap_start - reference_start,
                    overlap_end - reference_start,
                    device=device,
                )
                distances[rows, columns] = float("inf")

            local_k = min(k, reference_end - reference_start)
            local_distances, local_positions = distances.topk(
                local_k,
                dim=1,
                largest=False,
                sorted=False,
            )
            candidate_distances = torch.cat(
                (best_distances, local_distances),
                dim=1,
            )
            candidate_indices = torch.cat(
                (best_indices, local_positions + reference_start),
                dim=1,
            )
            best_distances, keep = candidate_distances.topk(
                k,
                dim=1,
                largest=False,
                sorted=True,
            )
            best_indices = candidate_indices.gather(1, keep)

        if torch.any(best_indices < 0) or torch.any(
            ~torch.isfinite(best_distances)
        ):
            raise RuntimeError("Failed to find k valid policy neighbors.")
        neighbor_indices[query_start:query_end] = best_indices

    return neighbor_indices


def build_policy_knn(embeddings: Tensor, cfg: Config) -> Tensor:
    return build_exact_policy_knn(
        embeddings,
        k=cfg.global_knn.k,
        query_chunk_size=cfg.global_knn.query_chunk_size,
        reference_chunk_size=cfg.global_knn.reference_chunk_size,
    )
