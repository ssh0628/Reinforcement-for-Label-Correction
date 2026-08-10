from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional as F


@dataclass(frozen=True, slots=True)
class CleanKNNResult:
    noisy_indices: Tensor
    neighbor_indices: Tensor
    neighbor_cosine_similarities: Tensor


def _validate_inputs(
    embeddings: Tensor,
    actions: Tensor,
    k: int,
    query_chunk_size: int,
    reference_chunk_size: int,
) -> None:
    if embeddings.ndim != 2:
        raise ValueError(f"embeddings must be [N, D], got {embeddings.shape}.")
    if embeddings.size(0) == 0:
        raise ValueError("embeddings must not be empty.")
    if not embeddings.is_floating_point():
        raise TypeError("embeddings must be floating-point tensors.")
    if actions.ndim != 1 or actions.size(0) != embeddings.size(0):
        raise ValueError("actions must be [N] and align with embeddings.")
    if torch.any((actions != 0) & (actions != 1)):
        raise ValueError("actions must contain only zero or one.")
    if k <= 0:
        raise ValueError("k must be positive.")
    if query_chunk_size <= 0 or reference_chunk_size <= 0:
        raise ValueError("KNN chunk sizes must be positive.")


@torch.inference_mode()
def build_exact_clean_knn(
    embeddings: Tensor,
    actions: Tensor,
    *,
    k: int,
    query_chunk_size: int,
    reference_chunk_size: int,
) -> CleanKNNResult:
    """Find each corrected sample's exact neighbors in the clean subset."""
    _validate_inputs(
        embeddings,
        actions,
        k,
        query_chunk_size,
        reference_chunk_size,
    )
    device = embeddings.device
    action_mask = actions.to(device=device, dtype=torch.bool)
    noisy_indices = action_mask.nonzero(as_tuple=False).flatten()
    clean_indices = (~action_mask).nonzero(as_tuple=False).flatten()
    clean_count = clean_indices.numel()
    if clean_count == 0:
        raise ValueError("NLA reward requires at least one clean sample.")

    effective_k = min(k, clean_count)
    noisy_count = noisy_indices.numel()
    if noisy_count == 0:
        empty_shape = (0, effective_k)
        return CleanKNNResult(
            noisy_indices=noisy_indices,
            neighbor_indices=torch.empty(
                empty_shape,
                dtype=torch.long,
                device=device,
            ),
            neighbor_cosine_similarities=torch.empty(
                empty_shape,
                dtype=torch.float32,
                device=device,
            ),
        )

    neighbor_indices = torch.empty(
        (noisy_count, effective_k),
        dtype=torch.long,
        device=device,
    )
    neighbor_cosine = torch.empty(
        (noisy_count, effective_k),
        dtype=torch.float32,
        device=device,
    )

    for query_start in range(0, noisy_count, query_chunk_size):
        query_end = min(query_start + query_chunk_size, noisy_count)
        query_ids = noisy_indices[query_start:query_end]
        queries = embeddings[query_ids].float()
        query_norms = queries.square().sum(dim=1, keepdim=True)
        query_count = query_end - query_start
        best_squared_distances = torch.full(
            (query_count, effective_k),
            float("inf"),
            device=device,
        )
        best_indices = torch.full(
            (query_count, effective_k),
            -1,
            dtype=torch.long,
            device=device,
        )

        for reference_start in range(0, clean_count, reference_chunk_size):
            reference_end = min(
                reference_start + reference_chunk_size,
                clean_count,
            )
            reference_ids = clean_indices[reference_start:reference_end]
            references = embeddings[reference_ids].float()
            squared_distances = (
                query_norms
                + references.square().sum(dim=1).unsqueeze(0)
                - 2.0 * queries @ references.transpose(0, 1)
            ).clamp_min_(0)

            local_k = min(effective_k, reference_end - reference_start)
            local_distances, local_positions = squared_distances.topk(
                local_k,
                dim=1,
                largest=False,
                sorted=False,
            )
            local_indices = reference_ids[local_positions]
            candidate_distances = torch.cat(
                (best_squared_distances, local_distances),
                dim=1,
            )
            candidate_indices = torch.cat(
                (best_indices, local_indices),
                dim=1,
            )
            best_squared_distances, keep_positions = candidate_distances.topk(
                effective_k,
                dim=1,
                largest=False,
                sorted=True,
            )
            best_indices = candidate_indices.gather(1, keep_positions)

        if torch.any(best_indices < 0) or torch.any(
            ~torch.isfinite(best_squared_distances)
        ):
            raise RuntimeError("Failed to find valid clean-subset neighbors.")

        nearest_features = embeddings[best_indices].float()
        cosine = (
            F.normalize(queries, dim=1).unsqueeze(1)
            * F.normalize(nearest_features, dim=2)
        ).sum(dim=2)
        neighbor_indices[query_start:query_end] = best_indices
        neighbor_cosine[query_start:query_end] = cosine

    return CleanKNNResult(
        noisy_indices=noisy_indices,
        neighbor_indices=neighbor_indices,
        neighbor_cosine_similarities=neighbor_cosine,
    )
