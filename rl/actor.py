"""Actor transition and policy-gradient update operations."""

from __future__ import annotations

from typing import Callable

import torch
from torch import Tensor, nn
from torch.optim import Optimizer

from rl.policy import CorrectionResult, LabelCorrectionPolicy


Preprocess = Callable[[Tensor, torch.device, Tensor, Tensor], Tensor]
Encode = Callable[[nn.Module, Tensor], Tensor]


def select_actor_queries(sample_count: int, update_samples: int, *, seed: int, step: int) -> Tensor:
    """Select a reproducible uniform query sample for one RL step."""
    if not 0 < update_samples <= sample_count:
        raise ValueError("update_samples must be in [1, sample_count].")
    if update_samples == sample_count:
        return torch.arange(sample_count)
    generator = torch.Generator().manual_seed(seed + step)
    return torch.randperm(sample_count, generator=generator)[:update_samples].sort().values


@torch.inference_mode()
def correct_from_embeddings(
    policy: LabelCorrectionPolicy, embeddings: Tensor, labels: Tensor, neighbors: Tensor
) -> CorrectionResult:
    correction = policy.correct_all(embeddings, labels, neighbors)
    if correction.actions.all():
        correction.actions[0] = False
        correction.corrected_labels[0] = labels[0]
    return correction


def _policy_embedding_gradients(
    policy: LabelCorrectionPolicy,
    embeddings: Tensor,
    label_state: Tensor,
    neighbors: Tensor,
    actions: Tensor,
    q_value: Tensor,
    *,
    microbatch_size: int,
    query_indices: Tensor,
) -> tuple[Tensor, Tensor, float]:
    """Differentiate the selected-query policy loss with respect to its embeddings."""
    query_count = query_indices.numel()
    embedding_leaf = embeddings.detach().clone().requires_grad_(True)
    label_state = label_state.detach().clone()
    neighbors = neighbors.detach().clone()
    actions = actions.detach().clone()
    detached_q = q_value.detach().clone()
    query_indices = query_indices.to(device=embeddings.device, dtype=torch.long, non_blocking=True)
    selected_neighbors = neighbors[query_indices]
    active_indices = torch.unique(torch.cat((query_indices, selected_neighbors.flatten())))
    total_loss = 0.0
    for start in range(0, query_count, microbatch_size):
        batch_queries = query_indices[start : start + microbatch_size]
        neighbor_indices = neighbors[batch_queries]
        policy_step = policy(
            embedding_leaf[batch_queries],
            embedding_leaf[neighbor_indices],
            label_state[batch_queries],
            label_state[neighbor_indices],
            actions=actions[batch_queries],
        )
        loss = -detached_q * policy_step.log_probabilities.sum() / query_count
        loss.backward()
        total_loss += float(loss.detach())

    if embedding_leaf.grad is None:
        raise RuntimeError("Actor policy loss did not produce embedding gradients.")
    return embedding_leaf.grad.detach(), active_indices.detach(), total_loss


def _backpropagate_embedding_gradients(
    model: nn.Module,
    raw_images: Tensor,
    embedding_gradients: Tensor,
    active_indices: Tensor,
    optimizer: Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    mean: Tensor,
    std: Tensor,
    *,
    batch_size: int,
    use_amp: bool,
    amp_dtype: torch.dtype,
    preprocess: Preprocess,
    encode: Encode,
) -> None:
    """Apply the cached embedding gradients to the backbone in bounded batches."""
    if embedding_gradients.ndim != 2 or embedding_gradients.size(0) != raw_images.size(0):
        raise ValueError("Embedding gradients must share the image sample dimension.")
    active_indices = active_indices.detach().to(device="cpu", dtype=torch.long)
    if active_indices.numel() == 0:
        raise ValueError("Actor update requires at least one active image.")
    optimizer.zero_grad(set_to_none=True)
    for start in range(0, active_indices.numel(), batch_size):
        batch_indices = active_indices[start : start + batch_size]
        batch = raw_images[batch_indices].contiguous()
        if device.type == "cuda":
            batch = batch.pin_memory()
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            images = preprocess(batch, device, mean, std)
            encoded = encode(model, images)
            gradient_indices = batch_indices.to(device=embedding_gradients.device, non_blocking=True)
            surrogate = (encoded.float() * embedding_gradients[gradient_indices]).sum()
        scaler.scale(surrogate).backward()
    scaler.step(optimizer)
    scaler.update()


def update_actor(
    model: nn.Module,
    policy: LabelCorrectionPolicy,
    optimizer: Optimizer,
    scaler: torch.amp.GradScaler,
    raw_images: Tensor,
    label_state: Tensor,
    embeddings: Tensor,
    neighbors: Tensor,
    actions: Tensor,
    q_value: Tensor,
    device: torch.device,
    mean: Tensor,
    std: Tensor,
    *,
    microbatch_size: int,
    query_indices: Tensor | None = None,
    use_amp: bool,
    amp_dtype: torch.dtype,
    preprocess: Preprocess,
    encode: Encode,
) -> float:
    """Apply one full or sampled policy-gradient update using microbatches."""
    sample_count = raw_images.size(0)
    model.eval()
    if microbatch_size <= 0:
        raise ValueError("Actor microbatch_size must be positive.")
    expected_shapes = (
        embeddings.ndim == 2 and embeddings.size(0) == sample_count,
        neighbors.ndim == 2 and neighbors.size(0) == sample_count,
        label_state.ndim == 2 and label_state.size(0) == sample_count,
        actions.ndim == 1 and actions.size(0) == sample_count,
    )
    if sample_count == 0 or not all(expected_shapes):
        raise ValueError("Actor inputs must cover the same non-empty training set.")
    if query_indices is None:
        query_indices = torch.arange(sample_count)
    else:
        query_indices = query_indices.detach().to(device="cpu", dtype=torch.long)
        if (
            query_indices.ndim != 1
            or query_indices.numel() == 0
            or torch.any(query_indices < 0)
            or torch.any(query_indices >= sample_count)
            or torch.unique(query_indices).numel() != query_indices.numel()
        ):
            raise ValueError("query_indices must contain unique in-range sample indices.")

    embedding_gradients, active_indices, loss = _policy_embedding_gradients(
        policy,
        embeddings,
        label_state,
        neighbors,
        actions,
        q_value,
        microbatch_size=microbatch_size,
        query_indices=query_indices,
    )
    _backpropagate_embedding_gradients(
        model,
        raw_images,
        embedding_gradients,
        active_indices,
        optimizer,
        scaler,
        device,
        mean,
        std,
        batch_size=microbatch_size,
        use_amp=use_amp,
        amp_dtype=amp_dtype,
        preprocess=preprocess,
        encode=encode,
    )
    return loss
