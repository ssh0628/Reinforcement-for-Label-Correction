"""Actor transition and policy-gradient update operations."""

from __future__ import annotations

from typing import Callable

import torch
from torch import Tensor, nn
from torch.optim import Optimizer

from rl.policy import CorrectionResult, LabelCorrectionPolicy


Preprocess = Callable[[Tensor, torch.device, Tensor, Tensor], Tensor]
Encode = Callable[[nn.Module, Tensor], Tensor]


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
) -> tuple[Tensor, float]:
    """Differentiate the dataset-level policy loss with respect to all embeddings."""
    sample_count = embeddings.size(0)
    embedding_leaf = embeddings.detach().clone().requires_grad_(True)
    detached_q = q_value.detach()
    total_loss = 0.0
    for start in range(0, sample_count, microbatch_size):
        end = min(start + microbatch_size, sample_count)
        neighbor_indices = neighbors[start:end]
        policy_step = policy(
            embedding_leaf[start:end],
            embedding_leaf[neighbor_indices],
            label_state[start:end],
            label_state[neighbor_indices],
            actions=actions[start:end],
        )
        loss = -detached_q * policy_step.log_probabilities.sum() / sample_count
        loss.backward()
        total_loss += float(loss.detach())

    if embedding_leaf.grad is None:
        raise RuntimeError("Actor policy loss did not produce embedding gradients.")
    return embedding_leaf.grad.detach(), total_loss


def _backpropagate_embedding_gradients(
    model: nn.Module,
    raw_images: Tensor,
    embedding_gradients: Tensor,
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
    optimizer.zero_grad(set_to_none=True)
    for start in range(0, raw_images.size(0), batch_size):
        end = min(start + batch_size, raw_images.size(0))
        batch = raw_images[start:end].contiguous()
        if device.type == "cuda":
            batch = batch.pin_memory()
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            images = preprocess(batch, device, mean, std)
            encoded = encode(model, images)
            surrogate = (encoded.float() * embedding_gradients[start:end]).sum()
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
    use_amp: bool,
    amp_dtype: torch.dtype,
    preprocess: Preprocess,
    encode: Encode,
) -> float:
    """Apply one exact dataset-level policy-gradient update using microbatches."""
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

    embedding_gradients, loss = _policy_embedding_gradients(
        policy,
        embeddings,
        label_state,
        neighbors,
        actions,
        q_value,
        microbatch_size=microbatch_size,
    )
    _backpropagate_embedding_gradients(
        model,
        raw_images,
        embedding_gradients,
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
