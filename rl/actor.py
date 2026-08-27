"""Actor transition and policy-gradient update operations."""

from __future__ import annotations

from typing import Callable

import torch
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.utils.checkpoint import checkpoint

from rl.policy import CorrectionResult, LabelCorrectionPolicy


Preprocess = Callable[[Tensor, torch.device, Tensor, Tensor], Tensor]
Encode = Callable[[nn.Module, Tensor], Tensor]


def select_queries(sample_count: int, batch_size: int, seed: int, step: int) -> Tensor:
    """Select one reproducible random Actor mini-batch for an RL step."""
    if not 0 < batch_size <= sample_count:
        raise ValueError("batch_size must be in [1, sample_count].")
    generator = torch.Generator().manual_seed(seed + step)
    return torch.randperm(sample_count, generator=generator)[:batch_size].sort().values


@torch.inference_mode()
def correct_from_embeddings(
    policy: LabelCorrectionPolicy, embeddings: Tensor, labels: Tensor, neighbors: Tensor
) -> CorrectionResult:
    correction = policy.correct_all(embeddings, labels, neighbors)
    if correction.actions.all():
        correction.actions[0] = False
        correction.corrected_labels[0] = labels[0]
    return correction


def _encode_unique_images(
    model: nn.Module,
    raw_images: Tensor,
    indices_cpu: Tensor,
    device: torch.device,
    mean: Tensor,
    std: Tensor,
    *,
    batch_size: int,
    use_amp: bool,
    amp_dtype: torch.dtype,
    preprocess: Preprocess,
    encode: Encode,
) -> Tensor:
    """Encode each selected image once without retaining backbone activations."""
    chunks: list[Tensor] = []
    for start in range(0, indices_cpu.numel(), batch_size):
        batch = raw_images[indices_cpu[start : start + batch_size]].contiguous()
        if device.type == "cuda":
            batch = batch.pin_memory()
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            images = preprocess(batch, device, mean, std)
            chunks.append(
                checkpoint(
                    lambda inputs: encode(model, inputs),
                    images,
                    use_reentrant=False,
                    preserve_rng_state=False,
                )
            )
    if not chunks:
        raise RuntimeError("Actor update received no images to encode.")
    return torch.cat(chunks)


def update_actor(
    model: nn.Module,
    policy: LabelCorrectionPolicy,
    optimizer: Optimizer,
    scaler: torch.amp.GradScaler,
    raw_images: Tensor,
    label_state: Tensor,
    neighbors_cpu: Tensor,
    actions: Tensor,
    q_value: Tensor,
    query_indices_cpu: Tensor,
    device: torch.device,
    mean: Tensor,
    std: Tensor,
    *,
    encoding_batch_size: int,
    use_amp: bool,
    amp_dtype: torch.dtype,
    preprocess: Preprocess,
    encode: Encode,
) -> float:
    """Apply one policy-gradient optimizer step to the selected query batch."""
    sample_count = raw_images.size(0)
    model.eval()
    query_count = query_indices_cpu.numel()
    if query_count == 0 or torch.any(query_indices_cpu < 0) or torch.any(query_indices_cpu >= sample_count):
        raise ValueError("Actor query indices must be a non-empty in-range vector.")

    selected_neighbors_cpu = neighbors_cpu[query_indices_cpu]
    combined = torch.cat((query_indices_cpu, selected_neighbors_cpu.flatten()))
    unique_indices_cpu, inverse = torch.unique(combined, sorted=False, return_inverse=True)
    unique_embeddings = _encode_unique_images(
        model, raw_images, unique_indices_cpu, device, mean, std,
        batch_size=encoding_batch_size, use_amp=use_amp, amp_dtype=amp_dtype,
        preprocess=preprocess, encode=encode,
    )
    query_indices = query_indices_cpu.to(device=device, non_blocking=True)
    neighbor_indices = selected_neighbors_cpu.to(device=device, non_blocking=True)
    inverse = inverse.to(device=device, non_blocking=True)
    neighbor_count = selected_neighbors_cpu.size(1)

    with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
        policy_step = policy(
            unique_embeddings[inverse[:query_count]],
            unique_embeddings[inverse[query_count:]].reshape(query_count, neighbor_count, -1),
            label_state[query_indices],
            label_state[neighbor_indices],
            actions=actions[query_indices],
        )
        loss = -q_value.detach() * policy_step.log_probabilities.mean()

    optimizer.zero_grad(set_to_none=True)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    return float(loss.detach())
