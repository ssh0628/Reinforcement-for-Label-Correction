"""Actor transition and policy-gradient update operations."""

from __future__ import annotations

from typing import Callable, Protocol

import torch
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.utils.checkpoint import checkpoint

from cifar_test.rl.policy import CorrectionResult, LabelCorrectionPolicy


class ChangeRecorder(Protocol):
    def begin_actor_update(self) -> None: ...

    def capture_unscaled_gradient(self) -> None: ...

    def finish_actor_update(self) -> None: ...


Preprocess = Callable[[Tensor, torch.device, Tensor, Tensor], Tensor]
Encode = Callable[[nn.Module, Tensor], Tensor]


def select_queries(
    sample_count: int, update_samples: int, seed: int, step: int, *, shuffle: bool = False
) -> Tensor:
    """Select a deterministic uniform subset for this RL step."""
    if not 0 < update_samples <= sample_count:
        raise ValueError("update_samples must be in [1, sample_count].")
    generator = torch.Generator().manual_seed(seed + step)
    if update_samples == sample_count and not shuffle:
        return torch.arange(sample_count)
    selected = torch.randperm(sample_count, generator=generator)[:update_samples]
    return selected if shuffle else selected.sort().values


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
    batch_size: int,
    step_mode: str,
    use_amp: bool,
    amp_dtype: torch.dtype,
    preprocess: Preprocess,
    encode: Encode,
    change_recorder: ChangeRecorder | None = None,
) -> float:
    """Apply a trajectory-level or true mini-batch policy-gradient update."""
    sample_count = raw_images.size(0)
    model.eval()
    query_count = query_indices_cpu.numel()
    if query_count == 0 or torch.any(query_indices_cpu < 0) or torch.any(query_indices_cpu >= sample_count):
        raise ValueError("Actor query indices must be a non-empty in-range vector.")
    if step_mode not in {"trajectory", "minibatch"}:
        raise ValueError("Actor step_mode must be 'trajectory' or 'minibatch'.")
    if change_recorder is not None:
        change_recorder.begin_actor_update()

    signal = q_value.detach()

    def batch_loss(batch_queries_cpu: Tensor) -> Tensor:
        batch_neighbors_cpu = neighbors_cpu[batch_queries_cpu]
        combined = torch.cat((batch_queries_cpu, batch_neighbors_cpu.flatten()))
        unique_indices_cpu, inverse = torch.unique(combined, sorted=False, return_inverse=True)
        unique_embeddings = _encode_unique_images(
            model, raw_images, unique_indices_cpu, device, mean, std, batch_size=batch_size,
            use_amp=use_amp, amp_dtype=amp_dtype, preprocess=preprocess, encode=encode,
        )
        batch_queries = batch_queries_cpu.to(device=device, non_blocking=True)
        batch_neighbors = batch_neighbors_cpu.to(device=device, non_blocking=True)
        inverse = inverse.to(device=device, non_blocking=True)
        batch_count = batch_queries_cpu.numel()
        neighbor_count = batch_neighbors_cpu.size(1)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            policy_step = policy(
                unique_embeddings[inverse[:batch_count]],
                unique_embeddings[inverse[batch_count:]].reshape(batch_count, neighbor_count, -1),
                label_state[batch_queries],
                label_state[batch_neighbors],
                actions=actions[batch_queries],
            )
            return -signal * policy_step.log_probabilities.mean()

    def optimizer_step(loss: Tensor) -> None:
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        if change_recorder is not None:
            scaler.unscale_(optimizer)
            change_recorder.capture_unscaled_gradient()
        scaler.step(optimizer)
        scaler.update()

    def finish_diagnostics() -> None:
        if change_recorder is not None:
            change_recorder.finish_actor_update()

    if step_mode == "trajectory":
        selected_neighbors_cpu = neighbors_cpu[query_indices_cpu]
        combined = torch.cat((query_indices_cpu, selected_neighbors_cpu.flatten()))
        unique_indices_cpu, inverse = torch.unique(combined, sorted=False, return_inverse=True)
        unique_embeddings = _encode_unique_images(
            model, raw_images, unique_indices_cpu, device, mean, std, batch_size=batch_size,
            use_amp=use_amp, amp_dtype=amp_dtype, preprocess=preprocess, encode=encode,
        )
        inverse = inverse.to(device=device, non_blocking=True)
        query_indices = query_indices_cpu.to(device=device, non_blocking=True)
        neighbor_indices = selected_neighbors_cpu.to(device=device, non_blocking=True)
        neighbor_count = selected_neighbors_cpu.size(1)
        query_embeddings = unique_embeddings[inverse[:query_count]]
        neighbor_embeddings = unique_embeddings[inverse[query_count:]].reshape(query_count, neighbor_count, -1)
        loss = torch.zeros((), device=device)
        for start in range(0, query_count, batch_size):
            end = min(start + batch_size, query_count)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                policy_step = policy(
                    query_embeddings[start:end],
                    neighbor_embeddings[start:end],
                    label_state[query_indices[start:end]],
                    label_state[neighbor_indices[start:end]],
                    actions=actions[query_indices[start:end]],
                )
                loss = loss - signal * policy_step.log_probabilities.sum() / query_count
        optimizer_step(loss)
        finish_diagnostics()
        return float(loss.detach())

    weighted_loss = 0.0
    for start in range(0, query_count, batch_size):
        batch_queries_cpu = query_indices_cpu[start : start + batch_size]
        loss = batch_loss(batch_queries_cpu)
        optimizer_step(loss)
        weighted_loss += float(loss.detach()) * batch_queries_cpu.numel()
    finish_diagnostics()
    return weighted_loss / query_count
