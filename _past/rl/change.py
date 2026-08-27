"""Optional actor-change diagnostics used by the cache-refresh experiment."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, fields

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.optim import Optimizer


@dataclass(frozen=True, slots=True)
class ChangeDiagnosticRow:
    epoch: int
    trajectory_step: int
    global_step: int
    reference_global_step: int
    steps_since_reference: int
    learning_rate: float
    probe_samples: int
    reference_parameter_norm: float
    step_gradient_norm: float
    step_lr_gradient_norm: float
    cumulative_lr_gradient_norm_before_update: float
    relative_cumulative_lr_gradient_norm_before_update: float
    cumulative_lr_gradient_norm_after_update: float
    relative_cumulative_lr_gradient_norm_after_update: float
    parameter_drift_norm_before_update: float
    relative_parameter_drift_before_update: float
    parameter_drift_norm_after_update: float
    relative_parameter_drift_after_update: float
    feature_mean_cosine_similarity: float
    feature_cosine_drift: float
    knn_neighbor_overlap: float


CHANGE_DIAGNOSTIC_FIELDS = tuple(field.name for field in fields(ChangeDiagnosticRow))


class ChangeDiagnosticsRecorder:
    def __init__(self, model: nn.Module, optimizer: Optimizer, probe_indices: Tensor) -> None:
        if probe_indices.ndim != 1 or probe_indices.numel() == 0:
            raise ValueError("probe_indices must be a non-empty vector.")
        head = getattr(model, "head", None)
        head_ids = {id(parameter) for parameter in head.parameters()} if isinstance(head, nn.Module) else set()
        self.parameters = [
            parameter for parameter in model.parameters() if parameter.requires_grad and id(parameter) not in head_ids
        ]
        if not self.parameters:
            raise RuntimeError("No actor-backbone parameters available to track.")
        if probe_indices.device != self.parameters[0].device:
            raise ValueError("probe_indices and actor-backbone parameters must share a device.")

        groups: dict[int, dict[str, object]] = {}
        for group in optimizer.param_groups:
            for parameter in group["params"]:
                if id(parameter) in groups:
                    raise RuntimeError("An actor parameter appears in multiple optimizer groups.")
                groups[id(parameter)] = group
        if any(id(parameter) not in groups for parameter in self.parameters):
            raise RuntimeError("Tracked actor parameters must belong to the actor optimizer.")

        self.optimizer_groups = [groups[id(parameter)] for parameter in self.parameters]
        self.probe_indices = probe_indices
        self.reference_parameters: list[Tensor] = []
        self.cumulative_updates: list[Tensor] = []
        self.actor_step_updates: list[Tensor] = []
        self.reference_embeddings: Tensor | None = None
        self.reference_neighbors: Tensor | None = None
        self.reference_parameter_norm = 0.0
        self.reference_global_step = 0
        self._cumulative_norm_after = 0.0
        self._parameter_drift_after = 0.0
        self._graph_metrics: dict[str, float] | None = None
        self._update_metrics: dict[str, float] | None = None

    @staticmethod
    def _vector_norm(tensors: list[Tensor]) -> float:
        if not tensors:
            return 0.0
        squared_norm = torch.zeros((), device=tensors[0].device)
        for tensor in tensors:
            squared_norm += tensor.float().square().sum()
        return math.sqrt(float(squared_norm))

    def _parameter_drift_norm(self) -> float:
        squared_norm = torch.zeros((), device=self.parameters[0].device)
        for parameter, reference in zip(self.parameters, self.reference_parameters, strict=True):
            squared_norm += (parameter.detach().float() - reference.float()).square().sum()
        return math.sqrt(float(squared_norm))

    def begin_trajectory(self, reference_global_step: int) -> None:
        if reference_global_step <= 0:
            raise ValueError("reference_global_step must be positive.")
        if not self.reference_parameters:
            self.reference_parameters = [parameter.detach().clone() for parameter in self.parameters]
            self.cumulative_updates = [
                torch.zeros_like(parameter, memory_format=torch.preserve_format)
                for parameter in self.parameters
            ]
            self.actor_step_updates = [
                torch.zeros_like(parameter, memory_format=torch.preserve_format)
                for parameter in self.parameters
            ]
        else:
            for parameter, reference, cumulative in zip(
                self.parameters, self.reference_parameters, self.cumulative_updates, strict=True
            ):
                reference.copy_(parameter.detach())
                cumulative.zero_()
        self.reference_parameter_norm = self._vector_norm(self.reference_parameters)
        if self.reference_parameter_norm == 0.0:
            raise RuntimeError("Actor-backbone parameter norm must be non-zero.")
        self.reference_embeddings = None
        self.reference_neighbors = None
        self.reference_global_step = reference_global_step
        self._cumulative_norm_after = 0.0
        self._parameter_drift_after = 0.0
        self._graph_metrics = None
        self._update_metrics = None

    def begin_actor_update(self) -> None:
        if not self.actor_step_updates:
            raise RuntimeError("Begin the trajectory before recording an actor update.")
        for update in self.actor_step_updates:
            update.zero_()
        self._update_metrics = None

    @torch.inference_mode()
    def observe_policy_graph(self, embeddings: Tensor, neighbors: Tensor) -> None:
        if self.probe_indices.numel() == embeddings.size(0):
            current_embeddings, current_neighbors = embeddings.float(), neighbors
        else:
            current_embeddings = embeddings.index_select(0, self.probe_indices).float()
            current_neighbors = neighbors.index_select(0, self.probe_indices)
        first = self.reference_embeddings is None
        if first:
            self.reference_embeddings = current_embeddings.clone()
            self.reference_neighbors = current_neighbors.clone()
        if self.reference_neighbors is None or self.reference_embeddings is None:
            raise RuntimeError("Reference policy graph was not initialized.")

        cosine = 1.0 if first else float(
            F.cosine_similarity(current_embeddings, self.reference_embeddings, dim=1).mean().clamp(-1.0, 1.0)
        )
        overlap = 1.0 if first else float(
            current_neighbors.unsqueeze(2)
            .eq(self.reference_neighbors.unsqueeze(1))
            .any(dim=2)
            .float()
            .mean()
        )
        self._graph_metrics = {
            "cumulative_before": self._cumulative_norm_after,
            "relative_cumulative_before": self._cumulative_norm_after / self.reference_parameter_norm,
            "parameter_drift_before": self._parameter_drift_after,
            "relative_parameter_drift_before": self._parameter_drift_after / self.reference_parameter_norm,
            "feature_cosine_similarity": cosine,
            "knn_neighbor_overlap": overlap,
        }

    def capture_unscaled_gradient(self) -> None:
        learning_rates = {float(group["lr"]) for group in self.optimizer_groups}
        if len(learning_rates) != 1:
            raise RuntimeError("Change diagnostics require one actor learning rate.")
        learning_rate = learning_rates.pop()
        for parameter, cumulative, step_update in zip(
            self.parameters, self.cumulative_updates, self.actor_step_updates, strict=True
        ):
            if parameter.grad is None:
                continue
            gradient = parameter.grad.detach().float()
            cumulative.add_(gradient, alpha=-learning_rate)
            step_update.add_(gradient, alpha=-learning_rate)

    def finish_actor_update(self) -> None:
        learning_rates = {float(group["lr"]) for group in self.optimizer_groups}
        if len(learning_rates) != 1:
            raise RuntimeError("Change diagnostics require one actor learning rate.")
        learning_rate = learning_rates.pop()
        step_update_norm = self._vector_norm(self.actor_step_updates)
        cumulative_after = self._vector_norm(self.cumulative_updates)
        self._cumulative_norm_after = cumulative_after
        self._update_metrics = {
            "learning_rate": learning_rate,
            "step_gradient_norm": step_update_norm / abs(learning_rate),
            "step_update_norm": step_update_norm,
            "cumulative_after": cumulative_after,
            "relative_cumulative_after": cumulative_after / self.reference_parameter_norm,
        }

    def finish_step(self, *, epoch: int, trajectory_step: int, global_step: int) -> ChangeDiagnosticRow:
        if self._graph_metrics is None or self._update_metrics is None:
            raise RuntimeError("Observe the policy graph and actor gradient before finishing a diagnostic step.")
        drift_after = self._parameter_drift_norm()
        self._parameter_drift_after = drift_after
        graph, update = self._graph_metrics, self._update_metrics
        row = ChangeDiagnosticRow(
            epoch=epoch,
            trajectory_step=trajectory_step,
            global_step=global_step,
            reference_global_step=self.reference_global_step,
            steps_since_reference=global_step - self.reference_global_step,
            learning_rate=update["learning_rate"],
            probe_samples=self.probe_indices.numel(),
            reference_parameter_norm=self.reference_parameter_norm,
            step_gradient_norm=update["step_gradient_norm"],
            step_lr_gradient_norm=update["step_update_norm"],
            cumulative_lr_gradient_norm_before_update=graph["cumulative_before"],
            relative_cumulative_lr_gradient_norm_before_update=graph["relative_cumulative_before"],
            cumulative_lr_gradient_norm_after_update=update["cumulative_after"],
            relative_cumulative_lr_gradient_norm_after_update=update["relative_cumulative_after"],
            parameter_drift_norm_before_update=graph["parameter_drift_before"],
            relative_parameter_drift_before_update=graph["relative_parameter_drift_before"],
            parameter_drift_norm_after_update=drift_after,
            relative_parameter_drift_after_update=drift_after / self.reference_parameter_norm,
            feature_mean_cosine_similarity=graph["feature_cosine_similarity"],
            feature_cosine_drift=1.0 - graph["feature_cosine_similarity"],
            knn_neighbor_overlap=graph["knn_neighbor_overlap"],
        )
        self._graph_metrics = self._update_metrics = None
        return row


def serialize_change_row(row: ChangeDiagnosticRow) -> dict[str, object]:
    return asdict(row)
