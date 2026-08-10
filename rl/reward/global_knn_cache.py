from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor

from setting.config import Config
from setting.dataset import NPYPathDataset, paths_fingerprint


SUPPORTED_ARTIFACT_VERSION = 1
VALIDATION_CHUNK_SIZE = 65_536
REQUIRED_KEYS = frozenset(
    {
        "version",
        "split",
        "k",
        "neighbor_indices",
        "neighbor_cosine_similarities",
        "labels",
        "paths_sha256",
        "sample_count",
        "feature_dim",
        "class_names",
        "model_name",
        "checkpoint_path",
        "embeddings",
    }
)


@dataclass(frozen=True, slots=True)
class GlobalKNNCache:
    cache_path: Path
    fixed_embeddings: Tensor
    neighbor_indices: Tensor
    neighbor_cosine_similarities: Tensor
    labels: Tensor

    @property
    def sample_count(self) -> int:
        return self.fixed_embeddings.size(0)

    @property
    def feature_dim(self) -> int:
        return self.fixed_embeddings.size(1)

    @property
    def k(self) -> int:
        return self.neighbor_indices.size(1)


def _require_tensor(artifact: dict[str, object], key: str) -> Tensor:
    value = artifact[key]
    if not isinstance(value, Tensor):
        raise TypeError(f"Global KNN cache field '{key}' must be a tensor.")
    return value


def _validate_metadata(
    artifact: dict[str, object],
    cfg: Config,
) -> None:
    missing = REQUIRED_KEYS.difference(artifact)
    if missing:
        raise ValueError(
            f"Global KNN cache is missing fields: {sorted(missing)}."
        )
    if artifact["version"] != SUPPORTED_ARTIFACT_VERSION:
        raise ValueError(
            "Unsupported global KNN artifact version: "
            f"{artifact['version']} != {SUPPORTED_ARTIFACT_VERSION}."
        )
    if artifact["split"] != cfg.global_knn.split:
        raise ValueError(
            f"Global KNN cache split mismatch: {artifact['split']} "
            f"!= {cfg.global_knn.split}."
        )
    if artifact["k"] != cfg.global_knn.k:
        raise ValueError(
            f"Global KNN cache k mismatch: {artifact['k']} != {cfg.global_knn.k}."
        )
    if tuple(artifact["class_names"]) != cfg.data.class_names:
        raise ValueError("Global KNN cache class names do not match the current config.")
    if artifact["model_name"] != cfg.model.name:
        raise ValueError(
            f"Global KNN cache model mismatch: {artifact['model_name']} != {cfg.model.name}."
        )

    stored_checkpoint = Path(str(artifact["checkpoint_path"])).resolve()
    configured_checkpoint = cfg.global_knn.checkpoint_path.resolve()
    if stored_checkpoint != configured_checkpoint:
        raise ValueError(
            "Global KNN cache checkpoint mismatch: "
            f"{stored_checkpoint} != {configured_checkpoint}."
        )


def _validate_tensors(
    embeddings: Tensor,
    neighbor_indices: Tensor,
    neighbor_cosine: Tensor,
    labels: Tensor,
    artifact: dict[str, object],
    dataset: NPYPathDataset,
    cfg: Config,
) -> None:
    sample_count = len(dataset)
    k = cfg.global_knn.k
    if embeddings.ndim != 2 or embeddings.size(0) != sample_count:
        raise ValueError(
            f"Global KNN cache embeddings must be [{sample_count}, D], "
            f"got {tuple(embeddings.shape)}."
        )
    if not embeddings.is_floating_point() or embeddings.size(1) == 0:
        raise TypeError("Global KNN cache embeddings must be non-empty floating tensors.")
    if neighbor_indices.shape != (sample_count, k):
        raise ValueError(
            f"Global KNN cache neighbor indices must be [{sample_count}, {k}], "
            f"got {tuple(neighbor_indices.shape)}."
        )
    if neighbor_indices.dtype != torch.long:
        raise TypeError("Global KNN cache neighbor indices must use torch.long.")
    if neighbor_cosine.shape != (sample_count, k):
        raise ValueError(
            f"Global KNN cache cosine similarities must be [{sample_count}, {k}], "
            f"got {tuple(neighbor_cosine.shape)}."
        )
    if not neighbor_cosine.is_floating_point():
        raise TypeError("Global KNN cache cosine similarities must be floating tensors.")
    if labels.shape != (sample_count,) or labels.dtype != torch.long:
        raise ValueError(
            f"Global KNN cache labels must be torch.long [{sample_count}], "
            f"got {tuple(labels.shape)} and {labels.dtype}."
        )

    if artifact["sample_count"] != sample_count:
        raise ValueError(
            f"Global KNN cache sample count mismatch: {artifact['sample_count']} "
            f"!= {sample_count}."
        )
    if artifact["feature_dim"] != embeddings.size(1):
        raise ValueError(
            f"Global KNN cache feature dimension mismatch: {artifact['feature_dim']} "
            f"!= {embeddings.size(1)}."
        )
    if not torch.equal(labels.cpu(), dataset.targets):
        raise ValueError("Global KNN cache labels do not match train_labels.npy order.")

    for start in range(0, sample_count, VALIDATION_CHUNK_SIZE):
        end = min(start + VALIDATION_CHUNK_SIZE, sample_count)
        embedding_chunk = embeddings[start:end]
        index_chunk = neighbor_indices[start:end]
        cosine_chunk = neighbor_cosine[start:end]
        if not torch.isfinite(embedding_chunk).all():
            raise ValueError("Global KNN cache embeddings contain NaN or infinity.")
        if torch.any(index_chunk < 0) or torch.any(index_chunk >= sample_count):
            raise ValueError("Global KNN cache KNN graph contains an out-of-range index.")
        row_indices = torch.arange(start, end).unsqueeze(1)
        if torch.any(index_chunk == row_indices):
            raise ValueError("Global KNN cache KNN graph contains a self-neighbor.")
        sorted_indices = index_chunk.sort(dim=1).values
        if torch.any(sorted_indices[:, 1:] == sorted_indices[:, :-1]):
            raise ValueError("Global KNN cache KNN graph contains duplicate neighbors.")
        if not torch.isfinite(cosine_chunk).all() or torch.any(
            cosine_chunk.abs() > 1.00001
        ):
            raise ValueError("Global KNN cache cosine similarities are invalid.")


def load_global_knn_cache(cfg: Config) -> GlobalKNNCache:
    """Load and validate the fixed f_omega reward cache on CPU."""
    cfg.validate()
    artifact_path = cfg.global_knn.artifact_path
    if not artifact_path.is_file():
        raise FileNotFoundError(f"Global KNN cache not found: {artifact_path}")

    artifact = torch.load(
        artifact_path,
        map_location="cpu",
        weights_only=True,
    )
    if not isinstance(artifact, dict):
        raise TypeError("Global KNN cache must contain a dictionary.")
    _validate_metadata(artifact, cfg)

    dataset = NPYPathDataset(cfg.data, cfg.global_knn.split)
    current_fingerprint = paths_fingerprint(dataset.paths)
    if artifact["paths_sha256"] != current_fingerprint:
        raise ValueError(
            "Global KNN cache image path order does not match the current paths NPY."
        )

    embeddings = _require_tensor(artifact, "embeddings")
    neighbor_indices = _require_tensor(artifact, "neighbor_indices")
    neighbor_cosine = _require_tensor(
        artifact,
        "neighbor_cosine_similarities",
    )
    labels = _require_tensor(artifact, "labels")
    _validate_tensors(
        embeddings,
        neighbor_indices,
        neighbor_cosine,
        labels,
        artifact,
        dataset,
        cfg,
    )
    return GlobalKNNCache(
        cache_path=artifact_path,
        fixed_embeddings=embeddings.contiguous(),
        neighbor_indices=neighbor_indices.contiguous(),
        neighbor_cosine_similarities=neighbor_cosine.contiguous(),
        labels=labels.contiguous(),
    )
