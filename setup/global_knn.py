from __future__ import annotations

from pathlib import Path

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from setting.config import CFG, Config
from setting.dataset import (
    NPYPathDataset,
    build_transforms,
    paths_fingerprint,
)
from setup.backbone import load_warmup_backbone
from setup.warmup import (
    loader_worker_options,
    move_images_to_device,
    resolve_device,
    seed_everything,
    worker_init_fn,
)


ARTIFACT_VERSION = 1


def build_embedding_loader(
    model: nn.Module,
    cfg: Config,
    device: torch.device,
) -> tuple[NPYPathDataset, DataLoader]:
    _, eval_transform = build_transforms(model, cfg.data)
    dataset = NPYPathDataset(
        cfg.data,
        cfg.global_knn.split,
        transform=eval_transform,
    )
    worker_options = loader_worker_options(cfg)
    loader = DataLoader(
        dataset,
        batch_size=cfg.loader.batch_size,
        shuffle=False,
        num_workers=cfg.loader.num_workers,
        pin_memory=cfg.loader.pin_memory and device.type == "cuda",
        drop_last=False,
        worker_init_fn=worker_init_fn,
        **worker_options,
    )
    return dataset, loader


@torch.inference_mode()
def extract_embeddings(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    cfg: Config,
    *,
    progress: bool = True,
) -> Tensor:
    if not hasattr(model, "forward_features") or not hasattr(
        model,
        "forward_head",
    ):
        raise TypeError(
            "The configured model must provide forward_features/forward_head."
        )

    amp_enabled = cfg.runtime.use_amp and device.type == "cuda"
    chunks: list[Tensor] = []
    samples_seen = 0
    for batch_index, (images, _, sample_indices) in enumerate(
        loader,
        start=1,
    ):
        expected_indices = torch.arange(
            samples_seen,
            samples_seen + sample_indices.numel(),
        )
        if not torch.equal(sample_indices, expected_indices):
            raise RuntimeError(
                "Embedding loader must preserve sequential dataset indices."
            )
        samples_seen += sample_indices.numel()
        images = move_images_to_device(images, device, cfg)
        with torch.autocast(device_type=device.type, enabled=amp_enabled):
            feature_map = model.forward_features(images)
            embeddings = model.forward_head(feature_map, pre_logits=True)
        if embeddings.ndim != 2:
            raise ValueError(
                f"Expected two-dimensional embeddings, got {embeddings.shape}."
            )
        chunks.append(embeddings.float().cpu())
        if progress:
            print(f"[EMBED] batch={batch_index}/{len(loader)}", end="\r")
    if progress:
        print()

    if not chunks:
        raise ValueError("The embedding loader produced no samples.")
    return torch.cat(chunks, dim=0).contiguous()


def _feature_chunk(
    feature_bank: Tensor,
    start: int,
    end: int,
    device: torch.device,
) -> Tensor:
    chunk = feature_bank[start:end]
    return chunk if chunk.device == device else chunk.to(device)


@torch.inference_mode()
def build_exact_global_knn(
    embeddings: Tensor,
    cfg: Config,
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    if embeddings.ndim != 2:
        raise ValueError(f"embeddings must be [N, D], got {embeddings.shape}.")
    sample_count = embeddings.size(0)
    k = cfg.global_knn.k
    if sample_count <= k:
        raise ValueError(
            f"Global KNN requires more than k={k} samples; found {sample_count}."
        )

    feature_bank = embeddings
    if cfg.global_knn.cache_features_on_device and device.type != "cpu":
        try:
            feature_bank = embeddings.to(device)
        except RuntimeError as exc:
            raise RuntimeError(
                "Could not cache embeddings on the KNN device. Set "
                "global_knn.cache_features_on_device=False to stream chunks."
            ) from exc

    neighbor_indices = torch.empty((sample_count, k), dtype=torch.long)
    neighbor_cosine = torch.empty((sample_count, k), dtype=torch.float32)
    query_chunk_size = cfg.global_knn.query_chunk_size
    reference_chunk_size = cfg.global_knn.reference_chunk_size

    for query_start in range(0, sample_count, query_chunk_size):
        query_end = min(query_start + query_chunk_size, sample_count)
        queries = _feature_chunk(feature_bank, query_start, query_end, device)
        query_norms = queries.square().sum(dim=1, keepdim=True)
        query_count = query_end - query_start
        best_squared_distances = torch.full(
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
            references = _feature_chunk(
                feature_bank,
                reference_start,
                reference_end,
                device,
            )
            reference_norms = references.square().sum(dim=1).unsqueeze(0)
            squared_distances = (
                query_norms
                + reference_norms
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
                squared_distances[rows, columns] = float("inf")

            local_k = min(k, reference_end - reference_start)
            local_distances, local_positions = squared_distances.topk(
                local_k,
                dim=1,
                largest=False,
                sorted=False,
            )
            local_indices = local_positions + reference_start
            candidate_distances = torch.cat(
                (best_squared_distances, local_distances),
                dim=1,
            )
            candidate_indices = torch.cat(
                (best_indices, local_indices),
                dim=1,
            )
            best_squared_distances, keep_positions = candidate_distances.topk(
                k,
                dim=1,
                largest=False,
                sorted=True,
            )
            best_indices = candidate_indices.gather(1, keep_positions)

        if torch.any(best_indices < 0) or torch.any(
            ~torch.isfinite(best_squared_distances)
        ):
            raise RuntimeError("Failed to find k valid non-self neighbors.")

        index_for_bank = (
            best_indices
            if feature_bank.device == device
            else best_indices.cpu()
        )
        nearest_features = feature_bank[index_for_bank]
        if nearest_features.device != device:
            nearest_features = nearest_features.to(device)
        cosine = (
            F.normalize(queries, dim=1).unsqueeze(1)
            * F.normalize(nearest_features, dim=2)
        ).sum(dim=2)

        neighbor_indices[query_start:query_end] = best_indices.cpu()
        neighbor_cosine[query_start:query_end] = cosine.float().cpu()
        print(f"[KNN] queries={query_end}/{sample_count}")

    return neighbor_indices, neighbor_cosine


def release_accelerator(model: nn.Module, device: torch.device) -> None:
    model.to("cpu")
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()


def save_artifact(
    dataset: NPYPathDataset,
    embeddings: Tensor,
    neighbor_indices: Tensor,
    neighbor_cosine: Tensor,
    checkpoint: dict[str, object],
    cfg: Config,
) -> Path:
    output_path = cfg.global_knn.artifact_path
    if output_path.exists() and not cfg.global_knn.overwrite:
        raise FileExistsError(
            f"Global KNN artifact already exists: {output_path}. "
            "Set global_knn.overwrite=True to replace it."
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    artifact: dict[str, object] = {
        "version": ARTIFACT_VERSION,
        "split": cfg.global_knn.split,
        "k": cfg.global_knn.k,
        "distance_metric": "euclidean",
        "attention_similarity": "cosine",
        "neighbor_indices": neighbor_indices,
        "neighbor_cosine_similarities": neighbor_cosine,
        "labels": dataset.targets.clone(),
        "paths_sha256": paths_fingerprint(dataset.paths),
        "sample_count": len(dataset),
        "feature_dim": embeddings.size(1),
        "class_names": cfg.data.class_names,
        "model_name": cfg.model.name,
        "checkpoint_path": str(cfg.global_knn.checkpoint_path),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_selection_metric": checkpoint.get("selection_metric"),
        "embeddings": embeddings,
    }

    temporary_path = output_path.with_suffix(f"{output_path.suffix}.tmp")
    try:
        torch.save(artifact, temporary_path)
        temporary_path.replace(output_path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return output_path


def main(cfg: Config = CFG) -> None:
    cfg.validate()
    seed_everything(cfg.runtime.seed)
    output_path = cfg.global_knn.artifact_path
    if output_path.exists() and not cfg.global_knn.overwrite:
        raise FileExistsError(
            f"Global KNN artifact already exists: {output_path}. "
            "Set global_knn.overwrite=True to replace it."
        )

    device = resolve_device(cfg)
    model, checkpoint = load_warmup_backbone(
        cfg,
        device,
        trainable=False,
    )
    dataset, loader = build_embedding_loader(model, cfg, device)
    print(f"device={device}")
    print(f"checkpoint={cfg.global_knn.checkpoint_path}")
    print(f"split={cfg.global_knn.split} samples={len(dataset)} k={cfg.global_knn.k}")

    embeddings = extract_embeddings(model, loader, device, cfg)
    if embeddings.size(0) != len(dataset):
        raise RuntimeError(
            f"Embedding count mismatch: {embeddings.size(0)} != {len(dataset)}."
        )
    del loader
    release_accelerator(model, device)

    indices, cosine = build_exact_global_knn(
        embeddings,
        cfg,
        device,
    )
    artifact_path = save_artifact(
        dataset,
        embeddings,
        indices,
        cosine,
        checkpoint,
        cfg,
    )
    print(f"[OK] Global KNN saved: {artifact_path}")


if __name__ == "__main__":
    main()
