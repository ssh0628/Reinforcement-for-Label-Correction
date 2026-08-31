"""Diagnose warm-up KNN quality without feeding clean labels into training."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torch import Tensor
from torch.nn import functional as F

from log.common import atomic_path, run_with_log, write_csv
from rl import engine
from setting import data as cifar


CONFIG = cifar.CONFIG
OUTPUT_DIR = CONFIG.warmup_log_dir
RUN_LOG_PATH = OUTPUT_DIR / "knn_quality.log"
QUALITY_CSV_PATH = OUTPUT_DIR / "knn_quality.csv"
PROJECTION_CSV_PATH = OUTPUT_DIR / "knn_projection.csv"
PROJECTION_PNG_PATH = OUTPUT_DIR / "knn_embedding_umap.png"
OUTPUT_PATHS = (RUN_LOG_PATH, QUALITY_CSV_PATH, PROJECTION_CSV_PATH, PROJECTION_PNG_PATH)

QUALITY_FIELDS = (
    "scope", "class_id", "samples", "noise_rate", "observed_label_clean_accuracy",
    "clean_neighbor_purity", "weighted_knn_clean_accuracy", "knn_gain",
    "noisy_recovery_potential", "clean_preservation_potential",
)
PROJECTION_FIELDS = (
    "sample_index", "umap_x", "umap_y", "clean_label", "observed_label", "is_noisy",
)
CLASS_COLORS = (
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
)


@torch.inference_mode()
def compute_knn_diagnostics(
    embeddings: Tensor,
    neighbors: Tensor,
    clean_labels: Tensor,
    observed_labels: Tensor,
    noise_mask: Tensor,
    *,
    temperature: float,
    chunk_size: int,
    num_classes: int,
) -> tuple[Tensor, Tensor]:
    """Return per-sample clean-neighbor purity and weighted KNN predictions."""
    sample_count = embeddings.size(0)
    expected = (sample_count,)
    if embeddings.ndim != 2 or neighbors.ndim != 2:
        raise ValueError("Embeddings and neighbors must be [N,D] and [N,K].")
    if any(tuple(tensor.shape) != expected for tensor in (clean_labels, observed_labels, noise_mask)):
        raise ValueError("KNN diagnostic labels and mask must have shape [N].")
    if temperature <= 0 or chunk_size <= 0:
        raise ValueError("Temperature and chunk size must be positive.")

    normalized = F.normalize(embeddings.float(), dim=1)
    purity = torch.empty(sample_count, dtype=torch.float32, device=embeddings.device)
    predictions = torch.empty(sample_count, dtype=torch.long, device=embeddings.device)
    for start in range(0, sample_count, chunk_size):
        end = min(start + chunk_size, sample_count)
        indices = neighbors[start:end]
        purity[start:end] = clean_labels[indices].eq(clean_labels[start:end, None]).float().mean(dim=1)
        cosine = (normalized[start:end, None] * normalized[indices]).sum(dim=2)
        attention = torch.softmax(cosine / temperature, dim=1)
        scores = torch.zeros((end - start, num_classes), dtype=torch.float32, device=embeddings.device)
        scores.scatter_add_(1, observed_labels[indices], attention)
        predictions[start:end] = scores.argmax(dim=1)
    return purity, predictions


def _optional_accuracy(correct: Tensor, mask: Tensor) -> float | None:
    return float(correct[mask].float().mean()) if bool(mask.any()) else None


def build_quality_rows(
    purity: Tensor,
    predictions: Tensor,
    clean_labels: Tensor,
    observed_labels: Tensor,
    noise_mask: Tensor,
    *,
    class_ids: tuple[int, ...],
) -> list[dict[str, object]]:
    """Build overall and per-class diagnostic rows."""
    rows: list[dict[str, object]] = []
    for class_id in (None, *class_ids):
        selected = torch.ones_like(noise_mask) if class_id is None else clean_labels.eq(class_id)
        noisy = selected & noise_mask
        clean = selected & ~noise_mask
        prediction_correct = predictions.eq(clean_labels)
        observed_correct = observed_labels.eq(clean_labels)
        baseline = float(observed_correct[selected].float().mean())
        knn_accuracy = float(prediction_correct[selected].float().mean())
        rows.append(
            {
                "scope": "overall" if class_id is None else "class",
                "class_id": "" if class_id is None else class_id,
                "samples": int(selected.sum()),
                "noise_rate": float(noise_mask[selected].float().mean()),
                "observed_label_clean_accuracy": baseline,
                "clean_neighbor_purity": float(purity[selected].mean()),
                "weighted_knn_clean_accuracy": knn_accuracy,
                "knn_gain": knn_accuracy - baseline,
                "noisy_recovery_potential": _optional_accuracy(prediction_correct, noisy),
                "clean_preservation_potential": _optional_accuracy(prediction_correct, clean),
            }
        )
    return rows


def select_balanced_visualization_indices(
    clean_labels: Tensor, *, sample_count: int, class_ids: tuple[int, ...], seed: int
) -> Tensor:
    """Select the same number of visualization samples from every clean class."""
    if sample_count % len(class_ids):
        raise ValueError("Visualization sample count must be divisible by class count.")
    generator = torch.Generator().manual_seed(seed)
    per_class = sample_count // len(class_ids)
    selected: list[Tensor] = []
    labels_cpu = clean_labels.cpu()
    for class_id in class_ids:
        indices = labels_cpu.eq(class_id).nonzero(as_tuple=False).flatten()
        if indices.numel() < per_class:
            raise ValueError(f"Class {class_id} has fewer than {per_class} samples.")
        selected.append(indices[torch.randperm(indices.numel(), generator=generator)[:per_class]])
    combined = torch.cat(selected)
    return combined[torch.randperm(combined.numel(), generator=generator)]


@torch.inference_mode()
def project_umap_2d(
    embeddings: Tensor,
    *,
    pca_dimensions: int,
    neighbors: int,
    min_dist: float,
    seed: int,
) -> Tensor:
    """Reduce embeddings with PCA before a deterministic CPU UMAP projection."""
    features = embeddings.float()
    dimensions = min(pca_dimensions, features.size(0) - 1, features.size(1))
    if dimensions < 2:
        raise ValueError("UMAP projection requires at least two PCA dimensions.")
    if dimensions < features.size(1):
        centered = features - features.mean(dim=0, keepdim=True)
        _, _, components = torch.pca_lowrank(centered, q=dimensions, center=False, niter=4)
        features = centered @ components[:, :dimensions]
    try:
        from umap import UMAP
    except ImportError as error:
        raise RuntimeError("Install the project dependencies to use UMAP: uv sync") from error
    reducer = UMAP(
        n_components=2,
        n_neighbors=neighbors,
        min_dist=min_dist,
        metric="euclidean",
        random_state=seed,
        n_jobs=1,
        low_memory=True,
    )
    coordinates = reducer.fit_transform(features.cpu().numpy())
    return torch.from_numpy(np.asarray(coordinates, dtype=np.float32))


def write_projection_csv(
    path: Path,
    indices: Tensor,
    coordinates: Tensor,
    clean_labels: Tensor,
    observed_labels: Tensor,
    noise_mask: Tensor,
) -> None:
    rows = [
        {
            "sample_index": int(index),
            "umap_x": float(point[0]),
            "umap_y": float(point[1]),
            "clean_label": int(clean_labels[index]),
            "observed_label": int(observed_labels[index]),
            "is_noisy": bool(noise_mask[index]),
        }
        for index, point in zip(indices.tolist(), coordinates.tolist())
    ]
    write_csv(path, rows, PROJECTION_FIELDS)


def _scaled_coordinates(coordinates: Tensor, width: float, height: float) -> list[tuple[float, float]]:
    lower = torch.quantile(coordinates, 0.01, dim=0)
    upper = torch.quantile(coordinates, 0.99, dim=0)
    scaled = (coordinates.clamp(lower, upper) - lower) / (upper - lower).clamp_min(1e-12)
    return [(float(x * width), float((1 - y) * height)) for x, y in scaled.tolist()]


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size=size)
    except OSError:
        return ImageFont.load_default()


def render_projection_png(
    path: Path,
    indices: Tensor,
    coordinates: Tensor,
    clean_labels: Tensor,
    observed_labels: Tensor,
    noise_mask: Tensor,
) -> None:
    """Render clean and observed-label views as a lightweight raster image."""
    panel_width, panel_height = 680, 580
    left_margin, top_margin, gap = 60, 105, 80
    canvas_width, canvas_height = 2 * panel_width + gap + 2 * left_margin, 790
    points = _scaled_coordinates(coordinates, panel_width, panel_height)
    global_indices = indices.tolist()
    palette = tuple(tuple(int(color[index : index + 2], 16) for index in (1, 3, 5)) for color in CLASS_COLORS)
    image = Image.new("RGB", (canvas_width, canvas_height), "white")
    draw = ImageDraw.Draw(image)
    title_font, subtitle_font, legend_font = _font(24), _font(16), _font(13)
    title = "Warm-up best embedding: UMAP projection"
    subtitle = "Same coordinates · circles clean · diamonds noisy"
    draw.text((canvas_width / 2, 24), title, fill="#202124", font=title_font, anchor="ma")
    draw.text((canvas_width / 2, 58), subtitle, fill="#4a4a4a", font=subtitle_font, anchor="ma")
    panel_origins = (left_margin, left_margin + panel_width + gap)
    panel_labels = (("GT clean labels", clean_labels), ("Observed noisy labels", observed_labels))
    for origin_x, (title, color_labels) in zip(panel_origins, panel_labels):
        draw.rectangle(
            (origin_x, top_margin, origin_x + panel_width, top_margin + panel_height),
            fill="#fafafa",
            outline="#c7c7c7",
            width=1,
        )
        draw.text(
            (origin_x + panel_width / 2, top_margin - 28), title,
            fill="#202124", font=subtitle_font, anchor="ma",
        )
        for offset, global_index in enumerate(global_indices):
            x, y = points[offset]
            color = palette[int(color_labels[global_index])]
            px, py = origin_x + x, top_margin + y
            if bool(noise_mask[global_index]):
                draw.polygon(((px, py - 2), (px + 2, py), (px, py + 2), (px - 2, py)), fill=color)
            else:
                draw.ellipse((px - 2, py - 2, px + 2, py + 2), fill=color)

    legend_y = top_margin + panel_height + 42
    for class_id in range(len(CLASS_COLORS)):
        x = left_margin + class_id * 142
        rgb = palette[class_id]
        draw.ellipse((x - 5, legend_y - 5, x + 5, legend_y + 5), fill=rgb)
        draw.text((x + 11, legend_y), f"class {class_id}", fill="#202124", font=legend_font, anchor="lm")
    with atomic_path(path) as temporary_path:
        image.save(temporary_path, format="PNG", optimize=True)


def main() -> None:
    device = engine.initialize_cuda_runtime(CONFIG.knn_quality.seed)
    started = time.perf_counter()

    raw_images, clean_labels_cpu = cifar.load_selected_cifar10_train()
    observed_labels_cpu, noise_mask_cpu = engine.load_noisy_label_artifacts(clean_labels_cpu)
    model = cifar.build_model()
    checkpoint = engine.load_warmup_checkpoint(model, CONFIG.warmup_checkpoint_path, device)
    mean, std = engine.normalization_tensors(device)
    embeddings = engine.extract_all_embeddings(model, raw_images, device, mean, std)
    neighbors = engine.build_neighbor_indices(embeddings)
    clean_labels = clean_labels_cpu.to(device, non_blocking=True)
    observed_labels = observed_labels_cpu.to(device, non_blocking=True)
    noise_mask = noise_mask_cpu.to(device, non_blocking=True)
    purity, predictions = compute_knn_diagnostics(
        embeddings, neighbors, clean_labels, observed_labels, noise_mask,
        temperature=CONFIG.knn.temperature, chunk_size=CONFIG.knn.correction_chunk_size,
        num_classes=cifar.NUM_CLASSES,
    )
    rows = build_quality_rows(
        purity, predictions, clean_labels, observed_labels, noise_mask, class_ids=cifar.CLASSES
    )
    write_csv(QUALITY_CSV_PATH, rows, QUALITY_FIELDS)

    visual_indices = select_balanced_visualization_indices(
        clean_labels_cpu, sample_count=CONFIG.knn_quality_visualization_samples,
        class_ids=cifar.CLASSES, seed=CONFIG.knn_quality.seed,
    )
    coordinates = project_umap_2d(
        embeddings[visual_indices.to(device)],
        pca_dimensions=CONFIG.knn_quality.pca_dimensions,
        neighbors=CONFIG.knn_quality.umap_neighbors,
        min_dist=CONFIG.knn_quality.umap_min_dist,
        seed=CONFIG.knn_quality.seed,
    )
    write_projection_csv(
        PROJECTION_CSV_PATH, visual_indices, coordinates, clean_labels_cpu,
        observed_labels_cpu, noise_mask_cpu,
    )
    render_projection_png(
        PROJECTION_PNG_PATH, visual_indices, coordinates, clean_labels_cpu,
        observed_labels_cpu, noise_mask_cpu,
    )

    overall = rows[0]
    elapsed = time.perf_counter() - started
    print(
        f"checkpoint_epoch={int(checkpoint['deployment_epoch'])} samples={clean_labels.numel()} "
        f"k={CONFIG.knn.k} temperature={CONFIG.knn.temperature}"
    )
    print(
        f"clean_neighbor_purity={float(overall['clean_neighbor_purity']):.6f} "
        f"weighted_knn_clean_accuracy={float(overall['weighted_knn_clean_accuracy']):.6f} "
        f"knn_gain={float(overall['knn_gain']):+.6f}"
    )
    print(
        f"noisy_recovery_potential={float(overall['noisy_recovery_potential']):.6f} "
        f"clean_preservation_potential={float(overall['clean_preservation_potential']):.6f} "
        f"seconds={elapsed:.3f}"
    )
    for path in (QUALITY_CSV_PATH, PROJECTION_CSV_PATH, PROJECTION_PNG_PATH):
        print(f"saved={path}")


def run_with_file_logging() -> None:
    cifar.require_files((*cifar.NOISE_ARTIFACT_PATHS, CONFIG.warmup_checkpoint_path), stage="KNN quality")
    cifar.require_available_outputs(
        OUTPUT_PATHS, overwrite=CONFIG.runtime.overwrite_knn_quality, stage="KNN quality"
    )
    run_with_log(RUN_LOG_PATH, main)
