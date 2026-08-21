from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import timm
import torch
from PIL import Image
from timm.data import resolve_model_data_config
from torch import Tensor, nn
from torch.utils.data import Dataset
from torchvision import transforms

from setting.config import DataConfig


def _load_npy(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"NPY file not found: {path}")
    return np.load(path, allow_pickle=False, mmap_mode="r")


def _as_path(value: object) -> str:
    return str(value)


def file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"File not found while hashing: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def structured_fingerprint(value: object) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def dataset_manifest_fingerprint(paths: Iterable[object]) -> str:
    """Hash ordered image paths and change-sensitive file metadata."""
    digest = hashlib.sha256(b"rlnlc-image-manifest-v1\n")
    for value in paths:
        path = Path(_as_path(value))
        try:
            stat = path.stat()
        except OSError as exc:
            raise FileNotFoundError(f"Image not found while fingerprinting dataset: {path}") from exc
        for field in (str(path), str(stat.st_size), str(stat.st_mtime_ns)):
            digest.update(field.encode("utf-8"))
            digest.update(b"\0")
    return digest.hexdigest()


def evaluation_preprocessing_signature(data: DataConfig, model_name: str) -> dict[str, object]:
    pretrained_cfg = timm.get_pretrained_cfg(model_name)
    return {
        "implementation": "aspect_letterbox-v1",
        "image_size": data.image_size,
        "letterbox_fill": data.letterbox_fill,
        "resampling": "bicubic",
        "normalization_mean": tuple(pretrained_cfg.mean),
        "normalization_std": tuple(pretrained_cfg.std),
        "timm_version": timm.__version__,
    }


class AspectLetterbox:
    """Resize without distortion, then pad symmetrically to a square."""

    def __init__(self, size: int, fill: tuple[int, int, int]) -> None:
        self.size = size
        self.fill = fill

    def __call__(self, image: Image.Image) -> Image.Image:
        width, height = image.size
        if width <= 0 or height <= 0:
            raise ValueError(f"Invalid image size: {(width, height)}")

        scale = min(self.size / width, self.size / height)
        resized_width = min(self.size, max(1, round(width * scale)))
        resized_height = min(self.size, max(1, round(height * scale)))
        resized = image.resize((resized_width, resized_height), resample=Image.Resampling.BICUBIC)

        canvas = Image.new("RGB", (self.size, self.size), self.fill)
        left = (self.size - resized_width) // 2
        top = (self.size - resized_height) // 2
        canvas.paste(resized, (left, top))
        return canvas


def build_transforms(model: nn.Module, data: DataConfig):
    model_data = resolve_model_data_config(model)
    normalize = transforms.Normalize(model_data["mean"], model_data["std"])
    letterbox = AspectLetterbox(data.image_size, data.letterbox_fill)
    train_transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(p=data.horizontal_flip_p),
            transforms.RandomVerticalFlip(p=data.vertical_flip_p),
            transforms.RandomRotation(data.rotation_degrees, fill=data.letterbox_fill),
            transforms.ColorJitter(*data.color_jitter),
            letterbox,
            transforms.ToTensor(),
            normalize,
        ]
    )
    eval_transform = transforms.Compose([letterbox, transforms.ToTensor(), normalize])
    return train_transform, eval_transform


class NPYPathDataset(Dataset[tuple[Tensor, int, int]]):
    def __init__(self, data: DataConfig, split: str, transform=None) -> None:
        self.data = data
        self.transform = transform
        self.paths = _load_npy(data.paths_file(split))
        labels = _load_npy(data.labels_file(split))

        if self.paths.ndim != 1 or labels.ndim != 1:
            raise ValueError(f"[{split}] paths and labels must be one-dimensional arrays.")
        if len(self.paths) != len(labels):
            raise ValueError(f"[{split}] paths/labels length mismatch: {len(self.paths)} != {len(labels)}.")

        self.labels = np.array(labels, dtype=np.int64, copy=True)
        if len(self.labels) == 0:
            raise ValueError(f"[{split}] dataset is empty.")
        if self.labels.min() < 0 or self.labels.max() >= len(data.class_names):
            raise ValueError(f"[{split}] labels must be in [0, {len(data.class_names) - 1}].")
        self.targets = torch.from_numpy(self.labels)

        non_absolute = next(
            (_as_path(path) for path in self.paths if not Path(_as_path(path)).is_absolute()), None
        )
        if non_absolute is not None:
            raise ValueError(f"[{split}] image path is not absolute: {non_absolute}")

        self._validate_class_metadata()

    def _validate_class_metadata(self) -> None:
        classes_file = self.data.npy_dir / "classes.json"
        if not classes_file.is_file():
            return
        metadata = json.loads(classes_file.read_text(encoding="utf-8"))
        actual = tuple(metadata.get("classes", ()))
        if actual != self.data.class_names:
            raise ValueError(f"classes.json mismatch: {actual} != {self.data.class_names}.")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> tuple[Tensor, int, int]:
        image_path = _as_path(self.paths[index])
        label = int(self.labels[index])

        try:
            with Image.open(image_path) as source:
                image = source.convert("RGB")
        except (OSError, ValueError) as exc:
            raise RuntimeError(f"Failed to load image: {image_path}") from exc

        if self.transform is not None:
            image = self.transform(image)
        return image, label, index
