from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from setting.config import CLASS_NAMES


@dataclass(frozen=True, slots=True)
class NPYPathingConfig:
    dataset_root: Path = Path("/absolute/path/to/dataset_root")
    output_dir: Path = Path("/absolute/path/to/npy")

    class_names: tuple[str, ...] = CLASS_NAMES
    split_names: tuple[str, ...] = ("train", "val", "test")
    split_ratios: tuple[float, ...] = (0.70, 0.15, 0.15)
    image_extensions: tuple[str, ...] = (
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".tif",
        ".tiff",
        ".webp",
    )
    recursive: bool = True
    seed: int = 42
    overwrite: bool = False


CFG = NPYPathingConfig()


@dataclass(frozen=True, slots=True)
class SamplePair:
    image_path: Path
    class_name: str
    label: int


def validate_config(cfg: NPYPathingConfig) -> None:
    if cfg.class_names != CLASS_NAMES:
        raise ValueError(f"class_names must be {CLASS_NAMES}.")
    if not cfg.dataset_root.is_absolute():
        raise ValueError("dataset_root must be an absolute path.")
    if not cfg.output_dir.is_absolute():
        raise ValueError("output_dir must be an absolute path.")
    if not cfg.dataset_root.is_dir():
        raise FileNotFoundError(f"Dataset root not found: {cfg.dataset_root}")
    if len(cfg.split_names) != len(cfg.split_ratios):
        raise ValueError("split_names and split_ratios must have the same length.")
    if len(set(cfg.split_names)) != len(cfg.split_names):
        raise ValueError(f"split_names contains duplicates: {cfg.split_names}")
    if set(cfg.split_names) != {"train", "val", "test"}:
        raise ValueError("split_names must contain train, val, and test.")
    if any(ratio <= 0 for ratio in cfg.split_ratios):
        raise ValueError(f"split_ratios must be positive: {cfg.split_ratios}")
    if not cfg.image_extensions:
        raise ValueError("image_extensions must not be empty.")
    if any(not extension.startswith(".") for extension in cfg.image_extensions):
        raise ValueError("Every image extension must start with '.'.")


def list_images(class_dir: Path, cfg: NPYPathingConfig) -> list[Path]:
    iterator = class_dir.rglob("*") if cfg.recursive else class_dir.glob("*")
    extensions = {extension.lower() for extension in cfg.image_extensions}
    return sorted(
        (
            path.resolve()
            for path in iterator
            if path.is_file()
            and path.suffix.lower() in extensions
            and not path.name.startswith((".", ".nfs"))
        ),
        key=str,
    )


def scan_dataset(cfg: NPYPathingConfig) -> dict[str, list[SamplePair]]:
    samples_by_class: dict[str, list[SamplePair]] = {}
    used_images: set[Path] = set()

    for label, class_name in enumerate(cfg.class_names):
        class_dir = cfg.dataset_root / class_name
        if not class_dir.is_dir():
            raise FileNotFoundError(f"Class directory not found: {class_dir}")

        class_samples: list[SamplePair] = []
        for image_path in list_images(class_dir, cfg):
            if image_path in used_images:
                raise ValueError(f"Duplicate image path: {image_path}")
            used_images.add(image_path)
            class_samples.append(
                SamplePair(
                    image_path=image_path,
                    class_name=class_name,
                    label=label,
                )
            )

        if len(class_samples) < len(cfg.split_names):
            raise ValueError(
                f"{class_name} needs at least {len(cfg.split_names)} images "
                f"to populate every split; found {len(class_samples)}."
            )
        samples_by_class[class_name] = class_samples

    return samples_by_class


def calculate_split_counts(
    sample_count: int,
    split_ratios: tuple[float, ...],
) -> list[int]:
    """Allocate at least one class sample to each split."""
    split_count = len(split_ratios)
    if sample_count < split_count:
        raise ValueError(
            f"sample_count={sample_count} is smaller than split_count={split_count}."
        )

    ratio_sum = sum(split_ratios)
    exact = [sample_count * ratio / ratio_sum for ratio in split_ratios]
    counts = [int(np.floor(value)) for value in exact]
    remainder = sample_count - sum(counts)
    order = sorted(
        range(split_count),
        key=lambda index: (exact[index] - counts[index], -index),
        reverse=True,
    )
    for index in order[:remainder]:
        counts[index] += 1

    for empty_index in [
        index for index, count in enumerate(counts) if count == 0
    ]:
        donors = [index for index, count in enumerate(counts) if count > 1]
        if not donors:
            raise RuntimeError("Could not populate every split.")
        donor = max(
            donors,
            key=lambda index: (counts[index] - exact[index], counts[index]),
        )
        counts[donor] -= 1
        counts[empty_index] += 1
    return counts


def build_stratified_splits(
    samples_by_class: dict[str, list[SamplePair]],
    cfg: NPYPathingConfig,
) -> dict[str, list[SamplePair]]:
    splits = {split: [] for split in cfg.split_names}

    for class_index, class_name in enumerate(cfg.class_names):
        samples = list(samples_by_class[class_name])
        random.Random(cfg.seed + class_index).shuffle(samples)
        counts = calculate_split_counts(len(samples), cfg.split_ratios)
        start = 0
        for split, count in zip(cfg.split_names, counts):
            end = start + count
            splits[split].extend(samples[start:end])
            start = end
        if start != len(samples):
            raise RuntimeError(f"Not all {class_name} samples were assigned.")

    for split_index, split in enumerate(cfg.split_names):
        random.Random(cfg.seed + 10_000 + split_index).shuffle(splits[split])
    return splits


def validate_splits(
    samples_by_class: dict[str, list[SamplePair]],
    splits: dict[str, list[SamplePair]],
    cfg: NPYPathingConfig,
) -> None:
    source_paths = {
        sample.image_path
        for samples in samples_by_class.values()
        for sample in samples
    }
    split_sets = {
        split: {sample.image_path for sample in samples}
        for split, samples in splits.items()
    }
    for index, left in enumerate(cfg.split_names):
        for right in cfg.split_names[index + 1 :]:
            overlap = split_sets[left] & split_sets[right]
            if overlap:
                raise RuntimeError(
                    f"Data leakage between {left} and {right}: {next(iter(overlap))}"
                )
    assigned_paths = set().union(*split_sets.values())
    if assigned_paths != source_paths:
        raise RuntimeError("Split assignment did not preserve every source sample.")

    for split, samples in splits.items():
        present_labels = {sample.label for sample in samples}
        expected_labels = set(range(len(cfg.class_names)))
        if present_labels != expected_labels:
            raise RuntimeError(
                f"{split} does not contain every class: {present_labels}"
            )


def output_filenames(cfg: NPYPathingConfig) -> list[str]:
    filenames = ["classes.json", "split_config.json"]
    for split in cfg.split_names:
        filenames.extend(
            (
                f"{split}_paths.npy",
                f"{split}_labels.npy",
            )
        )
    return filenames


def ensure_output_is_writable(cfg: NPYPathingConfig) -> None:
    existing = [
        cfg.output_dir / filename
        for filename in output_filenames(cfg)
        if (cfg.output_dir / filename).exists()
    ]
    if existing and not cfg.overwrite:
        formatted = "\n".join(f"  - {path}" for path in existing)
        raise FileExistsError(
            "Output files already exist. Set overwrite=True to replace them:\n"
            f"{formatted}"
        )


def class_counts(
    samples: list[SamplePair],
    class_names: tuple[str, ...],
) -> dict[str, int]:
    counts = {class_name: 0 for class_name in class_names}
    for sample in samples:
        counts[sample.class_name] += 1
    return counts


def build_report(
    samples_by_class: dict[str, list[SamplePair]],
    splits: dict[str, list[SamplePair]],
    cfg: NPYPathingConfig,
) -> dict[str, object]:
    return {
        "dataset_root": str(cfg.dataset_root.resolve()),
        "output_dir": str(cfg.output_dir.resolve()),
        "classes": list(cfg.class_names),
        "class_to_index": {
            class_name: index for index, class_name in enumerate(cfg.class_names)
        },
        "split_names": list(cfg.split_names),
        "split_ratios": list(cfg.split_ratios),
        "seed": cfg.seed,
        "recursive": cfg.recursive,
        "sampling_strategy": "runtime_sqrt_sampler",
        "static_downsampling": False,
        "source_counts": {
            class_name: len(samples_by_class[class_name])
            for class_name in cfg.class_names
        },
        "split_counts": {
            split: class_counts(samples, cfg.class_names)
            for split, samples in splits.items()
        },
        "split_totals": {
            split: len(samples) for split, samples in splits.items()
        },
        "total_samples": sum(len(samples) for samples in samples_by_class.values()),
    }


def save_outputs(
    samples_by_class: dict[str, list[SamplePair]],
    splits: dict[str, list[SamplePair]],
    cfg: NPYPathingConfig,
) -> None:
    ensure_output_is_writable(cfg)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    report = build_report(samples_by_class, splits, cfg)

    with TemporaryDirectory(prefix=".npy_pathing_", dir=cfg.output_dir) as temp:
        temp_dir = Path(temp)
        (temp_dir / "classes.json").write_text(
            json.dumps({"classes": list(cfg.class_names)}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (temp_dir / "split_config.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        for split, samples in splits.items():
            image_paths = np.asarray(
                [str(sample.image_path) for sample in samples], dtype=np.str_
            )
            labels = np.asarray(
                [sample.label for sample in samples], dtype=np.int64
            )
            if len(image_paths) != len(labels):
                raise RuntimeError(f"{split}: path/label length mismatch.")
            np.save(temp_dir / f"{split}_paths.npy", image_paths)
            np.save(temp_dir / f"{split}_labels.npy", labels)

        for filename in output_filenames(cfg):
            (temp_dir / filename).replace(cfg.output_dir / filename)


def print_summary(
    samples_by_class: dict[str, list[SamplePair]],
    splits: dict[str, list[SamplePair]],
    cfg: NPYPathingConfig,
) -> None:
    print(f"[OK] dataset_root={cfg.dataset_root.resolve()}")
    print(f"[OK] output_dir={cfg.output_dir.resolve()}")
    print("[OK] Static downsampling: disabled")
    print("[OK] Rebalancing: runtime sqrt sampler")
    print("[SOURCE]")
    for class_name in cfg.class_names:
        print(f"  {class_name}: {len(samples_by_class[class_name])}")
    for split in cfg.split_names:
        counts = class_counts(splits[split], cfg.class_names)
        summary = ", ".join(
            f"{class_name}={counts[class_name]}" for class_name in cfg.class_names
        )
        print(f"[{split.upper()}] total={len(splits[split])} | {summary}")


def main(cfg: NPYPathingConfig = CFG) -> None:
    validate_config(cfg)
    samples_by_class = scan_dataset(cfg)
    splits = build_stratified_splits(samples_by_class, cfg)
    validate_splits(samples_by_class, splits, cfg)
    save_outputs(samples_by_class, splits, cfg)
    print_summary(samples_by_class, splits, cfg)


if __name__ == "__main__":
    main()
