from __future__ import annotations

import random
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import timm
import torch
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, WeightedRandomSampler

from setting.config import CFG, BestCheckpoint, Config
from setting.dataset import NPYPathDataset, build_transforms


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def worker_init_fn(_worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def resolve_device(cfg: Config) -> torch.device:
    if cfg.runtime.device != "auto":
        return torch.device(cfg.runtime.device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_model(cfg: Config) -> nn.Module:
    return timm.create_model(
        cfg.model.name,
        pretrained=cfg.model.pretrained,
        num_classes=cfg.num_classes,
        drop_rate=cfg.model.drop_rate,
        drop_path_rate=cfg.model.drop_path_rate,
    )


def get_class_counts(labels: Tensor, num_classes: int) -> Tensor:
    counts = torch.bincount(labels, minlength=num_classes)
    if counts.numel() != num_classes or torch.any(counts == 0):
        raise ValueError(
            f"Every class must contain at least one training sample: {counts.tolist()}."
        )
    return counts


def build_sqrt_sampler(
    dataset: NPYPathDataset,
    class_counts: Tensor,
    cfg: Config,
) -> WeightedRandomSampler | None:
    if not cfg.train.use_sqrt_sampler:
        return None
    class_weights = class_counts.to(torch.float64).rsqrt()
    sample_weights = class_weights[dataset.targets]
    num_samples = (
        len(dataset)
        if cfg.train.sampler_num_samples is None
        else cfg.train.sampler_num_samples
    )
    if num_samples <= 0:
        raise ValueError("sampler_num_samples must be positive.")
    if not cfg.train.sampler_replacement and num_samples > len(dataset):
        raise ValueError(
            "sampler_num_samples cannot exceed dataset size without replacement."
        )
    generator = torch.Generator().manual_seed(cfg.runtime.seed)
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=num_samples,
        replacement=cfg.train.sampler_replacement,
        generator=generator,
    )


def loader_worker_options(cfg: Config) -> dict[str, object]:
    if cfg.loader.num_workers == 0:
        return {"persistent_workers": False}
    return {
        "prefetch_factor": cfg.loader.prefetch_factor,
        "persistent_workers": cfg.loader.persistent_workers,
    }


def build_dataloaders(
    model: nn.Module,
    cfg: Config,
    device: torch.device,
):
    train_transform, eval_transform = build_transforms(model, cfg.data)
    train_dataset = NPYPathDataset(cfg.data, "train", transform=train_transform)
    val_dataset = NPYPathDataset(cfg.data, "val", transform=eval_transform)
    class_counts = get_class_counts(train_dataset.targets, cfg.num_classes)
    sampler = build_sqrt_sampler(train_dataset, class_counts, cfg)
    train_sample_count = (
        len(sampler) if sampler is not None else len(train_dataset)
    )
    if (
        cfg.loader.train_drop_last
        and train_sample_count < cfg.loader.warmup_batch_size
    ):
        raise ValueError(
            "Warmup drop_last=True would discard the entire training set: "
            f"{train_sample_count} samples < batch size "
            f"{cfg.loader.warmup_batch_size}."
        )
    worker_options = loader_worker_options(cfg)
    pin_memory = cfg.loader.pin_memory and device.type == "cuda"

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.loader.warmup_batch_size,
        sampler=sampler,
        shuffle=sampler is None,
        num_workers=cfg.loader.num_workers,
        pin_memory=pin_memory,
        drop_last=cfg.loader.train_drop_last,
        worker_init_fn=worker_init_fn,
        **worker_options,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.loader.warmup_batch_size,
        shuffle=False,
        num_workers=cfg.loader.num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        worker_init_fn=worker_init_fn,
        **worker_options,
    )
    return train_loader, val_loader, class_counts


def get_head(model: nn.Module) -> nn.Module:
    head = getattr(model, "head", None)
    if not isinstance(head, nn.Module):
        raise TypeError("The configured model does not expose a .head module.")
    return head


def freeze_backbone(model: nn.Module) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in get_head(model).parameters():
        parameter.requires_grad = True


def unfreeze_model(model: nn.Module) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = True


def set_train_mode(model: nn.Module, backbone_frozen: bool) -> None:
    if backbone_frozen:
        model.eval()
        get_head(model).train()
    else:
        model.train()


def build_optimizer(
    parameters: Iterable[nn.Parameter],
    lr: float,
    cfg: Config,
) -> Optimizer:
    parameters = tuple(parameters)
    if not parameters:
        raise ValueError("The optimizer received no parameters.")
    return torch.optim.AdamW(
        parameters,
        lr=lr,
        weight_decay=cfg.train.weight_decay,
        betas=cfg.train.adamw_betas,
        eps=cfg.train.adamw_eps,
    )


def build_criterion(
    class_counts: Tensor,
    device: torch.device,
    cfg: Config,
) -> nn.CrossEntropyLoss:
    weights = None
    if cfg.train.use_weighted_ce:
        weights = class_counts.to(device=device, dtype=torch.float32).rsqrt()
    return nn.CrossEntropyLoss(
        weight=weights,
        label_smoothing=cfg.train.label_smoothing,
    )


def loss_reduction_denominator(
    targets: Tensor,
    criterion: nn.CrossEntropyLoss,
) -> Tensor:
    """Return the denominator used by mean-reduced cross entropy."""
    if criterion.weight is None:
        return torch.tensor(
            targets.numel(),
            device=targets.device,
            dtype=torch.float64,
        )
    return criterion.weight[targets].sum(dtype=torch.float64)


def update_confusion_matrix(
    predictions: Tensor,
    targets: Tensor,
    confusion_matrix: Tensor,
    num_classes: int,
) -> None:
    predictions = predictions.detach().to(dtype=torch.long)
    targets = targets.detach().to(dtype=torch.long)
    flat_indices = targets * num_classes + predictions
    confusion_matrix += torch.bincount(
        flat_indices,
        minlength=num_classes * num_classes,
    ).reshape(num_classes, num_classes)


def finalize_metrics(
    loss_numerator: Tensor,
    loss_denominator: Tensor,
    confusion_matrix: Tensor,
) -> dict[str, float]:
    matrix = confusion_matrix.to(torch.float64)
    true_positives = matrix.diag()
    actual_per_class = matrix.sum(dim=1)
    predicted_per_class = matrix.sum(dim=0)
    recall = true_positives / actual_per_class.clamp_min(1)
    precision = true_positives / predicted_per_class.clamp_min(1)
    denominator = precision + recall
    per_class_f1 = torch.where(
        denominator > 0,
        2 * precision * recall / denominator,
        torch.zeros_like(denominator),
    )
    accuracy = true_positives.sum() / matrix.sum().clamp_min(1)
    loss, accuracy, balanced_accuracy, macro_f1 = torch.stack(
        (
            loss_numerator
            / loss_denominator.clamp_min(
                torch.finfo(loss_denominator.dtype).tiny
            ),
            accuracy,
            recall.mean(),
            per_class_f1.mean(),
        )
    ).tolist()
    return {
        "loss": loss,
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "macro_f1": macro_f1,
    }


def move_images_to_device(
    images: Tensor,
    device: torch.device,
    cfg: Config,
) -> Tensor:
    memory_format = (
        torch.channels_last
        if cfg.runtime.use_channels_last and device.type == "cuda"
        else torch.preserve_format
    )
    images = images.to(
        device=device,
        non_blocking=True,
        memory_format=memory_format,
    )
    return images


def move_batch_to_device(
    images: Tensor,
    targets: Tensor,
    device: torch.device,
    cfg: Config,
) -> tuple[Tensor, Tensor]:
    images = move_images_to_device(images, device, cfg)
    targets = targets.to(device=device, non_blocking=True)
    return images, targets


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.CrossEntropyLoss,
    optimizer: Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    backbone_frozen: bool,
    cfg: Config,
) -> dict[str, float]:
    set_train_mode(model, backbone_frozen)
    loss_numerator = torch.zeros((), device=device, dtype=torch.float64)
    loss_denominator = torch.zeros((), device=device, dtype=torch.float64)
    confusion_matrix = torch.zeros(
        (cfg.num_classes, cfg.num_classes),
        device=device,
        dtype=torch.long,
    )
    amp_enabled = cfg.runtime.use_amp and device.type == "cuda"

    for images, targets, _ in loader:
        images, targets = move_batch_to_device(images, targets, device, cfg)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=amp_enabled):
            logits = model(images)
            loss = criterion(logits, targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        batch_denominator = loss_reduction_denominator(targets, criterion)
        loss_numerator += loss.detach().to(torch.float64) * batch_denominator
        loss_denominator += batch_denominator
        update_confusion_matrix(
            logits.argmax(dim=1), targets, confusion_matrix, cfg.num_classes
        )
    return finalize_metrics(
        loss_numerator,
        loss_denominator,
        confusion_matrix,
    )


@torch.inference_mode()
def validate_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.CrossEntropyLoss,
    device: torch.device,
    cfg: Config,
) -> dict[str, float]:
    model.eval()
    loss_numerator = torch.zeros((), device=device, dtype=torch.float64)
    loss_denominator = torch.zeros((), device=device, dtype=torch.float64)
    confusion_matrix = torch.zeros(
        (cfg.num_classes, cfg.num_classes),
        device=device,
        dtype=torch.long,
    )
    amp_enabled = cfg.runtime.use_amp and device.type == "cuda"

    for images, targets, _ in loader:
        images, targets = move_batch_to_device(images, targets, device, cfg)
        with torch.autocast(device_type=device.type, enabled=amp_enabled):
            logits = model(images)
            loss = criterion(logits, targets)

        batch_denominator = loss_reduction_denominator(targets, criterion)
        loss_numerator += loss.detach().to(torch.float64) * batch_denominator
        loss_denominator += batch_denominator
        update_confusion_matrix(
            logits.argmax(dim=1), targets, confusion_matrix, cfg.num_classes
        )
    return finalize_metrics(
        loss_numerator,
        loss_denominator,
        confusion_matrix,
    )


def save_checkpoint(
    path: Path,
    epoch: int,
    model: nn.Module,
    class_counts: Tensor,
    validation_metrics: dict[str, float],
    cfg: Config,
    selection_metric: str | None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "epoch": epoch,
        "model": model.state_dict(),
        "class_names": cfg.data.class_names,
        "class_counts": class_counts,
        "validation_metrics": validation_metrics,
        "selection_metric": selection_metric,
    }
    torch.save(checkpoint, path)


def initial_best_value(item: BestCheckpoint) -> float:
    return float("-inf") if item.mode == "max" else float("inf")


def is_improved(value: float, best_value: float, mode: str) -> bool:
    return value > best_value if mode == "max" else value < best_value


def print_epoch(
    epoch: int,
    optimizer: Optimizer,
    train_metrics: dict[str, float],
    validation_metrics: dict[str, float],
    cfg: Config,
) -> None:
    lr = optimizer.param_groups[0]["lr"]
    print(
        f"epoch={epoch + 1:02d}/{cfg.train.epochs} "
        f"lr={lr:.3e} "
        f"train_loss={train_metrics['loss']:.4f} "
        f"train_macro_f1={train_metrics['macro_f1']:.4f} "
        f"train_bal_acc={train_metrics['balanced_accuracy']:.4f} "
        f"val_loss={validation_metrics['loss']:.4f} "
        f"val_macro_f1={validation_metrics['macro_f1']:.4f} "
        f"val_bal_acc={validation_metrics['balanced_accuracy']:.4f}"
    )


def main(cfg: Config = CFG) -> None:
    cfg.validate()
    seed_everything(cfg.runtime.seed)
    device = resolve_device(cfg)
    model = build_model(cfg).to(device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = cfg.runtime.cudnn_benchmark
        if cfg.runtime.use_channels_last:
            model = model.to(memory_format=torch.channels_last)
    train_loader, val_loader, class_counts = build_dataloaders(
        model,
        cfg,
        device,
    )
    criterion = build_criterion(class_counts, device, cfg)
    amp_enabled = cfg.runtime.use_amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    freeze_backbone(model)
    optimizer = build_optimizer(
        model.parameters(),
        cfg.train.lr_head,
        cfg,
    )
    scheduler: CosineAnnealingLR | None = None
    best_values = {
        item.metric: initial_best_value(item) for item in cfg.checkpoint.best
    }

    print(f"device={device}")
    print(f"npy_dir={cfg.data.npy_dir}")
    print(f"classes={cfg.data.class_names}")
    print(f"class_counts={class_counts.tolist()}")
    print("preprocess=aspect_letterbox")
    print(
        f"channels_last="
        f"{cfg.runtime.use_channels_last and device.type == 'cuda'}"
    )
    print(f"sqrt_sampler={cfg.train.use_sqrt_sampler}")
    print(f"inverse_sqrt_wce={cfg.train.use_weighted_ce}")

    for epoch in range(cfg.train.epochs):
        if epoch == cfg.train.backbone_freeze_epochs:
            unfreeze_model(model)
            for parameter_group in optimizer.param_groups:
                parameter_group["lr"] = cfg.train.lr_unfrozen
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=cfg.train.cosine_t_max,
                eta_min=cfg.train.min_lr,
            )

        train_metrics = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            scaler,
            device,
            epoch < cfg.train.backbone_freeze_epochs,
            cfg,
        )
        validation_metrics = validate_one_epoch(
            model, val_loader, criterion, device, cfg
        )
        print_epoch(epoch, optimizer, train_metrics, validation_metrics, cfg)

        if scheduler is not None:
            scheduler.step()

        for item in cfg.checkpoint.best:
            current_value = validation_metrics[item.metric]
            if is_improved(current_value, best_values[item.metric], item.mode):
                best_values[item.metric] = current_value
                save_checkpoint(
                    cfg.runtime.output_dir / item.filename,
                    epoch,
                    model,
                    class_counts,
                    validation_metrics,
                    cfg,
                    selection_metric=item.metric,
                )

        save_checkpoint(
            cfg.runtime.output_dir / cfg.checkpoint.last_filename,
            epoch,
            model,
            class_counts,
            validation_metrics,
            cfg,
            selection_metric=None,
        )


if __name__ == "__main__":
    main()
