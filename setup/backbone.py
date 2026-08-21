from __future__ import annotations

from dataclasses import replace

import torch
from torch import nn

from setting.config import Config
from setup.warmup import build_model


def load_warmup_backbone(
    cfg: Config, device: torch.device, *, trainable: bool, remove_classifier: bool = False
) -> nn.Module:
    """Load the shared warmup initialization for f_omega or f_theta."""
    checkpoint_path = cfg.global_knn.checkpoint_path
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Warmup checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or "model" not in checkpoint:
        raise ValueError(f"Invalid warmup checkpoint: {checkpoint_path}")
    checkpoint_classes = tuple(checkpoint.get("class_names", ()))
    if checkpoint_classes != cfg.data.class_names:
        raise ValueError(
            f"Checkpoint classes do not match config: {checkpoint_classes} != {cfg.data.class_names}."
        )

    model_cfg = replace(cfg, model=replace(cfg.model, pretrained=False))
    model = build_model(model_cfg)
    model.load_state_dict(checkpoint["model"], strict=True)
    if remove_classifier:
        reset_classifier = getattr(model, "reset_classifier", None)
        if not callable(reset_classifier):
            raise TypeError("The configured model cannot remove its classifier.")
        reset_classifier(0)

    model = model.to(device)
    model.train(trainable)
    for parameter in model.parameters():
        parameter.requires_grad = trainable

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = cfg.runtime.cudnn_benchmark
        if cfg.runtime.use_channels_last:
            model = model.to(memory_format=torch.channels_last)
    return model
