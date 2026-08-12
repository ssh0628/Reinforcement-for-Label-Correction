from __future__ import annotations

import torch
from torch import Tensor, nn

from rl.actor.policy import LabelCorrectionPolicy
from setting.config import Config
from setup.backbone import load_warmup_backbone


class PolicyActor(nn.Module):
    """Trainable f_theta and the parameter-free RLNLC policy equations."""

    def __init__(
        self,
        feature_extractor: nn.Module,
        cfg: Config,
    ) -> None:
        super().__init__()
        if not hasattr(feature_extractor, "forward_features") or not hasattr(
            feature_extractor,
            "forward_head",
        ):
            raise TypeError(
                "feature_extractor must provide forward_features/forward_head."
            )
        self.feature_extractor = feature_extractor
        self.policy = LabelCorrectionPolicy(cfg.policy)
        self.use_channels_last = cfg.runtime.use_channels_last

    def encode(self, images: Tensor) -> Tensor:
        if images.ndim != 4:
            raise ValueError(f"images must be [B, C, H, W], got {images.shape}.")
        if images.size(0) == 0:
            raise ValueError("images must not be empty.")
        if self.use_channels_last and images.device.type == "cuda":
            images = images.contiguous(memory_format=torch.channels_last)
        feature_map = self.feature_extractor.forward_features(images)
        embeddings = self.feature_extractor.forward_head(
            feature_map,
            pre_logits=True,
        )
        if embeddings.ndim != 2:
            raise ValueError(
                f"Expected two-dimensional embeddings, got {embeddings.shape}."
            )
        return embeddings


def load_policy_actor(
    cfg: Config,
    device: torch.device,
) -> PolicyActor:
    feature_extractor = load_warmup_backbone(
        cfg,
        device,
        trainable=True,
        remove_classifier=True,
    )
    return PolicyActor(feature_extractor, cfg)
