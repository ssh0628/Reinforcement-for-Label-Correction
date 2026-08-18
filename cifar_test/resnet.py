"""CIFAR-sized ResNet backbones with the feature hooks RLNLC needs."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torchvision.models.resnet import BasicBlock, ResNet


class CifarResNet18(ResNet):
    """ResNet-18 with a 3x3 stride-1 stem and no max pooling."""

    def __init__(self, num_classes: int) -> None:
        if num_classes <= 1:
            raise ValueError("num_classes must be greater than one.")
        super().__init__(
            block=BasicBlock,
            layers=(2, 2, 2, 2),
            num_classes=num_classes,
        )
        self.conv1 = nn.Conv2d(
            3,
            64,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.maxpool = nn.Identity()
        self.feature_dim = self.fc.in_features
        self._reset_cifar_parameters()

    @property
    def head(self) -> nn.Module:
        """Expose the classifier using the interface expected by warm-up."""
        return self.fc

    def _reset_cifar_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight,
                    mode="fan_out",
                    nonlinearity="relu",
                )
            elif isinstance(module, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def forward_features(self, images: Tensor) -> Tensor:
        features = self.conv1(images)
        features = self.bn1(features)
        features = self.relu(features)
        features = self.maxpool(features)
        features = self.layer1(features)
        features = self.layer2(features)
        features = self.layer3(features)
        return self.layer4(features)

    def forward_head(
        self,
        features: Tensor,
        *,
        pre_logits: bool = False,
    ) -> Tensor:
        embeddings = torch.flatten(self.avgpool(features), 1)
        return embeddings if pre_logits else self.fc(embeddings)

    def reset_classifier(self, num_classes: int) -> None:
        if num_classes < 0:
            raise ValueError("num_classes must be non-negative.")
        self.fc = (
            nn.Linear(self.feature_dim, num_classes)
            if num_classes > 0
            else nn.Identity()
        )

    def _forward_impl(self, images: Tensor) -> Tensor:
        return self.forward_head(self.forward_features(images))


def build_cifar_resnet18(
    pretrained: bool,
    num_classes: int,
) -> CifarResNet18:
    if pretrained:
        raise ValueError("The paper-aligned CIFAR ResNet-18 uses no pretraining.")
    return CifarResNet18(num_classes=num_classes)
