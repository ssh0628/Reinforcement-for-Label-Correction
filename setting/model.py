"""CIFAR-sized ResNet backbones with the feature hooks RLNLC needs."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torchvision.models.resnet import BasicBlock, ResNet


class CifarResNet(ResNet):
    def __init__(self, layers: tuple[int, int, int, int], num_classes: int) -> None:
        if num_classes <= 1:
            raise ValueError("num_classes must be greater than one.")
        super().__init__(block=BasicBlock, layers=layers, num_classes=num_classes)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.maxpool = nn.Identity()
        self.feature_dim = self.fc.in_features
        self._reset_cifar_parameters()

    @property
    def head(self) -> nn.Module:
        return self.fc

    def _reset_cifar_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
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

    def forward_head(self, features: Tensor, *, pre_logits: bool = False) -> Tensor:
        embeddings = torch.flatten(self.avgpool(features), 1)
        return embeddings if pre_logits else self.fc(embeddings)

    def _forward_impl(self, images: Tensor) -> Tensor:
        features = self.forward_features(images)
        return self.forward_head(features)


class CifarResNet18(CifarResNet):
    def __init__(self, num_classes: int) -> None:
        super().__init__((2, 2, 2, 2), num_classes)


class CifarResNet34(CifarResNet):
    def __init__(self, num_classes: int) -> None:
        super().__init__((3, 4, 6, 3), num_classes)


def build_cifar_resnet(model_name: str, pretrained: bool, num_classes: int) -> CifarResNet:
    if pretrained:
        raise ValueError("The paper-aligned CIFAR ResNet uses no pretraining.")
    builders = {
        "cifar_resnet18": CifarResNet18,
        "cifar_resnet34": CifarResNet34,
    }
    try:
        return builders[model_name](num_classes=num_classes)
    except KeyError as error:
        raise ValueError(f"Unsupported model name: {model_name!r}.") from error
