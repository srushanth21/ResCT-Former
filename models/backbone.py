from typing import Tuple

import torch
import torch.nn as nn
from torchvision import models


class ResNet34Backbone(nn.Module):
    """ResNet34 backbone returning stage features (c1..c4)."""

    def __init__(self):
        super().__init__()
        m = models.resnet34(weights=None)

        self.stem = nn.Sequential(m.conv1, m.bn1, m.relu, m.maxpool)
        self.layer1 = m.layer1  # 64
        self.layer2 = m.layer2  # 128
        self.layer3 = m.layer3  # 256
        self.layer4 = m.layer4  # 512

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.stem(x)
        c1 = self.layer1(x)
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)
        c4 = self.layer4(c3)
        return c1, c2, c3, c4
