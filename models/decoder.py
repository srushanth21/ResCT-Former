from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class SegFormerStyleDecoder(nn.Module):
    """SegFormer-style decoder.

    - 1x1 conv unify channels
    - Upsample to same size
    - Concatenate
    - Conv3x3 -> Conv1x1
    - Final upsample
    """

    def __init__(self, c1: int = 64, c2: int = 128, c3: int = 256, c4: int = 512, embed_dim: int = 256):
        super().__init__()
        self.linear_c1 = nn.Conv2d(c1, embed_dim, kernel_size=1)
        self.linear_c2 = nn.Conv2d(c2, embed_dim, kernel_size=1)
        self.linear_c3 = nn.Conv2d(c3, embed_dim, kernel_size=1)
        self.linear_c4 = nn.Conv2d(c4, embed_dim, kernel_size=1)

        self.fuse = nn.Sequential(
            nn.Conv2d(embed_dim * 4, embed_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dim, embed_dim // 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(embed_dim // 2),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Conv2d(embed_dim // 2, 1, kernel_size=1)

    def forward(
        self,
        c1: torch.Tensor,
        c2: torch.Tensor,
        c3: torch.Tensor,
        c4: torch.Tensor,
        out_size: Tuple[int, int],
    ) -> torch.Tensor:
        target_hw = c1.shape[2:]

        p1 = self.linear_c1(c1)
        p2 = F.interpolate(self.linear_c2(c2), size=target_hw, mode="bilinear", align_corners=False)
        p3 = F.interpolate(self.linear_c3(c3), size=target_hw, mode="bilinear", align_corners=False)
        p4 = F.interpolate(self.linear_c4(c4), size=target_hw, mode="bilinear", align_corners=False)

        x = torch.cat([p4, p3, p2, p1], dim=1)
        x = self.fuse(x)
        x = self.classifier(x)
        x = F.interpolate(x, size=out_size, mode="bilinear", align_corners=False)
        return x
