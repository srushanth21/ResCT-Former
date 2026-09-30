from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class PixelShuffleUp(nn.Module):
    """Learned 2x upsampler using PixelShuffle (sub-pixel convolution).

    Unlike bilinear interpolation which blurs edges, PixelShuffle learns
    an upsampling kernel that can preserve sharp boundaries. It rearranges
    elements from channels into spatial dimensions — no checkerboard artifacts
    like transposed convolutions.

    Architecture per stage:
        Conv3x3 (C → 4C) → BN → ReLU → PixelShuffle(2) → (4C → C at 2x resolution)
    """

    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels * 4, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(channels * 4)
        self.act = nn.ReLU(inplace=True)
        self.shuffle = nn.PixelShuffle(upscale_factor=2)
        # After PixelShuffle(2): channels*4 / (2*2) = channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.shuffle(self.act(self.bn(self.conv(x))))


class ProgressiveDecoder(nn.Module):
    """Progressive U-Net style decoder with learned PixelShuffle upsampling.

    The intermediate 2x upsamples (between FPN levels) use bilinear interpolation
    since those fuse with skip connections anyway. The critical final 4x upsample
    (from 1/4 resolution to full resolution) uses two stages of learned
    PixelShuffle for sharp boundary reconstruction.
    """

    def __init__(self, c1: int = 64, c2: int = 128, c3: int = 256, c4: int = 512, embed_dim: int = 256, exp_mode: str = "A0"):
        super().__init__()
        self.exp_mode = exp_mode.upper()

        # Project all incoming features to a common embedding dimension
        self.proj4 = nn.Conv2d(c4, embed_dim, kernel_size=1)
        self.proj3 = nn.Conv2d(c3, embed_dim, kernel_size=1)
        self.proj2 = nn.Conv2d(c2, embed_dim, kernel_size=1)
        self.proj1 = nn.Conv2d(c1, embed_dim, kernel_size=1)

        # Fusion blocks for each upsampling step
        self.fuse43 = self._make_fuse_block(embed_dim)
        self.fuse32 = self._make_fuse_block(embed_dim)
        self.fuse21 = self._make_fuse_block(embed_dim)

        # Classifier at 1/4 resolution
        self.classifier = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(embed_dim // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dim // 2, embed_dim // 4, kernel_size=1),
        )

        # Learned 4x upsample: two stages of PixelShuffle(2)
        mid_ch = embed_dim // 4  # 64 channels after classifier
        self.upsample_2x_1 = PixelShuffleUp(mid_ch)  # 64ch @ H/4 → 64ch @ H/2
        self.upsample_2x_2 = PixelShuffleUp(mid_ch)  # 64ch @ H/2 → 64ch @ H

        # Final 1x1 head to produce single-channel logits
        self.head = nn.Conv2d(mid_ch, 1, kernel_size=1)
        
        # Fallback bilinear head for A2 mode
        self.fallback_head = nn.Conv2d(mid_ch, 1, kernel_size=1)

    def _make_fuse_block(self, dim: int):
        return nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True)
        )

    def forward(
        self,
        c1: torch.Tensor,
        c2: torch.Tensor,
        c3: torch.Tensor,
        c4: torch.Tensor,
        out_size: Tuple[int, int],
    ) -> torch.Tensor:

        # Project to embed_dim
        p4 = self.proj4(c4)
        p3 = self.proj3(c3)
        p2 = self.proj2(c2)
        p1 = self.proj1(c1)

        # Step 1: Upsample p4 and fuse with p3
        up4 = F.interpolate(p4, size=p3.shape[2:], mode="bilinear", align_corners=False)
        f43 = self.fuse43(up4 + p3)

        # Step 2: Upsample f43 and fuse with p2
        up3 = F.interpolate(f43, size=p2.shape[2:], mode="bilinear", align_corners=False)
        f32 = self.fuse32(up3 + p2)

        # Step 3: Upsample f32 and fuse with p1
        up2 = F.interpolate(f32, size=p1.shape[2:], mode="bilinear", align_corners=False)
        f21 = self.fuse21(up2 + p1)

        # Classify at 1/4 resolution (embed_dim → embed_dim//4 channels)
        feat = self.classifier(f21)

        if self.exp_mode == "A2":
            # Ablation A2: No PixelShuffle, just naive 4x bilinear
            logits = self.fallback_head(feat)
            logits = F.interpolate(logits, size=out_size, mode="bilinear", align_corners=False)
        else:
            # Learned 4x upsample via PixelShuffle (2x + 2x)
            feat = self.upsample_2x_1(feat)  # H/4 → H/2
            feat = self.upsample_2x_2(feat)  # H/2 → H
            logits = self.head(feat)

            # Safety: if PixelShuffle output doesn't exactly match target size
            if logits.shape[2:] != out_size:
                logits = F.interpolate(logits, size=out_size, mode="bilinear", align_corners=False)

        return logits

