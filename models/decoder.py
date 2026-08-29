from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ProgressiveDecoder(nn.Module):
    """Progressive U-Net style decoder for sharp high-resolution edge preservation.
    
    Instead of a massive 8x interpolation of all features at once, this decoder
    progressively upsamples features 2x at a time, fusing deep semantic context
    with high-resolution spatial edges at every step.
    """

    def __init__(self, c1: int = 64, c2: int = 128, c3: int = 256, c4: int = 512, embed_dim: int = 256):
        super().__init__()
        
        # Project all incoming features to a common embedding dimension
        self.proj4 = nn.Conv2d(c4, embed_dim, kernel_size=1)
        self.proj3 = nn.Conv2d(c3, embed_dim, kernel_size=1)
        self.proj2 = nn.Conv2d(c2, embed_dim, kernel_size=1)
        self.proj1 = nn.Conv2d(c1, embed_dim, kernel_size=1)
        
        # Fusion blocks for each upsampling step
        self.fuse43 = self._make_fuse_block(embed_dim)
        self.fuse32 = self._make_fuse_block(embed_dim)
        self.fuse21 = self._make_fuse_block(embed_dim)
        
        # Final classifier
        self.classifier = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(embed_dim // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dim // 2, 1, kernel_size=1)
        )

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
        
        # Classify at 1/4 resolution
        logits_low = self.classifier(f21)
        
        # Final 4x upsample to original resolution
        logits_full = F.interpolate(logits_low, size=out_size, mode="bilinear", align_corners=False)
        
        return logits_full
