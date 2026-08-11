import torch
import torch.nn as nn


class CEFF(nn.Module):
    """Cross-level Edge Feature Fusion (CEFF) - Non-Linear Fusion Update.

    Instead of linear subtraction, this uses concatenation and a spatial convolution
    to learn complex, non-linear change features from perfectly aligned feature maps.
    """

    def __init__(self, channels: int, reduction: int = 2):
        super().__init__()
        # We replace the channel attention MLP with a non-linear spatial fusion block
        self.fuse = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, f_pre: torch.Tensor, f_post: torch.Tensor) -> torch.Tensor:
        # Concatenate pre and post along the channel dimension -> [B, 2C, H, W]
        fcat = torch.cat([f_pre, f_post], dim=1)
        # Learn non-linear differences -> [B, C, H, W]
        change_features = self.fuse(fcat)
        return change_features
