import torch
import torch.nn as nn


class CEFF(nn.Module):
    """Cross-level Edge Feature Fusion (CEFF) - Symmetric Non-Linear Fusion Update.

    Uses Absolute Difference (which is perfectly mathematically symmetric) 
    followed by a spatial convolution to learn complex, non-linear change features 
    from perfectly aligned feature maps without confusing the network on temporal swaps.
    """

    def __init__(self, channels: int, reduction: int = 2, dropout: float = 0.1, exp_mode: str = "A0"):
        super().__init__()
        self.exp_mode = exp_mode.upper()
        self.fuse = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
        )

    def forward(self, f_pre: torch.Tensor, f_post: torch.Tensor) -> torch.Tensor:
        # Symmetric difference -> [B, C, H, W]
        diff = torch.abs(f_pre - f_post)
        
        if self.exp_mode == "A4":
            # Ablation A4: Skip learned fusion, just use raw difference
            return diff
            
        # Learn non-linear differences -> [B, C, H, W]
        change_features = self.fuse(diff)
        return change_features

