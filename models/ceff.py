import torch
import torch.nn as nn


class CEFF(nn.Module):
    """Cross-level Edge Feature Fusion (CEFF).

    - GAP(F_pre + F_post)
    - Shared MLP
    - Softmax -> weights
    - Fuse features
    """

    def __init__(self, channels: int, reduction: int = 2):
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels * 2),
        )

    def forward(self, f_pre: torch.Tensor, f_post: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = f_pre.shape
        s = f_pre + f_post
        g = self.gap(s).view(b, c)
        w = self.mlp(g).view(b, 2, c, 1, 1)
        w = torch.softmax(w, dim=1)
        w_pre = w[:, 0]
        w_post = w[:, 1]
        return f_pre * w_pre + f_post * w_post
