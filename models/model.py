import torch
import torch.nn as nn
import torch.nn.functional as F

from models.backbone import ResNet34Backbone
from models.cross_temporal_attn import CrossTemporalAttention
from models.ceff import CEFF
from models.decoder import ProgressiveDecoder


class AdaptiveScratchFormerCD(nn.Module):
    """Adaptive-ScratchFormer-CD with Cross-Temporal Attention.

    Architecture:
        1. Shared ResNet34 backbone extracts features from pre and post independently.
        2. CrossTemporalAttention at stages 3 & 4: pre features attend to post
           features (and vice versa), learning temporal relationships.
        3. CEFF computes symmetric change features from the cross-attended features.
        4. ProgressiveDecoder upsamples step-by-step for sharp boundaries.
        5. Deep Supervision on intermediate stages during training.
    """

    def __init__(
        self,
        embed_dim: int = 256,
        ssa_heads: int = 8,
        ssa_reduce_ratio: int = 2,
        ssa_max_offset: float = 2.0,
        exp_mode: str = "E0",
    ):
        super().__init__()

        self.backbone = ResNet34Backbone()

        # Cross-Temporal Attention at stages 3 & 4
        self.cross_attn3 = CrossTemporalAttention(
            256, num_heads=ssa_heads, reduce_ratio=ssa_reduce_ratio,
        )
        self.cross_attn4 = CrossTemporalAttention(
            512, num_heads=ssa_heads, reduce_ratio=ssa_reduce_ratio,
        )

        # CEFF (symmetric abs diff + conv)
        self.ceff1 = CEFF(64)
        self.ceff2 = CEFF(128)
        self.ceff3 = CEFF(256)
        self.ceff4 = CEFF(512)

        self.decoder = ProgressiveDecoder(embed_dim=embed_dim)

        # Deep Supervision Classifiers
        self.classifier_f2 = nn.Conv2d(128, 1, kernel_size=1)
        self.classifier_f3 = nn.Conv2d(256, 1, kernel_size=1)

    def forward(self, pre: torch.Tensor, post: torch.Tensor):
        # 1. EXTRACT features with shared backbone
        pre_c1, pre_c2, pre_c3, pre_c4 = self.backbone(pre)
        post_c1, post_c2, post_c3, post_c4 = self.backbone(post)

        # 2. CROSS-TEMPORAL ATTENTION: each image attends to the other
        pre_c3, post_c3 = self.cross_attn3(pre_c3, post_c3)
        pre_c4, post_c4 = self.cross_attn4(pre_c4, post_c4)

        # 3. FUSE: symmetric change features
        f1 = self.ceff1(pre_c1, post_c1)
        f2 = self.ceff2(pre_c2, post_c2)
        f3 = self.ceff3(pre_c3, post_c3)
        f4 = self.ceff4(pre_c4, post_c4)

        # 4. DECODE
        logits = self.decoder(f1, f2, f3, f4, out_size=pre.shape[2:])

        # 5. DEEP SUPERVISION
        if self.training:
            out_size = pre.shape[2:]
            logits_f2 = F.interpolate(self.classifier_f2(f2), size=out_size, mode="bilinear", align_corners=False)
            logits_f3 = F.interpolate(self.classifier_f3(f3), size=out_size, mode="bilinear", align_corners=False)
            return logits, logits_f2, logits_f3

        return logits

    def get_offset_reg_loss(self) -> torch.Tensor:
        """Kept for backward compatibility with train.py. Returns 0."""
        return torch.zeros((), device=next(self.parameters()).device)

