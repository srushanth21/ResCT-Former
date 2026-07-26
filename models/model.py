import torch
import torch.nn as nn

from models.backbone import ResNet34Backbone
from models.deformable_ssa import DeformableSSA
from models.ceff import CEFF
from models.decoder import SegFormerStyleDecoder


class AdaptiveScratchFormerCD(nn.Module):
    """Adaptive-ScratchFormer-CD (Deformable SSA Version)."""

    def __init__(
        self,
        embed_dim: int = 256,
        ssa_heads: int = 8,
        ssa_reduce_ratio: int = 2,
        ssa_max_offset: float = 2.0,
    ):
        super().__init__()

        self.backbone = ResNet34Backbone()

        # Deformable SSA only at stage 3 & 4
        self.dssa3 = DeformableSSA(256, num_heads=ssa_heads, reduce_ratio=ssa_reduce_ratio, max_offset=ssa_max_offset)
        self.dssa4 = DeformableSSA(512, num_heads=ssa_heads, reduce_ratio=ssa_reduce_ratio, max_offset=ssa_max_offset)

        # CEFF
        self.ceff1 = CEFF(64)
        self.ceff2 = CEFF(128)
        self.ceff3 = CEFF(256)
        self.ceff4 = CEFF(512)

        self.decoder = SegFormerStyleDecoder(embed_dim=embed_dim)

        self._offset_reg_loss: torch.Tensor | None = None

    def forward(self, pre: torch.Tensor, post: torch.Tensor) -> torch.Tensor:
        pre_c1, pre_c2, pre_c3, pre_c4 = self.backbone(pre)
        post_c1, post_c2, post_c3, post_c4 = self.backbone(post)

        pre_c3 = self.dssa3(pre_c3)
        reg_pre_c3 = self.dssa3.last_offset_reg
        post_c3 = self.dssa3(post_c3)
        reg_post_c3 = self.dssa3.last_offset_reg

        pre_c4 = self.dssa4(pre_c4)
        reg_pre_c4 = self.dssa4.last_offset_reg
        post_c4 = self.dssa4(post_c4)
        reg_post_c4 = self.dssa4.last_offset_reg

        regs = [r for r in [reg_pre_c3, reg_post_c3, reg_pre_c4, reg_post_c4] if r is not None]
        if len(regs) == 0:
            self._offset_reg_loss = None
        else:
            self._offset_reg_loss = sum(regs)

        f1 = self.ceff1(pre_c1, post_c1)
        f2 = self.ceff2(pre_c2, post_c2)
        f3 = self.ceff3(pre_c3, post_c3)
        f4 = self.ceff4(pre_c4, post_c4)

        logits = self.decoder(f1, f2, f3, f4, out_size=pre.shape[2:])
        return logits

    def get_offset_reg_loss(self) -> torch.Tensor:
        if self._offset_reg_loss is None:
            # keep dtype/device consistent
            return torch.zeros((), device=next(self.parameters()).device)
        return self._offset_reg_loss
