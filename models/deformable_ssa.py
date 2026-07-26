import torch
import torch.nn as nn
import torch.nn.functional as F


class DeformableSSA(nn.Module):
    """Deformable Sparse Self-Attention (Deformable SSA).

    Steps:
      1) offset = tanh(conv3x3(F)) * max_offset
      2) build normalized grid [-1,1]
      3) grid_sample
      4) token reduction via avg_pool2d (gamma = reduce_ratio)
      5) flatten -> [B, N, C]
      6) manual multi-head self-attention (no nn.Transformer)
      7) reshape + upsample
      8) residual connection

    Notes:
      - offset conv initialized to 0
      - sampling grid clamped to [-1, 1]
    """

    def __init__(
        self,
        channels: int,
        num_heads: int = 8,
        reduce_ratio: int = 2,
        max_offset: float = 2.0,
        attn_dropout: float = 0.0,
        proj_dropout: float = 0.0,
    ):
        super().__init__()
        if channels % num_heads != 0:
            raise ValueError(f"channels ({channels}) must be divisible by num_heads ({num_heads})")

        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.scale = self.head_dim ** -0.5

        self.reduce_ratio = int(reduce_ratio)
        self.max_offset = float(max_offset)

        self.offset_conv = nn.Conv2d(channels, 2, kernel_size=3, padding=1)
        nn.init.zeros_(self.offset_conv.weight)
        nn.init.zeros_(self.offset_conv.bias)

        self.qkv = nn.Linear(channels, channels * 3, bias=True)
        self.norm1 = nn.LayerNorm(channels)
        self.attn_drop = nn.Dropout(attn_dropout)
        self.proj = nn.Linear(channels, channels, bias=True)
        self.proj_drop = nn.Dropout(proj_dropout)

        self.out_proj = nn.Conv2d(channels, channels, kernel_size=1)

        self.last_offset: torch.Tensor | None = None
        self.last_offset_reg: torch.Tensor | None = None

    def _base_grid(self, h: int, w: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        # Pixel-center aligned base grid for align_corners=False
        if w > 1:
            xs = torch.linspace(-1.0 + 1.0 / w, 1.0 - 1.0 / w, steps=w, device=device, dtype=dtype)
        else:
            xs = torch.zeros(1, device=device, dtype=dtype)
        if h > 1:
            ys = torch.linspace(-1.0 + 1.0 / h, 1.0 - 1.0 / h, steps=h, device=device, dtype=dtype)
        else:
            ys = torch.zeros(1, device=device, dtype=dtype)

        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        grid = torch.stack([grid_x, grid_y], dim=-1)  # [H,W,2]
        return grid

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape

        # 1) Offset prediction
        offset_px = torch.tanh(self.offset_conv(x)) * self.max_offset  # [B,2,H,W] in pixel units

        # Save for regularization (keeps graph)
        self.last_offset = offset_px
        self.last_offset_reg = (offset_px.square()).mean()

        # Convert pixel offsets -> normalized offsets for grid in [-1,1]
        # Required conversion:
        #   offset_x_norm = offset_x / (W/2)
        #   offset_y_norm = offset_y / (H/2)
        w_div = max(float(w) / 2.0, 1.0)
        h_div = max(float(h) / 2.0, 1.0)
        offset_norm = torch.empty_like(offset_px)
        offset_norm[:, 0] = offset_px[:, 0] / w_div
        offset_norm[:, 1] = offset_px[:, 1] / h_div

        # 2) Normalized grid [-1,1]
        base_grid = self._base_grid(h, w, x.device, x.dtype).unsqueeze(0).expand(b, -1, -1, -1)
        offset_grid = offset_norm.permute(0, 2, 3, 1)  # [B,H,W,2]
        sampling_grid = base_grid + offset_grid
        sampling_grid = torch.nan_to_num(sampling_grid, nan=0.0, posinf=1.0, neginf=-1.0)
        sampling_grid = torch.clamp(sampling_grid, -1.0, 1.0)

        # 3) Deformable sampling
        x_sampled = F.grid_sample(
            x,
            sampling_grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=False,
        )

        # 4) Token reduction
        if self.reduce_ratio > 1:
            x_reduced = F.avg_pool2d(x_sampled, kernel_size=self.reduce_ratio, stride=self.reduce_ratio)
        else:
            x_reduced = x_sampled

        br, cr, hr, wr = x_reduced.shape
        tokens = x_reduced.flatten(2).transpose(1, 2).contiguous()  # [B,N,C]
        # Normalize tokens before computing QKV
        tokens = self.norm1(tokens)
        n = tokens.shape[1]

        # 5-6) Manual MHA
        qkv = self.qkv(tokens)  # [B,N,3C]
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(b, n, self.num_heads, self.head_dim).transpose(1, 2)  # [B,heads,N,hd]
        k = k.view(b, n, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, n, self.num_heads, self.head_dim).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B,heads,N,N]
        attn = attn - attn.amax(dim=-1, keepdim=True)
        attn = torch.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        out = attn @ v  # [B,heads,N,hd]
        out = out.transpose(1, 2).contiguous().view(b, n, c)  # [B,N,C]

        out = self.proj(out)
        out = self.proj_drop(out)

        # 7) Reshape + upsample
        out_map = out.transpose(1, 2).contiguous().view(b, c, hr, wr)
        if hr != h or wr != w:
            out_map = F.interpolate(out_map, size=(h, w), mode="bilinear", align_corners=False)

        # 8) Residual
        out_map = torch.nan_to_num(out_map, nan=0.0, posinf=0.0, neginf=0.0)
        out_map = self.out_proj(out_map) + x
        return out_map
