import torch
import torch.nn as nn
import torch.nn.functional as F


class DeformableSSA(nn.Module):
    """Deformable Sparse Self-Attention (Deformable SSA) - True Cross-Resolution."""

    def __init__(
        self,
        channels: int,
        num_heads: int = 8,
        reduce_ratio: int = 2,
        max_offset: float = 2.0,
        attn_dropout: float = 0.0,
        proj_dropout: float = 0.0,
        max_tokens: int = 16384,  # Supports up to 1024x1024 input images
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

        # 1) Offset prediction (Predicts from downsampled grid)
        self.offset_conv = nn.Conv2d(channels, 2, kernel_size=3, padding=1)
        nn.init.zeros_(self.offset_conv.weight)
        nn.init.zeros_(self.offset_conv.bias)

        # 2) Stop Bleeding Fix 2: Depthwise Strided Conv downsampler
        if self.reduce_ratio > 1:
            self.downsampler = nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                stride=self.reduce_ratio,
                padding=1,
                groups=channels,
                bias=False
            )
        else:
            self.downsampler = None

        # 3) Stop Bleeding Fix 3: Positional Embeddings
        # Replaced Absolute Embeddings with Depthwise Convolution (Mix-PE).
        # Absolute embeddings destroy translation invariance and cause severe 
        # spatial overfitting (memorizing train set locations), capping Val IoU.
        self.pos_embed_q = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels)
        self.pos_embed_kv = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels)

        # 4) Stop Bleeding Fix 1: Mandatory Normalization
        self.norm1 = nn.LayerNorm(channels)
        self.norm2 = nn.LayerNorm(channels)

        # Separate projections for High-Res Q and Low-Res KV
        self.q_proj = nn.Linear(channels, channels, bias=True)
        self.kv_proj = nn.Linear(channels, channels * 2, bias=True)
        
        self.attn_drop = nn.Dropout(attn_dropout)
        self.proj = nn.Linear(channels, channels, bias=True)
        self.proj_drop = nn.Dropout(proj_dropout)

        # FFN Block
        self.mlp = nn.Sequential(
            nn.Linear(channels, 4 * channels),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(4 * channels, channels),
            nn.Dropout(0.1),
        )

        self.last_offset: torch.Tensor | None = None
        self.last_offset_reg: torch.Tensor | None = None

    def _base_grid(self, h: int, w: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if w > 1:
            xs = torch.linspace(-1.0 + 1.0 / w, 1.0 - 1.0 / w, steps=w, device=device, dtype=dtype)
        else:
            xs = torch.zeros(1, device=device, dtype=dtype)
        if h > 1:
            ys = torch.linspace(-1.0 + 1.0 / h, 1.0 - 1.0 / h, steps=h, device=device, dtype=dtype)
        else:
            ys = torch.zeros(1, device=device, dtype=dtype)

        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        grid = torch.stack([grid_x, grid_y], dim=-1)
        return grid

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        n_q = h * w

        # --- Q Generation (High Resolution) ---
        x_spatial = x  # [B, C, H, W]
        tokens = x.flatten(2).transpose(1, 2)  # [B, H*W, C]
        
        # Add Pos Embed Q (Depthwise Conv)
        q_pos = self.pos_embed_q(x_spatial).flatten(2).transpose(1, 2)
        tokens_q = tokens + q_pos
        
        # Pre-LN 1
        tokens_norm = self.norm1(tokens_q)
        
        # Q projection
        q = self.q_proj(tokens_norm)  # [B, H*W, C]

        # --- KV Generation (Low Resolution & Deformed) ---
        x_norm_spatial = tokens_norm.transpose(1, 2).contiguous().view(b, c, h, w)
        
        if self.reduce_ratio > 1 and self.downsampler is not None:
            x_down = self.downsampler(x_norm_spatial)
        else:
            x_down = x_norm_spatial
            
        h_k, w_k = x_down.shape[2], x_down.shape[3]
        n_k = h_k * w_k
        
        # Offset prediction from x_down
        offset_px = torch.tanh(self.offset_conv(x_down)) * self.max_offset
        self.last_offset = offset_px
        self.last_offset_reg = (offset_px.square()).mean()
        
        w_div = max(float(w_k) / 2.0, 1.0)
        h_div = max(float(h_k) / 2.0, 1.0)
        offset_norm = torch.empty_like(offset_px)
        offset_norm[:, 0] = offset_px[:, 0] / w_div
        offset_norm[:, 1] = offset_px[:, 1] / h_div
        
        base_grid = self._base_grid(h_k, w_k, x.device, x.dtype).unsqueeze(0).expand(b, -1, -1, -1)
        offset_grid = offset_norm.permute(0, 2, 3, 1)
        sampling_grid = base_grid + offset_grid
        sampling_grid = torch.nan_to_num(sampling_grid, nan=0.0, posinf=1.0, neginf=-1.0)
        sampling_grid = torch.clamp(sampling_grid, -1.0, 1.0)
        
        # Deformable sampling
        x_sampled = F.grid_sample(
            x_down,
            sampling_grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=False,
        )
        
        # Flatten KV
        tokens_k = x_sampled.flatten(2).transpose(1, 2)  # [B, H_k*W_k, C]
        
        # Add Pos Embed KV (Depthwise Conv)
        kv_pos = self.pos_embed_kv(x_sampled).flatten(2).transpose(1, 2)
        tokens_k = tokens_k + kv_pos
        
        # KV projection
        kv = self.kv_proj(tokens_k)  # [B, H_k*W_k, 2C]
        k, v = kv.chunk(2, dim=-1)

        # --- Multi-Head Self-Attention ---
        q = q.view(b, n_q, self.num_heads, self.head_dim).transpose(1, 2)  # [B, heads, N_q, hd]
        k = k.view(b, n_k, self.num_heads, self.head_dim).transpose(1, 2)  # [B, heads, N_k, hd]
        v = v.view(b, n_k, self.num_heads, self.head_dim).transpose(1, 2)

        if hasattr(F, "scaled_dot_product_attention"):
            attn_out = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_drop.p if self.training else 0.0
            )
        else:
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn - attn.amax(dim=-1, keepdim=True)
            attn = torch.softmax(attn, dim=-1)
            attn = self.attn_drop(attn)
            attn_out = attn @ v

        attn_out = attn_out.transpose(1, 2).contiguous().view(b, n_q, c)  # [B, N_q, C]
        attn_out = self.proj(attn_out)
        attn_out = self.proj_drop(attn_out)

        # --- Residual & FFN ---
        # Residual 1 (High-Res tokens preserved)
        tokens_res = tokens + attn_out

        # Pre-LN 2 & FFN (Residual 2)
        tokens_norm2 = self.norm2(tokens_res)
        ffn_out = self.mlp(tokens_norm2)
        tokens_out = tokens_res + ffn_out

        # Reshape to Image (Matches original size perfectly, NO interpolation needed!)
        out_map = tokens_out.transpose(1, 2).contiguous().view(b, c, h, w)
        return out_map
