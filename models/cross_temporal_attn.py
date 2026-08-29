import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossTemporalAttention(nn.Module):
    """Bidirectional Cross-Temporal Attention for Change Detection.

    Instead of self-attention on individual images, this module lets
    pre features directly attend to post features (and vice versa).

    This is the key architectural difference that allows models like BIT
    to reach 80%+ IoU: the network learns *relationships* between the
    two time steps, not just pixel-level differences.
    """

    def __init__(
        self,
        channels: int,
        num_heads: int = 8,
        reduce_ratio: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        if channels % num_heads != 0:
            raise ValueError(f"channels ({channels}) must be divisible by num_heads ({num_heads})")

        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.reduce_ratio = int(reduce_ratio)

        # Pre-LN for Q and KV
        self.norm_q = nn.LayerNorm(channels)
        self.norm_kv = nn.LayerNorm(channels)

        # Projections (shared for both directions — parameter efficient & symmetric)
        self.q_proj = nn.Linear(channels, channels, bias=True)
        self.kv_proj = nn.Linear(channels, channels * 2, bias=True)
        self.proj = nn.Linear(channels, channels, bias=True)
        self.proj_drop = nn.Dropout(dropout)

        # KV downsampler for efficiency (depthwise strided conv)
        if self.reduce_ratio > 1:
            self.downsampler = nn.Conv2d(
                channels, channels,
                kernel_size=3, stride=self.reduce_ratio, padding=1,
                groups=channels, bias=False,
            )
        else:
            self.downsampler = None

        # Positional encoding (depthwise conv — translation equivariant)
        self.pos_q = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels)
        self.pos_kv = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels)

        # FFN block
        self.norm2 = nn.LayerNorm(channels)
        self.mlp = nn.Sequential(
            nn.Linear(channels, 4 * channels),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(4 * channels, channels),
            nn.Dropout(dropout),
        )

    def _cross_attend(self, x_q: torch.Tensor, x_kv: torch.Tensor) -> torch.Tensor:
        """Single direction: Q from x_q attends to KV from x_kv.

        Args:
            x_q:  [B, C, H, W] — the image whose features will be enriched
            x_kv: [B, C, H, W] — the other temporal image providing context

        Returns:
            [B, C, H, W] — x_q enriched with temporal context from x_kv
        """
        b, c, h, w = x_q.shape
        n_q = h * w

        # ---- Q path (full resolution) ----
        x_q_pos = x_q + self.pos_q(x_q)
        tokens_q_raw = x_q.flatten(2).transpose(1, 2)           # [B, HW, C] for residual
        tokens_q = x_q_pos.flatten(2).transpose(1, 2)           # [B, HW, C]

        # ---- KV path (reduced resolution for efficiency) ----
        if self.downsampler is not None:
            x_kv_down = self.downsampler(x_kv)
        else:
            x_kv_down = x_kv

        x_kv_pos = x_kv_down + self.pos_kv(x_kv_down)
        tokens_kv = x_kv_pos.flatten(2).transpose(1, 2)         # [B, H'W', C]
        n_k = tokens_kv.shape[1]

        # ---- Pre-LN + Projections ----
        q = self.q_proj(self.norm_q(tokens_q))                   # [B, HW, C]
        kv = self.kv_proj(self.norm_kv(tokens_kv))               # [B, H'W', 2C]
        k, v = kv.chunk(2, dim=-1)

        # ---- Multi-Head Cross-Attention ----
        q = q.view(b, n_q, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(b, n_k, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, n_k, self.num_heads, self.head_dim).transpose(1, 2)

        if hasattr(F, "scaled_dot_product_attention"):
            attn_out = F.scaled_dot_product_attention(q, k, v)
        else:
            scale = self.head_dim ** -0.5
            attn = (q @ k.transpose(-2, -1)) * scale
            attn = attn - attn.amax(dim=-1, keepdim=True)
            attn = torch.softmax(attn, dim=-1)
            attn_out = attn @ v

        attn_out = attn_out.transpose(1, 2).contiguous().view(b, n_q, c)
        attn_out = self.proj_drop(self.proj(attn_out))

        # ---- Residual 1 ----
        tokens_res = tokens_q_raw + attn_out

        # ---- FFN (Pre-LN) + Residual 2 ----
        tokens_out = tokens_res + self.mlp(self.norm2(tokens_res))

        return tokens_out.transpose(1, 2).contiguous().view(b, c, h, w)

    def forward(self, x_pre: torch.Tensor, x_post: torch.Tensor):
        """Bidirectional cross-temporal attention.

        Args:
            x_pre:  [B, C, H, W] — features from the pre-change image
            x_post: [B, C, H, W] — features from the post-change image

        Returns:
            pre_attended:  [B, C, H, W] — pre features enriched with post context
            post_attended: [B, C, H, W] — post features enriched with pre context
        """
        pre_attended = self._cross_attend(x_pre, x_post)   # pre asks: "what's in post?"
        post_attended = self._cross_attend(x_post, x_pre)   # post asks: "what's in pre?"
        return pre_attended, post_attended
