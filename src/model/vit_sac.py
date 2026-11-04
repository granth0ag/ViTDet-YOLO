# import torch
# import torch.nn as nn
# from src.model.vit import ViTDetBackbone


# class SAC(nn.Module):
#     """Switchable atrous locality. Input/output: [B, H, W, C] (what norm1 returns)."""
#     def __init__(self, dim, rates=(1, 3, 5)):
#         super().__init__()
#         self.rates = tuple(rates)
#         self.branches = nn.ModuleList(
#             nn.Conv2d(dim, dim, 3, padding=r, dilation=r, groups=dim) for r in self.rates)
#         self.pool = nn.AvgPool2d(5, stride=1, padding=2)
#         self.switch = nn.Conv2d(dim, len(self.rates), 1)
#         nn.init.zeros_(self.switch.weight); nn.init.zeros_(self.switch.bias)  # gates start 1/R
#         self.gamma = nn.Parameter(torch.zeros(dim))                           # starts as baseline
#         self.register_buffer("gate_sum", torch.zeros(len(self.rates)), persistent=False)
#         self.n = 0

#     def forward(self, y):
#         f = y.permute(0, 3, 1, 2)                              # [B, C, H, W]
#         g = self.switch(self.pool(f)).float().softmax(1)       # [B, R, H, W], fp32 under AMP
#         out = sum(g[:, i:i + 1] * b(f) for i, b in enumerate(self.branches))
#         if self.training:
#             self.gate_sum += g.mean((0, 2, 3)).detach(); self.n += 1
#         return y + self.gamma * out.permute(0, 2, 3, 1)

#     def pop_gates(self):
#         g = self.gate_sum / max(self.n, 1)
#         self.gate_sum.zero_(); self.n = 0
#         return g


# class ViTDetSACBackbone(ViTDetBackbone):
#     def __init__(self, *args, sac_rates=(1, 3, 5), **kw):
#         super().__init__(*args, **kw)
#         for blk in self.vit.blocks:
#             blk.norm1 = nn.Sequential(blk.norm1, SAC(self.embed_dim, sac_rates))

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from src.model.vit import ViTDetBackbone, window_partition, window_unpartition


class SAC(nn.Module):
    """One switchable-atrous stream (used 3x per block: q, k, v).
    y: [B,H,W,C] channels-last, f: same as [B,C,H,W], p: avg-pooled f. Returns [B,H,W,C]."""
    def __init__(self, dim, rates=(1, 3, 5)):
        super().__init__()
        self.rates = tuple(rates)
        self.branches = nn.ModuleList(
            nn.Conv2d(dim, dim, 3, padding=r, dilation=r, groups=dim) for r in self.rates)
        self.switch = nn.Conv2d(dim, len(self.rates), 1)
        nn.init.zeros_(self.switch.weight); nn.init.zeros_(self.switch.bias)   # gates start 1/R
        for b in self.branches:
            nn.init.zeros_(b.weight); nn.init.zeros_(b.bias)
            b.weight.data[:, 0, 1, 1] = 1.0 
        self.gamma = nn.Parameter(torch.zeros(dim))                            # starts as baseline
        self.register_buffer("gate_sum", torch.zeros(len(self.rates)), persistent=False)
        self.n = 0

    def forward(self, y, f, p):
        g = self.switch(p).float().softmax(1)                                  # [B,R,H,W], fp32
        out = sum(g[:, i:i + 1] * b(f) for i, b in enumerate(self.branches))
        if self.training:
            self.gate_sum += g.mean((0, 2, 3)).detach(); self.n += 1
        # return y + self.gamma * out.permute(0, 2, 3, 1)
        return out.permute(0, 2, 3, 1)

    def pop_gates(self):
        g = self.gate_sum / max(self.n, 1)
        self.gate_sum.zero_(); self.n = 0
        return g


class QKVSAC(nn.Module):
    """Per-block container: separate SAC for q, k, v. Pool is shared (parameter-free)."""
    def __init__(self, dim, rates):
        super().__init__()
        self.q, self.k, self.v = SAC(dim, rates), SAC(dim, rates), SAC(dim, rates)
        self.pool = nn.AvgPool2d(5, stride=1, padding=2)

    def forward(self, y):
        f = y.permute(0, 3, 1, 2)
        p = self.pool(f)
        return self.q(y, f, p), self.k(y, f, p), self.v(y, f, p)


class ViTDetSACBackbone(ViTDetBackbone):
    def __init__(self, *args, sac_rates=(1, 3, 5), **kw):
        super().__init__(*args, **kw)
        for blk in self.vit.blocks:
            blk.qkv_sac = QKVSAC(self.embed_dim, sac_rates)

    @staticmethod
    def _attend(attn, q, k, v):
        q, k, v = (rearrange(t, "b n (h d) -> b h n d", h=attn.num_heads) for t in (q, k, v))
        out = F.scaled_dot_product_attention(q, k, v)
        return attn.proj(rearrange(out, "b h n d -> b n (h d)"))

    def forward(self, x):
        x = rearrange(self.vit.patch_embed.proj(x), "b c h w -> b h w c")
        x = x + self.pos_embed
        B, H, W, C = x.shape
        ws = self.window_size

        for i, blk in enumerate(self.vit.blocks):
            y = blk.norm1(x)
            yq, yk, yv = blk.qkv_sac(y)                       # full grid, BEFORE window partition

            # pointwise 1x1 = pretrained qkv weights, sliced per stream
            Wt, bias = blk.attn.qkv.weight, blk.attn.qkv.bias
            wq, wk, wv = Wt.chunk(3, 0)
            bq, bk, bv = bias.chunk(3, 0) if bias is not None else (None, None, None)
            q, k, v = F.linear(yq, wq, bq), F.linear(yk, wk, bk), F.linear(yv, wv, bv)

            if i in self.global_idx:
                q, k, v = (rearrange(t, "b h w c -> b (h w) c") for t in (q, k, v))
                out = rearrange(self._attend(blk.attn, q, k, v), "b (h w) c -> b h w c", h=H, w=W)
            else:
                q, k, v = (window_partition(t, ws) for t in (q, k, v))
                out = window_unpartition(self._attend(blk.attn, q, k, v), ws, H, W)

            x = x + blk.drop_path1(blk.ls1(out))
            x = x + blk.drop_path2(blk.ls2(blk.mlp(blk.norm2(x))))

        x = self.vit.norm(x)
        return rearrange(x, "b h w c -> b c h w")