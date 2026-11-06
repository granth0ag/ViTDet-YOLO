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

# # import torch
# # import torch.nn as nn
# # import torch.nn.functional as F
# # from einops import rearrange

# # from src.model.vit import ViTDetBackbone, window_partition, window_unpartition


# # class SAC(nn.Module):
# #     """Ab use nahi hota. Sirf isliye rakha hai ki train.py ka `from src.model.vit_sac import SAC` aur
# #     isinstance(m, SAC) na tute. LIFE run mein model ke andar iska koi instance nahi banta."""
# #     def __init__(self, dim, rates=(1, 3, 5)):
# #         super().__init__()
# #         self.rates = tuple(rates)
# #         self.branches = nn.ModuleList(
# #             nn.Conv2d(dim, dim, 3, padding=r, dilation=r, groups=dim) for r in self.rates)
# #         self.switch = nn.Conv2d(dim, len(self.rates), 1)
# #         nn.init.zeros_(self.switch.weight); nn.init.zeros_(self.switch.bias)
# #         for b in self.branches:
# #             nn.init.zeros_(b.weight); nn.init.zeros_(b.bias)
# #             b.weight.data[:, 0, 1, 1] = 1.0
# #         self.gamma = nn.Parameter(torch.zeros(dim))
# #         self.register_buffer("gate_sum", torch.zeros(len(self.rates)), persistent=False)
# #         self.n = 0

# #     def forward(self, y, f, p):
# #         g = self.switch(p).float().softmax(1)
# #         out = sum(g[:, i:i + 1] * b(f) for i, b in enumerate(self.branches))
# #         if self.training:
# #             self.gate_sum += g.mean((0, 2, 3)).detach(); self.n += 1
# #         return out.permute(0, 2, 3, 1)

# #     def pop_gates(self):
# #         g = self.gate_sum / max(self.n, 1)
# #         self.gate_sum.zero_(); self.n = 0
# #         return g


# # class _GateStub(nn.Module):
# #     """Sirf train.py ke logging loop ke liye (LIFE mein gates/gamma hote hi nahi). Zeros return karta hai."""
# #     def __init__(self, dim, rates):
# #         super().__init__()
# #         self.rates = tuple(rates)
# #         self.register_buffer("gamma", torch.zeros(dim), persistent=False)

# #     def pop_gates(self):
# #         return torch.zeros(len(self.rates))


# # class DSC(nn.Module):
# #     """Depthwise separable conv: depthwise k x k, phir pointwise 1x1."""
# #     def __init__(self, dim, k):
# #         super().__init__()
# #         self.dw = nn.Conv2d(dim, dim, k, padding=k // 2, groups=dim)
# #         self.pw = nn.Conv2d(dim, dim, 1)

# #     def forward(self, x):
# #         return self.pw(self.dw(x))


# # class QKVSAC(nn.Module):
# #     """LIFE module (qkv linear ki jagah).
# #         F1 = PW(x) | F2 = DSC3(F1) | F3 = DSC5(F2)
# #         har F_i ke 3 channel-chunks (q_i, k_i, v_i);  Q = cat(q1,q2,q3), K = cat(k1,k2,k3), V = cat(v1,v2,v3)
# #     Input y [B,H,W,C] -> q, k, v har ek [B,H,W,C]."""
# #     def __init__(self, dim, rates):
# #         super().__init__()
# #         assert dim % 3 == 0, "LIFE channels ko 3 hisson mein baanta hai"
# #         self.conv1 = nn.Conv2d(dim, dim, 1)
# #         self.conv2 = DSC(dim, 3)
# #         self.conv3 = DSC(dim, 5)
# #         self.q, self.k, self.v = _GateStub(dim, rates), _GateStub(dim, rates), _GateStub(dim, rates)

# #     def forward(self, y):
# #         x = y.permute(0, 3, 1, 2)                               # [B,C,H,W]
# #         f1 = self.conv1(x)
# #         f2 = self.conv2(f1)
# #         f3 = self.conv3(f2)
# #         q1, k1, v1 = f1.chunk(3, dim=1)
# #         q2, k2, v2 = f2.chunk(3, dim=1)
# #         q3, k3, v3 = f3.chunk(3, dim=1)
# #         q = torch.cat([q1, q2, q3], dim=1).permute(0, 2, 3, 1)  # -> [B,H,W,C]
# #         k = torch.cat([k1, k2, k3], dim=1).permute(0, 2, 3, 1)
# #         v = torch.cat([v1, v2, v3], dim=1).permute(0, 2, 3, 1)
# #         return q, k, v


# # class ViTDetSACBackbone(ViTDetBackbone):
# #     def __init__(self, *args, sac_rates=(1, 3, 5), **kw):
# #         super().__init__(*args, **kw)
# #         for blk in self.vit.blocks:
# #             blk.qkv_sac = QKVSAC(self.embed_dim, sac_rates)

# #     @staticmethod
# #     def _attend(attn, q, k, v):
# #         q, k, v = (rearrange(t, "b n (h d) -> b h n d", h=attn.num_heads) for t in (q, k, v))
# #         out = F.scaled_dot_product_attention(q, k, v)
# #         return attn.proj(rearrange(out, "b h n d -> b n (h d)"))

# #     def forward(self, x):
# #         x = rearrange(self.vit.patch_embed.proj(x), "b c h w -> b h w c")
# #         x = x + self.pos_embed
# #         B, H, W, C = x.shape
# #         ws = self.window_size

# #         for i, blk in enumerate(self.vit.blocks):
# #             y = blk.norm1(x)
# #             q, k, v = blk.qkv_sac(y)                            # LIFE: full grid, window partition se pehle

# #             if i in self.global_idx:
# #                 q, k, v = (rearrange(t, "b h w c -> b (h w) c") for t in (q, k, v))
# #                 out = rearrange(self._attend(blk.attn, q, k, v), "b (h w) c -> b h w c", h=H, w=W)
# #             else:
# #                 q, k, v = (window_partition(t, ws) for t in (q, k, v))
# #                 out = window_unpartition(self._attend(blk.attn, q, k, v), ws, H, W)

# #             x = x + blk.drop_path1(blk.ls1(out))
# #             x = x + blk.drop_path2(blk.ls2(blk.mlp(blk.norm2(x))))

# #         x = self.vit.norm(x)
# #         return rearrange(x, "b h w c -> b c h w")