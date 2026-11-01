import torch
import torch.nn as nn
from src.model.vit import ViTDetBackbone


class SAC(nn.Module):
    """Switchable atrous locality. Input/output: [B, H, W, C] (what norm1 returns)."""
    def __init__(self, dim, rates=(1, 3, 5)):
        super().__init__()
        self.rates = tuple(rates)
        self.branches = nn.ModuleList(
            nn.Conv2d(dim, dim, 3, padding=r, dilation=r, groups=dim) for r in self.rates)
        self.pool = nn.AvgPool2d(5, stride=1, padding=2)
        self.switch = nn.Conv2d(dim, len(self.rates), 1)
        nn.init.zeros_(self.switch.weight); nn.init.zeros_(self.switch.bias)  # gates start 1/R
        self.gamma = nn.Parameter(torch.zeros(dim))                           # starts as baseline
        self.register_buffer("gate_sum", torch.zeros(len(self.rates)), persistent=False)
        self.n = 0

    def forward(self, y):
        f = y.permute(0, 3, 1, 2)                              # [B, C, H, W]
        g = self.switch(self.pool(f)).float().softmax(1)       # [B, R, H, W], fp32 under AMP
        out = sum(g[:, i:i + 1] * b(f) for i, b in enumerate(self.branches))
        if self.training:
            self.gate_sum += g.mean((0, 2, 3)).detach(); self.n += 1
        return y + self.gamma * out.permute(0, 2, 3, 1)

    def pop_gates(self):
        g = self.gate_sum / max(self.n, 1)
        self.gate_sum.zero_(); self.n = 0
        return g


class ViTDetSACBackbone(ViTDetBackbone):
    def __init__(self, *args, sac_rates=(1, 3, 5), **kw):
        super().__init__(*args, **kw)
        for blk in self.vit.blocks:
            blk.norm1 = nn.Sequential(blk.norm1, SAC(self.embed_dim, sac_rates))