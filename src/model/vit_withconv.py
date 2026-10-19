# src/model/vit_withconv.py
import copy

import torch.nn as nn
from einops import rearrange

from src.model.vit import ViTDetBackbone


class LocalMlp(nn.Module):
    """
    LocalViT-style FFN (grid input/output [B, H, W, C]):
        1x1 conv (d -> 4d) -> act -> 3x3 depthwise conv -> [act] -> 1x1 conv (4d -> d)

    Weights pretrained timm Mlp se copy hote hain, aur depthwise conv identity se start hota hai.
    act_after_dw=False pe init pe ye BILKUL pretrained MLP jaisa chalta hai.
    act_after_dw=True  pe paper wala (dono jagah activation), par init pe pretrained function thoda badal jaata hai.
    """

    def __init__(self, mlp, act_after_dw=False):
        super().__init__()
        dim, hidden = mlp.fc1.in_features, mlp.fc1.out_features       # 384, 1536
        self.expand = nn.Conv2d(dim, hidden, kernel_size=1)
        self.dw = nn.Conv2d(hidden, hidden, kernel_size=3, padding=1, groups=hidden)
        self.project = nn.Conv2d(hidden, dim, kernel_size=1)
        self.act = copy.deepcopy(mlp.act)                              # GELU (pretrained wala)
        self.act2 = copy.deepcopy(mlp.act) if act_after_dw else nn.Identity()

        with nn.no_grad() if False else __import__("torch").no_grad():
            self.expand.weight.copy_(mlp.fc1.weight[:, :, None, None])
            self.expand.bias.copy_(mlp.fc1.bias)
            self.project.weight.copy_(mlp.fc2.weight[:, :, None, None])
            self.project.bias.copy_(mlp.fc2.bias)
            self.dw.weight.zero_()
            self.dw.weight[:, 0, 1, 1] = 1.0                           # identity kernel
            self.dw.bias.zero_()

    def forward(self, x):                                              # [B, H, W, C]
        x = rearrange(x, "b h w c -> b c h w")
        x = self.act(self.expand(x))
        x = self.act2(self.dw(x))
        x = self.project(x)
        return rearrange(x, "b c h w -> b h w c")


class ViTDetConvBackbone(ViTDetBackbone):
    """ViTDetBackbone + LocalViT-style conv MLP. forward() parent wala hi hai."""

    def __init__(self, img_size=640, window_size=10, global_idx=(2, 5, 8, 11),
                 drop_path_rate=0.0, act_after_dw=False, conv_blocks=None):
        super().__init__(img_size, window_size, global_idx, drop_path_rate)
        idx = range(len(self.vit.blocks)) if conv_blocks is None else conv_blocks
        for i in idx:
            blk = self.vit.blocks[i]
            blk.mlp = LocalMlp(blk.mlp, act_after_dw)