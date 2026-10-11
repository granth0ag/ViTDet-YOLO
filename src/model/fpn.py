import torch
import torch.nn as nn


class SimpleFeaturePyramid(nn.Module):
    """ViT stride-16 map -> [P2 (stride 4)], P3 (8), P4 (16), P5 (32), sab out_ch channels."""

    def __init__(self, in_ch=384, out_ch=128, use_p2=False):
        super().__init__()
        self.use_p2 = use_p2
        self.strides = (4, 8, 16, 32) if use_p2 else (8, 16, 32)

        def head(c):
            return nn.Sequential(
                nn.Conv2d(c, out_ch, 1, bias=False), nn.GroupNorm(8, out_ch),
                nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False), nn.GroupNorm(8, out_ch))

        if use_p2:   # 2 deconvs: x4 upsample
            self.to_p2 = nn.Sequential(
                nn.ConvTranspose2d(in_ch, in_ch // 2, 2, 2), nn.GroupNorm(8, in_ch // 2), nn.GELU(),
                nn.ConvTranspose2d(in_ch // 2, in_ch // 4, 2, 2))
            self.head_p2 = head(in_ch // 4)
        self.to_p3 = nn.ConvTranspose2d(in_ch, in_ch // 2, 2, 2)
        self.to_p4 = nn.Identity()
        self.to_p5 = nn.MaxPool2d(2, 2)
        self.head_p3, self.head_p4, self.head_p5 = head(in_ch // 2), head(in_ch), head(in_ch)

    def forward(self, x):
        outs = [self.head_p3(self.to_p3(x)), self.head_p4(self.to_p4(x)), self.head_p5(self.to_p5(x))]
        if self.use_p2:
            outs.insert(0, self.head_p2(self.to_p2(x)))
        return outs