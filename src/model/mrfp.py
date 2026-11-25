
import torch
import torch.nn as nn
import torch.nn.functional as F


class MRFP(nn.Module):
    """The project's residual MRFP; full keeps the original state dict and math."""

    def __init__(self, ch=128, variant="full"):
        super().__init__()
        if variant not in ("full", "no-dilation", "no-switch", "no-dilation-no-switch"):
            raise ValueError(f"Unsupported residual MRFP variant: {variant}")
        self.variant = variant
        hidden = ch // 2
        dilation = 1 if variant in ("no-dilation", "no-dilation-no-switch") else 3
        use_switch = variant in ("full", "no-dilation")

        self.reduce = nn.Conv2d(ch, hidden, 1)

        self.conv3 = nn.Conv2d(
            hidden, hidden, 3, padding=1, dilation=1
        )
        self.conv_dilated = nn.Conv2d(
            hidden, hidden, 3, padding=dilation, dilation=dilation
        )

        self.switch = nn.Conv2d(hidden, 2, 1) if use_switch else None
        if self.switch is not None:
            nn.init.zeros_(self.switch.weight)
            nn.init.zeros_(self.switch.bias)

        self.expand = nn.Conv2d(hidden, ch, 1)
        self.gamma = nn.Parameter(torch.tensor(0.1))

    def forward(self, x):
        z = F.gelu(self.reduce(x))

        a = self.conv3(z)
        b = self.conv_dilated(z)

        if self.switch is not None:
            gates = torch.softmax(
                self.switch(
                    F.avg_pool2d(z, 5, stride=1, padding=2)
                ),
                dim=1
            )
            z = gates[:, 0:1] * a + gates[:, 1:2] * b
        else:
            z = 0.5 * a + 0.5 * b
        z = self.expand(F.gelu(z))

        return x + self.gamma * z


class SwitchableDepthwiseConv(nn.Module):
    """Neck-local SAC-style DW branches at dilation 1/3 with learned gates.

    As in this project's SAC, branches have independent convolution weights
    and zero-initialized switch logits. There is no backbone or residual here.
    """

    def __init__(self, channels, kernel_size):
        super().__init__()
        self.rates = (1, 3)
        self.branches = nn.ModuleList([
            nn.Conv2d(channels, channels, kernel_size,
                      padding=(kernel_size // 2) * rate, dilation=rate, groups=channels)
            for rate in self.rates
        ])
        self.switch = nn.Conv2d(channels, len(self.rates), 1)
        nn.init.zeros_(self.switch.weight)
        nn.init.zeros_(self.switch.bias)

    def forward(self, x):
        # Keep the softmax in fp32 under AMP, like the existing backbone SAC.
        gates = self.switch(F.avg_pool2d(x, 5, stride=1, padding=2)).float().softmax(1)
        return sum(gates[:, i:i + 1] * branch(x) for i, branch in enumerate(self.branches))


class CoMerMRFP(nn.Module):
    """ViT-CoMer's MRFP computation adapted to a list of pyramid feature maps.

    Shared pointwise projections implement the paper's per-token FC layers.
    Each level splits the reduced channels between independent DW 3x3/5x5
    filters, concatenates them, then uses its own BatchNorm and GELU.
    The additional GELU before expansion follows the official MRFP forward.
    The plain-dw reference has no switch, dilation, or residual/gamma.
    plain-dw-sac adds switches and dilation inside each kernel group only.
    single-dw replaces the split 3x3/5x5 filters with one 3x3 DW over all
    reduced channels. Projections, BN, activations and level structure stay
    identical across these three variants.

    Reference (MRFP, MultiDWConv, CTIBlock's hidden_features=dim*0.5):
    https://github.com/Traffic-X/ViT-CoMer/blob/main/detection/mmdet_custom/models/backbones/comer_modules.py
    This is a neck-module comparison, not the full ViT-CoMer CNN/CTI backbone.
    """

    def __init__(self, ch=128, num_levels=3, variant="plain-dw"):
        super().__init__()
        if variant not in ("plain-dw", "plain-dw-sac", "single-dw"):
            raise ValueError(f"Unsupported CoMer MRFP variant: {variant}")
        if ch < 4 or ch % 4:
            raise ValueError("ViT-CoMer-style MRFP requires channels divisible by 4")
        self.variant = variant
        hidden = ch // 2
        self.reduce = nn.Conv2d(ch, hidden, 1)

        def spatial_conv(channels, kernel):
            if variant == "plain-dw-sac":
                return SwitchableDepthwiseConv(channels, kernel)
            return nn.Conv2d(channels, channels, kernel, padding=kernel // 2, groups=channels)

        self.levels = nn.ModuleList()
        for _ in range(num_levels):
            spatial = ({"dw": spatial_conv(hidden, 3)} if variant == "single-dw" else {
                "dw3": spatial_conv(hidden // 2, 3),
                "dw5": spatial_conv(hidden // 2, 5),
            })
            self.levels.append(nn.ModuleDict({**spatial, "norm": nn.BatchNorm2d(hidden)}))
        self.expand = nn.Conv2d(hidden, ch, 1)

    def forward(self, features):
        if len(features) != len(self.levels):
            raise ValueError("Number of input features must match the configured pyramid levels")
        outputs = []
        for feature, level in zip(features, self.levels):
            z = self.reduce(feature)
            if self.variant == "single-dw":
                z = level["dw"](z)
            else:
                a, b = z.chunk(2, dim=1)
                z = torch.cat((level["dw3"](a), level["dw5"](b)), dim=1)
            z = F.gelu(level["norm"](z))
            outputs.append(self.expand(F.gelu(z)))
        return outputs


class MRFPNeck(nn.Module):
    def __init__(self, neck, ch=128, variant="full"):
        super().__init__()
        self.fpn = neck
        self.strides = neck.strides
        self.variant = variant
        self.mrfp = (CoMerMRFP(ch, len(self.strides), variant=variant)
                     if variant in ("plain-dw", "plain-dw-sac", "single-dw")
                     else MRFP(ch, variant=variant))

    def forward(self, x):
        features = self.fpn(x)
        if isinstance(self.mrfp, CoMerMRFP):
            return self.mrfp(features)
        return [self.mrfp(feature) for feature in features]
