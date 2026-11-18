
import torch
import torch.nn as nn
import torch.nn.functional as F


class MRFP(nn.Module):
    def __init__(self, ch=128):
        super().__init__()
        hidden = ch // 2

        self.reduce = nn.Conv2d(ch, hidden, 1)

        self.conv3 = nn.Conv2d(
            hidden, hidden, 3, padding=1, dilation=1
        )
        self.conv_dilated = nn.Conv2d(
            hidden, hidden, 3, padding=3, dilation=3
        )

        self.switch = nn.Conv2d(hidden, 2, 1)
        nn.init.zeros_(self.switch.weight)
        nn.init.zeros_(self.switch.bias)

        self.expand = nn.Conv2d(hidden, ch, 1)
        self.gamma = nn.Parameter(torch.tensor(0.1))

    def forward(self, x):
        z = F.gelu(self.reduce(x))

        a = self.conv3(z)
        b = self.conv_dilated(z)

        gates = torch.softmax(
            self.switch(
                F.avg_pool2d(z, 5, stride=1, padding=2)
            ),
            dim=1
        )

        z = gates[:, 0:1] * a + gates[:, 1:2] * b
        z = self.expand(F.gelu(z))

        return x + self.gamma * z


class MRFPNeck(nn.Module):
    def __init__(self, neck, ch=128):
        super().__init__()
        self.fpn = neck
        self.strides = neck.strides
        self.mrfp = MRFP(ch)

    def forward(self, x):
        features = self.fpn(x)
        return [self.mrfp(feature) for feature in features]
