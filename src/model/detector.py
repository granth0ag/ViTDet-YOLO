import torch
import torch.nn as nn
from types import SimpleNamespace
from ultralytics.nn.modules.head import Detect


class YoloHead(nn.Module):
    """Takes [p3, p4, p5] (all 128 ch) -> raw YOLOv8 predictions."""
    def __init__(self, num_classes=80, ch=(128, 128, 128), strides=(8, 16, 32)):
        super().__init__()
        self.detect = Detect(nc=num_classes, ch=ch)
        # Keep the Detect stride on the model device without changing checkpoint keys.
        del self.detect.stride
        self.detect.register_buffer("stride", torch.tensor(strides, dtype=torch.float32), persistent=False)
        self.detect.bias_init()          # sensible init for cls/box biases

    def forward(self, feats):            # feats = [p3, p4, p5]
        return self.detect(list(feats))


class DetectionModel(nn.Module):
    """Thin wrapper because v8DetectionLoss expects model.model[-1] and model.args."""
    def __init__(self, backbone, neck, head, center_channels=None):
        super().__init__()
        self.backbone, self.neck, self.head = backbone, neck, head
        self.model = nn.ModuleList([head.detect])                       # loss looks here
        self.args = SimpleNamespace(box=7.5, cls=0.5, dfl=1.5)          # loss gains

        self.center_head = (
            nn.Conv2d(center_channels, 2, kernel_size=1) if center_channels is not None else None
        )
        if self.center_head is not None:
            nn.init.zeros_(self.center_head.weight)
            nn.init.zeros_(self.center_head.bias)

    def forward(self, x):
        feats = self.neck(self.backbone(x))
        preds = self.head(feats)
        if self.training and self.center_head is not None:
            # Level and spatial order match Ultralytics' flattened anchor order.
            offsets = torch.cat([self.center_head(f).flatten(2) for f in feats], dim=2)
            return preds, offsets.transpose(1, 2)
        return preds
