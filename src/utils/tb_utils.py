# src/utils/tb_utils.py
import numpy as np
import torch
from PIL import Image, ImageDraw
from torchmetrics.detection import MeanAveragePrecision
try:
    from ultralytics.utils.nms import non_max_suppression
except ImportError:
    from ultralytics.utils.ops import non_max_suppression

MEAN = torch.tensor((0.485, 0.456, 0.406)).view(3, 1, 1)
STD = torch.tensor((0.229, 0.224, 0.225)).view(3, 1, 1)


def to_xyxy(t, size):
    """[n,5] (cls, cx, cy, w, h) normalized -> xyxy pixels, labels."""
    cx, cy, w, h = t[:, 1] * size, t[:, 2] * size, t[:, 3] * size, t[:, 4] * size
    return torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], 1), t[:, 0].long()


@torch.no_grad()
def evaluate_map(model, loader, device, img_size=320, conf=0.001, iou=0.65):
    """Returns torchmetrics dict (map, map_50, map_small, ...)."""
    model.eval()
    metric = MeanAveragePrecision(box_format="xyxy", iou_type="bbox")
    for images, targets, _ in loader:
        y, _ = model(images.to(device))
        dets = non_max_suppression(y, conf_thres=conf, iou_thres=iou)
        preds, gts = [], []
        for d, t in zip(dets, targets):
            preds.append({"boxes": d[:, :4].cpu(), "scores": d[:, 4].cpu(),
                          "labels": d[:, 5].long().cpu()})
            boxes, labels = to_xyxy(t, img_size)
            gts.append({"boxes": boxes, "labels": labels})
        metric.update(preds, gts)
    return metric.compute()


@torch.no_grad()
def detection_images(model, loader, device, n=4, img_size=320, conf=0.25, iou=0.45):
    """n images with GT (green) + predictions (red). Returns uint8 tensor [n,3,H,W]."""
    model.eval()
    images, targets, _ = next(iter(loader))
    images, targets = images[:n], targets[:n]
    y, _ = model(images.to(device))
    dets = non_max_suppression(y, conf_thres=conf, iou_thres=iou)

    out = []
    for img, t, d in zip(images, targets, dets):
        pil = Image.fromarray(((img * STD + MEAN).clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8))
        draw = ImageDraw.Draw(pil)
        gt_boxes, gt_labels = to_xyxy(t, img_size)
        for b, c in zip(gt_boxes.tolist(), gt_labels.tolist()):
            draw.rectangle(b, outline="lime", width=2)
        for x1, y1, x2, y2, cf, c in d.cpu().tolist():
            draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
            draw.text((x1 + 2, y1 + 2), f"{int(c)} {cf:.2f}", fill="red")
        out.append(torch.from_numpy(np.array(pil)).permute(2, 0, 1))
    return torch.stack(out)