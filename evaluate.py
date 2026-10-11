# evaluate.py
from pathlib import Path

import torch
from torchmetrics.detection import MeanAveragePrecision
try:
    from ultralytics.utils.nms import non_max_suppression
except ImportError:
    from ultralytics.utils.ops import non_max_suppression

from src.dataset import build_coco_subset_loaders
from src.model.vit import ViTDetBackbone
from src.model.fpn import SimpleFeaturePyramid
from src.model.detector import YoloHead, DetectionModel

# ---------------- config ----------------
IMG_SIZE = 320
BATCH_SIZE = 8
CKPT = "last.pt"         # best.pt nahi: test se checkpoint chunna test leak hota
CONF_THRES = 0.001       # mAP ke liye low rakhte hain, warna recall kat jaata hai
IOU_THRES = 0.65
DATA_ROOT = "/kaggle/temp/coco_subset" if Path("/kaggle").exists() else "data/coco_subset"
device = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------- data (sirf test split) ----------------
_, test_loader = build_coco_subset_loaders(
    root=DATA_ROOT, batch_size=BATCH_SIZE, img_size=IMG_SIZE, augment=False, num_workers=2)
print(f"test images: {len(test_loader.dataset)}")

# ---------------- model ----------------
model = DetectionModel(
    ViTDetBackbone(img_size=IMG_SIZE),
    SimpleFeaturePyramid(in_ch=384, out_ch=128),
    YoloHead(num_classes=80, ch=(128, 128, 128)),
).to(device)
model.load_state_dict(torch.load(CKPT, map_location=device))
model.eval()


def targets_to_xyxy(t, size):
    """[n,5] (cls, cx, cy, w, h) normalized -> boxes xyxy pixels, labels."""
    cx, cy, w, h = t[:, 1] * size, t[:, 2] * size, t[:, 3] * size, t[:, 4] * size
    boxes = torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=1)
    return boxes, t[:, 0].long()


@torch.no_grad()
def evaluate():
    metric = MeanAveragePrecision(box_format="xyxy", iou_type="bbox")

    for images, targets, _ in test_loader:
        y, _ = model(images.to(device))                     # [B, 84, 2100], boxes xywh pixels
        dets = non_max_suppression(y, conf_thres=CONF_THRES, iou_thres=IOU_THRES)

        preds, gts = [], []
        for d, t in zip(dets, targets):
            preds.append({"boxes": d[:, :4].cpu(),
                          "scores": d[:, 4].cpu(),
                          "labels": d[:, 5].long().cpu()})
            boxes, labels = targets_to_xyxy(t, IMG_SIZE)
            gts.append({"boxes": boxes, "labels": labels})

        metric.update(preds, gts)

    return metric.compute()


if __name__ == "__main__":
    r = evaluate()
    print(f"mAP@[.5:.95] : {r['map'].item():.4f}")
    print(f"mAP@0.5      : {r['map_50'].item():.4f}")
    print(f"mAP@0.75     : {r['map_75'].item():.4f}")
    print(f"mAP small    : {r['map_small'].item():.4f}")
    print(f"mAP medium   : {r['map_medium'].item():.4f}")
    print(f"mAP large    : {r['map_large'].item():.4f}")
    print(f"mAR@100      : {r['mar_100'].item():.4f}")