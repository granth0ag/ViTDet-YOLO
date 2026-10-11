# evaluate.py
from pathlib import Path

import torch
from torchmetrics.detection import MeanAveragePrecision
try:
    from ultralytics.utils.nms import non_max_suppression
except ImportError:
    from ultralytics.utils.ops import non_max_suppression

from src.dataset import build_folder_loaders, NUM_CLASSES, CLASS_NAMES, DATA_ROOT
from src.model.build import build_model

# ---------------- config  ----------------
IMG_SIZE = 640
USE_P2 = False
BATCH_SIZE = 4
TAG = f"wildlife_{IMG_SIZE}_{'p2' if USE_P2 else 'p3p5'}"
CKPT = f"best_{TAG}.pt"   # best val mAP wala checkpoint (val se chuna gaya, to test leak nahi)
CONF_THRES = 0.001        # mAP ke liye low rakhte hain, warna recall kat jaata hai
IOU_THRES = 0.65
device = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------- data (sirf test split) ----------------
(test_loader,) = build_folder_loaders(
    DATA_ROOT, splits=("test",), batch_size=BATCH_SIZE, img_size=IMG_SIZE,
    augment=False, num_workers=2)
print(f"test images: {len(test_loader.dataset)} | checkpoint: {CKPT}")

# ---------------- model ----------------
model = build_model(IMG_SIZE, NUM_CLASSES, use_p2=USE_P2).to(device)
model.load_state_dict(torch.load(CKPT, map_location=device))
model.eval()


def targets_to_xyxy(t, size):
    """[n,5] (cls, cx, cy, w, h) normalized -> boxes xyxy pixels, labels."""
    cx, cy, w, h = t[:, 1] * size, t[:, 2] * size, t[:, 3] * size, t[:, 4] * size
    boxes = torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=1)
    return boxes, t[:, 0].long()


@torch.no_grad()
def evaluate():
    metric = MeanAveragePrecision(box_format="xyxy", iou_type="bbox", class_metrics=True)
    metric.warn_on_many_detections = False

    for images, targets, _ in test_loader:
        y, _ = model(images.to(device))                     # [B, 4+nc, anchors], boxes xywh pixels
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
    print(f"mAP small    : {r['map_small'].item():.4f}   (-1 = is size ka koi GT nahi)")
    print(f"mAP medium   : {r['map_medium'].item():.4f}")
    print(f"mAP large    : {r['map_large'].item():.4f}")
    print(f"mAR@100      : {r['mar_100'].item():.4f}")

    if r["map_per_class"].numel() > 1:
        print("per-class mAP@[.5:.95]:")
        for c, ap in zip(r["classes"].tolist(), r["map_per_class"].tolist()):
            print(f"  {CLASS_NAMES[c]:10s} {ap:.4f}")