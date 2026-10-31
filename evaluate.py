# evaluate.py
import argparse

import torch
from torchmetrics.detection import MeanAveragePrecision
try:
    from ultralytics.utils.nms import non_max_suppression
except ImportError:
    from ultralytics.utils.ops import non_max_suppression

from src.dataset import build_folder_loaders, DATASETS
from src.model.build import build_model, run_tag

# ---------------- args ----------------
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default="wildlife", choices=list(DATASETS))
parser.add_argument("--img-size", type=int, default=None, help="override the dataset's default")
parser.add_argument("--aug", default="D", choices=["A", "B", "C", "D"], help="train.py wala --aug")
# CHANGED: --backbone hata diya (vit_conv scrap), patch-embed flags add kiye
parser.add_argument("--patch-embed", default="linear", choices=["linear", "conv"],
                    help="train.py wala --patch-embed")
parser.add_argument("--pe-ckpt", default=None,
                    help="train.py wala --pe-ckpt. SIRF run_tag (checkpoint ka naam) ke liye; "
                         "weights yahan load nahi hote, full checkpoint mein pehle se hain")
parser.add_argument("--seed", type=int, default=42, help="train.py wala --seed")
parser.add_argument("--split", default=None, choices=["val", "test"],
                    help="default: test (agar dataset mein hai), warna val (e.g. foci)")
parser.add_argument("--ckpt", default="best", choices=["best", "last"])
parser.add_argument("--conf", type=float, default=0.001, help="NMS confidence threshold")
args = parser.parse_args()

# ---------------- config (IMG_SIZE, USE_P2 train.py jaise EXACT same) ----------------
DS = DATASETS[args.dataset]
IMG_SIZE = args.img_size or DS["img_size"]
CLASS_NAMES, DATA_ROOT = DS["classes"], DS["root"]
NUM_CLASSES = len(CLASS_NAMES)

# split: user ne diya to wahi, warna test (agar is dataset mein hai), warna val
SPLITS = DS.get("splits", ("train", "val", "test"))
args.split = args.split or ("test" if "test" in SPLITS else "val")
assert args.split in SPLITS, f"{args.dataset} mein '{args.split}' split nahi hai (splits: {SPLITS})"

USE_P2 = False
BATCH_SIZE = 4
IOU_THRES = 0.65

# CHANGED: naya run_tag signature (patch_embed, pe_ckpt, aug, seed)
TAG = run_tag(args.dataset, IMG_SIZE, USE_P2, args.patch_embed, args.pe_ckpt, args.aug, args.seed)
CKPT = f"{args.ckpt}_{TAG}.pt"

device = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------- data ----------------
(loader,) = build_folder_loaders(
    DATA_ROOT, splits=(args.split,), batch_size=BATCH_SIZE, img_size=IMG_SIZE,
    augment=False, num_workers=2)
print(f"{TAG} | {args.split}: {len(loader.dataset)} images | checkpoint: {CKPT} | conf {args.conf}")

# ---------------- model ----------------
# CHANGED: pe_ckpt yahan PASS NAHI karte; load_state_dict fine-tuned patch-embed weights la deta hai
model = build_model(IMG_SIZE, NUM_CLASSES, use_p2=USE_P2, patch_embed=args.patch_embed).to(device)
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
    # alag metric sirf IoU 0.25 ke liye (main 'map' ko 0.25 se kharab na kare)
    metric25 = MeanAveragePrecision(box_format="xyxy", iou_type="bbox", iou_thresholds=[0.25])
    metric.warn_on_many_detections = False
    metric25.warn_on_many_detections = False

    for images, targets, _ in loader:
        # AMP for inference/evaluation
        with torch.amp.autocast("cuda", enabled=(device == "cuda")):
            y, _ = model(images.to(device))                 # [B, 4+nc, anchors], boxes xywh pixels

        dets = non_max_suppression(y, conf_thres=args.conf, iou_thres=IOU_THRES)

        preds, gts = [], []
        for d, t in zip(dets, targets):
            preds.append({"boxes": d[:, :4].cpu(),
                          "scores": d[:, 4].cpu(),
                          "labels": d[:, 5].long().cpu()})
            boxes, labels = targets_to_xyxy(t, IMG_SIZE)
            gts.append({"boxes": boxes, "labels": labels})

        metric.update(preds, gts)
        metric25.update(preds, gts)

    return metric.compute(), metric25.compute()


if __name__ == "__main__":
    r, r25 = evaluate()
    print(f"mAP@[.5:.95] : {r['map'].item():.4f}")
    print(f"mAP@0.25     : {r25['map'].item():.4f}")
    print(f"mAP@0.5      : {r['map_50'].item():.4f}")
    print(f"mAP@0.75     : {r['map_75'].item():.4f}")
    print(f"mAP small    : {r['map_small'].item():.4f}   (-1 = is size ka koi GT nahi)")
    print(f"mAP medium   : {r['map_medium'].item():.4f}")
    print(f"mAP large    : {r['map_large'].item():.4f}")
    print(f"mAR@100      : {r['mar_100'].item():.4f}")
    print(f"mAP@0.25 small/medium/large : {r25['map_small'].item():.4f} / "
          f"{r25['map_medium'].item():.4f} / {r25['map_large'].item():.4f}")

    if r["map_per_class"].numel() > 1:
        print("per-class mAP@[.5:.95]:")
        for c, ap in zip(r["classes"].tolist(), r["map_per_class"].tolist()):
            print(f"  {CLASS_NAMES[c]:10s} {ap:.4f}")

    # ablation table mein paste karne ke liye ek line (TAG mein ab seed bhi hai)
    print(f"\nTABLE | {TAG} | {args.split} {args.ckpt} | mAP@0.25 {r25['map'].item():.4f} "
          f"| mAP@0.5 {r['map_50'].item():.4f} | mAP {r['map'].item():.4f} "
          f"| mAP@0.75 {r['map_75'].item():.4f} | mAR@100 {r['mar_100'].item():.4f}")