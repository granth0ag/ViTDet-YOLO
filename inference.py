
# inference.py
import argparse
from pathlib import Path

import torch
import torchvision.transforms as T
from PIL import Image, ImageDraw
try:
    from ultralytics.utils.nms import non_max_suppression
except ImportError:
    from ultralytics.utils.ops import non_max_suppression

from src.dataset import DATASETS, MEAN, STD, IMG_EXT
from src.model.build import build_model, run_tag

# ---------------- config (IMG_SIZE, USE_P2 train.py jaise EXACT same) ----------------
IMG_SIZE = 640
USE_P2 = False
IOU_THRES = 0.45
device = "cuda" if torch.cuda.is_available() else "cpu"

to_tensor = T.Compose([T.ToTensor(), T.Normalize(mean=MEAN, std=STD)])


def load_model(tag, backbone, ckpt_kind="best"):
    ckpt = f"{ckpt_kind}_{tag}.pt"
    model = build_model(IMG_SIZE, NUM_CLASSES, use_p2=USE_P2, backbone=backbone).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    print(f"loaded {ckpt}")
    return model.eval()


def letterbox(image, S=IMG_SIZE):
    """dataset.py ka non-augment letterbox (same). Returns canvas, scale, pad_x, pad_y."""
    w0, h0 = image.size
    scale = S / max(w0, h0)
    new_w, new_h = round(w0 * scale), round(h0 * scale)
    pad_x, pad_y = (S - new_w) // 2, (S - new_h) // 2
    canvas = Image.new("RGB", (S, S), (114, 114, 114))
    canvas.paste(image.resize((new_w, new_h), Image.BILINEAR), (pad_x, pad_y))
    return canvas, scale, pad_x, pad_y


@torch.no_grad()
@torch.no_grad()
def detect(model, image, conf=0.25):
    """PIL image -> [n, 6] (x1, y1, x2, y2, conf, cls) in ORIGINAL image pixels."""
    w0, h0 = image.size
    canvas, scale, pad_x, pad_y = letterbox(image)
    x = to_tensor(canvas).unsqueeze(0).to(device)

    with torch.amp.autocast("cuda", enabled=(device == "cuda")):
        y, _ = model(x)

    det = non_max_suppression(y, conf_thres=conf, iou_thres=IOU_THRES)[0].cpu()

    # letterbox pixels -> original pixels
    det[:, [0, 2]] = ((det[:, [0, 2]] - pad_x) / scale).clamp(0, w0)
    det[:, [1, 3]] = ((det[:, [1, 3]] - pad_y) / scale).clamp(0, h0)
    return det


def draw(image, det):
    image = image.copy()
    d = ImageDraw.Draw(image)
    lw = max(2, max(image.size) // 300)
    for x1, y1, x2, y2, conf, cls in det.tolist():
        label = f"{CLASS_NAMES[int(cls)]} {conf:.2f}"
        d.rectangle([x1, y1, x2, y2], outline="lime", width=lw)
        d.text((x1 + 3, y1 + 3), label, fill="white")
    return image


if __name__ == "__main__":
    # python inference.py --dataset wildlife --aug D --backbone conv
    # python inference.py --dataset wildlife --aug D --backbone conv img1.jpg folder/

    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="*", help="image files ya folders")
    ap.add_argument("--dataset", default="wildlife", choices=list(DATASETS))
    ap.add_argument("--aug", default="D", choices=["A", "B", "C", "D"], help="train.py wala --aug")
    ap.add_argument("--backbone", default="vit", choices=["vit", "conv"], help="train.py wala --backbone")
    ap.add_argument("--ckpt", default="best", choices=["best", "last"])
    ap.add_argument("--conf", type=float, default=0.25, help="detections ka confidence threshold")
    args = ap.parse_args()

    # dataset config
    DS = DATASETS[args.dataset]
    NUM_CLASSES = len(DS["classes"])
    CLASS_NAMES = DS["classes"]
    DATA_ROOT = DS["root"]

    paths = []
    for a in args.paths:
        p = Path(a)
        paths += sorted(q for q in p.iterdir() if q.suffix.lower() in IMG_EXT) if p.is_dir() else [p]
    if not paths:                                              # default: test split ki pehli 8 images
        test_dir = Path(DATA_ROOT) / "images" / "test"
        paths = sorted(q for q in test_dir.iterdir() if q.suffix.lower() in IMG_EXT)[:8]

    tag = run_tag(args.dataset, IMG_SIZE, USE_P2, args.backbone, args.aug)
    out_dir = Path("outputs") / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    model = load_model(tag, args.backbone, args.ckpt)
    for p in paths:
        image = Image.open(p).convert("RGB")
        det = detect(model, image, conf=args.conf)
        draw(image, det).save(out_dir / p.name)
        names = sorted({CLASS_NAMES[int(c)] for c in det[:, 5].tolist()})
        print(f"{p.name}: {len(det)} detections {names}")
    print("saved in", out_dir)
