# inference.py
import sys
from pathlib import Path

import torch
import torchvision.transforms as T
from PIL import Image, ImageDraw
try:
    from ultralytics.utils.nms import non_max_suppression
except ImportError:
    from ultralytics.utils.ops import non_max_suppression

from src.dataset import NUM_CLASSES, CLASS_NAMES, DATA_ROOT, MEAN, STD, IMG_EXT
from src.model.build import build_model

# ---------------- config (IMG_SIZE aur USE_P2 train.py jaise EXACT same) ----------------
IMG_SIZE = 640
USE_P2 = False
TAG = f"wildlife_{IMG_SIZE}_{'p2' if USE_P2 else 'p3p5'}"
CKPT = f"best_{TAG}.pt"
CONF_THRES = 0.25
IOU_THRES = 0.45
OUT_DIR = Path("outputs")
device = "cuda" if torch.cuda.is_available() else "cpu"

to_tensor = T.Compose([T.ToTensor(), T.Normalize(mean=MEAN, std=STD)])


def load_model():
    model = build_model(IMG_SIZE, NUM_CLASSES, use_p2=USE_P2).to(device)
    model.load_state_dict(torch.load(CKPT, map_location=device))
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
def detect(model, image):
    """PIL image -> [n, 6] (x1, y1, x2, y2, conf, cls) in ORIGINAL image pixels."""
    w0, h0 = image.size
    canvas, scale, pad_x, pad_y = letterbox(image)
    x = to_tensor(canvas).unsqueeze(0).to(device)

    y, _ = model(x)
    det = non_max_suppression(y, conf_thres=CONF_THRES, iou_thres=IOU_THRES)[0].cpu()

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
    # usage: python inference.py img1.jpg img2.jpg     (ya koi folder)
    paths = []
    for a in sys.argv[1:]:
        p = Path(a)
        paths += sorted(q for q in p.iterdir() if q.suffix.lower() in IMG_EXT) if p.is_dir() else [p]
    if not paths:                                              # default: test split ki pehli 8 images
        test_dir = Path(DATA_ROOT) / "images" / "test"
        paths = sorted(q for q in test_dir.iterdir() if q.suffix.lower() in IMG_EXT)[:8]

    OUT_DIR.mkdir(exist_ok=True)
    model = load_model()
    for p in paths:
        image = Image.open(p).convert("RGB")
        det = detect(model, image)
        draw(image, det).save(OUT_DIR / p.name)
        names = sorted({CLASS_NAMES[int(c)] for c in det[:, 5].tolist()})
        print(f"{p.name}: {len(det)} detections {names}")