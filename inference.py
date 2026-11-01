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


# ---------------- args ----------------
ap = argparse.ArgumentParser()

ap.add_argument("--dataset", default="wildlife", choices=list(DATASETS))
ap.add_argument("--img-size", type=int, default=None,
                help="override the dataset's default")
ap.add_argument("--aug", default="D", choices=["A", "B", "C", "D"],
                help="train.py wala --aug")
ap.add_argument("--sac-rates", default="",
                help="comma-separated rates for SAC")
ap.add_argument("--seed", type=int, default=42,
                help="train.py wala --seed")
ap.add_argument("--ckpt", default="best", choices=["best", "last"])
ap.add_argument("--conf", type=float, default=0.25,
                help="detections ka confidence threshold")
ap.add_argument("paths", nargs="*", help="image files ya folders")

args = ap.parse_args()


# ---------------- config ----------------
rates = tuple(int(r) for r in args.sac_rates.split(",")) if args.sac_rates else None

DS = DATASETS[args.dataset]
IMG_SIZE = args.img_size or DS["img_size"]
CLASS_NAMES = DS["classes"]
DATA_ROOT = DS["root"]
NUM_CLASSES = len(CLASS_NAMES)

USE_P2 = False
IOU_THRES = 0.45

TAG = run_tag(
    args.dataset,
    IMG_SIZE,
    USE_P2,
    args.aug,
    args.seed,
    sac=rates
)

CKPT = f"{args.ckpt}_{TAG}.pt"
OUT_DIR = Path("outputs") / TAG
OUT_DIR.mkdir(parents=True, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

to_tensor = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=MEAN, std=STD)
])


# ---------------- model ----------------
def load_model():
    model = build_model(
        IMG_SIZE,
        NUM_CLASSES,
        use_p2=USE_P2,
        sac_rates=rates
    ).to(device)

    model.load_state_dict(torch.load(CKPT, map_location=device))
    model.eval()

    print(f"loaded {CKPT}")
    return model


def letterbox(image, S=IMG_SIZE):
    """Same non-augment letterbox as dataset.py."""
    w0, h0 = image.size

    scale = S / max(w0, h0)
    new_w = round(w0 * scale)
    new_h = round(h0 * scale)

    pad_x = (S - new_w) // 2
    pad_y = (S - new_h) // 2

    canvas = Image.new("RGB", (S, S), (114, 114, 114))
    canvas.paste(
        image.resize((new_w, new_h), Image.BILINEAR),
        (pad_x, pad_y)
    )

    return canvas, scale, pad_x, pad_y


@torch.no_grad()
def detect(model, image, conf=0.25):
    """PIL image -> [n, 6] in ORIGINAL image pixels."""
    w0, h0 = image.size

    canvas, scale, pad_x, pad_y = letterbox(image)

    x = to_tensor(canvas).unsqueeze(0).to(device)

    with torch.amp.autocast("cuda", enabled=(device == "cuda")):
        y, _ = model(x)

    det = non_max_suppression(
        y,
        conf_thres=conf,
        iou_thres=IOU_THRES
    )[0].cpu()

    if len(det):
        det[:, [0, 2]] = (
            (det[:, [0, 2]] - pad_x) / scale
        ).clamp(0, w0)

        det[:, [1, 3]] = (
            (det[:, [1, 3]] - pad_y) / scale
        ).clamp(0, h0)

    return det


def draw(image, det):
    image = image.copy()
    d = ImageDraw.Draw(image)

    lw = max(2, max(image.size) // 300)

    for x1, y1, x2, y2, conf, cls in det.tolist():
        label = f"{CLASS_NAMES[int(cls)]} {conf:.2f}"

        d.rectangle(
            [x1, y1, x2, y2],
            outline="lime",
            width=lw
        )

        d.text(
            (x1 + 3, y1 + 3),
            label,
            fill="white"
        )

    return image


# ---------------- main ----------------
if __name__ == "__main__":

    paths = []

    for a in args.paths:
        p = Path(a)

        if p.is_dir():
            paths += sorted(
                q for q in p.iterdir()
                if q.suffix.lower() in IMG_EXT
            )
        else:
            paths.append(p)

    if not paths:
        # test split if available, otherwise val
        splits = DS.get("splits", ("train", "val", "test"))
        split = "test" if "test" in splits else "val"

        split_dir = Path(DATA_ROOT) / "images" / split

        paths = sorted(
            q for q in split_dir.iterdir()
            if q.suffix.lower() in IMG_EXT
        )[:8]

    print(f"run: {TAG} | sac={rates}")
    print(f"dataset={args.dataset} | img_size={IMG_SIZE} | aug={args.aug} | seed={args.seed}")
    print(f"checkpoint: {CKPT}")

    model = load_model()

    for p in paths:
        image = Image.open(p).convert("RGB")

        det = detect(
            model,
            image,
            conf=args.conf
        )

        draw(image, det).save(OUT_DIR / p.name)

        names = sorted({
            CLASS_NAMES[int(c)]
            for c in det[:, 5].tolist()
        })

        print(f"{p.name}: {len(det)} detections {names}")

    print("saved in", OUT_DIR)
