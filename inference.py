# inference.py
import argparse
from pathlib import Path

import torch
from PIL import Image, ImageDraw

from src.dataset import DATASETS, IMG_EXT
from src.experiments import (
    add_inference_options, add_training_options, inference_options,
    validate_experiment_dataset,
)
from src.model.build import build_model, run_tag
from src.utils.prediction import detect_image


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

add_training_options(ap)
add_inference_options(ap)
args = ap.parse_args()
options = inference_options(ap, args)
validate_experiment_dataset(
    ap, args.dataset, args.nwd_loss or args.aux_center_loss or args.sahi or args.soft_nms,
)
if not 0 <= args.conf <= 1:
    ap.error("--conf must be in [0, 1]")


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
    sac=rates, nwd=args.nwd_loss, aux_center=args.aux_center_loss,
)

CKPT = f"{args.ckpt}_{TAG}.pt"
post_tag = ("_sahi" if options.sahi else "") + ("_softnms" if options.soft_nms else "")
OUT_DIR = Path("outputs") / (TAG + post_tag)
OUT_DIR.mkdir(parents=True, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------- model ----------------
def load_model():
    model = build_model(
        IMG_SIZE,
        NUM_CLASSES,
        use_p2=USE_P2,
        sac_rates=rates, aux_center=args.aux_center_loss, pretrained=False,
    ).to(device)

    model.load_state_dict(torch.load(CKPT, map_location=device))
    model.eval()

    print(f"loaded {CKPT}")
    return model


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
        with Image.open(p) as source:
            image = source.convert("RGB")

        with torch.inference_mode(), torch.amp.autocast("cuda", enabled=(device == "cuda")):
            det = detect_image(model, image, IMG_SIZE, device, args.conf, IOU_THRES, options).cpu()

        draw(image, det).save(OUT_DIR / p.name)

        names = sorted({
            CLASS_NAMES[int(c)]
            for c in det[:, 5].tolist()
        })

        print(f"{p.name}: {len(det)} detections {names}")

    print("saved in", OUT_DIR)
