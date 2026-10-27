# prepare_foci.py
"""
FOCI (virus patch detection, Pascal VOC-style XML) -> YOLO format, same layout as prepare_voc.py:

    <BASE>/foci/images/{train,val}/*.jpg
    <BASE>/foci/labels/{train,val}/*.txt      (class_id cx cy w h, normalized, class_id 0 = virus)

Raw dataset structure:
    <raw>/JPEGImages/JPEGImages/
    <raw>/Virus_anno/Annotations/

    The sibling directories cell_Pic1/ and Xml_cell/ are not used.
    The test/ directory is not used because an official test split has not been confirmed.

Split:
    Original image/XML pairs are shuffled with seed=42 and split into 90% train / 10% validation.
    No separate test split is created. Augmentation and resizing are handled by the training pipeline.

Usage (from the project root):
    python prepare_foci.py --raw /path/to/Foci_Dataset
    python prepare_foci.py --raw /path/to/Foci_Dataset --val-ratio 0.1 --seed 42 --force
    python prepare_foci.py --img-dir /x/JPEGImages/JPEGImages --ann-dir /x/Virus_anno/Annotations

The CLASSES list below must match DATASETS["foci"]["classes"] in src/dataset.py.
"""
import argparse
import json
import os
import random
import shutil
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **kw):
        return x

# ---------------- output paths (same BASE logic as prepare_voc.py) ----------------
BASE = Path("/kaggle/temp") if Path("/kaggle").exists() else Path("data")
OUT = BASE / "foci"

CLASSES = ["virus"]
CLASS_ID = {c: i for i, c in enumerate(CLASSES)}
IMG_EXTS = {".jpg", ".jpeg", ".png"}

EXPECTED_BOXES = 4216          # Expected count from raw annotation analysis; warning only.


# ---------------- locate + pair ----------------
def _find_one(raw, name):
    """Find a uniquely named directory recursively under the raw dataset path."""
    raw = Path(raw)
    hits = sorted(p for p in raw.rglob(name) if p.is_dir())
    if raw.name == name:
        hits.insert(0, raw)
    if not hits:
        raise SystemExit(f"Directory '{name}/' was not found under {raw}.")
    if len(hits) > 1:
        raise SystemExit(
            f"Multiple '{name}/' directories were found: {[str(h) for h in hits]}. "
            f"Specify the correct directories with --img-dir and --ann-dir."
        )
    return hits[0]


def find_dirs(raw, img_dir=None, ann_dir=None):
    """Locate the image and annotation directories in the FOCI dataset structure."""
    raw = Path(raw) if raw else None

    if img_dir:
        img_dir = Path(img_dir)
    elif raw:
        # Dataset structure: Foci_Dataset/JPEGImages/JPEGImages/
        img_dir = raw / "JPEGImages" / "JPEGImages"
    else:
        img_dir = None

    if ann_dir:
        ann_dir = Path(ann_dir)
    elif raw:
        # Dataset structure: Foci_Dataset/Virus_anno/Annotations/
        ann_dir = raw / "Virus_anno" / "Annotations"
    else:
        ann_dir = None

    if img_dir is None or not img_dir.is_dir():
        raise SystemExit(f"Image directory not found: {img_dir}")
    if ann_dir is None or not ann_dir.is_dir():
        raise SystemExit(f"Annotation directory not found: {ann_dir}")
    print(f"images:      {img_dir}\nannotations: {ann_dir}")
    return img_dir, ann_dir


def pair_files(img_dir, ann_dir):
    """Match XML files to images by file stem. Returns sorted [(stem, img_path, xml_path)]."""
    imgs = {}
    for p in sorted(img_dir.iterdir()):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            imgs.setdefault(p.stem, p)
    xmls = sorted(p for p in ann_dir.iterdir() if p.is_file() and p.suffix.lower() == ".xml")

    pairs, no_image = [], []
    for x in xmls:
        if x.stem in imgs:
            pairs.append((x.stem, imgs[x.stem], x))
        else:
            no_image.append(x.name)
    no_xml = sorted(set(imgs) - {s for s, _, _ in pairs})

    print(f"found: {len(imgs)} images, {len(xmls)} xmls -> {len(pairs)} pairs")
    if no_image:
        print(f"WARNING: {len(no_image)} xml files have no image (skipped), e.g. {no_image[:3]}")
    if no_xml:
        print(f"WARNING: {len(no_xml)} images have no xml (skipped), e.g. {no_xml[:3]}")
    if not pairs:
        raise SystemExit("No image/XML pairs found. Check --img-dir / --ann-dir.")
    return pairs


# ---------------- conversion ----------------
def convert(img, xml, keep_difficult, seen_names):
    """Returns label lines. Boxes: VOC 1-based inclusive pixels -> normalized cx cy w h."""
    with Image.open(img) as im:
        W, H = im.size                                     # real image size (the dataloader sees this too)
    root = ET.parse(xml).getroot()

    lines = []
    for obj in root.iter("object"):
        name = (obj.findtext("name") or "").strip()
        seen_names[name] += 1                              # single-class dataset: every box -> class 0
        if int((obj.findtext("difficult") or "0").strip()) and not keep_difficult:
            continue
        bb = obj.find("bndbox")
        xmin, ymin, xmax, ymax = (float(bb.findtext(k)) for k in ("xmin", "ymin", "xmax", "ymax"))
        x1, y1 = max(0.0, xmin - 1), max(0.0, ymin - 1)    # 1-based inclusive -> 0-based edges
        x2, y2 = min(float(W), xmax), min(float(H), ymax)
        if x2 <= x1 or y2 <= y1:
            continue
        lines.append(f"0 {(x1 + x2) / 2 / W:.6f} {(y1 + y2) / 2 / H:.6f} "
                     f"{(x2 - x1) / W:.6f} {(y2 - y1) / H:.6f}")
    return lines


def place(src, dst):
    """Hard link (no extra disk), copy if linking is not possible."""
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def write_split(split, items, keep_difficult, seen_names):
    (OUT / "images" / split).mkdir(parents=True, exist_ok=True)
    (OUT / "labels" / split).mkdir(parents=True, exist_ok=True)
    stats = {"images": 0, "boxes": 0, "empty": 0}
    for stem, img, xml in tqdm(items, desc=split):
        lines = convert(img, xml, keep_difficult, seen_names)
        place(img, OUT / "images" / split / f"{stem}{img.suffix.lower()}")
        (OUT / "labels" / split / f"{stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
        stats["images"] += 1
        stats["boxes"] += len(lines)
        stats["empty"] += int(not lines)
    return stats


# ---------------- checks ----------------
def verify():
    """Image/label pairing, class ids, coordinate ranges, no train/val overlap."""
    stems = {}
    for split in ("train", "val"):
        imgs = {p.stem for p in (OUT / "images" / split).iterdir()}
        lbls = {p.stem for p in (OUT / "labels" / split).iterdir()}
        assert imgs == lbls, f"{split}: images and labels do not match"
        stems[split] = imgs
        for f in (OUT / "labels" / split).glob("*.txt"):
            for line in f.read_text().splitlines():
                c, cx, cy, w, h = line.split()
                assert 0 <= int(c) < len(CLASSES), f"{f.name}: class id {c}"
                assert all(0.0 <= float(v) <= 1.0 for v in (cx, cy, w, h)), f"{f.name}: coords outside [0,1]"
                assert float(w) > 0 and float(h) > 0, f"{f.name}: zero-size box"
    assert not (stems["train"] & stems["val"]), "train and val overlap"
    print("checks ok: pairing, class ids, coordinates, no overlap between splits")


def save_checks(n):
    """n random train images with boxes drawn -> <OUT>/_check/ (for eyeballing)."""
    if n <= 0:
        return
    chk = OUT / "_check"
    chk.mkdir(exist_ok=True)
    files = sorted((OUT / "images" / "train").iterdir())
    for p in random.Random(0).sample(files, min(n, len(files))):
        im = Image.open(p).convert("RGB")
        W, H = im.size
        d = ImageDraw.Draw(im)
        for line in (OUT / "labels" / "train" / f"{p.stem}.txt").read_text().splitlines():
            c, cx, cy, w, h = line.split()
            cx, cy, w, h = float(cx) * W, float(cy) * H, float(w) * W, float(h) * H
            d.rectangle([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], outline="lime", width=2)
            d.text((cx - w / 2 + 3, cy - h / 2 + 3), CLASSES[int(c)], fill="yellow")
        im.save(chk / f"{p.stem}.jpg")
    print(f"box-check images saved in {chk}")


# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", help="path to Foci_Dataset (contains JPEGImages/ and Virus_anno/)")
    ap.add_argument("--img-dir", help="override image directory")
    ap.add_argument("--ann-dir", help="override annotation directory")
    ap.add_argument("--val-ratio", type=float, default=0.1, help="fraction of pairs held out for validation")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--keep-difficult", action="store_true", help="keep objects flagged difficult (default: drop)")
    ap.add_argument("--force", action="store_true", help="delete existing output and rebuild")
    ap.add_argument("--check", type=int, default=8, help="number of box-check images (0 = skip)")
    args = ap.parse_args()

    if not (0.0 < args.val_ratio < 1.0):
        raise SystemExit("--val-ratio must be between 0 and 1")
    if not (args.raw or (args.img_dir and args.ann_dir)):
        raise SystemExit("Provide --raw, or both --img-dir and --ann-dir.")

    opts = {"val_ratio": args.val_ratio, "seed": args.seed, "keep_difficult": args.keep_difficult}
    marker = OUT / ".done"
    if args.force and OUT.exists():
        shutil.rmtree(OUT)
    if marker.exists():
        prev = json.loads(marker.read_text())
        if prev["options"] == opts:
            print(f"{OUT} already exists: {prev['counts']}")
            return
        raise SystemExit(f"{OUT} was built with different options {prev['options']}. Use --force to rebuild.")
    if OUT.exists():
        shutil.rmtree(OUT)                                 # previous run was interrupted

    img_dir, ann_dir = find_dirs(args.raw, args.img_dir, args.ann_dir)
    pairs = pair_files(img_dir, ann_dir)                   # sorted -> deterministic before the shuffle

    random.Random(args.seed).shuffle(pairs)
    n_val = max(1, round(len(pairs) * args.val_ratio))
    val_items, train_items = pairs[:n_val], pairs[n_val:]

    seen_names = Counter()
    counts = {}
    for split, items in (("train", train_items), ("val", val_items)):
        s = write_split(split, items, args.keep_difficult, seen_names)
        counts[split] = s
        print(f"{split}: {s['images']} images | {s['boxes']} boxes | {s['empty']} images with no boxes")

    print("object names in XML (all mapped to class 0):", dict(seen_names))
    unexpected = [n for n in seen_names if n not in CLASS_ID]
    if unexpected:
        print(f"WARNING: names not in CLASSES {CLASSES}: {unexpected}. Confirm they are all virus patches.")

    total = sum(c["boxes"] for c in counts.values())
    if total != EXPECTED_BOXES:
        print(f"WARNING: wrote {total} boxes, expected {EXPECTED_BOXES} "
              f"(difficult objects dropped or clipped boxes can explain a gap)")

    verify()
    save_checks(args.check)
    marker.write_text(json.dumps({"options": opts, "counts": counts}))
    print(f"done -> {OUT}")


if __name__ == "__main__":
    main()