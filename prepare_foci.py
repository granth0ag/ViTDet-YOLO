
"""
FOCI (virus patch detection, Pascal VOC-style XML) -> YOLO format.

Uses ONLY:
    <RAW>/JPEGImages/JPEGImages/
    <RAW>/Virus_anno/Annotations/

Ignores:
    <RAW>/Xml_cell/
    <RAW>/cell_Pic1/
    <RAW>/test/

Output:
    <BASE>/foci/images/train/*.jpg
    <BASE>/foci/images/val/*.jpg
    <BASE>/foci/labels/train/*.txt
    <BASE>/foci/labels/val/*.txt
    <BASE>/foci/_check/*.jpg

YOLO label format:
    class_id cx cy w h

There is exactly one class:
    virus = 0

Images are NOT resized. Their original dimensions are retained.
The expected original image size is 1024 x 768 x 3.

By default:
    - all images found under JPEGImages/JPEGImages are used
    - a fixed-seed random validation split is created
    - difficult objects are ALWAYS kept
    - no separate test split is created
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


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

# Output is intentionally kept in the project data directory, just like
# the requested VOC layout:
#
#     data/foci/images/train
#     data/foci/images/val
#     data/foci/labels/train
#     data/foci/labels/val
#     data/foci/_check
#
BASE = Path("data")
OUT = BASE / "foci"

# RAW is discovered automatically below. If you prefer a fixed location,
# replace this with the folder containing JPEGImages/ and Virus_anno/.
RAW = BASE / "foci_raw"
IMAGE_RAW = RAW / "JPEGImages" / "JPEGImages"
ANNOTATION_RAW = RAW / "Virus_anno" / "Annotations"

CLASSES = ["virus"]
CLASS_ID = {"virus": 0}




# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def find_raw_root():
    """
    Finds the FOCI dataset root.

    Expected structure:
        <raw>/JPEGImages/JPEGImages/
        <raw>/Virus_anno/Annotations/

    If BASE/foci_raw does not exist, try to locate a directory containing
    both JPEGImages and Virus_anno below BASE.
    """
    if IMAGE_RAW.is_dir() and ANNOTATION_RAW.is_dir():
        return RAW

    candidates = []

    # Search both the project data directory and Kaggle's mounted input
    # directory. This allows the script to work directly with a dataset
    # mounted like:
    #
    # /kaggle/input/.../Foci_Dataset/
    #
    # without touching Xml_cell, cell_Pic1, or test.
    search_roots = [BASE]
    if Path("/kaggle/input").is_dir():
        search_roots.append(Path("/kaggle/input"))

    for search_root in search_roots:
        for p in search_root.rglob("JPEGImages"):
            if not p.is_dir():
                continue

            # The actual image folder is expected to be JPEGImages/JPEGImages.
            image_dir = p / "JPEGImages"
            if not image_dir.is_dir():
                continue

            root = p.parent
            anno_dir = root / "Virus_anno" / "Annotations"
            if anno_dir.is_dir():
                candidates.append(root)

    if len(candidates) == 1:
        return candidates[0]

    if len(candidates) > 1:
        raise SystemExit(
            "Multiple possible FOCI roots found:\n"
            + "\n".join(str(p) for p in candidates)
            + "\nSet RAW in prepare_foci.py to the correct dataset root."
        )

    raise SystemExit(
        "FOCI raw dataset not found.\n\n"
        f"Expected images:      {IMAGE_RAW}\n"
        f"Expected annotations: {ANNOTATION_RAW}\n\n"
        "Set RAW in prepare_foci.py to the folder containing "
        "JPEGImages/ and Virus_anno/."
    )


def image_files(image_dir):
    """Return all supported image files recursively."""
    exts = {
        ".jpg", ".jpeg", ".JPG", ".JPEG",
        ".tif", ".tiff", ".TIF", ".TIFF",
    }
    return sorted(
        p for p in image_dir.rglob("*")
        if p.is_file() and p.suffix in exts
    )


def annotation_for_image(annotation_dir, image_path):
    """
    Find the XML annotation corresponding to an image.

    First tries the exact image stem:
        abc.jpg -> abc.xml

    Then recursively searches for that stem if the annotations are nested.
    """
    exact = annotation_dir / f"{image_path.stem}.xml"
    if exact.exists():
        return exact

    matches = list(annotation_dir.rglob(f"{image_path.stem}.xml"))

    if len(matches) == 1:
        return matches[0]

    if len(matches) == 0:
        raise SystemExit(
            f"No annotation XML found for image: {image_path.name}"
        )

    raise SystemExit(
        f"Multiple annotation XML files found for {image_path.name}:\n"
        + "\n".join(str(p) for p in matches)
    )


# ---------------------------------------------------------------------
# VOC XML -> YOLO conversion
# ---------------------------------------------------------------------

def convert(image_path, annotation_path, keep_difficult=True):
    """
    Returns:
        (YOLO label lines, image path)

    VOC boxes:
        xmin, ymin, xmax, ymax

    YOLO:
        class_id cx cy w h

    Coordinates are normalized to [0, 1].

    FOCI has one class only:
        virus -> class 0

    Difficult objects are always kept.
    """
    with Image.open(image_path) as im:
        W, H = im.size
        channels = len(im.getbands())



    if channels != 3:
        raise SystemExit(
            f"{image_path.name}: expected 3-channel image, got {channels} channels"
        )

    root = ET.parse(annotation_path).getroot()

    lines = []

    for obj in root.iter("object"):
        name = (obj.findtext("name") or "").strip().lower()

        if name != "virus":
            raise SystemExit(
                f"unknown class '{name}' in {annotation_path.name}; "
                "FOCI is expected to contain only the class 'virus'"
            )

        # DO NOT discard difficult objects.
        # The argument is retained only to make the function mirror
        # prepare_voc.py's interface.
        if not keep_difficult:
            # Intentionally never used by main; difficult objects are kept.
            pass

        bb = obj.find("bndbox")
        if bb is None:
            continue

        try:
            xmin = float(bb.findtext("xmin"))
            ymin = float(bb.findtext("ymin"))
            xmax = float(bb.findtext("xmax"))
            ymax = float(bb.findtext("ymax"))
        except (TypeError, ValueError):
            raise SystemExit(
                f"Invalid bounding box in {annotation_path.name}"
            )

        # VOC-style 1-based inclusive coordinates -> 0-based edges.
        x1 = max(0.0, xmin - 1.0)
        y1 = max(0.0, ymin - 1.0)
        x2 = min(float(W), xmax)
        y2 = min(float(H), ymax)

        if x2 <= x1 or y2 <= y1:
            continue

        cx = ((x1 + x2) / 2.0) / W
        cy = ((y1 + y2) / 2.0) / H
        w = (x2 - x1) / W
        h = (y2 - y1) / H

        lines.append(
            f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"
        )

    return lines, image_path


# ---------------------------------------------------------------------
# Image placement
# ---------------------------------------------------------------------

def place(src, dst):
    """
    Hard-link the image when possible to avoid an unnecessary duplicate.
    Fall back to a normal copy if hard links are unavailable.
    """
    if dst.exists():
        return

    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


# ---------------------------------------------------------------------
# Split writing
# ---------------------------------------------------------------------

def write_split(split, items):
    (OUT / "images" / split).mkdir(parents=True, exist_ok=True)
    (OUT / "labels" / split).mkdir(parents=True, exist_ok=True)

    stats = {
        "images": 0,
        "boxes": 0,
        "empty": 0,
        "per_class": Counter(),
    }

    for image_path, annotation_path in tqdm(items, desc=split):
        lines, img = convert(
            image_path,
            annotation_path,
            keep_difficult=True,
        )

        # Keep a simple unique name based on the image filename.
        # If duplicate stems exist, fail instead of silently overwriting.
        name = image_path.stem

        dst_img = OUT / "images" / split / f"{name}.jpg"
        dst_lbl = OUT / "labels" / split / f"{name}.txt"

        if dst_img.exists() or dst_lbl.exists():
            raise SystemExit(
                f"Duplicate output filename detected: {name}. "
                "The raw dataset contains duplicate image stems."
            )

        place(img, dst_img)
        dst_lbl.write_text(
            "\n".join(lines) + ("\n" if lines else "")
        )

        stats["images"] += 1
        stats["boxes"] += len(lines)
        stats["empty"] += int(not lines)
        stats["per_class"].update(
            int(line.split()[0]) for line in lines
        )

    return stats


# ---------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------

def verify():
    """Check image/label pairing, class IDs, coordinates and split overlap."""

    for split in ("train", "val"):
        img_dir = OUT / "images" / split
        lbl_dir = OUT / "labels" / split

        imgs = {p.stem for p in img_dir.iterdir() if p.is_file()}
        lbls = {p.stem for p in lbl_dir.iterdir() if p.is_file()}

        assert imgs == lbls, (
            f"{split}: images and labels do not match"
        )

        for f in lbl_dir.glob("*.txt"):
            for line in f.read_text().splitlines():
                c, cx, cy, w, h = line.split()

                assert 0 <= int(c) < len(CLASSES), (
                    f"{f.name}: invalid class id {c}"
                )

                assert all(
                    0.0 <= float(v) <= 1.0
                    for v in (cx, cy, w, h)
                ), (
                    f"{f.name}: coordinates outside [0, 1]"
                )

                assert float(w) > 0 and float(h) > 0, (
                    f"{f.name}: zero-size box"
                )

    tr = {
        p.stem
        for p in (OUT / "images" / "train").iterdir()
        if p.is_file()
    }
    va = {
        p.stem
        for p in (OUT / "images" / "val").iterdir()
        if p.is_file()
    }

    assert not (tr & va), "train and val splits overlap"

    print(
        "checks ok: pairing, class ids, coordinates, "
        "no train/val overlap"
    )


# ---------------------------------------------------------------------
# Visual sanity checks
# ---------------------------------------------------------------------

def save_checks(n):
    """
    Draw converted YOLO boxes on n random training images.

    Saved to:
        <OUT>/_check/
    """
    if n <= 0:
        return

    chk = OUT / "_check"
    chk.mkdir(exist_ok=True)

    files = sorted(
        p for p in (OUT / "images" / "train").iterdir()
        if p.is_file()
    )

    if not files:
        print("no training images available for visual checks")
        return

    rng = random.Random(0)
    selected = rng.sample(files, min(n, len(files)))

    for p in selected:
        im = Image.open(p).convert("RGB")
        W, H = im.size
        d = ImageDraw.Draw(im)

        label_file = OUT / "labels" / "train" / f"{p.stem}.txt"

        for line in label_file.read_text().splitlines():
            c, cx, cy, w, h = line.split()

            cx = float(cx) * W
            cy = float(cy) * H
            w = float(w) * W
            h = float(h) * H

            x1 = cx - w / 2
            y1 = cy - h / 2
            x2 = cx + w / 2
            y2 = cy + h / 2

            d.rectangle(
                [x1, y1, x2, y2],
                outline="lime",
                width=2,
            )

            d.text(
                (x1 + 3, y1 + 3),
                CLASSES[int(c)],
                fill="yellow",
            )

        im.save(chk / f"{p.stem}.jpg")

    print(f"box-check images saved in {chk}")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--val-size",
        type=int,
        default=14,
        help="number of images held out for validation",
    )

    ap.add_argument(
        "--seed",
        type=int,
        default=42,
        help="random seed used for the train/val split",
    )

    ap.add_argument(
        "--force",
        action="store_true",
        help="delete existing output and rebuild it",
    )

    ap.add_argument(
        "--cleanup",
        action="store_true",
        help="delete raw FOCI data after preparation",
    )

    ap.add_argument(
        "--check",
        type=int,
        default=8,
        help="number of train images for visual box checks (0 = skip)",
    )

    args = ap.parse_args()

    # Locate raw dataset.
    global RAW, IMAGE_RAW, ANNOTATION_RAW

    RAW = find_raw_root()
    IMAGE_RAW = RAW / "JPEGImages" / "JPEGImages"
    ANNOTATION_RAW = RAW / "Virus_anno" / "Annotations"

    if not IMAGE_RAW.is_dir():
        raise SystemExit(f"image directory not found: {IMAGE_RAW}")

    if not ANNOTATION_RAW.is_dir():
        raise SystemExit(
            f"annotation directory not found: {ANNOTATION_RAW}"
        )

    images = image_files(IMAGE_RAW)

    if not images:
        raise SystemExit(f"no supported image files found under {IMAGE_RAW}")

    print(f"raw image directory: {IMAGE_RAW}")
    print(f"raw annotation directory: {ANNOTATION_RAW}")
    print(f"found {len(images)} images")

    # Resolve image -> XML pairs.
    # Skip images that do not have an annotation instead of stopping
    # the entire preparation process.
    items = []
    missing_annotations = []

    for image_path in images:
        exact = ANNOTATION_RAW / f"{image_path.stem}.xml"

        if exact.exists():
            annotation_path = exact
        else:
            matches = list(
                ANNOTATION_RAW.rglob(f"{image_path.stem}.xml")
            )

            if len(matches) == 1:
                annotation_path = matches[0]
            elif len(matches) == 0:
                missing_annotations.append(image_path.name)
                continue
            else:
                raise SystemExit(
                    f"Multiple annotation XML files found for {image_path.name}:\n"
                    + "\n".join(str(p) for p in matches)
                )

        items.append((image_path, annotation_path))

    if missing_annotations:
        print(
            f"skipping {len(missing_annotations)} image(s) without annotation:"
        )
        for name in missing_annotations:
            print(f"  - {name}")

    print(f"using {len(items)} annotated images")

    if args.val_size <= 0:
        raise SystemExit("--val-size must be greater than 0")

    if not items:
        raise SystemExit("no annotated images found")

    if args.val_size >= len(items):
        raise SystemExit(
            f"--val-size={args.val_size} must be smaller than "
            f"the number of annotated images ({len(items)})"
        )

    # Deterministic split.
    items.sort(key=lambda x: x[0].name)
    random.Random(args.seed).shuffle(items)

    val_items = items[:args.val_size]
    train_items = items[args.val_size:]

    opts = {
        "val_size": args.val_size,
        "seed": args.seed,
        "keep_difficult": True,
        "class": "virus",
        "image_size": [1024, 768, 3],
    }

    marker = OUT / ".done"

    if args.force and OUT.exists():
        shutil.rmtree(OUT)

    if marker.exists():
        prev = json.loads(marker.read_text())

        if prev["options"] == opts:
            print(
                f"{OUT} already prepared: {prev['counts']}"
            )
            return

        raise SystemExit(
            f"{OUT} was created with different options: "
            f"{prev['options']}. Use --force to rebuild."
        )

    if OUT.exists():
        shutil.rmtree(OUT)

    # Write train and validation only.
    counts = {}

    for split, split_items in (
        ("train", train_items),
        ("val", val_items),
    ):
        s = write_split(split, split_items)

        counts[split] = {
            "images": s["images"],
            "boxes": s["boxes"],
            "empty": s["empty"],
        }

        print(
            f"{split}: {s['images']} images | "
            f"{s['boxes']} boxes | "
            f"{s['empty']} images with no boxes"
        )

        if split == "train":
            print(
                "  boxes per class (train):",
                ", ".join(
                    f"{CLASSES[c]} {n}"
                    for c, n in sorted(s["per_class"].items())
                ),
            )

    verify()
    save_checks(args.check)

    marker.write_text(
        json.dumps(
            {
                "options": opts,
                "counts": counts,
            },
            indent=2,
        )
    )

    if args.cleanup:
        shutil.rmtree(RAW)
        print("raw data deleted")

    print(f"done -> {OUT}")


if __name__ == "__main__":
    main()
