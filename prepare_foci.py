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