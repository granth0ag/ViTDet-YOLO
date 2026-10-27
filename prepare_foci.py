# prepare_foci.py
"""
FOCI (virus patch detection, Pascal VOC-style XML) -> YOLO format, same layout as prepare_voc.py:

    <BASE>/foci/images/{train,val}/*.jpg
    <BASE>/foci/labels/{train,val}/*.txt      (cls cx cy w h, normalized, class id 0 = virus)

Raw dataset (local, .gitignore mein daalo):
    <raw>/JPEGImages/   +   <raw>/Annotations/
    (cell_Pic1/ aur Xml_cell/ JPEGImages/Annotations ke subsets hain -> use nahi karte)
    (test/ ka official test split hona confirm nahi hai -> use nahi karte)

Split:
    original image/XML pairs ko seed=42 se shuffle karke 90% train / 10% val.
    Koi test split nahi. Augmentation/resizing yahan NAHI hota (training pipeline ka kaam hai).

Usage (project root):
    python prepare_foci.py --raw /path/to/Foci_Dataset
    python prepare_foci.py --raw /path/to/Foci_Dataset --val-ratio 0.1 --seed 42 --force

CLASSES below must match DATASETS["foci"]["classes"] in src/dataset.py.
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

# ---------------- paths (same BASE logic as prepare_voc.py) ----------------
BASE = Path("/kaggle/temp") if Path("/kaggle").exists() else Path("data")
OUT = BASE / "foci"

CLASSES = ["virus"]
CLASS_ID = {c: i for i, c in enumerate(CLASSES)}
IMG_EXTS = {".jpg", ".jpeg", ".png"}

EXPECTED_BOXES = 4216          # raw annotation analysis se (sirf warning ke liye)


# ---------------- locate + pair ----------------
def find_dirs(raw):
    """JPEGImages/ aur Annotations/ dhoondho (raw ke andar ya ek level nested)."""
    raw = Path(raw)
    for root in [raw] + [p for p in raw.rglob("*") if p.is_dir()]:
        if (root / "JPEGImages").is_dir() and (root / "Annotations").is_dir():
            return root / "JPEGImages", root / "Annotations"
    raise SystemExit(f"{raw} ke andar JPEGImages/ + Annotations/ nahi mile.")


def _index(folder, exts):
    out = {}
    for p in sorted(folder.iterdir()):
        if p.name.startswith(".") or not p.is_file() or p.suffix.lower() not in exts:
            continue
        if p.stem in out:
            raise SystemExit(f"duplicate stem '{p.stem}' in {folder}: {out[p.stem].name} vs {p.name}")
        out[p.stem] = p
    return out


def pair_files(img_dir, ann_dir):
    imgs, xmls = _index(img_dir, IMG_EXTS), _index(ann_dir, {".xml"})
    no_xml, no_img = sorted(set(imgs) - set(xmls)), sorted(set(xmls) - set(imgs))
    print(f"JPEGImages: {len(imgs)} images | Annotations: {len(xmls)} XMLs")
    if no_xml:
        print(f"  {len(no_xml)} images bina XML ke (use nahi honge): {no_xml[:10]}{' ...' if len(no_xml) > 10 else ''}")
    if no_img:
        print(f"  {len(no_img)} XMLs bina image ke (use nahi honge): {no_img[:10]}{' ...' if len(no_img) > 10 else ''}")
    stems = sorted(set(imgs) & set(xmls))
    return [(s, imgs[s], xmls[s]) for s in stems]


# ---------------- conversion ----------------
def convert(img, xml):
    """Returns (label lines, info). Boxes: VOC xmin/ymin/xmax/ymax pixels -> normalized cx cy w h."""
    with Image.open(img) as im:
        W, H = im.size                                     # asli image size, hardcode nahi
    root = ET.parse(xml).getroot()
    info = {"size": (W, H), "name_mismatch": False, "size_mismatch": False,
            "other_class": Counter(), "clipped": 0, "dropped": 0, "px": []}

    fn = (root.findtext("filename") or "").strip()
    if fn and Path(fn).stem != img.stem:
        info["name_mismatch"] = True
    sz = root.find("size")
    if sz is not None:
        try:
            if (int(sz.findtext("width")), int(sz.findtext("height"))) != (W, H):
                info["size_mismatch"] = True
        except (TypeError, ValueError):
            info["size_mismatch"] = True

    lines = []
    for obj in root.iter("object"):
        name = (obj.findtext("name") or "").strip()
        if name not in CLASS_ID:
            info["other_class"][name] += 1
            continue
        bb = obj.find("bndbox")
        xmin, ymin, xmax, ymax = (float(bb.findtext(k)) for k in ("xmin", "ymin", "xmax", "ymax"))
        if not (xmin < xmax and ymin < ymax):
            info["dropped"] += 1
            continue
        x1, y1, x2, y2 = max(0.0, xmin), max(0.0, ymin), min(float(W), xmax), min(float(H), ymax)
        if (x1, y1, x2, y2) != (xmin, ymin, xmax, ymax):
            info["clipped"] += 1
        if x2 <= x1 or y2 <= y1:
            info["dropped"] += 1
            continue
        lines.append(f"{CLASS_ID[name]} {(x1 + x2) / 2 / W:.6f} {(y1 + y2) / 2 / H:.6f} "
                     f"{(x2 - x1) / W:.6f} {(y2 - y1) / H:.6f}")
        info["px"].append((x2 - x1, y2 - y1))
    return lines, info


def place(src, dst):
    """Hard link (no extra disk), copy agar link possible na ho."""
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def write_split(split, items, agg):
    (OUT / "images" / split).mkdir(parents=True, exist_ok=True)
    (OUT / "labels" / split).mkdir(parents=True, exist_ok=True)
    stats = {"images": 0, "boxes": 0, "empty": 0}
    for stem, img, xml in tqdm(items, desc=split):
        lines, info = convert(img, xml)
        place(img, OUT / "images" / split / img.name)
        (OUT / "labels" / split / f"{stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
        stats["images"] += 1
        stats["boxes"] += len(lines)
        stats["empty"] += int(not lines)
        agg["sizes"][info["size"]] += 1
        agg["other_class"].update(info["other_class"])
        agg["clipped"] += info["clipped"]
        agg["dropped"] += info["dropped"]
        agg["px"].extend(info["px"])
        if info["name_mismatch"]:
            agg["name_mismatch"].append(stem)
        if info["size_mismatch"]:
            agg["size_mismatch"].append(stem)
    return stats


# ---------------- checks ----------------
def verify():
    """Image/label pairing, class ids, coordinate ranges, no overlap between splits."""
    for split in ("train", "val"):
        imgs = {p.stem for p in (OUT / "images" / split).iterdir()}
        lbls = {p.stem for p in (OUT / "labels" / split).iterdir()}
        assert imgs == lbls, f"{split}: images aur labels match nahi karte"
        for f in (OUT / "labels" / split).glob("*.txt"):
            for line in f.read_text().splitlines():
                c, cx, cy, w, h = line.split()
                assert int(c) == 0, f"{f.name}: class id {c} (sirf 0 = virus hona chahiye)"
                assert 0.0 <= float(cx) <= 1.0 and 0.0 <= float(cy) <= 1.0, f"{f.name}: cx/cy [0,1] se bahar"
                assert 0.0 < float(w) <= 1.0 and 0.0 < float(h) <= 1.0, f"{f.name}: w/h (0,1] se bahar"
    tr = {p.stem for p in (OUT / "images" / "train").iterdir()}
    va = {p.stem for p in (OUT / "images" / "val").iterdir()}
    assert not (tr & va), "train aur val overlap kar rahe hain"
    print("checks ok: pairing, class id == 0, coordinates, no train/val overlap")


def save_checks(n):
    """n random train images with boxes drawn -> <OUT>/_check/ (eyeball karne ke liye)."""
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
        im.save(chk / f"{p.stem}.jpg")
    print(f"box-check images saved in {chk}")


# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True, help="Foci_Dataset folder (JPEGImages/ + Annotations/ wala)")
    ap.add_argument("--val-ratio", type=float, default=0.1, help="val fraction (default 0.1 -> 90/10)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--force", action="store_true", help="existing output delete karke dobara banao")
    ap.add_argument("--check", type=int, default=8, help="box-check images ki sankhya (0 = skip)")
    args = ap.parse_args()

    opts = {"val_ratio": args.val_ratio, "seed": args.seed}
    marker = OUT / ".done"
    if args.force and OUT.exists():
        shutil.rmtree(OUT)
    if marker.exists():
        prev = json.loads(marker.read_text())
        if prev["options"] == opts:
            print(f"{OUT} pehle se bana hai: {prev['counts']}")
            return
        raise SystemExit(f"{OUT} alag options se bana hai {prev['options']}. --force se dobara banao.")
    if OUT.exists():
        shutil.rmtree(OUT)                                 # pichhla run beech mein ruka tha

    img_dir, ann_dir = find_dirs(args.raw)
    pairs = pair_files(img_dir, ann_dir)                   # sorted -> deterministic
    assert pairs, "koi image/XML pair nahi mili"
    print(f"matched pairs: {len(pairs)}")

    # split original pairs PEHLE (augmentation baad mein, sirf train pe, training pipeline mein)
    random.Random(args.seed).shuffle(pairs)
    n_val = round(len(pairs) * args.val_ratio)
    val_items, train_items = pairs[:n_val], pairs[n_val:]

    agg = {"sizes": Counter(), "other_class": Counter(), "clipped": 0, "dropped": 0,
           "px": [], "name_mismatch": [], "size_mismatch": []}
    counts = {}
    for split, items in (("train", train_items), ("val", val_items)):
        s = write_split(split, items, agg)
        counts[split] = s
        print(f"{split}: {s['images']} images | {s['boxes']} boxes | {s['empty']} images with no boxes")
    counts["total"] = {k: counts["train"][k] + counts["val"][k] for k in ("images", "boxes", "empty")}
    print(f"total: {counts['total']['images']} images | {counts['total']['boxes']} boxes")

    # ---- reports ----
    print("image sizes (W,H):", dict(agg["sizes"]))
    if agg["other_class"]:
        print("WARNING: non-virus objects ignore kiye:", dict(agg["other_class"]))
    if agg["clipped"] or agg["dropped"]:
        print(f"boxes clipped to image: {agg['clipped']} | dropped (invalid/zero-size): {agg['dropped']}")
    if agg["name_mismatch"]:
        print(f"WARNING: XML <filename> image se match nahi karta ({len(agg['name_mismatch'])}): {agg['name_mismatch'][:10]}")
    if agg["size_mismatch"]:
        print(f"WARNING: XML <size> asli image size se alag hai ({len(agg['size_mismatch'])}): {agg['size_mismatch'][:10]}")
    if counts["total"]["boxes"] != EXPECTED_BOXES:
        print(f"WARNING: {counts['total']['boxes']} boxes mile, raw analysis mein {EXPECTED_BOXES} the "
              f"(unmatched images/XMLs ya dropped boxes ki wajah se ho sakta hai)")
    if agg["px"]:
        ws, hs = sorted(w for w, _ in agg["px"]), sorted(h for _, h in agg["px"])
        print(f"box px: width median {ws[len(ws) // 2]:.0f} (min {ws[0]:.0f}, max {ws[-1]:.0f}) | "
              f"height median {hs[len(hs) // 2]:.0f} (min {hs[0]:.0f}, max {hs[-1]:.0f})")

    verify()
    save_checks(args.check)
    marker.write_text(json.dumps({"options": opts, "counts": counts}))
    print(f"done -> {OUT}")


if __name__ == "__main__":
    main()