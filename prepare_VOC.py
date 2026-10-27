# prepare_voc.py
"""
Pascal VOC -> YOLO format, in the same layout as prepare_wildlife.py:

    <BASE>/voc/images/{train,val,test}/*.jpg
    <BASE>/voc/labels/{train,val,test}/*.txt      (cls cx cy w h, normalized, class ids 0..19)

Split:
    train = VOC2007 trainval + VOC2012 trainval  minus  <val-size> random held-out images
    val   = the held-out images (fixed seed)
    test  = VOC2007 test (4952 images)

Usage (project root):
    python prepare_voc.py
    python prepare_voc.py --keep-difficult --val-size 500 --force

CLASSES below must match DATASETS["voc"]["classes"] in src/dataset.py.
"""
import argparse
import json
import os
import random
import shutil
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **kw):
        return x

# ---------------- paths (same BASE logic as prepare_wildlife.py) ----------------
BASE = Path("/kaggle/temp") if Path("/kaggle").exists() else Path("data")
RAW = BASE / "voc_raw"
OUT = BASE / "voc"

URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0/"
ZIPS = ["VOCtrainval_06-Nov-2007.zip", "VOCtest_06-Nov-2007.zip", "VOCtrainval_11-May-2012.zip"]

CLASSES = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
           "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa",
           "train", "tvmonitor"]
CLASS_ID = {c: i for i, c in enumerate(CLASSES)}

EXPECTED = {"2007 trainval": 5011, "2012 trainval": 11540, "2007 test": 4952}


# ---------------- download + extract ----------------
def _progress_hook():
    state = {"last": -10}

    def hook(blocks, block_size, total):
        if total <= 0:
            return
        pct = min(100, int(blocks * block_size * 100 / total))
        if pct >= state["last"] + 10:
            state["last"] = pct - pct % 10
            print(f"    {pct}%", flush=True)
    return hook


def download(name):
    dest = RAW / name
    if dest.exists():
        return dest
    RAW.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    for attempt in range(1, 4):
        try:
            print(f"downloading {name} (attempt {attempt}/3)")
            urllib.request.urlretrieve(URL + name, part, reporthook=_progress_hook())
            part.rename(dest)
            return dest
        except Exception as e:
            print(f"  failed: {e}")
            part.unlink(missing_ok=True)
    raise SystemExit(f"{URL + name} download nahi hua. Internet on hai? (Kaggle: Session options -> Internet)")


def extract(name):
    marker = RAW / f".extracted_{name}"
    if marker.exists():
        return
    zp = download(name)
    print(f"extracting {name} ...")
    try:
        with zipfile.ZipFile(zp) as zf:
            zf.extractall(RAW)             # 2007 trainval + test dono VOCdevkit/VOC2007 mein merge hote hain
    except zipfile.BadZipFile:
        zp.unlink(missing_ok=True)
        raise SystemExit(f"{name} corrupt tha, delete kar diya. Script dobara chala.")
    marker.touch()


def find_year_dir(year):
    for p in RAW.rglob(f"VOC{year}"):
        if p.is_dir() and (p / "Annotations").exists():
            return p
    raise SystemExit(f"VOC{year} folder nahi mila {RAW} ke andar.")


def read_ids(year_dir, split):
    f = year_dir / "ImageSets" / "Main" / f"{split}.txt"
    if not f.exists():
        raise SystemExit(f"{f} nahi mili.")
    return f.read_text().split()


# ---------------- conversion ----------------
def convert(year_dir, image_id, keep_difficult):
    """Returns (label lines, image path). Boxes: VOC 1-based inclusive pixels -> normalized cx cy w h."""
    img = year_dir / "JPEGImages" / f"{image_id}.jpg"
    with Image.open(img) as im:
        W, H = im.size                                     # asli image size (dataloader bhi yahi dekhta hai)
    root = ET.parse(year_dir / "Annotations" / f"{image_id}.xml").getroot()

    lines = []
    for obj in root.iter("object"):
        name = obj.findtext("name").strip()
        if name not in CLASS_ID:
            raise SystemExit(f"unknown class '{name}' in {image_id}")
        if int((obj.findtext("difficult") or "0").strip()) and not keep_difficult:
            continue
        bb = obj.find("bndbox")
        xmin, ymin, xmax, ymax = (float(bb.findtext(k)) for k in ("xmin", "ymin", "xmax", "ymax"))
        x1, y1 = max(0.0, xmin - 1), max(0.0, ymin - 1)    # 1-based inclusive -> 0-based edges
        x2, y2 = min(float(W), xmax), min(float(H), ymax)
        if x2 <= x1 or y2 <= y1:
            continue
        lines.append(f"{CLASS_ID[name]} {(x1 + x2) / 2 / W:.6f} {(y1 + y2) / 2 / H:.6f} "
                     f"{(x2 - x1) / W:.6f} {(y2 - y1) / H:.6f}")
    return lines, img


def place(src, dst):
    """Hard link (no extra disk), copy agar link possible na ho."""
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def write_split(split, items, keep_difficult):
    (OUT / "images" / split).mkdir(parents=True, exist_ok=True)
    (OUT / "labels" / split).mkdir(parents=True, exist_ok=True)
    stats = {"images": 0, "boxes": 0, "empty": 0, "per_class": Counter()}
    for year, year_dir, image_id in tqdm(items, desc=split):
        lines, img = convert(year_dir, image_id, keep_difficult)
        name = f"VOC{year}_{image_id}"
        place(img, OUT / "images" / split / f"{name}.jpg")
        (OUT / "labels" / split / f"{name}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
        stats["images"] += 1
        stats["boxes"] += len(lines)
        stats["empty"] += int(not lines)
        stats["per_class"].update(int(l.split()[0]) for l in lines)
    return stats


# ---------------- checks ----------------
def verify():
    """Image/label pairing, class ids, coordinate ranges."""
    for split in ("train", "val", "test"):
        imgs = {p.stem for p in (OUT / "images" / split).iterdir()}
        lbls = {p.stem for p in (OUT / "labels" / split).iterdir()}
        assert imgs == lbls, f"{split}: images aur labels match nahi karte"
        for f in (OUT / "labels" / split).glob("*.txt"):
            for line in f.read_text().splitlines():
                c, cx, cy, w, h = line.split()
                assert 0 <= int(c) < len(CLASSES), f"{f.name}: class id {c}"
                assert all(0.0 <= float(v) <= 1.0 for v in (cx, cy, w, h)), f"{f.name}: coords [0,1] se bahar"
                assert float(w) > 0 and float(h) > 0, f"{f.name}: zero-size box"
    tr = {p.stem for p in (OUT / "images" / "train").iterdir()}
    va = {p.stem for p in (OUT / "images" / "val").iterdir()}
    te = {p.stem for p in (OUT / "images" / "test").iterdir()}
    assert not (tr & va) and not (tr & te) and not (va & te), "splits overlap kar rahe hain"
    print("checks ok: pairing, class ids, coordinates, no overlap between splits")


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
            d.text((cx - w / 2 + 3, cy - h / 2 + 3), CLASSES[int(c)], fill="yellow")
        im.save(chk / f"{p.stem}.jpg")
    print(f"box-check images saved in {chk}")


# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-size", type=int, default=1000, help="trainval se held-out val images")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--keep-difficult", action="store_true", help="difficult objects bhi rakho (default: drop)")
    ap.add_argument("--force", action="store_true", help="existing output delete karke dobara banao")
    ap.add_argument("--cleanup", action="store_true", help="kaam ke baad raw zips + extracted folders delete karo")
    ap.add_argument("--check", type=int, default=8, help="box-check images ki sankhya (0 = skip)")
    args = ap.parse_args()

    opts = {"val_size": args.val_size, "seed": args.seed, "keep_difficult": args.keep_difficult}
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

    for z in ZIPS:
        extract(z)
    d07, d12 = find_year_dir("2007"), find_year_dir("2012")

    ids07_tv, ids12_tv, ids07_test = read_ids(d07, "trainval"), read_ids(d12, "trainval"), read_ids(d07, "test")
    found = {"2007 trainval": len(ids07_tv), "2012 trainval": len(ids12_tv), "2007 test": len(ids07_test)}
    print("found:", found)
    for k, v in EXPECTED.items():
        if found[k] != v:
            print(f"WARNING: {k} mein {found[k]} images mili, expected {v}")

    pool = [("2007", d07, i) for i in ids07_tv] + [("2012", d12, i) for i in ids12_tv]
    pool.sort(key=lambda t: (t[0], t[2]))                  # deterministic order, phir fixed-seed shuffle
    random.Random(args.seed).shuffle(pool)
    val_items, train_items = pool[:args.val_size], pool[args.val_size:]
    test_items = [("2007", d07, i) for i in sorted(ids07_test)]

    counts = {}
    for split, items in (("train", train_items), ("val", val_items), ("test", test_items)):
        s = write_split(split, items, args.keep_difficult)
        counts[split] = {"images": s["images"], "boxes": s["boxes"], "empty": s["empty"]}
        print(f"{split}: {s['images']} images | {s['boxes']} boxes | {s['empty']} images with no boxes")
        if split == "train":
            print("  boxes per class (train):",
                  ", ".join(f"{CLASSES[c]} {n}" for c, n in sorted(s["per_class"].items())))

    verify()
    save_checks(args.check)
    marker.write_text(json.dumps({"options": opts, "counts": counts}))

    if args.cleanup:
        shutil.rmtree(RAW)                                 # images hard link/copy ho chuki hain
        print("raw data deleted")
    print(f"done -> {OUT}")


if __name__ == "__main__":
    main()