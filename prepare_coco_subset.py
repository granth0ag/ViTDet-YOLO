
"""
COCO val2017 se N_IMAGES chun ke YOLO-format subset banata hai, train/test split ke saath.

Output layout:
    <OUT>/images/{train,test}/*.jpg
    <OUT>/labels/{train,test}/*.txt     (class cx cy w h, normalized, class 0-79)

Run (project root se, ek baar):  python prepare_coco_subset.py
"""
import random
import shutil
import urllib.request
import zipfile
from pathlib import Path

# ---------------- config ----------------
N_IMAGES = 4000
SPLITS = {"train": 0.9, "test": 0.1}
SEED = 42

# Kaggle pe /kaggle/temp (output mein nahi judta), laptop pe ./data
BASE = Path("/kaggle/temp") if Path("/kaggle").exists() else Path("data")
RAW = BASE / "coco_raw"
OUT = BASE / "coco_subset"

IMAGES_ZIP = ("val2017.zip", "http://images.cocodataset.org/zips/val2017.zip")
LABELS_ZIP = ("coco2017labels.zip",
              "https://github.com/ultralytics/assets/releases/download/v0.0.0/coco2017labels.zip")


def fetch(name, url):
    """Download + extract a zip into RAW (skipped if the zip already exists)."""
    RAW.mkdir(parents=True, exist_ok=True)
    zp = RAW / name
    if not zp.exists():
        print(f"downloading {name} ...")
        try:
            urllib.request.urlretrieve(url, zp)
        except Exception as e:
            zp.unlink(missing_ok=True)
            raise SystemExit(f"Download fail hua: {url}\n{e}\n"
                             "URL check kar (Ultralytics assets releases / cocodataset.org).")
    print(f"extracting {name} ...")
    with zipfile.ZipFile(zp) as zf:
        zf.extractall(RAW)


def find_dirs():
    """Image dir aur label dir dhundho, na mile to download karo."""
    img_dir = RAW / "val2017"
    if not img_dir.exists():
        fetch(*IMAGES_ZIP)

    def label_dir():
        # .../labels/val2017 (zip ke andar ka exact path alag ho sakta hai)
        return next((p for p in RAW.rglob("val2017")
                     if p.is_dir() and p.parent.name == "labels"), None)

    lbl_dir = label_dir()
    if lbl_dir is None:
        fetch(*LABELS_ZIP)
        lbl_dir = label_dir()
    if lbl_dir is None:
        raise SystemExit("labels/val2017 folder nahi mila, zip ka structure check kar.")
    return img_dir, lbl_dir


def valid_label(path):
    """True agar file exist kare aur kam se kam ek proper (5 column) line ho."""
    if not path.exists():
        return False
    lines = [l.split() for l in path.read_text().strip().splitlines() if l.strip()]
    return len(lines) > 0 and all(len(l) == 5 for l in lines)


def main():
    if (OUT / "images" / "train").exists() and any((OUT / "images" / "train").iterdir()):
        print(f"{OUT} pehle se bana hai, skip. (dobara banana ho to folder delete kar)")
        return

    img_dir, lbl_dir = find_dirs()

    ids = [p.stem for p in sorted(img_dir.glob("*.jpg")) if valid_label(lbl_dir / f"{p.stem}.txt")]
    print("labelled images available:", len(ids))
    assert len(ids) >= N_IMAGES, f"sirf {len(ids)} images mili, N_IMAGES kam kar"

    random.Random(SEED).shuffle(ids)
    ids = ids[:N_IMAGES]

    n_train = int(len(ids) * SPLITS["train"])
    split_ids = {"train": ids[:n_train], "test": ids[n_train:]}

    for split, sids in split_ids.items():
        (OUT / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUT / "labels" / split).mkdir(parents=True, exist_ok=True)
        for s in sids:
            shutil.copy(img_dir / f"{s}.jpg", OUT / "images" / split / f"{s}.jpg")
            shutil.copy(lbl_dir / f"{s}.txt", OUT / "labels" / split / f"{s}.txt")
        print(f"{split}: {len(sids)} images")

    # ---- sanity check: class ids 0-79 ----
    cls = [int(l.split()[0])
           for f in (OUT / "labels").rglob("*.txt")
           for l in f.read_text().splitlines() if l.strip()]
    print(f"boxes: {len(cls)} | class id range: {min(cls)}..{max(cls)}")
    assert 0 <= min(cls) and max(cls) <= 79, "class ids 0-79 se bahar hain"
    print("saved to", OUT)


if __name__ == "__main__":
    main()