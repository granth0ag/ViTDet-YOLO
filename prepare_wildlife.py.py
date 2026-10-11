# prepare_wildlife.py
import shutil
import urllib.request
import zipfile
from pathlib import Path

BASE = Path("/kaggle/temp") if Path("/kaggle").exists() else Path("data")
RAW, OUT = BASE / "wildlife_raw", BASE / "wildlife"
URL = "https://ultralytics.com/assets/african-wildlife.zip"
SPLITS = {"train": "train", "valid": "val", "val": "val", "test": "test"}
EXT = {".jpg", ".jpeg", ".png"}


def main():
    if (OUT / "images" / "train").exists():
        print(f"{OUT} pehle se bana hai, skip")
        return

    RAW.mkdir(parents=True, exist_ok=True)
    zp = RAW / "african-wildlife.zip"
    if not zp.exists():
        print("downloading ...")
        urllib.request.urlretrieve(URL, zp)
    with zipfile.ZipFile(zp) as zf:
        zf.extractall(RAW)

    counts = {"train": 0, "val": 0, "test": 0}
    for img in sorted(RAW.rglob("*")):
        if img.suffix.lower() not in EXT:
            continue
        parts = img.relative_to(RAW).parts
        split = next((SPLITS[p] for p in parts if p in SPLITS), None)
        if split is None or "images" not in parts:
            continue
        i = len(parts) - 1 - parts[::-1].index("images")      # last 'images' folder
        lbl = (RAW / Path(*parts[:i], "labels", *parts[i + 1:])).with_suffix(".txt")
        if not lbl.exists():
            continue
        (OUT / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUT / "labels" / split).mkdir(parents=True, exist_ok=True)
        shutil.copy(img, OUT / "images" / split / img.name)
        shutil.copy(lbl, OUT / "labels" / split / lbl.name)
        counts[split] += 1
    print(counts)

    ids = [int(l.split()[0]) for f in (OUT / "labels").rglob("*.txt")
           for l in f.read_text().splitlines() if l.strip()]
    print(f"boxes: {len(ids)} | class ids: {min(ids)}..{max(ids)}")
    assert min(ids) >= 0 and max(ids) <= 3, "class ids 0-3 se bahar hain"
    print("saved to", OUT)


if __name__ == "__main__":
    main()