"""Prepare cell_Pic1 + Xml_cell as the single-class foci_cell dataset."""

import argparse
import hashlib
import json
import random
import shutil
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

from PIL import Image, ImageDraw

from prepare_foci import convert, place


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(raw_root, output, val_size=5, seed=42):
    raw_root, output = Path(raw_root).resolve(), Path(output).resolve()
    images_dir, xml_dir = raw_root / "cell_Pic1", raw_root / "Xml_cell"
    if not images_dir.is_dir() or not xml_dir.is_dir():
        raise ValueError("--raw-root must contain cell_Pic1/ and Xml_cell/")
    if output == raw_root or raw_root in output.parents or output in raw_root.parents:
        raise ValueError("Output must be separate from the raw dataset")

    def indexed(folder, extensions):
        files = {}
        for path in sorted(folder.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in extensions:
                continue
            if path.stem in files:
                raise ValueError(f"Duplicate stem {path.stem} under {folder}")
            files[path.stem] = path
        return files

    images = indexed(images_dir, {".jpg", ".jpeg", ".png"})
    annotations = indexed(xml_dir, {".xml"})
    if images.keys() != annotations.keys():
        raise ValueError(f"Missing XMLs: {sorted(images.keys() - annotations.keys())}; "
                         f"missing images: {sorted(annotations.keys() - images.keys())}")
    if not 0 < val_size < len(images):
        raise ValueError(f"val-size must be between 1 and {len(images) - 1}")

    records = {}
    for stem, image in images.items():
        xml = annotations[stem]
        tree = ET.parse(xml).getroot()
        with Image.open(image) as im:
            width, height = im.size
        if (int(tree.findtext("size/width")), int(tree.findtext("size/height"))) != (width, height):
            raise ValueError(f"Image/XML dimensions differ for {image.name}")
        for obj in tree.findall("object"):
            box = obj.find("bndbox")
            if box is None:
                raise ValueError(f"Missing bounding box in {xml.name}")
            x1, y1, x2, y2 = [float(box.findtext(k)) for k in ("xmin", "ymin", "xmax", "ymax")]
            if not (1 <= x1 <= x2 <= width and 1 <= y1 <= y2 <= height):
                raise ValueError(f"Invalid or out-of-bounds cell box in {xml.name}")
        lines, _ = convert(image, xml, source_class="cell")
        records[stem] = dict(image=image.name, image_sha256=digest(image),
                             xml_sha256=digest(xml), boxes=len(lines),
                             labels="\n".join(lines) + ("\n" if lines else ""))

    stems = sorted(images)
    random.Random(seed).shuffle(stems)
    splits = {"train": stems[val_size:], "val": stems[:val_size]}
    manifest = dict(format_version=1, source=str(raw_root), seed=seed, val_size=val_size,
                    classes=["cell_patch"], source_class="cell", splits=splits,
                    coordinate_convention="VOC 1-based inclusive to normalized YOLO edges",
                    files=records,
                    counts={s: dict(images=len(ids), boxes=sum(records[i]["boxes"] for i in ids))
                            for s, ids in splits.items()})

    def verify(folder):
        for split, ids in splits.items():
            image_dir, label_dir = folder / "images" / split, folder / "labels" / split
            if {p.name for p in image_dir.iterdir()} != {records[i]["image"] for i in ids}:
                raise ValueError(f"Prepared images differ for {split}")
            if {p.name for p in label_dir.iterdir()} != {i + ".txt" for i in ids}:
                raise ValueError(f"Prepared labels differ for {split}")
            for stem in ids:
                record = records[stem]
                if digest(image_dir / record["image"]) != record["image_sha256"]:
                    raise ValueError(f"Prepared image differs: {stem}")
                if (label_dir / f"{stem}.txt").read_text() != record["labels"]:
                    raise ValueError(f"Prepared label differs: {stem}")

    if output.exists():
        marker = output / ".done"
        if not marker.is_file() or json.loads(marker.read_text()) != manifest:
            raise ValueError(f"{output} contains different data/options; select a new --output directory")
        verify(output)
        print(f"Reusing verified cell data at {output}")
        return manifest

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".foci_cell-", dir=output.parent))
    try:
        for split, ids in splits.items():
            (staging / "images" / split).mkdir(parents=True)
            (staging / "labels" / split).mkdir(parents=True)
            for stem in ids:
                place(images[stem], staging / "images" / split / images[stem].name)
                (staging / "labels" / split / f"{stem}.txt").write_text(records[stem]["labels"])
        verify(staging)
        (staging / "_check").mkdir()
        for stem in splits["val"]:
            with Image.open(images[stem]) as source:
                im = source.convert("RGB")
            draw = ImageDraw.Draw(im)
            for line in records[stem]["labels"].splitlines():
                _, cx, cy, w, h = map(float, line.split())
                draw.rectangle(((cx - w / 2) * im.width, (cy - h / 2) * im.height,
                                (cx + w / 2) * im.width, (cy + h / 2) * im.height),
                               outline="lime", width=2)
            im.save(staging / "_check" / f"{stem}.jpg")
        (staging / ".done").write_text(json.dumps(manifest, indent=2))
        staging.rename(output)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    print(f"Prepared {output}: {manifest['counts']}")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    base = Path("/kaggle/temp") if Path("/kaggle").exists() else Path("data")
    parser.add_argument("--output", type=Path, default=base / "foci_cell")
    parser.add_argument("--val-size", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42, help="fixed dataset split seed")
    args = parser.parse_args()
    try:
        prepare(args.raw_root, args.output, args.val_size, args.seed)
    except (ValueError, OSError, ET.ParseError) as error:
        parser.exit(1, f"Cell preparation failed: {error}\n")


if __name__ == "__main__":
    main()
