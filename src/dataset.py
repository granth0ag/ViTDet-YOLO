# dataset.py
import random
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset

COCO128_URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0/coco128.zip"
IMG_SIZE = 320
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
# ImageNet statistics for mean/std, common practice for transfer learning.


def download_coco128(root="data"):
    """Download + extract COCO128 into data/coco128/ (skipped if it already exists)."""
    root = Path(root)
    dataset_dir = root / "coco128"
    if dataset_dir.exists():
        return dataset_dir
    root.mkdir(parents=True, exist_ok=True)
    zip_path = root / "coco128.zip"
    urllib.request.urlretrieve(COCO128_URL, zip_path)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(root)
    zip_path.unlink()
    return dataset_dir


class CocoDataset(Dataset):
    """
    Returns (image, target, image_path):
        image  : [3, 320, 320], ImageNet-normalized
        target : [num_objects, 5] -> [class_id, cx, cy, w, h], normalized to the 320x320 image
        path   : str
    """

    def __init__(self, image_paths, img_size=IMG_SIZE, augment=False):
        self.image_paths = list(image_paths)
        self.img_size = img_size
        self.augment = augment  # horizontal flip only

        # PIL -> tensor [3,H,W] in [0,1], then ImageNet normalize
        self.transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=MEAN, std=STD),
        ])

    def __len__(self):
        return len(self.image_paths)

    @staticmethod
    def read_labels(image_path):
        """Read the matching YOLO .txt file -> array [N, 5].
        .../images/<split>/x.jpg  ->  .../labels/<split>/x.txt (sirf last 'images' folder badalta hai)"""
        parts = list(Path(image_path).parts)
        i = len(parts) - 1 - parts[::-1].index("images")
        parts[i] = "labels"
        label_path = Path(*parts).with_suffix(".txt")

        if not label_path.exists():
            return np.zeros((0, 5), dtype=np.float32)
        rows = [list(map(float, line.split()))
                for line in label_path.read_text().strip().splitlines()]
        return np.array(rows, dtype=np.float32).reshape(-1, 5)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        image = Image.open(path).convert("RGB")
        w0, h0 = image.size
        labels = self.read_labels(path)  # cls, cx, cy, w, h (relative to the ORIGINAL image)

        # letterbox: resize keeping aspect ratio, pad to a square
        S = self.img_size
        scale = S / max(w0, h0)
        new_w, new_h = round(w0 * scale), round(h0 * scale)
        image = image.resize((new_w, new_h), Image.BILINEAR)

        pad_x = (S - new_w) // 2
        pad_y = (S - new_h) // 2
        canvas = Image.new("RGB", (S, S), (114, 114, 114))
        canvas.paste(image, (pad_x, pad_y))

        # boxes: original-relative -> pixels -> letterboxed pixels -> 320-relative
        target = labels.copy()
        target[:, 1] = (labels[:, 1] * w0 * scale + pad_x) / S  # cx
        target[:, 2] = (labels[:, 2] * h0 * scale + pad_y) / S  # cy
        target[:, 3] = (labels[:, 3] * w0 * scale) / S          # w
        target[:, 4] = (labels[:, 4] * h0 * scale) / S          # h

        # optional horizontal flip
        if self.augment and random.random() < 0.5:
            canvas = canvas.transpose(Image.FLIP_LEFT_RIGHT)
            target[:, 1] = 1.0 - target[:, 1]

        image = self.transform(canvas)
        return image, torch.from_numpy(target), str(path)


def collate_fn(batch):
    """Stack images; keep targets as a list (different object counts per image)."""
    images, targets, paths = zip(*batch)
    return torch.stack(images, 0), list(targets), list(paths)


def build_dataloaders(root="data", batch_size=8, img_size=IMG_SIZE, val_fraction=0.0,
                      augment=False, num_workers=0, seed=42):
    """COCO128 (sanity check). val_fraction=0.0 -> train and val use the SAME images."""
    dataset_dir = download_coco128(root)
    image_dir = dataset_dir / "images" / "train2017"
    all_paths = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})

    if val_fraction > 0:
        random.Random(seed).shuffle(all_paths)
        n_val = int(len(all_paths) * val_fraction)
        val_paths, train_paths = all_paths[:n_val], all_paths[n_val:]
    else:
        train_paths = val_paths = all_paths

    train_ds = CocoDataset(train_paths, img_size, augment=augment)
    val_ds = CocoDataset(val_paths, img_size, augment=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, collate_fn=collate_fn)
    return train_loader, val_loader


def build_coco_subset_loaders(root="data/coco_subset", batch_size=8, img_size=IMG_SIZE,
                              augment=True, num_workers=4):
    """
    prepare_coco_subset.py se bane folder ke liye:
        <root>/images/{train,test}/*.jpg  +  <root>/labels/{train,test}/*.txt
    `augment` sirf train pe lagta hai. Returns (train_loader, test_loader).
    """
    root = Path(root)

    def paths(split):
        d = root / "images" / split
        return sorted(p for p in d.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})

    train_ds = CocoDataset(paths("train"), img_size, augment=augment)
    test_ds = CocoDataset(paths("test"), img_size, augment=False)

    kw = dict(batch_size=batch_size, num_workers=num_workers, collate_fn=collate_fn)
    train_loader = DataLoader(train_ds, shuffle=True, **kw)
    test_loader = DataLoader(test_ds, shuffle=False, **kw)
    return train_loader, test_loader