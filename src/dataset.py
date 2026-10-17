# dataset.py
import random
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from PIL import Image, ImageEnhance, ImageOps


def _enh(enhancer):
    return lambda im, m: enhancer(im).enhance(1 + random.uniform(-m, m))


# RandAugment-lite: sirf photometric ops (boxes pe asar nahi)
PHOTO_OPS = [_enh(ImageEnhance.Brightness), _enh(ImageEnhance.Contrast),
             _enh(ImageEnhance.Color), _enh(ImageEnhance.Sharpness),
             lambda im, m: ImageOps.autocontrast(im), lambda im, m: ImageOps.equalize(im)]

# ---------------- African Wildlife config ----------------
CLASS_NAMES = ["buffalo", "elephant", "rhino", "zebra"]       # class ids 0..3
NUM_CLASSES = len(CLASS_NAMES)
# prepare_wildlife.py ka output folder (images/{train,val,test}, labels/{train,val,test})
DATA_ROOT = "/kaggle/temp/wildlife" if Path("/kaggle").exists() else "data/wildlife"

IMG_SIZE = 640
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
# ImageNet statistics for mean/std, common practice for transfer learning.
IMG_EXT = {".jpg", ".jpeg", ".png"}


class YoloDataset(Dataset):
    """
    Returns (image, target, path). target: [n, 5] = [cls, cx, cy, w, h] normalized to S x S.
    augment=False -> centered letterbox (val/test).
    augment=True  -> mosaic, zoom/shift, mixup, colour jitter, RandAugment-lite, flip, random erasing.
    """

    def __init__(self, image_paths, img_size=IMG_SIZE, augment=False, mosaic=1.0, mixup=0.0,
                 erase=0.0, photo_ops=0, photo_mag=0.4, zoom=(0.5, 1.3)):
        self.image_paths = list(image_paths)
        self.img_size, self.augment = img_size, augment
        self.mosaic_p, self.mixup_p, self.erase_p = mosaic, mixup, erase
        self.photo_ops, self.photo_mag, self.zoom = photo_ops, photo_mag, zoom

        self.color = T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.02)
        self.eraser = T.RandomErasing(p=1.0, scale=(0.02, 0.15), ratio=(0.3, 3.3), value="random")
        self.transform = T.Compose([T.ToTensor(), T.Normalize(mean=MEAN, std=STD)])

    def __len__(self):
        return len(self.image_paths)

    @staticmethod
    def read_labels(image_path):
        """.../images/<split>/x.jpg -> .../labels/<split>/x.txt -> array [N, 5]"""
        parts = list(Path(image_path).parts)
        i = len(parts) - 1 - parts[::-1].index("images")
        parts[i] = "labels"
        label_path = Path(*parts).with_suffix(".txt")
        if not label_path.exists():
            return np.zeros((0, 5), dtype=np.float32)
        rows = [list(map(float, line.split())) for line in label_path.read_text().strip().splitlines()]
        return np.array(rows, dtype=np.float32).reshape(-1, 5)

    # ---------- helpers ----------
    def _load(self, path):
        """PIL image, cls [n], boxes xyxy in ORIGINAL pixels [n, 4]."""
        image = Image.open(path).convert("RGB")
        w0, h0 = image.size
        lab = self.read_labels(path)
        if len(lab):
            boxes = np.stack([(lab[:, 1] - lab[:, 3] / 2) * w0, (lab[:, 2] - lab[:, 4] / 2) * h0,
                              (lab[:, 1] + lab[:, 3] / 2) * w0, (lab[:, 2] + lab[:, 4] / 2) * h0],
                             1).astype(np.float32)
        else:
            boxes = np.zeros((0, 4), dtype=np.float32)
        return image, lab[:, 0], boxes

    @staticmethod
    def _clip_filter(boxes, cls, S):
        """Canvas pe clip; tiny ya mostly-cropped boxes hatao."""
        if not len(boxes):
            return boxes, cls
        full = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        boxes = boxes.clip(0, S)
        bw, bh = boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]
        keep = (bw > 2) & (bh > 2) & (bw * bh > 0.3 * full)
        return boxes[keep], cls[keep]

    def _single(self, idx):
        """Letterbox (augment pe random zoom + shift/crop). Returns canvas, cls, boxes (canvas pixels)."""
        S = self.img_size
        image, cls, boxes = self._load(self.image_paths[idx])
        w0, h0 = image.size
        scale = S / max(w0, h0)
        if self.augment:
            scale *= random.uniform(*self.zoom)
        new_w, new_h = max(1, round(w0 * scale)), max(1, round(h0 * scale))
        image = image.resize((new_w, new_h), Image.BILINEAR)
        if self.augment:
            pad_x = random.randint(min(0, S - new_w), max(0, S - new_w))
            pad_y = random.randint(min(0, S - new_h), max(0, S - new_h))
        else:
            pad_x, pad_y = (S - new_w) // 2, (S - new_h) // 2
        canvas = Image.new("RGB", (S, S), (114, 114, 114))
        canvas.paste(image, (pad_x, pad_y))
        if len(boxes):
            boxes = boxes * scale + np.array([pad_x, pad_y, pad_x, pad_y], dtype=np.float32)
        boxes, cls = self._clip_filter(boxes, cls, S)
        return canvas, cls, boxes

    def _mosaic(self, idx):
        """4 images ek 2S x 2S canvas pe, centre (xc, yc) ke 4 quadrants mein; phir S x S crop."""
        S = self.img_size
        ids = [idx] + random.choices(range(len(self)), k=3)
        xc, yc = (int(random.uniform(0.5 * S, 1.5 * S)) for _ in range(2))
        canvas = Image.new("RGB", (2 * S, 2 * S), (114, 114, 114))
        all_boxes, all_cls = [], []
        for k, j in enumerate(ids):
            image, cls, boxes = self._load(self.image_paths[j])
            w0, h0 = image.size
            r = S / max(w0, h0) * random.uniform(0.5, 1.0)       # tile thoda chhota, taaki zyada objects fit hon
            w, h = max(1, round(w0 * r)), max(1, round(h0 * r))
            image = image.resize((w, h), Image.BILINEAR)
            ox, oy = [(xc - w, yc - h), (xc, yc - h), (xc - w, yc), (xc, yc)][k]
            canvas.paste(image, (ox, oy))
            if len(boxes):
                all_boxes.append(boxes * r + np.array([ox, oy, ox, oy], dtype=np.float32))
                all_cls.append(cls)
        off = S // 2
        if all_boxes:
            boxes, cls = np.concatenate(all_boxes) - off, np.concatenate(all_cls)
        else:
            boxes, cls = np.zeros((0, 4), np.float32), np.zeros((0,), np.float32)
        boxes, cls = self._clip_filter(boxes, cls, S)
        return canvas.crop((off, off, off + S, off + S)), cls, boxes

    def _base(self, idx):
        if self.augment and random.random() < self.mosaic_p:
            return self._mosaic(idx)
        return self._single(idx)

    # ---------- main ----------
    def __getitem__(self, idx):
        S = self.img_size
        canvas, cls, boxes = self._base(idx)

        if self.augment and random.random() < self.mixup_p:                 # MixUp: 2 samples blend, labels concat
            canvas2, cls2, boxes2 = self._base(random.randrange(len(self)))
            canvas = Image.blend(canvas, canvas2, float(np.random.beta(32.0, 32.0)))
            boxes, cls = np.concatenate([boxes, boxes2]), np.concatenate([cls, cls2])

        if self.augment:
            canvas = self.color(canvas)
            for op in random.sample(PHOTO_OPS, self.photo_ops):             # RandAugment-lite
                canvas = op(canvas, self.photo_mag)
            if random.random() < 0.5:                                       # horizontal flip
                canvas = canvas.transpose(Image.FLIP_LEFT_RIGHT)
                if len(boxes):
                    boxes = np.stack([S - boxes[:, 2], boxes[:, 1], S - boxes[:, 0], boxes[:, 3]], 1)

        if len(boxes):
            target = np.stack([cls, (boxes[:, 0] + boxes[:, 2]) / 2 / S, (boxes[:, 1] + boxes[:, 3]) / 2 / S,
                               (boxes[:, 2] - boxes[:, 0]) / S, (boxes[:, 3] - boxes[:, 1]) / S],
                              1).astype(np.float32)
        else:
            target = np.zeros((0, 5), dtype=np.float32)

        image = self.transform(canvas)
        if self.augment and random.random() < self.erase_p:                 # Random Erasing (boxes unchanged)
            image = self.eraser(image)
        return image, torch.from_numpy(target), str(self.image_paths[idx])

def collate_fn(batch):
    """Stack images; keep targets as a list (different object counts per image)."""
    images, targets, paths = zip(*batch)
    return torch.stack(images, 0), list(targets), list(paths)


def build_folder_loaders(root=DATA_ROOT, splits=("train", "val", "test"), batch_size=8,
                         img_size=IMG_SIZE, augment=True, num_workers=4, aug=None):
    root, loaders, aug = Path(root), [], (aug or {})
    for s in splits:
        paths = sorted(p for p in (root / "images" / s).iterdir() if p.suffix.lower() in IMG_EXT)
        assert paths, f"{root}/images/{s} khaali hai, prepare_wildlife.py chalaya?"
        ds = YoloDataset(paths, img_size, augment=(augment and s == "train"), **aug)
        loaders.append(DataLoader(ds, batch_size=batch_size, shuffle=(s == "train"),
                                  num_workers=num_workers, collate_fn=collate_fn))
    return loaders