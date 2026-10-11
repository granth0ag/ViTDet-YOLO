# dataset.py
import random
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset

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
    YOLO-format detection dataset.
    Returns (image, target, image_path):
        image  : [3, S, S], ImageNet-normalized
        target : [num_objects, 5] -> [class_id, cx, cy, w, h], normalized to the S x S image
        path   : str

    augment=False -> plain letterbox (centered).
    augment=True  -> random zoom + random shift/crop + colour jitter + horizontal flip.
    """

    def __init__(self, image_paths, img_size=IMG_SIZE, augment=False):
        self.image_paths = list(image_paths)
        self.img_size = img_size
        self.augment = augment

        self.color = T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.02)
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
        S = self.img_size

        # boxes -> xyxy pixels in the original image
        cls = labels[:, 0]
        if len(labels):
            boxes = np.stack([(labels[:, 1] - labels[:, 3] / 2) * w0, (labels[:, 2] - labels[:, 4] / 2) * h0,
                              (labels[:, 1] + labels[:, 3] / 2) * w0, (labels[:, 2] + labels[:, 4] / 2) * h0],
                             1).astype(np.float32)
        else:
            boxes = np.zeros((0, 4), dtype=np.float32)

        # resize keeping aspect ratio (augment pe extra random zoom)
        scale = S / max(w0, h0)
        if self.augment:
            scale *= random.uniform(0.5, 1.3)
        new_w, new_h = max(1, round(w0 * scale)), max(1, round(h0 * scale))
        image = image.resize((new_w, new_h), Image.BILINEAR)

        # placement: centered (eval) / random shift, bada ho to crop (augment)
        if self.augment:
            pad_x = random.randint(min(0, S - new_w), max(0, S - new_w))
            pad_y = random.randint(min(0, S - new_h), max(0, S - new_h))
        else:
            pad_x, pad_y = (S - new_w) // 2, (S - new_h) // 2
        canvas = Image.new("RGB", (S, S), (114, 114, 114))
        canvas.paste(image, (pad_x, pad_y))  # negative offset pe PIL khud crop karta hai

        # boxes: transform, clip to canvas, tiny / mostly-cropped boxes hatao
        if len(boxes):
            boxes = boxes * scale + np.array([pad_x, pad_y, pad_x, pad_y], dtype=np.float32)
            full = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            boxes = boxes.clip(0, S)
            bw, bh = boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]
            keep = (bw > 2) & (bh > 2) & (bw * bh > 0.3 * full)
            boxes, cls = boxes[keep], cls[keep]

        # colour jitter + horizontal flip (sirf augment pe)
        if self.augment:
            canvas = self.color(canvas)
            if random.random() < 0.5:
                canvas = canvas.transpose(Image.FLIP_LEFT_RIGHT)
                if len(boxes):
                    boxes = np.stack([S - boxes[:, 2], boxes[:, 1], S - boxes[:, 0], boxes[:, 3]], 1)

        # xyxy pixels -> [cls, cx, cy, w, h] normalized to S
        if len(boxes):
            target = np.stack([cls,
                               (boxes[:, 0] + boxes[:, 2]) / 2 / S,
                               (boxes[:, 1] + boxes[:, 3]) / 2 / S,
                               (boxes[:, 2] - boxes[:, 0]) / S,
                               (boxes[:, 3] - boxes[:, 1]) / S], 1).astype(np.float32)
        else:
            target = np.zeros((0, 5), dtype=np.float32)

        return self.transform(canvas), torch.from_numpy(target), str(path)


def collate_fn(batch):
    """Stack images; keep targets as a list (different object counts per image)."""
    images, targets, paths = zip(*batch)
    return torch.stack(images, 0), list(targets), list(paths)


def build_folder_loaders(root=DATA_ROOT, splits=("train", "val", "test"), batch_size=8,
                         img_size=IMG_SIZE, augment=True, num_workers=4):
    """
    <root>/images/<split>/*.jpg  +  <root>/labels/<split>/*.txt   (prepare_wildlife.py ka output)
    `augment` sirf "train" split pe lagta hai. Returns list of loaders, `splits` ke order mein.
    """
    root, loaders = Path(root), []
    for s in splits:
        paths = sorted(p for p in (root / "images" / s).iterdir() if p.suffix.lower() in IMG_EXT)
        assert paths, f"{root}/images/{s} khaali hai, prepare_wildlife.py chalaya?"
        ds = YoloDataset(paths, img_size, augment=(augment and s == "train"))
        loaders.append(DataLoader(ds, batch_size=batch_size, shuffle=(s == "train"),
                                  num_workers=num_workers, collate_fn=collate_fn))
    return loaders