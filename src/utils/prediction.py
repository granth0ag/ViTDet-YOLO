"""Sliced inference and suppression for the optional detector ablations."""

import numpy as np
import torch
from PIL import Image
from torchvision.ops import batched_nms, box_convert

from src.dataset import MEAN, STD

try:
    from ultralytics.utils.nms import non_max_suppression
except ImportError:
    from ultralytics.utils.ops import non_max_suppression


MAX_DET = 300
MAX_CANDIDATES = 30000


def letterbox_geometry(image, size):
    width, height = image.size
    scale = size / max(width, height)
    new_w, new_h = max(1, round(width * scale)), max(1, round(height * scale))
    return scale, (size - new_w) // 2, (size - new_h) // 2, new_w, new_h


def letterbox_tensor(image, size):
    scale, pad_x, pad_y, new_w, new_h = letterbox_geometry(image, size)
    canvas = Image.new("RGB", (size, size), (114, 114, 114))
    canvas.paste(image.resize((new_w, new_h), Image.Resampling.BILINEAR), (pad_x, pad_y))
    tensor = torch.from_numpy(np.array(canvas)).permute(2, 0, 1).float() / 255
    tensor = (tensor - tensor.new_tensor(MEAN)[:, None, None]) / tensor.new_tensor(STD)[:, None, None]
    return tensor, (scale, pad_x, pad_y)


def decode_candidates(predictions, conf):
    """Decoded YOLO xywh and class scores -> one xyxy candidate per anchor."""
    for image in predictions.transpose(1, 2).float():
        scores, labels = image[:, 4:].max(dim=1)
        keep = scores > conf
        boxes = box_convert(image[keep, :4], in_fmt="cxcywh", out_fmt="xyxy")
        yield torch.cat((boxes, scores[keep, None], labels[keep, None]), dim=1)


def gaussian_soft_nms(detections, conf, sigma):
    """Greedily rerank candidates after same-class Gaussian score decay."""
    data = detections.detach().cpu().numpy().copy()
    areas = (data[:, 2] - data[:, 0]) * (data[:, 3] - data[:, 1])
    scores = data[:, 4].copy()
    selected = []
    for _ in range(min(MAX_DET, len(data))):
        index = scores.argmax()
        if scores[index] <= conf:
            break
        data[index, 4] = scores[index]
        selected.append(index)
        scores[index] = 0
        rest = np.flatnonzero((scores > conf) & (data[:, 5] == data[index, 5]))
        intersection = (
            np.minimum(data[rest, 2:4], data[index, 2:4])
            - np.maximum(data[rest, :2], data[index, :2])
        ).clip(min=0).prod(axis=1)
        union = areas[rest] + areas[index] - intersection
        iou = intersection / np.maximum(union, 1e-7)
        scores[rest] *= np.exp(-(iou ** 2) / sigma)
    return torch.from_numpy(data[selected]).to(detections.device)


def suppress(candidates, conf, iou, options):
    if len(candidates) > MAX_CANDIDATES:
        candidates = candidates[candidates[:, 4].topk(MAX_CANDIDATES).indices]
    if options.soft_nms:
        return gaussian_soft_nms(candidates, conf, options.soft_nms_sigma)
    keep = batched_nms(candidates[:, :4], candidates[:, 4], candidates[:, 5].long(), iou)
    return candidates[keep[:MAX_DET]]


def detect_batch(model, images, conf, iou, options):
    decoded, _ = model(images)
    if options.soft_nms:
        return [suppress(d, conf, iou, options) for d in decode_candidates(decoded, conf)]
    return non_max_suppression(decoded, conf_thres=conf, iou_thres=iou, max_det=MAX_DET)


def to_source(detections, geometry, width, height):
    """Map letterboxed xyxy to bounded coordinates in the source image."""
    scale, pad_x, pad_y = geometry
    detections = detections.clone()
    detections[:, 0:4:2] = ((detections[:, 0:4:2] - pad_x) / scale).clamp(0, width)
    detections[:, 1:4:2] = ((detections[:, 1:4:2] - pad_y) / scale).clamp(0, height)
    return detections


def detect_image(model, image, size, device, conf, iou, options, batch_size=4):
    """PIL input -> detections in source-image pixels."""
    width, height = image.size
    if not options.sahi:
        tensor, geometry = letterbox_tensor(image, size)
        detections = detect_batch(model, tensor[None].to(device), conf, iou, options)[0]
        return to_source(detections, geometry, width, height)

    from sahi.slicing import get_slice_bboxes

    tile_size = options.tile_size or max(1, (max(width, height) + 1) // 2)
    full_image = (0, 0, width, height)
    tiles = get_slice_bboxes(
        image_height=height, image_width=width,
        slice_height=min(height, tile_size), slice_width=min(width, tile_size),
        overlap_height_ratio=options.tile_overlap, overlap_width_ratio=options.tile_overlap,
        auto_slice_resolution=False,
    )
    views = [full_image] + [tuple(tile) for tile in tiles if tuple(tile) != full_image]
    candidates = []
    for start in range(0, len(views), batch_size):
        batch = views[start:start + batch_size]
        inputs, geometries = zip(*(letterbox_tensor(image.crop(tile), size) for tile in batch))
        decoded, _ = model(torch.stack(inputs).to(device))
        for detection, tile, geometry in zip(decode_candidates(decoded, conf), batch, geometries):
            x0, y0, x1, y1 = tile
            detection = to_source(detection, geometry, x1 - x0, y1 - y0)
            valid = (detection[:, 2] > detection[:, 0]) & (detection[:, 3] > detection[:, 1])
            detection = detection[valid]
            detection[:, :4] += detection.new_tensor((x0, y0, x0, y0))
            candidates.append(detection)
    return suppress(torch.cat(candidates), conf, iou, options)
