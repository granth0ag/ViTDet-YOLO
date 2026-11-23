import torch
from torchvision.ops import batched_nms


def sliced_detect(model, image, predict_tile, conf, tile_size, overlap, merge_iou):
    """Predict each PIL crop, restore xyxy coordinates and suppress same-class duplicates."""
    if tile_size <= 0 or not 0 <= overlap < 1 or not 0 <= merge_iou <= 1:
        raise ValueError("Expected positive tile size, overlap in [0,1), and merge IoU in [0,1]")
    try:
        from sahi.slicing import get_slice_bboxes
    except ImportError as exc:
        raise ImportError("Install the optional dependency with 'pip install sahi'") from exc

    tiles = get_slice_bboxes(image.height, image.width, slice_height=tile_size,
                             slice_width=tile_size, overlap_height_ratio=overlap,
                             overlap_width_ratio=overlap)
    results = []
    for left, top, right, bottom in tiles:
        det = predict_tile(model, image.crop((left, top, right, bottom)), conf).float()
        if len(det):
            det[:, :4] += det.new_tensor((left, top, left, top))
            results.append(det)
    if not results:
        return torch.empty((0, 6), dtype=torch.float32)
    det = torch.cat(results)
    keep = batched_nms(det[:, :4], det[:, 4], det[:, 5].long(), merge_iou)
    return det[keep]
