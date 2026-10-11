# ViTDet-style Backbone + YOLOv8 Head for Object Detection

A small object detector that combines a **plain Vision Transformer backbone with windowed attention**
(ViTDet-style), a **simple feature pyramid**, and the **YOLOv8 anchor-free detection head**.
Trained and evaluated on the African Wildlife dataset (4 classes).

<!-- PASTE: detection output grid image here -->
![Detections](assets/detections.jpg)

## Architecture

```
image (640x640)
  -> ViT-S/16 backbone (ImageNet-pretrained, timm)
       windowed attention (10x10 windows) + 4 global-attention blocks (cross-window propagation)
  -> Simple feature pyramid (built from the last stride-16 map)
       P3 (stride 8, deconv) | P4 (stride 16, identity) | P5 (stride 32, max-pool)
  -> YOLOv8 Detect head (decoupled, anchor-free, DFL)
  -> NMS -> boxes + classes
```

- **Backbone / neck:** inspired by *Exploring Plain Vision Transformer Backbones for Object Detection* (ViTDet, Li et al., ECCV 2022). Implemented from scratch on top of timm weights.
- **Head, loss, NMS:** from the [Ultralytics](https://github.com/ultralytics/ultralytics) YOLOv8 implementation.
- ViTDet-inspired, not a reproduction: ImageNet-supervised ViT-S instead of MAE-pretrained ViT-B, no relative position embeddings, fewer pyramid levels.

<!-- OPTIONAL: PASTE architecture diagram here -->

## Dataset

[African Wildlife](https://docs.ultralytics.com/datasets/detect/african-wildlife/): 4 classes (buffalo, elephant, rhino, zebra), YOLO-format labels, train / val / test = <X / Y / Z> images.

```bash
python prepare_wildlife.py
```

## Project structure

```
.
├── prepare_wildlife.py     # dataset download + arrangement
├── train.py                # training loop + TensorBoard logging
├── evaluate.py             # mAP on the test split
├── inference.py            # draw detections on images
├── smoke.py                # quick pipeline sanity check
└── src
    ├── dataset.py          # dataset, letterbox, augmentation
    ├── model/              # vit.py, fpn.py, detector.py, build.py
    ├── losses/             # detection_loss.py
    └── utils/              # tb_utils.py
```

## Usage

```bash
pip install -r requirements.txt
python prepare_wildlife.py
python train.py
python evaluate.py
python inference.py <image_or_folder>
tensorboard --logdir runs
```

## Training setup

| Setting | Value |
|---|---|
| Input size | 640 x 640 (letterbox) |
| Optimizer | AdamW, weight decay 0.05, cosine schedule, grad clip 10 |
| Learning rate | backbone 1e-4, neck + head 1e-3 |
| Batch size / epochs | <..> / <..> |
| Regularization | drop path 0.1 |
| Augmentation | random zoom, shift/crop, colour jitter, horizontal flip |

Hyperparameters follow common fine-tuning practice and were not tuned.

## Results

Test split, best checkpoint (selected on the validation split):

| Metric | Value |
|---|---|
| mAP@0.5 | <..> |
| mAP@[.5:.95] | <..> |
| mAP@0.75 | <..> |
| mAR@100 | <..> |

### Training curves

<!-- PASTE: loss curves (loss/total, box, cls, dfl) -->
![Loss curves](assets/loss_curves.png)

<!-- PASTE: validation mAP curves -->
![Validation metrics](assets/metric_curves.png)

### Sample detections

<!-- PASTE: 4-6 test-image detections (green = ground truth, red = prediction if from TensorBoard) -->
![Sample detections](assets/sample_detections.png)

## Notes and limitations

- Single run, single seed. Numbers are a baseline, not a tuned result.
- Head and pyramid are trained from scratch; only the ViT backbone is pretrained.
- <What you observed: overfitting after epoch N, weakest class, etc.>

## References

- Li et al., *Exploring Plain Vision Transformer Backbones for Object Detection*, ECCV 2022.
- Ultralytics YOLOv8: https://github.com/ultralytics/ultralytics
- timm: https://github.com/huggingface/pytorch-image-models
