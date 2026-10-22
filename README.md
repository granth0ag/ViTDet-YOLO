# ViTDet-YOLO: A ViTDet-Style Detector with a Simple Feature Pyramid and YOLOv8 Head

A from-scratch object detection pipeline combining a **plain Vision Transformer backbone**, a **Simple Feature Pyramid**, and the **YOLOv8 anchor-free detection head**.

The project is a controlled framework for studying **Vision Transformer representations for object detection** and for testing whether **lightweight convolutional inductive biases** improve detection performance.

The pipeline is evaluated on **Pascal VOC** and the **African Wildlife** dataset. A convolution-enhanced variant, inspired by the local feature extraction in **LocalViT**, is compared against the plain ViT baseline on both datasets.

---

## Project Overview

```text
                    Baseline
                       │
                       ▼
Image → ViT Backbone → Simple Feature Pyramid → YOLOv8 Head → Predictions

                Convolution-enhanced
                       │
                       ▼
Image → ViT Backbone + Local Conv → Simple Feature Pyramid → YOLOv8 Head → Predictions
```

The goal is not to reproduce ViTDet exactly, but to build a **ViTDet-style detection pipeline** and use it as a controlled environment for studying the effect of local convolutional processing.

### Research question

> **Can lightweight convolutional inductive biases improve the object detection performance of a plain Vision Transformer without replacing its global attention-based representation?**

---

## Architecture

### Baseline

```text
Input Image
    ↓
ViT-S/16 Backbone
    ↓
Last stride-16 feature map
    ↓
Simple Feature Pyramid
    ├── P3: stride 8  → upsampling / deconvolution
    ├── P4: stride 16 → identity
    └── P5: stride 32 → max pooling
    ↓
YOLOv8 Detect Head
    ↓
NMS
    ↓
Bounding Boxes + Classes
```

The backbone is an **ImageNet-pretrained ViT-S/16** from `timm`, using windowed self-attention with periodic global-attention blocks for cross-window information propagation.

The design follows **ViTDet**, particularly its use of a plain ViT backbone with a Simple Feature Pyramid for dense prediction. Note that this is not a conventional FPN: the multi-scale maps are generated from the single stride-16 feature map (deconvolution / identity / pooling), with no top-down fusion or lateral connections.

### Convolution-enhanced variant

The second variant adds lightweight convolutional processing to the ViT feature representation, inspired by the local feature extraction approach of **LocalViT**. The convolutional module gives the backbone an explicit local inductive bias while keeping the ViT-based detection pipeline intact.

Both models share the same:

- detection head
- feature pyramid
- datasets
- training framework
- evaluation metrics

so the main architectural difference is the local convolutional processing.

---

## Relation to ViTDet

This implementation is **ViTDet-inspired, not a reproduction of ViTDet**. Differences from the original setup:

- ViT-S/16 instead of the larger MAE-pretrained ViT-B
- ImageNet-supervised initialization
- YOLOv8 detection head instead of the original detection framework
- No relative position embeddings
- Different attention/window configuration
- A smaller feature pyramid
- Custom training and loss integration

The purpose is a lightweight, independently implemented pipeline for controlled experiments, not matching the original benchmark numbers.

---

## Experiments

| Variant | Backbone | Local convolution | Purpose |
|---|---|---|---|
| **ViT Baseline** | ViT-S/16 | No | Establish baseline |
| **Conv-enhanced ViT** | ViT-S/16 | Yes | Evaluate local inductive bias |

### Datasets

**Pascal VOC** — a standard multi-class detection benchmark with bounding-box annotations across 20 object categories.

**African Wildlife** — four categories (Buffalo, Elephant, Rhino, Zebra) with YOLO-format annotations.

---

## Experimental Setup

| Setting | Value |
|---|---|
| Backbone | ViT-S/16 |
| Initialization | ImageNet pretrained |
| Detection head | YOLOv8 Detect |
| Feature pyramid | Simple Feature Pyramid |
| Input resolution | 640 × 640 (African Wildlife) |
| Optimizer | AdamW |
| Weight decay | 0.05 |
| LR schedule | Cosine |
| Gradient clipping | 10 |
| Backbone learning rate | 1e-4 |
| Neck + head learning rate | 1e-3 |
| Batch size | 4 |
| Drop path | 0.1 |

Augmentations: random zoom, shift/crop, colour jitter, horizontal flip.

Hyperparameters were set once and **not extensively tuned for either architecture**.

---

## Results

### African Wildlife (test split)

| Metric | ViT Baseline | Conv-enhanced |
|---|---:|---:|
| mAP@0.5 | 0.9519 | **0.9534** |
| mAP@[0.5:0.95] | **0.7172** | 0.7166 |
| mAP@0.75 | **0.8493** | 0.8390 |
| mAR@100 | 0.7587 | **0.7671** |

On African Wildlife the two models are essentially tied. The conv-enhanced model is marginally better at mAP@0.5 and mAR@100, while the baseline is marginally better at mAP@[0.5:0.95] and mAP@0.75, so the VOC pattern does not repeat here. These differences are well within what single-run variation could produce.

### Pascal VOC

| Metric | ViT Baseline | Conv-enhanced |
|---|---:|---:|
| mAP@[0.5:0.95] | 0.5343 | **0.5420** |
| mAP@0.5 | **0.7692** | 0.7638 |
| mAP@0.75 | 0.5875 | **0.5979** |

On VOC, the convolutional module gives a small increase in overall mAP and AP@0.75, while mAP@0.5 decreases slightly, with only a minimal increase in computation time. This hints that local convolution may help localization quality more than coarse detection, but these are **preliminary single-run observations**, not evidence of a statistically significant improvement.

---

## Training Curves

### African Wildlife

<img width="751" height="433" alt="Training curves 1" src="https://github.com/user-attachments/assets/57458ee8-e8a3-45b3-83e6-d0f3f6d70459" />

<img width="751" height="433" alt="Training curves 2" src="https://github.com/user-attachments/assets/2b31ada5-6763-40ca-bd9f-3d61fb80a632" />

<img width="884" height="433" alt="Training curves 3" src="https://github.com/user-attachments/assets/c409265a-a45a-4899-87f0-ec5963531863" />

### Pascal VOC

**ViT Baseline**

<img width="1391" height="281" alt="Pascal VOC training curves, ViT baseline" src="https://github.com/user-attachments/assets/5c54b6a2-8ae6-42f7-a1b6-252f13939f70" />

**Conv-enhanced ViT**

<img width="1006" height="172" alt="Pascal VOC training curves, conv-enhanced ViT" src="https://github.com/user-attachments/assets/7db5f267-4f82-482e-8843-f962ea9b22ef" />

---

## Sample Detections

### African Wildlife

<img width="399" height="167" alt="Sample detections" src="https://github.com/user-attachments/assets/2c947232-0c58-4c08-b61f-60e2e7ee9ea0" />

### Pascal VOC

**ViT Baseline**

> **[ADD IMAGE HERE: Pascal VOC sample detections, ViT baseline]**

**Conv-enhanced ViT**

<img width="387" height="170" alt="Pascal VOC sample detections, conv-enhanced ViT" src="https://github.com/user-attachments/assets/df4c5de8-4e5f-40b3-b926-029f07ee50b3" />

---

## What This Project Investigates

The convolutional module is treated less as another architectural tweak and more as a probe of **local inductive bias** in detection:

- **Self-attention** gives global interactions between image patches.
- **Convolution** adds an explicit locality and spatially structured prior.
- Detection needs both semantic understanding and precise spatial reasoning.

The experiments ask whether a lightweight local convolutional component can supply useful spatial information to a ViT-based detector.

---

## Limitations

- Comparisons are based on single runs.
- No statistical significance is established.
- Hyperparameters were not exhaustively tuned for each variant.
- The convolutional enhancement is only one way of introducing local inductive bias.
- Results may depend on dataset characteristics and object scale.

Multi-seed runs, additional datasets, and computational-cost comparisons are needed to determine whether the observed differences are consistent.

---

## Project Structure

```text
.
├── prepare_wildlife.py     # dataset download + arrangement
├── prepare_VOC.py          # Pascal VOC preparation
├── train.py                # training loop + TensorBoard logging
├── evaluate.py             # evaluation
├── inference.py            # inference and visualization
├── smoke.py                # pipeline sanity check
└── src/
    ├── dataset.py          # dataset, preprocessing and augmentation
    ├── model/
    │   ├── vit.py          # ViT backbone
    │   ├── vit_withconv.py # convolution-enhanced ViT
    │   ├── sfp.py          # Simple Feature Pyramid
    │   ├── detector.py     # YOLOv8 detection head
    │   └── build.py        # model construction
    ├── losses/
    │   └── detection_loss.py
    └── utils/
        └── tb_utils.py
```

---

## Installation

```bash
pip install -r requirements.txt
```

## Dataset Preparation

```bash
# African Wildlife
python prepare_wildlife.py

# Pascal VOC
python prepare_VOC.py
```

## Training

```bash
python train.py
```

See the arguments in `train.py` for dataset, backbone, augmentation, resolution, and experiment configuration.

## Evaluation

```bash
python evaluate.py
```

## Inference

```bash
python inference.py <image_or_folder>
```

## TensorBoard

```bash
tensorboard --logdir runs
```

---

## References

- Li et al., **Exploring Plain Vision Transformer Backbones for Object Detection**, ECCV 2022 (ViTDet).
- **LocalViT: Bringing Locality to Vision Transformers**, 2021.
- Mehta and Rastegari, **MobileViT: Light-weight, General-purpose, and Mobile-friendly Vision Transformer**, ICLR 2022.
- **Ultralytics YOLOv8** implementation.
- **timm: PyTorch Image Models**.

---

## Status

An ongoing experimental project on **ViT-based object detection and local convolutional inductive biases**.

- [x] ViT-based detection pipeline
- [x] Simple Feature Pyramid
- [x] YOLOv8 detection head integration
- [x] African Wildlife evaluation
- [x] Pascal VOC evaluation
- [x] Convolution-enhanced ViT variant
- [x] Baseline vs convolutional comparison (VOC)
- [x] Baseline vs convolutional comparison (African Wildlife)
- [ ] Multi-seed evaluation
- [ ] Additional datasets
- [ ] Computational-cost analysis
- [ ] Further ablations of local feature extraction
