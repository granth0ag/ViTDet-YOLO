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

## MRFP paper ablations

The paper comparison uses **three necks with the same plain ViT backbone**:

| Experiment | CLI option | Spatial operation inside each pyramid level |
|---|---|---|
| Pure MRFP (ViT-CoMer-style) | `--mrfp plain-dw` | Split reduced channels into DW 3×3 and 5×5, dilation 1; concatenate |
| MRFP with SAC inside the neck | `--mrfp plain-dw-sac` | Same channel split and kernels; each kernel group has dilation 1/3 branches with a learned softmax switch |
| Single receptive-field fusion | `--mrfp single-dw` | One DW 3×3 over all reduced channels, dilation 1 |

All three retain the same shared 1×1 channel reduction/expansion, per-level
BatchNorm, activation order and pyramid levels. The final 1×1 projection fuses
channels in all three. None adds a residual/gamma inside the MRFP block.
`single-dw` removes the parallel 3×3/5×5 kernel choice, not the existing SFP
pyramid: “single receptive field” refers to the neck's DW kernel at each level.

For `plain-dw-sac`, the 3×3 group and 5×5 group each use two independently learned
DW convolutions with dilation 1 and 3. A 5×5 average pool followed by a 1×1
switch predicts two spatial gate weights; zero logits start at a 0.5/0.5 mixture.
Switches and spatial filters are independent across pyramid levels. This is a
SAC-style extension of the reference neck, following this project's independent
branch-weight convention. It does not enable SAC in the ViT attention blocks.

Compare pure MRFP with `single-dw` to study multi-kernel versus single-kernel
fusion, and pure MRFP with `plain-dw-sac` to study the added dilation/switching.
Parameter counts differ; reports include total and MRFP-only counts, so these
are not parameter-matched controls. Keep seeds, epochs, input size, augmentation,
splits and backbone fixed across runs.

### Run the three FOCI experiments

The existing runner trains the selected variants and produces one combined
report. Use `--skip-prepare` if the FOCI dataset is already prepared. Preview:

```bash
python vit_foci.py --skip-prepare --aug A --epochs 60 --seeds 42 43 44 \
  --mrfp-variants plain-dw plain-dw-sac single-dw --dry-run
```

Remove `--dry-run` to execute **three variants × three seeds = nine training
runs**, followed by evaluation. Set epochs and augmentation to the same values
as your other experiments. No dataset preparation or training happens in a dry
run. Omit `--skip-prepare` to use the existing preparation/reuse behavior.

For this comparison, leave `--sac-rates` empty: that existing option controls
**backbone** SAC. The `plain-dw-sac` variant itself enables **neck** SAC.
Use explicit variant names; `--mrfp` without a value still selects the older
`full` variant for compatibility.

Single runs and inference use the existing entry points:

```bash
python train.py --dataset foci --aug A --seed 42 --epochs 60 --mrfp plain-dw
python train.py --dataset foci --aug A --seed 42 --epochs 60 --mrfp plain-dw-sac
python train.py --dataset foci --aug A --seed 42 --epochs 60 --mrfp single-dw
python inference.py --dataset foci --aug A --seed 42 --mrfp plain-dw-sac path/to/image.png
```

Evaluate the three experiments together:

```bash
python evaluate.py --dataset foci --aug A --seeds 42 43 44 \
  --mrfp-variants plain-dw plain-dw-sac single-dw
```

Pure MRFP, the first listed configuration, is the paired-delta reference.
With explicit MRFP options, `evaluate.py` defaults to `--configs none` (plain
ViT). The FOCI runner evaluates whichever backbone it trained. `--configs`
continues to mean backbone SAC configurations, not neck variants.

Each explicit variant has a separate checkpoint and TensorBoard suffix, e.g.
`best_foci_640_p3p5_augA_s42_mrfp-plain-dw-sac.pt` and
`runs/foci_640_p3p5_augA_s42_mrfp-single-dw`. Reports include the variant list and
backbone configuration in their filenames. Pass the **same MRFP variant** to
training, evaluation and inference. Train each new architecture; old SAC-backbone
checkpoints should not be relabelled as these neck-only experiments.

### What is taken from ViT-CoMer

The official MRFP uses **both 3×3 and 5×5**, on separate halves of the reduced
channels. Its `MultiDWConv` has independent spatial filters and BatchNorm for each
of P3, P4 and P5. The shared projections reduce channels by half and restore them.
The order is projection → split-channel DW3/DW5 → concatenate → BatchNorm/GELU →
GELU → projection (the second GELU is present in the official MRFP forward).

`plain-dw` implements that module computation on the existing SFP feature maps,
using equivalent 1×1 projections instead of flattening into tokens. It retains
this project's PyTorch module initialization; it does not import ViT-CoMer's
whole-backbone initialization routine, CNN branch or CTI interactions. No new
MMDetection/MMCV dependency is introduced. This is a **ViT-CoMer-style MRFP module
reference in this detector**, not a reproduction of the full ViT-CoMer model.
If P2 is manually enabled, the same operation is extended to the fourth level.

Sources: [CVPR paper, Section 3.2 and Figure 4](https://openaccess.thecvf.com/content/CVPR2024/papers/Xia_ViT-CoMer_Vision_Transformer_with_Convolutional_Multi-scale_Feature_Interaction_for_Dense_CVPR_2024_paper.pdf)
and [official MRFP / MultiDWConv code](https://github.com/Traffic-X/ViT-CoMer/blob/main/detection/mmdet_custom/models/backbones/comer_modules.py#L81-L164).

### Compatibility and earlier controls

Commands and model construction calls that omit the MRFP option retain the
existing behavior: plain ViT + SFP with no SAC rates, or SAC backbone + the
original residual MRFP when rates are supplied. Their checkpoint names and
state-dict layouts are unchanged. `--mrfp-variant` remains an alias for `--mrfp`.

Earlier options remain available: `none` (SFP only), `full` (the original dense
3×3 residual MRFP with dilation 1/3 and a learned switch), `no-dilation`,
`no-switch`, and `no-dilation-no-switch`. Those controls operate on the older
residual MRFP; they are separate from the three paper experiments above.

Dataset preparation, labels, splits, augmentation, losses, optimizer settings
and prediction limits are unchanged. Training accepts the existing optional third
test loader while training/validating on the first two splits. No new FOCI sample
limit is applied. The standalone `vit_foci.ipynb` remains unchanged; use the Python
entry points above for these new variants.

CPU regression tests (synthetic data, no pretrained-weight or dataset downloads):

```bash
python -m unittest discover -s tests -v
```

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
