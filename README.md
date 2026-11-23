# ViTDet-YOLO: Microscopic Cell & Virus Patch Detection

**A PyTorch object-detection project combining a plain Vision Transformer, a Simple Feature Pyramid, and a YOLOv8 detection head.**

This repository focuses on detecting **small cell patches** and **virus-infected foci** in microscopy imagery. It includes an independently implemented, **ViTDet-inspired** detector and controlled experiments with local / multi-receptive-field convolutions. The aim is to build, evaluate, and document a practical detection pipeline—not to claim a new detection algorithm or a reproduction of the original ViTDet benchmark.

[Architecture](docs/ARCHITECTURE.md) · [Experimental details and seed-wise scores](docs/EXPERIMENTS.md) · [Earlier VOC / wildlife benchmarks](docs/LEGACY_BENCHMARKS.md)

## At a glance

- **Backbone:** ImageNet-pretrained ViT-S/16, local window attention with periodic global attention.
- **Feature pyramid:** stride-8 / stride-16 / stride-32 feature maps (P3–P5), all derived from the **same stride-16 ViT output**.
- **Detector:** anchor-free YOLOv8-style head with box regression and classification.
- **Variants:** plain baseline; switchable atrous convolutions (SAC) in the backbone or head; and a combined **MRFP-inspired refinement + QKV-SAC** variant.
- **Primary applications:** cell-patch detection (50 annotated images) and virus-patch detection (149 images).

## Pipeline

```text
Microscopy image
      |
      v
Pretrained plain ViT-S/16
(windowed + periodic global attention)
      |
      v
Single stride-16 feature map
      |
      v
Simple Feature Pyramid
      |             |             |
   P3 /8         P4 /16        P5 /32
      |             |             |
      +------+------+-------------+
             | 
     [optional shared MRFP-inspired
      receptive-field refinement]
             |
        YOLOv8 head
             |
     Bounding boxes / scores
```

For the **MRFP + SAC** configuration, QKV-SAC also operates **inside the ViT backbone**. The post-pyramid refinement module uses two 3×3 convolution branches (dilations 1 and 3), a spatially predicted softmax gate, and a learnable residual scale. Its parameters are **shared across P3, P4, and P5**. It is inspired by existing MRFP/SAC ideas, **not** an identical implementation of ViT-CoMer MRFP or DetectoRS SAC.

## Results

Metrics are percentages; differences in experiments and evaluation protocols are documented in [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md). **The cell-patch and virus-patch evaluations are separate tasks.**

### 1. Cell-patch detection

**50 annotated images (45 train / 5 validation).** Baseline, SAC-backbone and SAC-head scores are from epoch 200 of the single-seed study in the experimental report. The **MRFP + SAC run was configured for 200 epochs, but the available metric snapshot was recorded at epoch 195**; it is shown as observed rather than relabeled as a final-epoch score.

| Validation metric | Baseline | SAC backbone | SAC head | MRFP + SAC* |
|---|---:|---:|---:|---:|
| mAP@0.25 | 53.61 | 53.66 | 52.25 | **54.89** |
| mAP@0.50 | 45.57 | 43.74 | 42.28 | **46.42** |
| mAP@0.75 | 3.94 | **4.34** | 3.21 | 3.85 |
| mAP@0.50:0.95 | 15.33 | 15.28 | 13.34 | **15.48** |
| AR@100 | 20.50 | 20.43 | 18.51 | **20.60** |



### 2. Virus-patch detection

**149 images in the source dataset; 200-epoch experiments.** Mean validation scores across seeds **42, 43, and 44**:

| Validation metric | Baseline | MRFP + SAC | Change (pp) |
|---|---:|---:|---:|
| mAP@0.25 | 87.12 | **87.87** | +0.75 |
| mAP@0.50 | 78.84 | **80.47** | +1.63 |
| mAP@0.75 | 16.17 | **17.95** | +1.78 |
| mAP@0.50:0.95 | 31.84 | **32.67** | +0.83 |
| mAP small | **30.80** | 29.97 | −0.83 |
| AR@100 | 44.67 | **44.90** | +0.23 |

**Interpretation:** The combined variant improved mean AP@0.50 and AP@0.75 in these runs, while small-object AP decreased slightly. This is a limited validation result, not proof of statistical significance or a claim that the MRFP component alone caused the change. Full per-seed values and experiment provenance are in [the results log](docs/EXPERIMENTS.md).

## Run the project

### 1. Install

Python 3.10+ and a GPU are recommended for training.

```bash
git clone https://github.com/granth0ag/ViTDet-YOLO.git
cd ViTDet-YOLO
python -m pip install -r requirements.txt
```

### 2. Arrange data

The microscopy datasets are **not redistributed** in this repository. Obtain data and any required permissions from the original authors/sources. Put YOLO-format labels (class ID, normalized center x/y, width, height) in:

```text
data/
  foci/                  # virus-patch detection, one class: virus
    images/train/        labels/train/
    images/val/          labels/val/
  foci_cell/             # cell-patch detection, one class: cell_patch
    images/train/        labels/train/
    images/val/          labels/val/
```

Both tasks use 640×640 inputs by default. The legacy `foci` dataset key and corresponding checkpoint names remain unchanged. The separate `foci_cell` key makes the cell task explicit.

### 3. Train

```bash
# Virus-patch baseline
python train.py --dataset foci --aug A --seed 42 --epochs 200

# Virus-patch combined MRFP + SAC
python train.py --dataset foci --aug A --seed 42 --epochs 200 --sac-rates 1,3,5

# Cell-patch baseline
python train.py --dataset foci_cell --aug A --seed 42 --epochs 200

# Cell-patch combined MRFP + SAC
python train.py --dataset foci_cell --aug A --seed 42 --epochs 200 --sac-rates 1,3,5
```

**Compatibility note:** The existing `--sac-rates` flag and `_sac135` checkpoint suffix are retained for older runs. **In the current model builder, specifying `--sac-rates` activates both QKV-SAC and the MRFP-inspired neck**, not a standalone MRFP-only experiment.

### 4. Evaluate and visualize

```bash
python evaluate.py --dataset foci --aug A --seeds 42 43 44 --configs none 1,3,5 --split val --ckpt best
python inference.py --dataset foci --aug A --seed 42 --ckpt best path/to/image.png
```

These commands need locally trained checkpoints. The historical tables above include separately documented experiments and cannot be regenerated without their exact original checkpoints/configurations. No pretrained detector weights are included.

### 5. Run the cell detection ablation

Use non-viral cell images under `data/foci_cell`, with the class `cell_patch` and ID `0`.
The new experiment flags also work with `voc` and `wildlife`.

| Run | Addition to MRFP + QKV-SAC | Weights |
|---|---|---|
| E0 | Reference, with standard hard NMS | Existing `_sac135` |
| E1 | `--nwd-loss` adds a positive-anchor box loss of `0.5 × (1 − exp(−W₂ / 12.8))` to CIoU before the box gain | Train `_sac135_nwd` |
| E2 | `--aux-center-loss` adds a shared 1×1 center-offset head and a Smooth L1 loss with gain 0.25 | Train `_sac135_center` |
| E3 | `--sahi` resizes overlapping crops to model input size and merges them with full-image predictions using hard NMS | Reuse E0 |
| E4 | `--soft-nms` uses class-wise Gaussian score decay with sigma 0.5 | Reuse E0 |

Train E1 and E2 with the same settings as E0. Run the E0 command only if you need
to train the matching checkpoint. Change the seed in both training and evaluation
when running additional replicates.

```bash
python -m pip install -r requirements.txt
python train.py --dataset foci_cell --aug A --seed 42 --epochs 200 --sac-rates 1,3,5
python train.py --dataset foci_cell --aug A --seed 42 --epochs 200 --sac-rates 1,3,5 --nwd-loss
python train.py --dataset foci_cell --aug A --seed 42 --epochs 200 --sac-rates 1,3,5 --aux-center-loss

python evaluate.py --dataset foci_cell --aug A --seeds 42 --ckpt best --split val --ablations e0 e1 e2 e3 e4
```

The evaluator writes a Markdown comparison and a CSV with per-seed metrics under
`results_foci_cell_augA_val_ablation_e0-e1-e2-e3-e4.*`. All rows use the same
confidence threshold and detection cap (300 before COCO metric limits). Small-object
AP uses the full-image letterboxed coordinate system for every row. Reported
latency includes suppression and the additional SAHI image loading and crop passes.

To evaluate all four features together, train with both loss flags and append
`all` to the `--ablations` list. To use SAHI and Soft-NMS together with the E0
checkpoint, run evaluation with `--configs 1,3,5 --sahi --soft-nms` in place of
`--ablations ...`. The same inference flags work in the visualizer:

```bash
python inference.py --dataset foci_cell --aug A --seed 42 --sac-rates 1,3,5 --sahi --soft-nms path/to/cells.png
```

SAHI defaults to square crops half the original image's long side with 25% overlap.
Use `--tile-size`, `--tile-overlap`, and `--soft-nms-sigma` to override the inference settings.

## Repository layout

```text
src/
  model/
    vit.py         # plain ViT backbone with window/global attention
    vit_sac.py     # QKV-SAC backbone variant
    fpn.py         # Simple Feature Pyramid
    mrfp.py        # shared MRFP-inspired neck refinement
    detector.py    # YOLOv8 head integration
    build.py       # model assembly and checkpoint naming
  dataset.py       # data loading and augmentation
  utils/           # detection evaluation helpers
train.py           # training
evaluate.py        # seed-wise metrics and CSV summaries
inference.py       # bounding-box visualization
docs/              # architecture, result provenance, legacy benchmarks
```

## Limitations and scope

This is an **engineering and experimentation project**, not a peer-reviewed publication. The microscopy validation splits are small; the cell-patch comparison is not uniformly multi-seed; and the combined model contains both MRFP-inspired refinement and backbone SAC, so their individual contributions have **not** been isolated. No claims are made about clinical suitability or generalization to unseen microscopy laboratories.

## References and credits

- Li et al., [*Exploring Plain Vision Transformer Backbones for Object Detection*](https://arxiv.org/abs/2203.16527), ECCV 2022 (ViTDet).
- Xia et al., [*ViT-CoMer: Vision Transformer with Convolutional Multi-scale Feature Interaction for Dense Predictions*](https://openaccess.thecvf.com/content/CVPR2024/html/Xia_ViT-CoMer_Vision_Transformer_with_Convolutional_Multi-scale_Feature_Interaction_for_Dense_CVPR_2024_paper.html), CVPR 2024 (MRFP inspiration).
- Qiao et al., [*DetectoRS: Detecting Objects With Recursive Feature Pyramid and Switchable Atrous Convolution*](https://openaccess.thecvf.com/content/CVPR2021/html/Qiao_DetectoRS_Detecting_Objects_With_Recursive_Feature_Pyramid_and_Switchable_Atrous_CVPR_2021_paper.html), CVPR 2021.
- Singh et al., *Automatic Detection of Virus Infection Patterns in Foci Images Using Switchable Convolutions*, IEEE Access 2024.
- Singh and Mukherjee, [*Detection of Virus and Small Cell Patches in Foci Images Using Switchable Convolution and Feature Pyramid Networks*](https://arxiv.org/abs/2605.22290), 2026.
- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics), [timm](https://github.com/huggingface/pytorch-image-models), and [PyTorch](https://pytorch.org/) for the underlying components.

**Independent educational/research implementation.** Dataset and algorithm credit belongs to the original authors.
