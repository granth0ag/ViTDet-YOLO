# Architecture

## Motivation and scope

This project adapts a **plain Vision Transformer** for microscopic object detection by combining a ViTDet-style feature extractor and a YOLOv8 anchor-free detection head. It is not an exact ViTDet reproduction.

```text
image (640×640)
   |
ViT-S/16, ImageNet pretrained
   | windowed attention (10×10) + global blocks (2,5,8,11)
   |
single map [B,384,40,40]
   |
SimpleFeaturePyramid
   |-- P3 [B,128,80,80] : deconvolution + conv/GN
   |-- P4 [B,128,40,40] : identity + conv/GN
   '-- P5 [B,128,20,20] : max-pooling + conv/GN
   |
[legacy combined variant] shared MRFP-inspired refinement on P3,P4,P5
   |
YOLOv8 Detect (box distribution + class scores)
   |
NMS -> bounding boxes
```

For the **historical combined MRFP + SAC** model, the backbone itself uses QKV-SAC, as well as the post-SFP refinement neck. This matters for interpreting results: the improvement cannot be attributed uniquely to MRFP.

## Module details

- **Baseline:** `ViTDetBackbone` followed by `SimpleFeaturePyramid` and `YoloHead`.
- **QKV-SAC backbone:** three depthwise 3×3 dilation branches at rates 1, 3, 5 with spatial softmax gating before query/key/value projections. Inspired by DetectoRS SAC but *not* the original weight-locked formulation.
- **MRFP-inspired neck:** channel reduction (1×1), independent standard 3×3 convolutions at dilation 1 and 3, spatial softmax mixing, channel expansion (1×1), and residual scaling. The **same** refinement-module weights process all levels of the SFP. Unlike ViT-CoMer, no parallel CNN hierarchy or convolution-transformer interaction block is implemented.
- **Detector:** Ultralytics YOLOv8 `Detect` with its existing decoupled classification/box branches and standard box/cls/DFL losses.

## Exact builder behavior

When `mrfp_variant` is omitted, `build_model(sac_rates=None)` builds the plain backbone and SFP. `build_model(sac_rates=(1,3,5))` builds **QKV-SAC backbone plus MRFP-inspired SFP refinement**. The historical CLI `--sac-rates` is retained as a compatibility selector, not a complete description of all enabled modules. The original SAC-only or SAC-in-head experiments from the PDF report are **documented historical studies** and are not all independently selectable from the current builder.

## Neck-only MRFP ablations

Explicit `mrfp_variant` / `--mrfp` selection controls the neck independently of
backbone SAC. The dense paper experiments use the plain ViT backbone:

| CLI variant | Neck refinement |
|---|---|
| `plain-conv` | Split-channel normal 3×3/5×5, dilation 1, concatenation |
| `plain-conv-sac` | Same kernel groups with normal dilation 1/3 branches and learned switches inside the neck |
| `single-conv` | One normal 3×3 over all reduced channels, dilation 1 |

Each spatial convolution uses `groups=1`, mixing the input channels within its
branch. The corresponding `plain-dw`, `plain-dw-sac` and `single-dw` flags select
the earlier depthwise variants. All three dense variants use the same shared pointwise projections, per-level BatchNorm and
activation order, with no residual/gamma in the refinement block. Spatial filters
are independent across levels. These new experiments have separate checkpoint
tags and must not be identified with the historical combined-model scores.
See [the MRFP experiment guide](MRFP_ABLATIONS.md) for commands
and the distinction between neck SAC and `--sac-rates` backbone SAC.

## Research boundaries

The feature pyramid is *multi-resolution* but generated from a **single stride-16 transformer representation**, without conventional FPN lateral connections. Adding local convolutions does not reconstruct spatial details absent from the backbone representation. The recorded historical comparisons do not isolate the contribution of the MRFP-inspired neck from backbone SAC. 
