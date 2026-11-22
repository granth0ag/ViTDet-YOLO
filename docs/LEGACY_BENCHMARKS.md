# Earlier General Object Detection Experiments

These results are preserved from the project's November 2026 experimental report and original README. They are useful evidence of broader engineering experience but are **not the primary microscopy benchmarks**.

## Pascal VOC

20 classes, 1,000-image **held-out validation** set; checkpoint selection also used this set, so the reported numbers should not be treated as a separate test-set performance.

| Metric | Plain ViT | LocalViT-style FFN |
|---|---:|---:|
| mAP@0.50:0.95 | 53.43% | 54.20% |
| mAP@0.50 | 76.92% | 76.38% |
| mAP@0.75 | 58.75% | 59.79% |

## African Wildlife

Buffalo, elephant, rhino and zebra classes; 227-image **test set**.

| Metric | Plain ViT | LocalViT-style FFN |
|---|---:|---:|
| mAP@0.50 | 95.19% | 95.34% |
| mAP@0.50:0.95 | 71.72% | 71.66% |
| mAP@0.75 | 84.93% | 83.90% |
| AR@100 | 75.87% | 76.71% |

The LocalViT-style change did not consistently improve all metrics. These were single-run comparisons, not proof of a general advantage. The historical report describes a ViT-S/16 trained for 70 epochs (VOC) or 60 epochs (wildlife), with a local depthwise convolution in the transformer FFN.

### Earlier README visual material

The previous project README contained uploaded training-curve images and sample detections for these two datasets. They remain available in the original GitHub commit history. We avoid filling the main microscopy README with unrelated figures or claiming these images depict cell/virus patches.
