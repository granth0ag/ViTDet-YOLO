# Microscopy Experiment Record

This document separates **recorded empirical observations** from capabilities of the latest repository code. Scores below are percentages, not fractions. The results are preliminary evaluations on small validation sets; no statistically significant improvements are claimed.

## Datasets

| Task | Annotated images | Train | Validation | Class |
|---|---:|---:|---:|---|
| Virus-patch (foci) | 149 | 134 | 15 (14 valid images evaluated in report's baseline run) | virus |
| Cell-patch | 50 | 45 | 5 | cell_patch |

Both inputs were prepared for 640×640 detection. Dataset source and annotation credit: Amrita Singh and colleagues (see README references). A local YOLO-format copy is needed; original microscopy data are not redistributed here.

## Cell-patch detection

**Reported single-seed experiments:** baseline, SAC in QKV backbone, and SAC in YOLOv8 detection head, final training epoch 200; source: November 2026 experimental report, Tables 4–5 (sections 4.3–4.4). **Combined MRFP + SAC:** user-provided training log at **epoch 195** of a run scheduled for 200 epochs. Its score should not be misrepresented as the final epoch-200 evaluation.

| Metric | Baseline | QKV-SAC backbone | SAC in head | MRFP + SAC (epoch 195) |
|---|---:|---:|---:|---:|
| mAP@0.25 | 53.61 | 53.66 | 52.25 | **54.89** |
| mAP@0.50 | 45.57 | 43.74 | 42.28 | **46.42** |
| mAP@0.75 | 3.94 | **4.34** | 3.21 | 3.85 |
| mAP@0.50:0.95 | 15.33 | 15.28 | 13.34 | **15.48** |
| AR@100 | 20.50 | 20.43 | 18.51 | **20.60** |

Table 8 of the final report includes the combined run at epoch 195. The five-image validation split and the difference in evaluation epoch limit the comparison with the epoch-200 baseline.

## Virus-patch detection

**200 epochs; three seeds (42, 43, 44).** The enhanced run was initially labeled `SAC d=1,3,5` by the evaluation CSV, but its model builder couples QKV-SAC and MRFP-inspired neck refinement; it is therefore labeled **MRFP + SAC** here. Tables 6 and 7 of the November 2026 final report give the means and paired runs. Table 1 records the earlier 100-epoch single-run baseline.

| Metric | Baseline (mean) | MRFP + SAC (mean) | Difference, pp |
|---|---:|---:|---:|
| mAP@0.25 | 87.12 | 87.87 | +0.75 |
| mAP@0.50 | 78.84 | 80.47 | +1.63 |
| mAP@0.75 | 16.17 | 17.95 | +1.78 |
| mAP@0.50:0.95 | 31.84 | 32.67 | +0.83 |
| mAP small | 30.80 | 29.97 | −0.83 |
| AR@100 | 44.67 | 44.90 | +0.23 |

### Exact per-seed values available from the provided evaluation table

The following values appear in Table 7 of the final report; `mAP` denotes mAP@0.50:0.95. The displayed means above are rounded from these rows.

| Model | Seed | mAP@0.25 | mAP@0.50 | mAP@0.75 | mAP | mAP small | AR@100 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline | 42 | 86.93 | 78.75 | 15.02 | 31.56 | 30.67 | 43.82 |
| Baseline | 43 | 87.38 | 79.53 | 16.56 | 31.80 | 30.91 | 45.29 |
| Baseline | 44 | 87.06 | 78.24 | 16.94 | 32.17 | 30.83 | 44.91 |
| MRFP + SAC | 42 | 87.52 | 79.50 | 18.38 | 32.58 | 30.70 | 44.71 |
| MRFP + SAC | 43 | 88.85 | 81.49 | 17.80 | 32.95 | 29.18 | 45.27 |
| MRFP + SAC | 44 | 87.25 | 80.41 | 17.66 | 32.49 | 30.03 | 44.71 |

**Historical virus baseline from the PDF report (separate single run, best checkpoint):** mAP@0.25 86.81, mAP@0.50 78.55, mAP@0.75 13.85, mAP@0.50:0.95 30.41, and AR@100 43.94. These numbers use a different reported run and should not be interchanged with the above three-seed comparison.

## Evaluation / reproducibility notes

- Two microscopy tasks and runs were performed under different experiment conditions; **do not compare the absolute cell-patch and virus-patch scores as if they measured the same task**.
- The original report's cell-patch results are single-seed **last-epoch** measurements. The combined cell score is a **mid-training snapshot at epoch 195**. Final-epoch parity cannot be inferred.
- The virus comparison shows a three-seed mean, but the validation split is small. mAP small fell in the combined experiment; this trade-off is part of the record.
- Without an explicit `--mrfp` option, the repository's `--sac-rates` flag enables **both** QKV-SAC backbone and the original MRFP refinement, so the results **do not** establish that either component alone caused the gains.
- The older report and current repository may differ in exact settings/checkpoint provenance; preserve original logs and weights where possible.
- The NMS and max-detection settings influence crowded-object evaluation. Comparisons should be reproduced with identical versions and settings; no evaluation definitions were changed in this documentation cleanup.
- Do not claim medical deployment readiness, a statistically validated improvement, or a novel SAC algorithm from these results.
