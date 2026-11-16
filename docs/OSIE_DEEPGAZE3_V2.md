# DeepGaze III on the OSIE release

This is a reproducibility and robustness check of the nine-image run in `outputs/baseline_v1/`. It is exploratory. Those nine outcomes, the 35/24 scale, the MIT1003 center-bias template, and the information-gain definition were already chosen. No weight was trained, and nothing was tuned on the new scores.

The journal publication gate stays NO GO. The separate question of whether to begin an object-contour study is in `docs/OBJECT_REGION_STUDY_DECISION.md`.

## What was frozen

| Item | Setting |
| --- | --- |
| Code | DeepGaze commit `c7db17e2d1d7ea6468ffdee2cfaddf141095dcff`, class `DeepGazeIII` |
| Head | Release v1.1.0, `deepgaze3.pth`, SHA-256 `20b6dead18ce18f0117aff7e48c68737d5226935455e3844ae7f22216fd1a422` |
| Center bias | Release v1.0.0, `centerbias_mit1003.npy`, SHA-256 `051645b7da47884c138e81f28a4afbadd99b6c3262eaf086434a9626e3578be9`, nearest zoom, then logsumexp |
| Geometry | OSIE paper: about 24 px/deg in the 800×600 image. README training scale: 35 px/deg. PIL bilinear resize to 1167×875. Coordinates use resized size times the coordinate, divided by 800 or 600. |
| History | The four fixations immediately before the target, order `[-1, -2, -3, -4]`. Index 0 stays in the history and is not a target. Missing fixations are not allowed. |
| Metric | Natural-log density minus the center-bias log density, divided by ln(2), at the nearest resized pixel. |
| Images | All 700 JPEGs from the author-linked GitHub release. Every file is 800×600. SHA-256 values are in `outputs/osie_deepgaze3_v2/stimulus_manifest.csv`. |
| Fixations | `osie_fixations.mat`, SHA-256 `2aab9133a407e4eff3f57ab5799872b162dbc77ca462c0391762eb5e67deadb9` |
| Release zip | SHA-256 `3ea8d4285b41f615cfcd9fd8fa6c72ebc5530acf1e345e7d2720f389e422bad4` |

The nine JPEGs already scored in `outputs/baseline_v1/` are byte-identical to the copies in this release. The fast scanpath maps sample the same nearest-neighbor locations as `F.interpolate` on the official full-resolution maps. The checked density matched `DeepGazeIII.forward` with maximum absolute error 0.0. A half-pixel shortcut that multiplies by the precomputed ratio 35/24 was rejected because it moved one 1001 fixation by 0.13 bits.

The wrong-image control uses the next image in sorted-name order, for the same subject, whose scanpath is long enough to contain that index. A target with no such image is dropped from both columns. The rule was fixed in `configs/osie_deepgaze3_v2.yaml` before any score on an image outside the nine.

## Eligibility

Counts are in `outputs/osie_deepgaze3_v2/eligibility.json`. Index 0 is included in the below-4 count.

| Count | Value |
| --- | --- |
| Fixations in the mat file | 98,321 |
| Index below 4, not scored | 41,953 |
| Of which index 0 | 10,500 |
| Target outside the 800×600 canvas | 0 |
| No partner scanpath long enough | 11 |
| Partner coordinate outside the canvas | 0 |
| Scored paired targets | 56,357 |

## Separation from training data

The DeepGaze README says every model before MSDB was trained on MIT1003. OSIE is not named there. MIT1003 was not scored. COCO-FreeView was not scored. The DenseNet-201 backbone is the torchvision ImageNet-1K checkpoint. Overlap between these 700 photographs and ImageNet was not tested. That overlap is unverified.

The frozen uniform-bin B3 scores in `outputs/osie_v1/` and `outputs/protocolshift_b3_v1/` use a different probability reference. They are not compared with these center-bias bits.

## How the run was executed

```bash
PYTHONPATH=src TORCH_HOME=/tmp/torch .venv/bin/python scripts/run_osie_deepgaze3_v2.py
```

Backbone features are computed once per image. Histories are scored in batches of 16. Completed stems are appended to `completed_images.txt`, so a stopped run resumes without rescoring. This machine has no CUDA device, so GPU memory is not applicable. Peak CPU resident set is in `run_report.json`.

The primary table is the expansion subset: every scored image except the nine already in `outputs/baseline_v1/`. The nine are a reproduction check only. True information gain, model log density, and center-bias log density on those rows must stay within 0.001 bits of the frozen CSV. The old wrong-image column is not comparable, because that run did not require the same targets.

Image-equal and subject-equal means average fixations inside the group before the bootstrap (1,000 draws, seed 20261114). The crossed interval resamples subjects and images together and recomputes the subject-equal mean. It is a sensitivity check, not a mixed model.

## Measured summaries

The scoring code is commit `282b63f89d5537f253b13b4db96b31a80dea6f56`. Its parent is the nine-image baseline `f8eb165f0218b9151a4260c58e28455bff48595e`. The tables below were written by that scorer. The process then raised on a final print that asked for a timing field the report stores as `seconds_this_process`. The files already existed. The print in the commit that adds these tables uses that field. No density was recomputed for the correction.

## Measured summaries

Primary subset: 691 images and 55,700 fixations, excluding the nine. All three estimands for the paired difference exclude zero. Every one of the 691 image means is positive. The smallest is 1.021 bits and the largest is 7.892 bits. Unrounded values are in `outputs/osie_deepgaze3_v2/summary.csv`.

| Estimand | True image | Wrong image | Paired difference |
| --- | --- | --- | --- |
| Image-equal mean | 2.249 (SD 0.653, SE 0.025, interval 2.202 to 2.297) | −2.189 (SD 0.887, SE 0.034, interval −2.256 to −2.123) | 4.438 (SD 1.164, SE 0.044, interval 4.357 to 4.530) |
| Subject-equal mean | 2.276 (SD 0.247, SE 0.064, interval 2.160 to 2.396) | −2.183 (SD 0.051, SE 0.013, interval −2.208 to −2.158) | 4.459 (SD 0.258, SE 0.067, interval 4.315 to 4.571) |
| Subject-equal, images and subjects resampled | 2.255 (interval 2.121 to 2.384) | −2.182 (interval −2.265 to −2.107) | 4.437 (interval 4.269 to 4.595) |

The same three rows for all 700 images and 56,357 fixations are in the CSV. That pool still contains the nine already inspected images, so it is not a separate confirmation. Its image-equal paired difference is 4.431 bits, interval 4.348 to 4.520.

The nine-image true information gains, model log densities, and center-bias log densities match `outputs/baseline_v1/deepgaze3_osie_fixations.csv` on all 657 rows. The maximum absolute information-gain difference is 2.75e-6 bits. Head keys missing after the load: 0. Unexpected keys: 0.

No CUDA device was available, so there is no GPU-memory figure. Peak resident set was 7,052,464 KB. Histories were scored in batches of 16. The median recorded time was 30.1 seconds per image. Ten images recorded more than 120 seconds, up to 2,085 seconds, while the machine was paused; the other 690 average 30.6 seconds. The process wall time, including those pauses and the one image scored before this process, was 28,761 seconds. The per-image log is `image_seconds.csv`.

The figure is `outputs/osie_deepgaze3_v2/figures/paired_difference.png`. The right-hand bars are the image standard deviation, not the standard error.
