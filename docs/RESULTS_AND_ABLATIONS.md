# Results and ablations

All numbers in this file come from `outputs/protocolshift_b3_v1/`. The frozen B3 checksum matched the published subject-equal means within 1.2e-16: COCO-FreeView −0.31382255826016203 and OSIE 1.8482047247030444. The figure is `outputs/protocolshift_b3_v1/figures/b3_control_comparison.png`.

These splits were inspected before the controls were designed. The controls are exploratory. Subject intervals are percentile bootstraps of the subject means, 2,000 draws, seed 20261114. They are not crossed intervals. The crossed intervals for the frozen models remain in `outputs/audit_checkpoint/`. No multiple-comparison correction is applied; the table is a sensitivity list, not a family of confirmatory tests. No official published model was run, so none of these scores is evidence of superiority over DeepGaze, HAT, ScanDiff, or the ICCV 2025 dataset-bias model.

## One control at a time

Subject-equal B3, bits per fixation.

| Control | COCO-FreeView | Subject interval | OSIE | Subject interval |
| --- | ---: | --- | ---: | --- |
| Frozen native | −0.314 | −0.476 to −0.168 | 1.848 | 1.703 to 1.994 |
| COCO includes index 0; OSIE drops index 0 | 0.134 | −0.016 to 0.264 | 1.647 | 1.509 to 1.808 |
| Indices 1 through 8 only | 0.276 | 0.020 to 0.513 | 1.849 | 1.723 to 1.969 |
| Sigma 16 px | −2.271 | −2.410 to −2.151 | 1.268 | 1.102 to 1.428 |
| Sigma 24 px | −1.390 | −1.556 to −1.234 | 1.848 | 1.700 to 2.000 |
| Sigma 40 px | −0.314 | −0.480 to −0.165 | 1.897 | 1.769 to 2.018 |
| Sigma 64 px | 0.442 | 0.304 to 0.559 | 1.593 | 1.497 to 1.689 |
| Floor 0.01 | −1.104 | −1.304 to −0.926 | 1.563 | 1.391 to 1.749 |
| Floor 0.10 | 0.040 | −0.109 to 0.168 | 1.956 | 1.826 to 2.090 |
| Unit square, bandwidth 24/800 of width | −0.453 | −0.608 to −0.317 | 1.721 | 1.569 to 1.868 |
| Unit square, indices 1–8, same bandwidth | 0.127 | −0.102 to 0.344 | 1.719 | 1.596 to 1.840 |
| Unit square, bandwidth 40/1680 of width, indices 1–8 | −0.455 | −0.679 to −0.246 | 1.305 | 1.178 to 1.441 |

OSIE's sign stays positive in every row. COCO-FreeView's sign changes when late indices are dropped, when the kernel is widened to 64 px, when index 0 is added, or when the floor is raised to 0.10. In the index-0 and floor-0.10 rows the COCO interval includes values below zero. Canvas-fraction coordinates do not remove the gap. Sharper kernels make the COCO score more negative. That last fact agrees with the train-fold curve in `outputs/temporal_v2/`, where bandwidths below 24 px were worse for the pooled leave-one-observer map. Those train-fold scores were not used to pick a new validation model.

## Observer count

Twenty seeded subsets of 10 OSIE file positions, plus the fixed subset of positions 0 through 9, are in `osie_observer_subsets.csv`. None of the seeded native scores is negative. Their mean is 1.422 bits, with range 1.342 to 1.575. On indices 1 through 8 the seeded scores stay between 1.295 and 1.549. Matching the count of observers does not turn OSIE into the COCO-FreeView result.

## Joint control and the residual

The joint setting uses indices 1 through 8, a unit-square bandwidth of 0.03 of canvas width, and, for OSIE, 10 observers.

- COCO-FreeView, all 10 observers: 0.127 bits, subject interval −0.103 to 0.338.
- OSIE, mean of 20 seeded 10-observer subsets: 1.207 bits, subset range 1.118 to 1.372.
- OSIE, file positions 0 through 9: 1.193 bits, subject interval 1.056 to 1.345, on 10,941 fixations.

The difference between the OSIE subset mean and the COCO point estimate is 1.080 bits. The OSIE subset range and the COCO subject interval do not overlap. Those two intervals answer different questions, so the non-overlap is a description rather than a hypothesis test. The 1.080 bit difference is what remains after the controls that were run. It is not a causal effect of viewing time, instructions, or display geometry. Display geometry in degrees was not in the control.

## What the decomposition separates

Sensitive enough to move the COCO-FreeView overall sign: the set of indices, the pixel bandwidth, the probability floor, and inclusion of index 0.

Not sufficient to remove the gap: dropping OSIE index 0, using 10 OSIE observers, and expressing location as a fraction of the canvas.

Still unseparated: degrees of visual angle, image content, the 5-second versus 3-second sessions, and any difference among people who are not in these files.

## History transfer

See `docs/ZERO_SHOT_AND_ADAPTATION.md`. Cross-dataset unit-square history scores were 1.361 and 1.389, against within-dataset scores of 1.420 and 1.441. This ablation is about B4, not B3.

## Stimulus pilot

See `docs/VISUAL_FEATURE_EXPERIMENTS.md`. No feature ablation was fit, so there is no increment to report.
