# Results inventory

Full-precision values are the CSV files. Numbers below are rounded to three decimals. Intervals are subject-level percentile bootstrap intervals unless noted. No hypothesis has been adopted.

## Measured values: COCO-FreeView validation

Source: `outputs/temporal_v1/information_gain_summary.csv`, subset `valid|all_primary_indices`, estimand `subject_scanpath_equal`.

10 subjects, 86,811 fixations, 6,001 scanpaths, 603 images. Targets are released index 1 and later.

| Model | Bits | Interval | Role of the model |
| --- | ---: | --- | --- |
| B0 uniform | 0.000 | — | Geometric reference |
| B1 center | 0.870 | 0.763 to 0.985 | Train images only |
| B1, subject held out | 0.851 | 0.743 to 0.957 | Sensitivity |
| B2 same-image leave-one-observer | 1.617 | 1.472 to 1.743 | Human consensus, not an image-only model |
| B3 same index | -0.314 | -0.476 to -0.169 | Human consensus at one index |
| B4 previous fixation | 1.473 | 1.176 to 1.762 | Train kernel, target observer's previous fixation |
| B5 center plus history | 1.594 | 1.358 to 1.850 | Train mixture, weight 0.75 on history |
| B5c same-image plus history | 1.976 | 1.802 to 2.144 | Human consensus plus history, weight 0.50 |
| B6 wrong image | -0.332 | -0.457 to -0.205 | Negative control |
| B6 other fixation as history | 0.173 | -0.032 to 0.421 | Negative control; interval includes values below 0 |

Paired subject differences on the same rows (`paired_contrasts.csv`):

| Contrast | Bits | Interval |
| --- | ---: | --- |
| B1 − B0 | 0.870 | 0.764 to 0.982 |
| B4 − B1 | 0.602 | 0.391 to 0.800 |
| B5 − B1 | 0.723 | 0.563 to 0.884 |
| B2 − B1 | 0.747 | 0.576 to 0.896 |
| B3 − B2 | -1.931 | -1.992 to -1.877 |
| B2 − B6 wrong image | 1.949 | 1.788 to 2.095 |
| B4 − B6 order, nondegenerate, 86,804 fixations | 1.301 | 1.084 to 1.477 |

Balanced validation scanpaths, length at least 12: 5,417 scanpaths, 16,251 early fixations (indices 1–3), 38,685 late fixations (index 9 and later).

| Early minus late | Bits | Interval |
| --- | ---: | --- |
| B1 | 0.245 | 0.120 to 0.372 |
| B2 | 0.740 | 0.585 to 0.901 |
| B4 | 0.108 | -0.126 to 0.330 |
| B5 | 0.012 | -0.170 to 0.197 |

The B4 and B5 early-minus-late intervals include values below zero. That is a statement about this resample.

Index curve, validation, subject-equal, selected points from the same summary file:

| Index | B1 | B2 | B3 | B4 | Validation scanpaths at this index |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.462 | 2.175 | 1.773 | 1.605 | 6,001 |
| 3 | 0.823 | 1.956 | 0.428 | 1.512 | 5,981 |
| 6 | 0.764 | 1.639 | -0.262 | 1.468 | 5,924 |
| 9 | 0.772 | 1.454 | -0.673 | 1.510 | 5,781 |
| 15 | 0.800 | 1.285 | -1.697 | 1.452 | 3,144 |

B3 is positive at index 1 and negative from index 5 onward in this table.

Subject means of B1 on validation range from 0.520 (subject 109) to 1.258 (subject 102). The descriptive radius table has subject 102 as the smallest mean radius and subject 109 as the largest. That pairing is a cross-check, not a separate result.

Image-cluster interval for B1 on the same fixations is 0.836 to 0.888, with 603 images. It is narrower because it resamples images, not the ten observers who recur on every image.

Index 0, validation, not a primary target:

| Scoring rule | Bits | Interval |
| --- | ---: | --- |
| B1 map fit without index 0 | 2.383 | 2.367 to 2.398 |
| B1 map refit with index 0 | 3.647 | 3.557 to 3.724 |
| B2 using other observers' index 0 only | 6.049 | 5.883 to 6.200 |

Grid sensitivity, frozen pixel bandwidths, validation, B1 / B4: 21 by 13 gives 0.859 / 1.408; 42 by 26 gives 0.870 / 1.473; 84 by 52 gives 0.873 / 1.493.

Floor sensitivity does not change the order of B1, B2, and B4. At alpha 0.01, 0.05, and 0.10, B1 is 0.880, 0.870, and 0.853.

Selected bandwidths from the train folds: center 24 px, history 24 px, leave-one-observer 40 px, center/history mixture 0.75, same-image/history mixture 0.50. The 24 px choices are the edge of the candidate list. All five scores are in `hyperparameter_selection.csv`.

Failure-atlas cases and the numbers that selected them are in `outputs/temporal_v1/failure_atlas_cases.csv`. None of those validation images is in the local 18-image photograph subset. The figure is a coordinate plot. The low-agreement/high-center rule returned gaps of about 0.13 to 0.20 bits. The high-agreement/low-center rule returned B2-minus-B1 gaps of about 2.22 to 2.57 bits.

NSS on the primary validation rows: B1 1.157 (1.036 to 1.292), B2 2.121 (1.984 to 2.250), B4 2.265 (1.812 to 2.778). These are bin z-scores, not benchmark NSS.

B1 spatial reliability on validation, highest predicted decile of bins: predicted share 0.360, observed fixation share 0.362. Lowest decile: predicted share 0.006, observed share 0.001.

## Measured values: OSIE held-out images

See `docs/EXTERNAL_VALIDATION.md`. Those numbers are not COCO-FreeView numbers.

## Implementation facts

- Code path: `src/humangazelab/temporal.py`, `temporal_run.py`, `failure_atlas.py`, `osie.py`, `osie_run.py`
- Commands: `scripts/run_temporal_v1.py`, `scripts/run_osie_v1.py`
- Tests: `tests/test_temporal.py`, plus the original suite. After the temporal and OSIE tests, `pytest` reported 34 passed. After the checkpoint density and crossed-bootstrap tests, `pytest` reported 37 passed.
- Checkpoint tables, which do not replace the frozen outputs: `outputs/audit_checkpoint/` and `outputs/temporal_v2/`. The recomputed subject-equal means match the published COCO-FreeView and OSIE tables. See `docs/TECHNICAL_EVIDENCE_REPORT.md`.
- The COCO information-gain summary file was unchanged by a rerun after the index-0 history guard was added. SHA-256 `5ed78da664cac0b2e63b11b8b73747ef62cee613f5c511c550b42d6fb91b32aa`.
- Leave-one-observer count identity was checked on 25 images during the run, and a unit test checks that editing the held-out observer's coordinates does not change that observer's reference density.
- Train center-count mass matched the number of train primary fixations.
- No validation image received a fitting role.

## Researcher-approved assumptions

None of the temporal cutoffs, bandwidths, or hypotheses has a recorded approval from the investigator. The worksheet is blank on purpose.

## Unverified interpretations

- That the early-minus-late difference for B2 is a change in agreement rather than a change in where surviving scanpaths look.
- That B4's flat COCO-FreeView curve means oculomotor history is constant. The interval for the early-minus-late contrast includes values on both sides of zero, and the bandwidth sat on the edge of the candidate list.
- That B3's negative late values are a property of gaze rather than the kernel and the probability floor. The checkpoint shows the sign change while nine other observers are still present, and it shows the same decline inside long scanpaths. It does not explain the growing distance between observers.
- That the OSIE and COCO-FreeView patterns are the same effect.
- Any sentence that names a neural mechanism, a causal effect, or a population beyond these participants.

## AI-assisted work

The pre-specified grids, candidate lists, and epoch bounds are listed in the protocol. No hypothesis was adopted.

## Investigator-original contributions

The project direction, the dataset, the requirement that evaluation targets not enter fitting, the requirement that fixations not be treated as independent subjects, and the decision to withhold a hypothesis from the agent are the investigator's. The agent did not attribute the code to the investigator.
