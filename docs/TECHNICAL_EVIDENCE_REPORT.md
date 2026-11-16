# Technical evidence report

Checkpoint on the frozen temporal experiment. This file records dataset provenance, the models that were implemented, the checks that were run, the numbers those checks produced, and what remains unverified. It does not adopt a hypothesis, does not claim a new finding, and does not assess journal readiness or causality. Application answers are not in this file.

The frozen outputs are `outputs/temporal_v1/` and `outputs/osie_v1/`. This checkpoint writes only `outputs/audit_checkpoint/` and `outputs/temporal_v2/`. The descriptive tables in `outputs/tables/` and `outputs/figures/` were not rewritten. `main` was not moved.

## Dataset provenance

COCO-FreeView public train/validation JSON, SHA-256 `771167f4b5de5b9b58fb07c40d6e285d38b3b2ab9e4c6267df8a454a909da730`, 43,048 scanpaths, 10 subject ids, canvas 1680 by 1050. Primary targets exclude released index 0. Validation: 86,811 fixations, 6,001 scanpaths, 603 images. The file is gitignored. The readme's headline fixation count and the 1,000-image benchmark test set are not this file.

OSIE `fixations.mat` from `NUS-VIP/predicting-human-gaze-beyond-pixels`, SHA-256 `2aab9133a407e4eff3f57ab5799872b162dbc77ca462c0391762eb5e67deadb9`, 700 images, 15 subject slots (file positions, not original participant codes), 10,500 scanpaths. Held-out images are `crc32(name) % 5 == 4` (148 images, 20,685 fixations). That is not the authors' 500/200 split. Primary targets include index 0. History is undefined at index 0. The two datasets were not pooled.

## Exact implemented models

Information gain is log2 of the ratio of the predicted bin probability to the uniform probability on the grid. COCO-FreeView uses 42 by 26 bins. OSIE uses 20 by 15. Each map is a Gaussian blur of counts, renormalized on the canvas, then mixed with weight 0.05 toward the uniform distribution.

- B0: uniform on the grid.
- B1: center density fit on train images. A subject-held-out variant removes one participant from that fit.
- B2: leave-one-observer density on the same image, pooling fixations from the primary index onward.
- B3: leave-one-observer density using only the other observers' fixation at the same released index.
- B4: displacement kernel from the target observer's previous fixation. Undefined when there is no previous fixation.
- B5: mixture of B1 and B4. Frozen COCO weight on history: 0.75.
- B5c: mixture of B2 and B4. Frozen COCO weight on history: 0.50. Frozen OSIE weight on history: 0.25.
- B6: wrong-image map, and a non-predecessor fixation from the same scanpath.

B2, B3, B5c, and the wrong-image control all use human gaze on an image. B5c also uses the target observer's own previous fixation. None of them is an image-only predictor. Bandwidths and mixture weights in the frozen runs were chosen on training folds. The shared leave-one-observer bandwidth is selected on the pooled map (B2) and then reused for B3.

## Validated methods

Before this checkpoint the feature branch was clean at `afd45d6d4da47f3a60f012842da2c987b9716b23`, with the temporal sources and tests already committed. `pytest` had reported 34 passed. After two B3 tests and one crossed-bootstrap test, `pytest` reported 37 passed.

The B3 tests check three properties on synthetic scanpaths. The probability equals an explicit density of the other observers at that index. Moving another observer's index-0 location does not change B3 at index 1 when the primary index starts at 1. A blurred shortcut matches the normalized map, including the floor. Those tests passed. They do not by themselves explain the dataset difference.

Recomputing the frozen fits reproduced the published subject-equal means. All seven compared metrics on both datasets matched with absolute difference below 1e-9 (`outputs/audit_checkpoint/coco_checksum.csv`, `osie_checksum.csv`). Recomputed B3 is −0.31382255826016203 on COCO-FreeView validation and 1.8482047247030444 on OSIE held-out images.

## Numerical results

COCO-FreeView validation, subject-equal, frozen model: B1 0.870 bits (subject interval 0.763 to 0.985), B2 1.617 (1.472 to 1.743), B3 −0.314 (−0.476 to −0.169), B4 1.473 (1.176 to 1.762), B5c 1.976 (1.802 to 2.144).

B5c minus B2, same fixations, paired by subject: 0.359 bits (0.241 to 0.514). The image-equal interval is 0.370 to 0.406. The crossed subject-and-image interval is 0.239 to 0.518. Both models use the same-image leave-one-observer density. B5c mixes that density with the target observer's previous fixation. On OSIE the same contrast is 0.077 bits (0.038 to 0.116); the crossed interval is 0.036 to 0.119. The OSIE mixture puts weight 0.25 on history, so the two contrasts are not the same comparison.

B3 by index on COCO-FreeView validation (`coco_b3_by_index.csv`): index 1 has median 9 other observers, median nearest observer 60.3 px, 13.2% of probabilities near the floor, fixation-mean 1.772 bits. Index 5 still has median 9 other observers, nearest 96.6 px, floor fraction 0.279, mean −0.131 bits. Index 15 has median 5 other observers, nearest 168 px, floor fraction 0.495, mean −1.665 bits. A target in a bin the kernel does not cover is scored at log2(0.05) = −4.322 bits. Empty references, which return the uniform probability, are rare before the far tail.

OSIE held-out B3 (`osie_b3_by_index.csv`): index 0 has 14 other observers, nearest 14.9 px, floor fraction 0.028, mean 3.455 bits. The mean stays positive through index 8 (0.542 bits, nearest 37.8 px) and is negative at index 9 (−0.086 bits, 1,039 fixations). The frozen OSIE table that keeps the COCO 40 px leave-one-observer bandwidth, with densities still refit on OSIE, has B3 at 1.897 bits. Bandwidth does not flip that sign.

Scanpath length (`coco_survival_contrasts.csv`, `coco_length_sensitivity.csv`): among the 3,144 validation scanpaths that reach index 15, index-1 B3 is 1.811 bits, close to 1.773 on all 6,001 scanpaths. Early indices 1–3 minus indices 9 and later, inside those long scanpaths: B3 2.410 (2.045 to 2.728), B2 0.800 (0.634 to 0.970), B4 0.151 (−0.070 to 0.383). The B4 interval includes values below zero. The overall B3 mean on the long subset is −0.457, more negative than −0.314, because those scanpaths contribute more late fixations. The OSIE contrast restricted to scanpaths that reach index 15 has 3 scanpaths and one subject. It is not used.

Subject, image, and crossed widths for the COCO-FreeView subject-equal mean are in `coco_uncertainty.csv`. Crossed widths stay near the subject widths (B1 0.230 versus 0.222; B3 0.342 versus 0.307) and are wider than the image widths. On OSIE the crossed B1 width is 0.168 against a subject width of 0.118. The crossed procedure resamples subjects and images, then recomputes the subject-equal mean. It is not a mixed model.

Train-fold bandwidths below 24 px (`outputs/temporal_v2/train_fold_bandwidth.csv`; validation was not read): COCO center maximum remains 24 px (0.8780 bits; 16 px is 0.8777). COCO pooled leave-one-observer maximum remains 40 px (1.673); 24 px is 1.374 and 16 px is 0.652. History's maximum is 16 px on both datasets (COCO 1.4729 versus 1.4714 at 24 px; OSIE 1.4724 versus 1.4514). Scores at 4 px and 8 px match for center and history on these 40 px grids. Mixture weights selected with that history bandwidth stay at the frozen values. `temporal_v2` does not replace `temporal_v1`. The 16 px history kernel has no held-out score.

Nine OSIE JPEGs (`1001.jpg`, `1009.jpg`, `1116.jpg`, `1138.jpg`, `1189.jpg`, `1275.jpg`, `1521.jpg`, `1650.jpg`, `1660.jpg`) are 800 by 600. All 1,197 fixations on them fall inside the file. Drawn with y increasing downward, the paths lie on scene content (`figures/osie_fixation_overlay.png`). This is the released image frame, not the paper's 1680 by 1050 display mapping.

## Limitations

Ten COCO-FreeView observers recur in both splits. The intervals describe those observers. Index is not clock time, and the 5-second and 3-second protocols were not aligned. OSIE durations disagree with the 2014 paper's 100 ms cutoff and were not used. The failure atlas has no photograph underlay. Image features were not computed. MIT1003 raw samples were not converted to fixations. DeepGaze III was not run; its published likelihoods are not these numbers. The overlap with that work is the conditional-density question and the already published report that predictability changes along a scanpath. Details are in `docs/LIMITATIONS.md` and `docs/LITERATURE_GAP_MATRIX.md`.

## AI-assistance disclosure

The investigator chose the dataset and required that evaluation targets stay out of fitting, that fixations not be treated as independent subjects, and that no hypothesis be adopted. The worksheet has no adopted hypothesis.

## What remains unverified

- Whether a 16 px history bandwidth would change any held-out number. It was not scored on held-out images, on purpose.
- The paper's scaling of OSIE stimuli onto the monitor, and alignment for the other 691 OSIE images.
- A separate license sentence for the OSIE images. The retrieved license is the MIT license for the software. The JPEGs were not committed.
- Original OSIE participant codes and the authors' 500/200 split.
- Why same-index fixations are farther apart at later COCO-FreeView indices. The distance was measured. A cause was not.
- Any comparison against DeepGaze III on the same images, and any contribution of scene features.
- MIT1003 fixations. They were not derived.
- A population interval, a causal effect, or a result beyond these measurements.
