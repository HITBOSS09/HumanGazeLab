# Assumptions and decisions for the temporal experiment

The descriptive analysis has its own list in `docs/ASSUMPTIONS.md`. This file adds the decisions that the temporal experiment had to make. None of them has been marked as approved by the investigator. Items marked open are not resolved by the existence of a number.

## Measurement choices that were fixed before validation scores

1. Primary COCO-FreeView targets start at released index 1. Index 0 is history for index 1, and it is scored only in a labeled sensitivity.
2. The primary grid is 42 by 26 equal-area bins. The blur candidates are 24, 40, 64, 96, and 128 pixels. The floor is 0.05. Mixture candidates are 0, 0.25, 0.5, 0.75, and 1.
3. Train-image roles use `crc32(name) % 5`. Validation images are excluded.
4. COCO-FreeView groups are indices 1–3, 4–8, and 9 and later. The balanced subset is length at least 12.
5. OSIE held-out images are hash bucket 4. OSIE primary targets include released index 0, because the file does not label a forced fixation. History is undefined at that index.
6. OSIE early indices are 0–2 and late indices are 6 and later, inside OSIE only. Balanced length is at least 9.
7. The failure-atlas rules are the quartile rules in `configs/temporal_v1.yaml`.

## Open decisions

1. **Whether the bandwidth list should be extended downward.** The frozen primary fits still use the original list. A later train-fold curve, `outputs/temporal_v2/train_fold_bandwidth.csv`, adds 4, 8, 12, and 16 px and does not read validation scores. Center stays at 24 px. The pooled leave-one-observer score stays at 40 px on COCO-FreeView and 24 px on OSIE. History's train-fold maximum is 16 px. That value was not scored on held-out images, and 4 px is not distinguished from 8 px on these 40 px grids. Replacing the frozen primary model with 16 px would be a new confirmatory analysis. It was not done.
2. **Whether COCO-FreeView index 0 should remain outside the primary target set.** The readme says the sequences include the initial center fixation, and the descriptive audit found index 0 near the canvas center on most trials. The sensitivity shows that index 0 is much more predictable from a center model than later fixations are. The primary table still excludes it.
3. **Whether OSIE index 0 is a start-of-trial fixation.** Only about 20% of first OSIE fixations fall within 50 pixels of (400, 300). The paper describes a drift correction between trials. The file does not label the first sample as forced. The OSIE primary table keeps it, and history models skip it.
4. **The unit and filtering of OSIE durations.** The 2014 paper says fixations under 100 ms were removed. The parsed file has a minimum duration of 20. Durations were not used. The disagreement is unresolved.
5. **OSIE coordinate frame.** Every parsed coordinate lies in (0, 800) by (0, 600). Nine JPEGs from the authors' code repository are 800 by 600, and the released coordinates land on scene content when drawn with y increasing downward. The paper also says those stimuli were scaled onto a 1680 by 1050 monitor. That display mapping, including side padding, was not reconstructed.
6. **Subject identifiers in OSIE.** The file stores 15 subject slots per image and no original codes. The ids in the OSIE tables are positions 0–14.
7. **How to combine subject dependence and image dependence.** The frozen tables report a subject bootstrap and an image bootstrap separately. The checkpoint adds a crossed resample of both factors for the subject-equal mean (`outputs/audit_checkpoint/coco_uncertainty.csv` and `osie_uncertainty.csv`). On COCO-FreeView the crossed interval stays about as wide as the subject interval. It is not a mixed model, and the image interval is still not a substitute for the subject interval. Which interval a later report should lead with is unresolved.
8. **Whether any candidate hypothesis in the worksheet is the one to test.** None is adopted.

## Choices that were not taken

- Fixation durations were not treated as milliseconds in the temporal experiment.
- Test-set fixations were not used and were not imputed.
- A deep network was not trained.
- COCO-FreeView and OSIE fixation indices were not concatenated.
- The COCO-FreeView center density was not placed on the OSIE canvas. The frozen-bandwidth OSIE block refits densities on OSIE train images and reuses only the pixel bandwidths and mixture weights.
