# Experiment protocol: temporal_v1

This protocol records the measurement rules used for `outputs/temporal_v1/`. It is not a hypothesis. See `docs/HYPOTHESIS_DECISION_WORKSHEET.md`.

## Question the measurements are arranged to inform

How does the information that predicts a fixation, using other observers on the same image or using the preceding fixation, change across successive fixations in natural-image free viewing?

That sentence is the project direction supplied for this repository. It is not an adopted hypothesis.

## Data

- Source: public COCO-FreeView train and validation JSON. SHA-256 `771167f4b5de5b9b58fb07c40d6e285d38b3b2ab9e4c6267df8a454a909da730`.
- The test-fixation URL used by the earlier audit returned HTTP 404. No test fixations are used.
- Canvas: 1680 by 1050 pixels, from the dataset readme.
- Released index 0 is excluded from primary targets. It is retained as the previous location when the target is index 1.
- A sensitivity scores index 0 itself. Those rows are labeled and are not the primary target set.
- `T` is not used. Fixation index is array position. It is not converted to elapsed time.

## Information available to each predictor

| Predictor | What it is allowed to use | What it is not |
| --- | --- | --- |
| B0 uniform | The canvas bin count | A model of looking. Geometric reference only. |
| B1 center | Post-initial fixations on train images, all train subjects, after the bandwidth is frozen | The evaluation image's fixations. Validation images. |
| B1 subject held out | Train images of the other subjects | The target subject's fixations. This is a sensitivity, not the primary center model. |
| B2 same-image leave-one-observer | Post-initial fixations of the other observers on the evaluation image. Bandwidth frozen from train images | The target observer's fixations. An image-only predictor. |
| B3 index-conditioned leave-one-observer | Other observers' fixations at the same released index on the evaluation image | The target observer. Other indices. |
| B4 history | A displacement kernel fit on train-image transitions, placed at the target observer's own previous fixation, including index 0 when the target is index 1 | Any fixation that has not yet occurred. Other observers on the evaluation image. |
| B5 center plus history | A mixture of B1 and B4. The weight is frozen from the train mixture fold | Validation labels. |
| B5c same-image plus history | A mixture of B2 and B4, with its own frozen weight | An image-only predictor. |
| B6 image shuffle | The B2 construction on a different image in the same split. The pairing is a seeded derangement | The correct image. |
| B6 order shuffle | The B4 kernel with a non-predecessor fixation from the same scanpath as the conditioning location | A meaningful control when the scanpath has length 2. Those rows are flagged and removed from the paired contrast. |

## Hyperparameters

Chosen before the validation scores were computed, on train images only:

- Bandwidth candidates, in pixels: 24, 40, 64, 96, 128.
- Uniform floor `alpha = 0.05`. Sensitivity values 0.01 and 0.10 are reported and were not used to pick the primary model.
- Mixture weights: 0, 0.25, 0.5, 0.75, 1.
- Train roles: `crc32(image name) % 5`. Bucket 0 selects bandwidths. Bucket 1 selects mixture weights. Buckets 2, 3, and 4 fit. The validation split in the JSON is held out of every role.
- Ties keep the earlier, smaller candidate.
- After selection, B1 and B4 are refit on all train images. The refit does not see validation images.

The selected center and history bandwidths were the smallest candidate. That boundary is recorded in `hyperparameter_selection.csv`. The candidate list was not extended after the selection scores were seen.

## Density and metric

- Equal-area bins, primary grid 42 by 26. Alternate grids 21 by 13 and 84 by 52 are a robustness check for B1 and B4 only.
- Counts are blurred with a Gaussian in bin units, using `mode="constant"`. Mass that the blur places outside the canvas is discarded and the canvas density is renormalized. A uniform floor is then mixed in.
- Information gain, in bits per fixation, is `log2(p_bin) - log2(1 / n_bins)`.
- NSS is the z-score of the bin probability across bins of that map, at the bin that contains the fixation. For B4 the map is the conditional density given the previous bin. This is not the MIT benchmark's 1-degree NSS.
- Aggregation: mean inside a scanpath, then unweighted mean across a subject's scanpaths, then unweighted mean across subjects. The bootstrap resamples the subject means (2,000 draws, seed 20261114, mixed per row by the same rule as the descriptive analysis).
- An image-cluster bootstrap is also stored. It answers a different question and does not replace the subject interval.
- A fixation-weighted subject mean is stored for four overall validation rows.

## Temporal groups, COCO-FreeView only

- Early: indices 1–3.
- Middle: indices 4–8.
- Late: indices 9 and later.
- Balanced subset: released length at least 12, so indices 9, 10, and 11 exist.
- Sample size at each index is reported because later indices are the scanpaths that lasted that long.

## OSIE

OSIE is a separate protocol. See `docs/EXTERNAL_VALIDATION.md` and `configs/osie_v1.yaml`. Its indices, viewing duration, and canvas are not pooled with this protocol.

## Failure atlas

Validation-image means are ranked by rules fixed in `configs/temporal_v1.yaml` before the cases were inspected: quartile intersections for high B2 with low B1, low B2 with high B1, the largest absolute B2-minus-B4 gap, and the largest absolute change in the B1 image mean when index 0 is added as a target under the frozen primary B1 map. Photographs are drawn only when the file is already in the local 18-image subset. Otherwise the panel is a coordinate plot. The selected validation images were not in that subset, so the committed atlas is coordinate-only.
