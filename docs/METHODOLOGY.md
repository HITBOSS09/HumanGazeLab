# Methodology

Every number in `outputs/tables/metrics_summary.csv` is defined here. The configuration that sets the cutoffs is `configs/analysis.yaml`. Choices that are not forced by the file are listed again in [ASSUMPTIONS.md](ASSUMPTIONS.md).

## Canvas

Spatial calculations use a canvas of 1680 × 1050 pixels, the size stated in readme v1.0. The center is `(840, 525)`. All 665,005 released coordinates fall inside `[0, 1680] × [0, 1050]`. The 18 extracted stimuli are 1680 × 1050.

## Which fixations enter a summary

The readme says each sequence includes the initial center fixation and is ordered from first to last. Array position is treated as that order.

The primary rows exclude index 0. Rows whose metric name contains `_all_` include index 0. Both are computed. The primary tag follows `analysis.drop_initial_fixation` in the config, which is true in the committed config.

A scanpath with no defined value for a metric is omitted from that metric. The `n_scanpaths` column is the number that remain. Length-1 scanpaths (51 of them) have no post-initial fixation, so they are absent from the primary radius row (43,048 − 51 = 42,997).

## Weighting

Two estimands are reported.

**Subject.** Inside a subject, each scanpath has equal weight. Each subject id then has equal weight. The percentile bootstrap resamples subject ids with replacement (2,000 draws). The interval is the 2.5 and 97.5 percentiles of those draw means. `sd` is the sample standard deviation of the subject means (`ddof=1`). `sem` is that standard deviation divided by the square root of the number of subjects.

**Image.** The same construction with image name as the group.

These are not a mixed model. A subject estimand and an image estimand answer different weighting questions, and both are in the table.

`outputs/tables/descriptive_marginals.csv` is different again: each post-initial fixation, or each post-initial successive pair, has equal weight. Its mean radius is therefore not required to match `mean_radial_distance_post_px`.

The density maps give each subject equal weight after that subject's own fixations have been normalized to sum to 1. Longer scanpaths contribute more fixations to their subject's map. The SD panel is the sample standard deviation across the 10 subject maps.

## Radius and central share

Radius is Euclidean distance to `(840, 525)`.

The comparison for mean radius is the mean distance from the center to the center of every 1 px cell on the canvas. A Monte Carlo sample of 200,000 uniform points, seed `20261114`, is stored in `outputs/uniform_null.json` with its standard error. For this seed that sample mean sits about 1.07 px away from the grid mean; a larger sample (5,000,000 draws, a different seed) sat 0.02 px away. The table uses the grid.

The comparison for the share of fixations within radius `r` is `pi * r^2 / (1680 * 1050)`. The configured radii are 100, 200, and 400 px. Each is smaller than half the shorter side (525 px), so the disk is inside the canvas and is not clipped. The primary radius is 200 px. The other two are sensitivity rows.

## Successive distance

A successive distance is the Euclidean distance between fixation `i` and fixation `i+1` in the analyzed sequence. The scanpath value is the mean of those distances. The subject value is the mean of scanpath values.

## Within-scanpath shuffle

For a sequence with at least two points, the positions are permuted and the mean successive distance is recomputed. That is repeated `n_shuffles` times (20) and averaged. The reported contrast is shuffled mean minus released-order mean. Positions are unchanged; order changes.

One generator advances in `source_index` order. For each scanpath it draws the post-initial permutations first, then the full-sequence permutations. Sequences shorter than two points draw nothing. The seed is `20261114`.

The histogram `saccade_distance_real_vs_shuffled.png` is one permutation of each post-initial sequence at seed `20261115`. It is a picture of the pooled distances. It is not the subject-equal contrast.

## Early and late

**Ordinal.** On the analyzed sequence, the first `k` fixations are early and the remainder are late. Primary `k` is 3. Sensitivity values are 1, 2, and 5.

The radius contrast is the late mean radius minus the early mean radius.

The saccade contrast uses successive distances whose two endpoints are both early, and successive distances whose two endpoints are both late. The one saccade that crosses the split is in neither mean. A side of the split with fewer than two points has no saccade mean, and that scanpath is omitted from the saccade contrast.

**Temporal.** The onset of fixation `i` is the sum of released `T` values before it. Saccade time is not in the file, so this clock is fixation time only. A fixation is early when its onset is strictly below the threshold. Primary threshold is 1,500. Sensitivity thresholds are 1,000 and 2,000. The same boundary rule is used for saccades.

The temporal rows treat the released `T` numbers as milliseconds. The readme calls them durations and does not name the unit. See [ASSUMPTIONS.md](ASSUMPTIONS.md).

## Uncertainty seed

The bootstrap seed for a row is `bootstrap_seed` mixed with a checksum of the metric name, subset, and estimand. Adding a metric does not move the interval of another metric. The base seed is `20261114`. Each row records its seed.

## Subsets

Primary subject rows are computed on all scanpaths, on `split == train`, and on `split == valid`. Image rows and sensitivity rows use all scanpaths.

## Figures

Captions are in `outputs/figures/captions.txt`. Overlays place coordinates on a continuous canvas with extent `[0, width] × [0, height]`, y increasing downward. A pixel-center convention would move each mark by about half a pixel. Index 0 is drawn as an open red circle. Marker area increases with the released duration value.
