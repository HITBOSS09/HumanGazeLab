# SemanticGaze-Cal protocol

This file records the rules frozen in `configs/semantic_calibration_v1.yaml` before any DeepGaze region probability was used to change them. The run is exploratory. The 700-image OSIE information gains had already been inspected, so a new split of the same images is an internal check, not an untouched confirmation.

The journal publication gate stays **NO GO**. This protocol does not adopt a hypothesis and does not train a network.

## What is being asked

For each eligible target fixation and each semantic label, the frozen DeepGaze III density supplies a probability that the fixation lands inside that label's mask. The observed outcome is whether it does. The question is whether those probabilities match the observed frequencies after area, center distance, and location-matched controls. A mismatch on this binary event is not, by itself, a mismatch of the full spatial density, and a better event probability would not by itself be a better scanpath model.

## Frozen model

The forward is the official DeepGaze III class at commit `c7db17e2d1d7ea6468ffdee2cfaddf141095dcff`, weights release v1.1.0, and the MIT1003 center-bias template from release v1.0.0. Images are PIL-bilinear resized by 35/24, from 800×600 to 1167×875. History is the four preceding fixations, most recent first. Target index 0 is the released initial fixation and is never a target. A target requires index at least 4. Eligibility is `paired_targets` from the 700-image run: the same rows as `outputs/osie_deepgaze3_v2/deepgaze3_osie_fixations.csv`. Eleven scanpaths had no partner under that rule and were already excluded there. This run does not rescore information gain and does not overwrite that directory or `outputs/baseline_v1`.

Each new map is checked against the frozen log density at the same fine-grid pixel. A maximum absolute gap above 1e-3 nats stops the image.

## Annotations

`data/raw/external/osie_attrs.mat` is the MATLAB v7.3 file from the NUS-VIP release. It is gitignored. The adapter reads `attrNames` from the file. The verified order is text, face, emotion, sound, smell, taste, touch, motion, operability, watchability, touched, gazed. The comment in `computeSemanticMaps.m` lists a different order and is not used. A value greater than zero marks the attribute present, which is the rule in that m-file. Raw values 1, 2, and 3 stay in the manifest. They are not treated as probabilities.

A label mask is the union of object masks with a positive value. Labels overlap. They are separate binary events and are not forced to sum to one. `object_union` is the union of every annotated object. Pixels outside that union are unlabeled background. The 2014 paper left out regions that were too small, too blurry, or treated as sky, ocean, ground, or wall. Background is not coded as a confirmed absence of objects.

Images with an empty label are dropped from that label's calibration rows. They are not scored as structural zeroes.

## Coordinates

Analysis uses the original 800×600 canvas.

- Masks are stored by h5py as (800, 600) and transposed to (600, 800), row 0 at the top of the JPEG.
- A fixation is labeled by nearest-integer rounding on that canvas, clipped to the image.
- The model density lives on the 1167×875 grid and sums to one there. Each fine pixel is added to the original pixel that contains its center, using `(i + 0.5) * source / resized - 0.5`, then nearest integer. The projected map is renormalized. Integer original coordinates land on the same pixel the frozen lookup uses. Continuous coordinates can disagree by one pixel. Border radii 0, 1, and 12 pixels are the predeclared sensitivity. Radius 12 is half of the paper's 24 pixels per degree. Dilating the mask changes the outcome, not the predicted mass.

The area reference is the mask area divided by 600×800. That is the probability a uniform draw on the annotation grid falls in the mask. It is not the projection of a uniform fine-grid density, because the inverse bins are not equal.

## What is stored per target

Full maps are not written for the 56,357 targets. Each map is reduced to, for every label: predicted mass, center-bias mass, area probability, the outcome, outcomes after 1-pixel and 12-pixel dilation, and the mean mass and mean outcome on up to eight translated copies of the mask. One example projected map is kept for image 1001. Mean projected maps are kept only for the failure-atlas images, after those images have been selected by the rule below.

Shifts are drawn inside an 80-pixel radius, excluding the zero shift, and kept only when at least 80% of the mask area stays inside the canvas. At most 40 candidates are tried. If fewer than eight qualify, the match is marked impossible and the qualifying shifts are still reported. The generator is seeded per image from seed 20261115, so a resumed image does not change another image's shifts. The previous-fixation split is a distance of 100 pixels, written into the config before the calibration tables were read.

## Summaries

For each label, on images where the mask is present:

- mean predicted probability, observed frequency, their difference, and their ratio when the mean prediction exceeds 1e-8
- event Brier score
- event log loss, with probabilities clipped only to [1e-6, 1−1e-6]
- a 10-bin reliability curve on pooled fixations, labeled descriptive
- the same difference for the center-bias mass, the area reference, and the location-matched shifts
- image-equal and subject-equal bootstrap intervals, 1000 draws, seed 20261115
- a subject-equal point with both subjects and images resampled

Fixations are not treated as independent people. The image-equal interval resamples images. The crossed interval resamples subjects and images.

Three labels were named before scoring: `object_union`, `face`, and `text`. The other ten attributes are exploratory. A label present in fewer than 100 images is marked below that predeclared coverage bar. Intervals are not Bonferroni-adjusted. The three primary intervals are reported as a family that was named in advance. The exploratory labels are a separate family.

An image-level linear residual of the image-mean gap on area fraction and centroid distance is fit with five image-disjoint folds. Training rows are the other folds. That residual is a diagnostic. It is not a gaze model and it is not Stage D.

## Failure atlas

Before any failure picture is chosen, the rule is: labels face and text; images with at least 20 eligible fixations and a present mask; rank by the image-mean of observed minus predicted; take the two lowest and the two highest; break ties by image name. A case is not called a verified error until its mask, density, and fixations have been looked at. The atlas is complete only when every requested image has been scored.

## What this run does not do

The scorer does not fit a correction. The Stage D specification was written into the config after the 20-image reproduction check and before the full-sample calibration tables were read. It is an L2 logistic regression, penalty 1.0, 25 Newton steps, five image-disjoint folds, seed 20261115, labels `object_union`, `face`, and `text`. The comparisons are the unmodified region probability, an intercept-only base rate, Platt scaling of that probability, and Platt scaling plus area fraction, centroid distance, luminance contrast, fixation index, and distance from the previous fixation. Training rows are other images. The fit is not executed unless the descriptive tables still show a gap after the area, center-distance, and shift controls. It does not score MIT1003 or COCO-FreeView. It does not change the frozen checkpoint, the 35/24 scale, or the center-bias template.

## Command

From the repository root, with the gitignored OSIE files and the DeepGaze checkout and weights already on the machine:

```bash
PYTHONPATH=src TORCH_HOME=/tmp/torch SEMANTIC_STAGE=A .venv/bin/python scripts/run_semantic_calibration_v1.py
PYTHONPATH=src TORCH_HOME=/tmp/torch .venv/bin/python scripts/run_semantic_calibration_v1.py
```

Stage A is the predeclared image subset in `stage_a_stems`: every 50th sorted stem, the geometry-only alignment cases, and image 1001. The second command resumes from `outputs/semantic_calibration_v1/completed_images.txt` and scores the remaining images. Only a `run_manifest.json` with `"complete": true` is the full measurement. A Stage A summary uses the same filenames and is not that measurement.

Outputs go to `outputs/semantic_calibration_v1/`. The versioned event table is one CSV per label in `region_events/`, because the concatenated file exceeds the 100 MB host limit. Dependencies for the adapter include `h5py==3.16.0` because `attrs.mat` is MATLAB v7.3. Torch is required for scoring and is not pinned in `requirements.txt`; this environment used torch 2.15.0+cpu and torchvision 0.29.2+cpu. Tests that do not need the weights are `.venv/bin/pytest`.
