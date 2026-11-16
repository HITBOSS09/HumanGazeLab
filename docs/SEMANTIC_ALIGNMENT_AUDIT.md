# Semantic alignment audit

This file separates measurements from the decision to continue. Region-calibration numbers are not in this file. They are in `docs/SEMANTIC_RESULTS.md` only after a complete run.

## Provenance

The annotation file is the `data/attrs.mat` member of the NUS-VIP release `predicting-human-gaze-beyond-pixels`. The local copy is `data/raw/external/osie_attrs.mat`, gitignored.

| Item | Measured value |
| --- | --- |
| SHA-256 | `51e9747fc6364102080cd5bac8921a279fc3f9e2a022119ae691c4bee4e8b74e` |
| Size | 45,237,395 bytes |
| Format | MATLAB v7.3, read with h5py 3.16.0. `scipy.io.loadmat` refuses this file. |
| Images | 700, one annotation record each |
| Objects | 5,551 |
| Attribute names in the file | text, face, emotion, sound, smell, taste, touch, motion, operability, watchability, touched, gazed |
| Stimulus overlap | All 700 stems match `osie_stimuli_all`. No stimulus lacks an annotation. No annotation lacks a stimulus. |

The Matlab comment in `computeSemanticMaps.m` lists a different attribute order. The code path that builds the official semantic maps reads `attrNames` from the file. The adapter does the same and raises if the names differ from the tuple above.

Object count per image: minimum 2, median 7, maximum 27. Every mask is uint8 and binary after the orientation below. No empty mask was stored as an object. Feature values are not only 0 and 1. Face takes values 1, 2, and 3. Text takes 1 and 2. The binary union uses value > 0, which is the released Matlab rule. The raw scores remain in `annotation_manifest.csv`.

## Mask geometry

h5py reports each MATLAB mask as shape (800, 600). The oriented mask used everywhere below is the transpose, shape (600, 800), so row 0 is the top of the 800×600 JPEG. A flip up-down or left-right was compared on the same face masks and produced a much smaller fixation excess. The unflipped transpose is the orientation that puts face masks on faces.

On all 700 images, using fixations with index at least 4 (56,368 fixations; this count is not yet the paired-target filter):

| Quantity | Value |
| --- | --- |
| Mean fraction of pixels in any object | 0.3132 |
| Mean fixation hit rate on that union | 0.8121 |
| Object hits | 46,035 |
| Mean face-area fraction | 0.0314 |
| Face lift, images that contain a face | 12.970 |

Face lift is the mean, over images with a nonzero face mask, of (face hit rate) / (face area fraction). A lift near 1 would mean face fixations are no more common than the mask area. The measured lift is the reason alignment was accepted. The runner stops if this lift is not finite or is below 2.

Object masks overlap. The census counted 2,247 overlapping object pairs. About 3.6% of pixels lie in more than one object. Semantic labels overlap as well. A pixel is allowed to carry more than one label. Mean unlabeled fraction is 0.687. That complement is not a semantic class.

## Alignment cases

Cases were chosen from mask area, border contact, and label names only. No model score entered the sort. They are in `alignment_cases.csv`, and the overlays are `figures/alignment_*.png`.

The overlays that were inspected: the large text contour follows the ROAD CLOSED sign in image 1243; the largest object contour follows the waffle in image 1426; the highest-overlap contour follows the foreground people and book in image 1550; the smallest face is a 30-pixel region in image 1241, too small for the contour to be obvious at the exported scale, which is why the face-lift number is the quantitative check. Fixation dots on these overlays are the recorded locations, including the initial fixation. Scoring still excludes index 0 and indices below 4.

## Probability coordinates

The frozen model emits a log density on the 1167×875 grid. The finalizer subtracts the log-sum-exp, so the exponentiated map sums to one on that grid. Region mass needs the original canvas, because the masks are 600×800.

The projection assigns every fine pixel to exactly one original pixel by the center rule in the protocol, then renormalizes. Every original pixel receives at least one fine pixel. Counts are 1, 2, or 4. Dropping the fine pixels that are not the nearest neighbor of an integer coordinate would throw away about half of the grid, so that shortcut is not used.

For integer coordinates, the inverse bin of the frozen scaled pixel is that same integer. For continuous coordinates, nearest-integer labeling and the inverse bin of the scaled pixel can differ by one pixel. Observed labels use nearest integer on the original canvas. The 1-pixel border row is the sensitivity for that disagreement. It does not silently replace the label.

The paired evaluation set is the 56,357 rows of the frozen table, which is the 56,368 index-at-least-4 fixations minus the 11 scanpaths with no partner. Log density on each new forward is compared with `log_density_nats` in that table. The stop threshold is 1e-3 nats.

Stage A scored its 20 predeclared images and exited 0. Image 1001 differed by at most 1.335e-5 nats. The other 19 differed by 0. The full run then scored the remaining 680 images. Every one of those printed a maximum absolute log-density gap of 0.000e+00. The checkpoint reported `state_dict_missing` 0 and `state_dict_unexpected` 0. That check is reproduction of the frozen scalar log density. It is not the region-calibration result.

## Decision

Alignment is good enough to score region mass. The face masks sit on the faces people look at, the inspected contours sit on the objects, and the new forwards reproduce the frozen log densities inside the stop threshold on the images checked so far. This is not a claim that the semantic probabilities are calibrated.
