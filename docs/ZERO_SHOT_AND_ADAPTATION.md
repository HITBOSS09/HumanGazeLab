# Zero-shot transfer and adaptation

COCO-FreeView validation and the OSIE holdout were scored in `temporal_v1` and `osie_v1` before this branch. Any new score on those same fixations is exploratory. A reshuffle of those outcomes is not an independent confirmation. No untouched image partition was created, because every COCO-FreeView image in the public file is already in a train role or in the inspected validation split, and every OSIE image is already in the inspected hash split.

## History kernel on the unit square

`outputs/protocolshift_b3_v1/history_transfer.csv` fits a previous-fixation kernel on the source dataset's train images and scores the target dataset's validation images. Coordinates are fractions of the released canvas. The bandwidth is 0.03 of canvas width. Target validation fixations are not used to fit the kernel. This is not a degree-of-visual-angle transfer, and it is not a transfer of B3. B3 needs other observers on the same image, so it is not a cross-dataset predictor.

Subject-equal information gain, in bits:

| Source train | Target validation | Same dataset | Estimate | Fixations |
| --- | --- | --- | ---: | ---: |
| COCO-FreeView | COCO-FreeView | yes | 1.420 | 86,811 |
| COCO-FreeView | OSIE | no | 1.361 | 18,465 |
| OSIE | COCO-FreeView | no | 1.389 | 86,811 |
| OSIE | OSIE | yes | 1.441 | 18,465 |

The two cross-dataset scores sit within about 0.08 bits of the two within-dataset scores. That is a description of this kernel and this split. It is not a test of whether a deep saliency model transfers, and the ICCV 2025 paper reports a large drop for spatial models under a different normalization. These four numbers do not contradict that paper.

## Few-shot adaptation

The ICCV 2025 model adapts fewer than 20 dataset-bias parameters, with a large part of the reported gain available from about 50 samples. That procedure was not implemented. Adapting it on the already inspected OSIE holdout would not be a confirmatory few-shot result.

## MIT1003

A header request on 14 November 2026 to `https://people.csail.mit.edu/tjudd/WherePeopleLook/DATA.zip` returned HTTP 200 and content type `application/zip`. The publisher's readme, retrieved in the earlier audit, says the archive contains raw samples and that fixations come from `checkFixations.m`. The zip was not downloaded. No Python fixation detector was written. No MIT1003 subject table exists in this repository. Within-dataset, zero-shot, and few-shot evaluations on MIT1003 were therefore not run.

## Negative controls already frozen

The frozen experiments still contain the wrong-image and non-predecessor controls. They were not refit. They remain the negative controls for the density definitions. They are not negative controls for a feature model, because no feature model was fit.
