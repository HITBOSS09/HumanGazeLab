# Visual-feature experiments

No incremental feature model was fit. Increments that were not estimated:

1. Center bias. Already reported as frozen B1. Not refit here.
2. Center bias plus basic image features. Not estimated.
3. Image features plus gaze history. Not estimated.
4. Observer-reference models. The frozen B2 and B3 models are observer-reference models. They were not combined with image features.
5. A combined model. Not estimated.

The reason is practical and scientific. The verified local stimuli are 18 COCO-FreeView images from the descriptive subset and 9 OSIE images from the overlay sample. Object masks and bounding boxes for those files are not in the repository. A pretrained embedding was not justified: DeepGaze III, DeepGaze3.5-VL, HAT, ScanDiff, and the ICCV 2025 dataset-bias model already use deep visual features, and this repository did not run those implementations. No hypothesis has been approved. A feature model on 18 images would not be a fair baseline against those systems.

## Pilot that was run

`scripts/run_stimulus_pilot.py` reads the extracted JPEGs and the released coordinates. Every COCO-FreeView file in the subset is 1680 by 1050. Every OSIE file in the overlay sample is 800 by 600. For each fixation it averages grayscale luminance and gradient magnitude in an 8-pixel radius. The comparison is the same number of random pixels in the same image, drawn with seed 20261114. Output: `outputs/protocolshift_b3_v1/stimulus_pilot_contrast.csv`.

| Dataset | Images | Fixations | Luminance at fixations | Luminance at random pixels | Gradient at fixations | Gradient at random pixels | Mean radius, fixations | Mean radius, random pixels |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| COCO-FreeView | 18 | 2,800 | 0.434 | 0.341 | 0.029 | 0.016 | 300 px | 526 px |
| OSIE | 9 | 1,197 | 0.440 | 0.429 | 0.046 | 0.025 | 171 px | 267 px |

Fixations are closer to the canvas center than the random pixels. The gradient difference is therefore not a center-bias-controlled effect. The pilot does not say that edges predict gaze beyond center bias, and it does not support a V1 or feature-integration claim.

## Deployable versus annotation-using models

A deployable image-only predictor uses the photograph and parameters fit on other images. B1, in the frozen experiment, is that kind of model only in the weak sense of a center prior; it does not use the photograph's content. B2, B3, and B5c use gaze annotations on the evaluation image. B4 uses the target observer's previous fixation. The pilot is neither: it describes pixels at human fixations and does not output a probability for a new fixation.
