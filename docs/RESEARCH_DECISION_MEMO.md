# Research decision memo

This memo does not adopt a hypothesis. It does not claim a new architecture, a novel result, or a journal quartile. No manuscript is drafted.

Parent of this pass: `fd68b38e82626d50e6a6a9d4779d02f5056e0e1d`. Measurements are in `docs/BASELINE_REPRODUCTION.md` and `outputs/baseline_v1/`.

## Decision

**NO GO.**

The measured DeepGaze III result and the measured contrast excess are consistent with methods that are already published. They do not identify a gap that a new model would be needed to fill. The reserved MIT1003 final partition stays sealed.

## What the measurements show

Official DeepGaze III, at the OSIE paper's 24 px/deg rescaled to the README's 35 px/deg training scale, assigns 1.930 bits of image-mean information gain over the MIT1003 center-bias template on the nine local OSIE images (image SD 0.557). A wrong-image target under the same density and the same history scores −1.983 bits (image SD 0.868). The model is using the photograph. That is what the published scanpath model is for. The number is exploratory: these nine images were already inspected, and two of them sit in the earlier hash holdout.

The frozen COCO-FreeView B3 of −0.314 bits and the frozen OSIE B3 of 1.848 bits stay as they are. They are uniform-bin scores. They are not the DeepGaze center-bias score, and they are not evidence against DeepGaze III.

On 117 MIT1003 tuning images, a 16-pixel RMS-contrast excess over a radius-matched control is 0.0218 (image SD 0.0222, standard error of the image-mean 0.0021). The antipode control is 0.0236. Luminance does not show a positive excess (−0.022). Observers agree on the sign of the small contrast mean more tightly than images do. Local contrast at fixations is an old saliency finding. No object masks were available, so no object-level claim is made.

The authors' `checkFixations.m` reproduces the stored `fixationData` medians on 2,060 trials to about 1e-11, with no count mismatches. The conversion itself is not an open scientific question.

## What is actually still unresolved

1. COCO-FreeView still has no pixels-per-degree statement in its dataset paper. DeepGaze III was not scored on it. Until that scale is taken from a primary source, a cross-dataset model comparison with OSIE is blocked. The old uniform-bin gap is not a substitute.

2. MIT1003 still has three pixels-per-degree statements: the 2009 paper's viewing geometry, the m-file's 34 px/deg, and the DeepGaze README's 35 px/deg. Feature windows in this pass are pixels.

3. The official detector keeps negative blink samples inside the fixation median. On the tuning set, 854 of 14,728 medians fall outside the JPEG, almost all of them negative. Setting negatives to NaN changes the fixation count by 0.71 on average in the paired subset. Whether that changes the contrast excess has not been measured.

4. DeepGaze MSDB and DeepGaze3.5-VL were not executed. MSDB's checkpoint is the head only, and CLIP ResNet-50x64 was not loaded. Their papers already cover dataset-scale parameters and, for 3.5-VL, observer identity. Failing to run them is a resource blocker, not evidence that those methods leave a gap.

5. The nine-image DeepGaze III figure has not been repeated on OSIE images that this repository has not already scored. ImageNet overlap of those nine photographs was not tested. The DenseNet backbone is an ImageNet-1K network.

These five items do not, by themselves, justify a new architecture.

## Candidate questions

Neither question is adopted. Both are falsifiable on data that can be held out from the final MIT1003 partition.

1. On OSIE images that do not include the nine already scored here, does unmodified DeepGaze III, rescaled by 35/24 and scored against the same MIT1003 center-bias template, keep a same-image minus wrong-image information-gain difference whose image-level interval excludes zero? If the interval includes zero, the 1.93 bit exploratory figure does not describe OSIE beyond those nine photographs. If the difference stays large, the published model already accounts for image-specific prediction on OSIE, and that result is not a reason to train a new network.

2. On the MIT1003 tuning images only, if negative samples are set to NaN before `checkFixations.m`, does the image-mean RMS-contrast excess over the radius-matched control change by more than the standard error of that change? If it does not, blink handling is not what produces the contrast excess reported here. If it does, that excess is sensitive to a preprocessing choice the authors' m-file does not make. The final partition is not part of this question.

## Recommended experiment

Run candidate question 1, and only that, before any new model.

Use the unmodified DeepGaze III checkpoint and the same forward that matched `DeepGazeIII.forward` here. Exclude the nine image names already scored. Keep the wrong-image control and report the image-mean difference with the image-level standard error. Do not score MIT1003. Do not score COCO-FreeView until a primary source states its pixels per degree. Do not train a readout, a dataset slot, or a new backbone. A large difference confirms an existing model. A difference whose interval includes zero falsifies treating the present 1.93 bit figure as a property of OSIE. Either outcome leaves the publication gate at NO GO until the investigator decides there is a separate question.

Candidate question 2 is a validity check on the tuning partition. It can be run with the files already on disk. It is not the experiment that would establish a distinct contribution.
