# Semantic-region measurements

The numbers below are computed from `outputs/semantic_calibration_v1/` after `run_manifest.json` reported `"complete": true`. They are exploratory. The same 700 images had already been scored for information gain. This is not a confirmation set.

The fixation-level rows are stored as one CSV per label in `region_events/`. The single concatenated file is 139 MB, which the host rejects above 100 MB, so that file is not in git. Concatenating the thirteen label files rebuilds it. The per-image resume cache in `events/` is also local only.

The measurements are from the frozen DeepGaze III checkpoint and the released OSIE masks and fixations. Interpretations are marked as such. Nothing here is a claim of a new method.

## Sample

| Quantity | Value |
| --- | --- |
| Images | 700, all annotated |
| Subjects | 15 |
| Eligible fixations | 56,357, the paired targets from `outputs/osie_deepgaze3_v2` |
| Objects | 5,551 |
| Event rows | 732,641, which is 56,357 fixations times 13 labels |
| Face masks present | 521 images, 42,286 fixations |
| Text masks present | 246 images, 21,984 fixations |
| Any annotated object | 700 images, 56,357 fixations |

Images with an empty label are omitted from that label. They are not coded as zeroes. Unlabeled background is not a class.

The full-stage process took 22,863 seconds on CPU, about 34 seconds an image, after resuming the 20 Stage A images. CUDA was unavailable. The largest printed log-density disagreement with the frozen table was 1.335e-5 nats, on image 1001. The other 699 images printed 0.000e+00.

## Primary region events

Positive difference means the fixation fell inside the mask more often than the predicted mass. These three labels were named before the scores. Intervals are percentile bootstraps of 1,000 draws, seed 20261115. They are not Bonferroni-adjusted. Fixations are not the resampling unit.

| Label | Mean predicted | Observed frequency | Difference | Ratio | Brier | Log loss |
| --- | --- | --- | --- | --- | --- | --- |
| object union | 0.7485 | 0.8167 | +0.0682 | 1.091 | 0.1368 | 0.4292 |
| face | 0.2116 | 0.2516 | +0.0400 | 1.189 | 0.1562 | 0.4848 |
| text | 0.1514 | 0.2026 | +0.0513 | 1.339 | 0.1288 | 0.4055 |

Image-equal mean of the per-image difference, with the image bootstrap interval:

| Label | Image-equal | Interval |
| --- | --- | --- |
| object union | 0.0672 | 0.0614 to 0.0733 |
| face | 0.0381 | 0.0289 to 0.0467 |
| text | 0.0492 | 0.0378 to 0.0617 |

Subject-equal intervals: object union 0.0560 to 0.0825, face 0.0240 to 0.0601, text 0.0362 to 0.0671. Crossed subject-and-image intervals: object union 0.0536 to 0.0882, face 0.0216 to 0.0648, text 0.0305 to 0.0684. All three primary intervals exclude zero under all three resampling schemes. The sign is the same: the model assigns less mass to the mask than the fixation rate.

Subject-level face gaps range from −0.002 to +0.124. One subject is near zero. Object-union subject gaps are all positive, from +0.030 to +0.109. Text subject gaps range from −0.023 to +0.103.

## Controls

These are explanatory comparisons. They are not a second gaze model.

| Label | Model gap | Center-bias gap | Area gap | Shift gap |
| --- | --- | --- | --- | --- |
| object union | +0.0682 | +0.3822 | +0.4999 | +0.0043 |
| face | +0.0400 | +0.1883 | +0.2098 | −0.0073 |
| text | +0.0513 | +0.1674 | +0.1784 | −0.0058 |

The center-bias template and the area fraction both under-assign these regions by much more than DeepGaze III. The model already carries most of the excess. The remaining gap is the difference between 0.04 and 0.19 for faces, not the difference between the model and chance.

Location-matched shifts kept at least 80% of each mask inside the canvas. No primary fixation was left unmatched (`n_shift_unmatched_fixations` is 0). On those translated masks the model's mass matches the hit rate. The image-equal difference between the semantic gap and the shift gap is 0.046 for faces (interval 0.039 to 0.053), 0.055 for text (0.046 to 0.064), and 0.063 for the object union (0.058 to 0.069). The excess is on the annotated mask, not on a same-sized mask translated by up to 80 pixels.

A five-fold linear residual of the image-mean gap on area fraction and centroid distance leaves the image-to-image spread almost unchanged: face standard deviation 0.105 to 0.101, text 0.094 to 0.095, object union 0.080 to 0.077. The mean residual is about zero because the fit has an intercept. That intercept absorbs the average gap. It is not evidence that area and center distance explain the gap.

Dilating the mask and keeping the same predicted mass makes the gap larger, not smaller. Face goes from +0.040 at the strict pixel to +0.046 at 1 pixel and +0.101 at 12 pixels. Text goes from +0.051 to +0.063 and +0.148. A one-pixel labeling disagreement is not the source of the positive gap.

The gap stays positive at early targets (index 4 to 6) and later targets (index 7 and above). Face is +0.047 early and +0.032 later. Text is +0.076 early and +0.027 later. Object union is +0.085 early and +0.049 later. The text gap is larger when the previous fixation is within the predeclared 100 pixels (+0.170 on 757 fixations) than when it is farther (+0.047 on 21,227). The face gap does not split that way (+0.037 near, +0.040 far). The near-text count is small relative to the far count, and those 757 fixations are not 757 images.

Sorted-name halves agree. Face gaps are +0.044 and +0.033. Text gaps are +0.051 and +0.048. Object-union gaps are +0.067 and +0.067. The nine previously scored images give an object-union gap of +0.069 on all nine and a face gap of +0.033 on the five that contain a face. Only one of those nine contains text, and its gap is −0.047. One image is not a counterexample to the 245-image text result.

The multi-label columns in `control_comparisons.csv` split fixations by how many semantic labels contain the fixation. That split conditions on the outcome. It is not a calibration contrast, and it is not used below.

## Exploratory labels

The other ten attributes were not the predeclared comparison. Emotion (−0.006) and smell (−0.003) have image intervals that include zero. Motion and sound are positive and their image intervals include zero. Sound is present on 61 images and smell on 93, below the predeclared coverage minimum of 100. Gazed, taste, touch, touched, operability, and watchability are positive and their image intervals exclude zero. That is a description of the table. It is not twelve findings, and it was not adjusted for the number of labels.

## Failure atlas

The eight images were the two lowest and two highest image-mean gaps for face and for text, among images with at least 20 fixations. The rule was fixed before the pictures were opened. Overlays and mean conditional maps are in `figures/failure_density_*.png`.

The masks sit on the named regions in all eight. These are not alignment failures.

| Image | Label | Predicted | Observed | What the picture shows |
| --- | --- | --- | --- | --- |
| 1602 | face | 0.495 | 0.108 | The head mask is on the speaker. Fixations sit on the open book. The density has a blob on the head and a blob on the book. |
| 1485 | face | 0.516 | 0.198 | The mask is on the face. Many fixations sit on the monitor and the wall fixtures. The density is concentrated on the head. |
| 1218 | face | 0.492 | 0.849 | Four face masks sit on the children. Fixations cluster on those faces. The density covers the faces and still leaves about half its mass elsewhere. |
| 1649 | face | 0.271 | 0.698 | Head masks sit on the four people. Fixations cover the faces and the book. The density splits between the heads and the book. |
| 1495 | text | 0.572 | 0.223 | The mask is the station sign. Fixations sit on the two people. The density is on the sign. |
| 1470 | text | 0.618 | 0.300 | The mask is the ice-warning sign. Fixations sit on the pedestrians. The density is on the sign. |
| 1632 | text | 0.210 | 0.544 | The mask is the wall text. Fixations hit that text and the people. The density puts more mass on the people and the flag than on the words. |
| 1287 | text | 0.088 | 0.475 | The mask follows the book page, with a small second region. Fixations hit the page and the face. The density is on the face. |

Each tail has a competing region in the same picture. The discrepancy is real for these images. It is also what a scene with two salient things looks like when one number has to summarize a whole mask. These eight rows are the tails of a ranking, not a random sample of errors.

## Event-level recalibration

The descriptive gaps above survived the shift control and were not removed by area or center distance, so the predeclared Stage D fit was run. It is an L2 logistic regression, penalty 1.0, five image-disjoint folds, seed 20261115. Training rows are other images. Out-of-fold Brier score and log loss:

| Label | Unmodified Brier | Intercept only | Platt | Platt plus covariates |
| --- | --- | --- | --- | --- |
| object union | 0.1368 | 0.1498 | 0.1303 | 0.1296 |
| face | 0.1562 | 0.1883 | 0.1540 | 0.1519 |
| text | 0.1288 | 0.1617 | 0.1250 | 0.1222 |

The image-equal Brier change against the unmodified probability has an interval below zero for Platt and for the covariate model, on all three labels. The intercept-only base rate is worse. The covariate model, which adds area, centroid distance, luminance contrast, fixation index, and previous-fixation distance, is a few thousandths below Platt. A paired test of that extra drop was not predeclared and is not reported as a finding.

The pooled row in `stage_d_comparisons.csv` stacks the three labels, so fixations are repeated. Its intercept-only Brier of 0.250 is the cost of ignoring which label is being scored. It is not a model of a person.

A lower region-event Brier score is not a higher scanpath log-likelihood. No spatial density was modified. No claim is made that gaze prediction improved.
