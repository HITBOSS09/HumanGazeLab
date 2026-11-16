# Decision after the semantic-region audit

Two decisions are recorded here. They are not the same decision.

The measurements are in `docs/SEMANTIC_RESULTS.md`. The overlap with published work is in `docs/SEMANTIC_NOVELTY_REVIEW.md`. No hypothesis was adopted. No manuscript is drafted.

## Testing a correction

**GO for the predeclared event-level calibrator. That test has been run. NO GO for a spatial correction.**

The descriptive gaps for the object union, faces, and text are positive. Their image, subject, and crossed intervals exclude zero. Linear adjustment for area and centroid distance does not remove the image-to-image spread. Translated masks of the same area are calibrated, and the annotated masks are not. That was the bar set in `docs/SEMANTIC_CALIBRATION_PROTOCOL.md` for running the small cross-fit model. It was run.

The result of the test is a Brier change of about 0.002 to 0.007 from Platt scaling of the region probability, and a further few thousandths from the non-semantic covariates. The unmodified probability already beats the base rate. The gain is a recalibration of a binary event. It is not evidence that a normalized change to the 1167×875 density would raise scanpath likelihood. A spatial correction is not proposed.

## Journal publication

**NO GO.**

The gap is a few percentage points on a dataset whose DeepGaze III performance had already been inspected. Face attention, text as a fixation target, and object-level salience were reported with these masks in 2014. Observer differences along the same attributes were reported by de Haas, Iakovidis, Schwarzkopf, and Gegenfurtner in 2019. Calibration of a spatial fixation density, by equal-mass bins, was reported for DeepGaze IIE by Linardos et al. in 2021. A scalar improvement in region-event Brier score does not add a gaze model to that literature, and it was not evaluated as one.

## What is still open

The integral over a mask can be low while the density inside the mask is well placed, or the integral can be right while the mass sits on the wrong part of the object. The eight tail images show both patterns: sometimes the density is on the mask and people look elsewhere, and sometimes people look at the mask and the density is on a competitor. Those pictures were chosen as tails. They do not say which pattern dominates.

## The next experiment

One experiment would resolve that. Restrict the DeepGaze IIE equal-mass bin check to the interior of the face mask, and separately to the text mask. Use the conditional density that is already defined, stream it, and do not keep the maps. If the bins inside the mask receive equal fixation counts, the gap is the total mass and not the placement inside the region. If the bins are skewed, the gap is where the mass sits inside the region. Do not fit another calibrator on the way to that count. Any later adjustment would need images that have not already been used to choose it. These 700 have been inspected, so they cannot be that confirmation set.
