# Final scientific verdict

Completed outputs were written at 12:39 PM IST on 16 November 2026. The run manifest reports `"complete": true` after 14 hours 27 minutes on CPU. No DeepGaze III forward was repeated for this audit. No calibrator was fit. The decision rule is the one in `docs/SEMANTIC_SPATIAL_PROTOCOL.md` and `configs/semantic_spatial_v1.yaml`.

The numbers below were recomputed from `fixation_bins.csv` and from the frozen region-mass tables.

## Verification

`validation_checks.csv` records 13 checks. All passed.

| Check | Result |
| --- | --- |
| Completed images | 700 unique ids, 700 event files, no duplicates, no empty files |
| Correspondence | The 700 ids are the scored images in `outputs/osie_deepgaze3_v2` and the images in the frozen region-mass events |
| Rows | 112,714 = 56,357 eligible fixations × face and text. No duplicate keys |
| Log density | Maximum absolute gap versus the frozen table is 0 nats on all 56,357 fixations |
| Region mass | Maximum absolute gap versus the frozen face and text `p_model` is 0 |
| Weights | On 15,095 in-mask fixations, fractional weights sum to 1 within 4.9×10⁻¹⁰ |
| Quarters | Predicted share of each bin is 0.25 within 2.7×10⁻¹⁴ on every in-mask row |
| Method | Every event file uses `obs0`–`obs3`. That exact quarter is the fractional plateau rule. A whole-pixel raster assignment does not land on 0.25 at this tolerance |
| Intervals | Image-equal highest-bin intervals were recomputed with 1,000 draws and seed 20261115 and match `uncertainty.csv` |
| Checkpoint | DeepGaze commit `c7db17e`, state-dict missing 0, unexpected 0. Face lift on the 700 images is 12.970 |

Twenty-one in-mask fixations land on a density value that crosses a quarter, so their count is split. The other in-mask fixations land on a value that sits inside one quarter, so their weight is 0 or 1. Tied pixels still share one weight: mean tied mass is 0.545 on faces and 0.538 on text. Raster order was not used.

Eight face masks and thirteen text masks had no in-mask fixation, so the histogram uses 513 face images and 233 text images. The total-mass rows still use every image on which the mask is present: 521 and 246.

## Total region-mass error

This is the frozen image-equal gap, observed frequency minus predicted mask mass. Positive means the mask received too little probability. Intervals are the stored 1,000-draw image bootstrap. They were not refit.

| Label | Images | Eligible fixations | Image-equal gap | Interval |
| --- | --- | --- | --- | --- |
| Face | 521 | 42,286 | +0.0381 | 0.0289 to 0.0467 |
| Text | 246 | 21,984 | +0.0492 | 0.0378 to 0.0617 |

The fixation-pooled differences, which weight busy images more, are +0.0400 for faces and +0.0513 for text. The letter uses the image-equal rows.

Both intervals lie entirely above zero. The absolute gaps are 3.8 and 4.9 percentage points of eligible fixations. Subject-equal and crossed intervals from the frozen audit also exclude zero. The size is a few points of rate. It is large relative to its interval.

## Within-mask spatial error

Conditional on the fixation landing inside the mask, four bins each carry a predicted share of 0.25. Bin 0 is the highest conditional density. The prespecified contrast is the image-equal mean of (observed share of bin 0 minus 0.25).

Image-equal observed shares:

| Label | Highest | Bin 2 | Bin 3 | Lowest | In-mask fixations | Images | Subjects |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Face | 0.150 | 0.227 | 0.282 | 0.341 | 10,640 | 513 | 15 |
| Text | 0.126 | 0.179 | 0.278 | 0.418 | 4,455 | 233 | 15 |

| Label | Estimand | Estimate | Interval |
| --- | --- | --- | --- |
| Face | Image-equal | −0.1003 | −0.1104 to −0.0895 |
| Face | Subject-equal | −0.1097 | −0.1185 to −0.0994 |
| Face | Crossed subject and image | −0.1072 | −0.1240 to −0.0912 |
| Text | Image-equal | −0.1239 | −0.1410 to −0.1064 |
| Text | Subject-equal | −0.1282 | −0.1410 to −0.1165 |
| Text | Crossed subject and image | −0.1261 | −0.1468 to −0.1042 |

Every interval lies entirely below zero. The highest quarter receives about 15% of in-mask face fixations and about 13% of in-mask text fixations, against 25% predicted. The deficit is 10.0 and 12.4 percentage points. The histogram rises from the highest bin to the lowest. That is a large redistribution of the conditional distribution, and it is detectable under every prespecified resampling.

All 15 subjects have a negative gap on both labels. Face subject gaps run from −0.136 to −0.061. Text subject gaps run from −0.185 to −0.094. At image level the sign is the majority, not a constant: 444 of 513 face images and 206 of 233 text images are negative. The image standard deviations are 0.123 and 0.142.

Prespecified sensitivities leave the gap in place. Face early and later targets are −0.109 and −0.110. Text early and later are −0.136 and −0.117. Dropping the one-pixel rim leaves −0.109 and −0.131. Masks of at least 100 pixels, and of at least 1,000 pixels, stay within 0.01 of the full-sample gap. Pearson correlations of the image gap with area, centroid distance, and within-mask concentration are 0.051, 0.091, and −0.006 for faces, and 0.061, 0.066, and −0.002 for text.

No eligible fixation lands in a face mask and a text mask at once, so the overlap contrast has no rows. It is not an estimate.

## Shifted-mask control

Location-matched shifts use the frozen rule: eight shifts, 80-pixel radius, at least 80% of the mask inside the canvas, seed 20261115. The histogram is over fixations that land in the shifted mask. The protocol says this control does not change the letter. Same-side exclusion means the imbalance is also present in translated regions.

| Label | Image-equal gap | Interval | Shift hits | Images |
| --- | --- | --- | --- | --- |
| Face | −0.0954 | −0.1034 to −0.0875 | 41,583 | 513 |
| Text | −0.1223 | −0.1362 to −0.1073 | 15,264 | 234 |

Both intervals lie entirely below zero, on the same side as the real masks, and the sizes match the real-mask gaps. The highest-bin deficit appears inside a same-sized mask moved by up to 80 pixels. The spatial error is a property of these peaked conditional maps inside a region. It is not identified as a property of the face outline or the text outline.

## Verdict under the frozen rule

An interval excludes zero when both ends lie on the same side of zero. Total-mass evidence is the frozen image-equal gap. Within-mask evidence is the new image-equal highest-bin gap.

- Face: both intervals exclude zero. **C. Both errors.**
- Text: both intervals exclude zero. **C. Both errors.**

The call is not inconclusive. The intervals are several standard errors from zero, they agree across image, subject, and crossed resampling, and every subject has the same sign. The shift control qualifies the spatial half of C: the within-mask error is real on the annotated mask, and the same error is real on the shifted mask.

## What this is, as a research result

**Empirical finding.** On these 700 images, frozen DeepGaze III assigns too little total probability to face masks and to text masks, by about 4 and 5 percentage points. Conditional on the fixation falling inside the mask, too much of the remaining mass sits on the highest-density quartile: that quartile is under-occupied by about 10 and 12 percentage points. The same under-occupation appears in location-matched shifted masks.

**Novelty.** Low. Equal-mass spatial calibration of a fixation density was published for DeepGaze IIE by Linardos, Kümmerer, Wallis, and Bethge, ICCV 2021, arXiv:2105.12441. Face and text occupancy on these masks was published by Xu et al., Journal of Vision, 2014, and observer differences by de Haas et al., PNAS, 2019. Restricting the equal-mass cut to a mask, on a scanpath-conditional DeepGaze III map, is a different table. The shift result places that table with the known over-peaked density, rather than with a new fact about faces or text. The comparison with the sources already read is in `docs/SEMANTIC_SPATIAL_LITERATURE.md`.

**Methodological contribution.** The fractional plateau rule is the assignment required by this checkpoint. The frozen finalizer upsamples with nearest neighbors, so exact ties hold about half the conditional mass. Giving those pixels one shared weight makes the quarter well defined. That is a correction of the binning, not a gaze model and not a calibrator.

**Negative-control validity.** The shifted-mask control was predeclared, uses the same generator as the region-mass audit, and was not selected after the counts. It behaves as a control should: it reproduces the spatial gap and therefore blocks an outline-specific reading. It does not cancel the total-mass result. The earlier audit already found shifted masks calibrated for total mass and annotated masks not calibrated.

**Limitations.** The 700 images had already been scored for information gain and for region mass, so this is an exploratory description, not a confirmation set. There are 15 subjects. The within-mask rows are the fixations that landed inside the mask, which is a selected subset. Coordinates are pixels on the 800×600 canvas, not degrees. Maps were streamed and discarded; the audit checks the stored scalars, not a second forward. The overlap contrast is empty. No held-out image set was scored.

**Next experiment.** On images this project has not scored, compute the whole-image equal-mass histogram and the within-mask histogram with the same fractional rule, and do not fit a correction on OSIE. If the within-mask deficit matches the whole-image deficit, the mask result is the published spatial overconfidence read through a window. A monotone map, if one is trained, has to be trained on other images before either histogram is used to judge it.

## Conclusions

1. **Face verdict: C.** Total region-mass error and within-mask spatial error.
2. **Text verdict: C.** Total region-mass error and within-mask spatial error.
3. **Shifted-control outcome:** The highest-bin deficit is present in location-matched shifted masks, with intervals entirely below zero and sizes close to the real-mask gaps. The spatial error is not specific to the annotated outline.
4. **Strongest defensible finding:** On this already inspected OSIE set, frozen DeepGaze III under-assigns face and text masks by about 4 and 5 percentage points, and, inside the mask, under-fills the highest density quarter by about 10 and 12 percentage points. Shifted masks show the same quarter deficit.
5. **Journal paper: NO GO.** The measurements are real under the prespecified intervals. They do not add a method, a held-out confirmation, or a semantic-outline result beyond the spatial overconfidence literature and the region-mass audit already recorded. This is not a Q1 case and it is not a journal submission.
6. **Next research direction:** Compare the whole-image equal-mass histogram with the within-mask histogram on images this project has not scored, using the fractional plateau rule and no calibrator fitted on OSIE.
