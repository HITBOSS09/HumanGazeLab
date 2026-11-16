# Within-region spatial calibration

This protocol was written before any within-mask bin count. The region-mass results in `outputs/semantic_calibration_v1/` stay frozen. The DeepGaze III checkpoint, the 35/24 scale, the four-fixation history, and the paired-target eligibility stay frozen. No calibrator is fit.

The question is where the predicted mass sits inside a face mask or a text mask, given that the fixation landed inside that mask. It is a different question from how much mass the mask received.

## Bins

The conditional density is the projected map on the mask, divided by the mask's original probability mass. That conditional mass sums to one. Distinct conditional densities are ordered from highest to lowest. Pixels that share an exact density are exchangeable, because the frozen finalizer upsamples with nearest neighbors and those pixels are one plateau. A plateau's mass is split across the quarter boundaries it covers, and every pixel in it receives the same fractional weights. A fixation on that plateau counts in each touched bin by that fraction. Raster order is not a ranking.

Four bins each receive one quarter of the conditional mass. The predicted share of the highest bin is that quarter, not a count of whole pixels.

An earlier draft broke exact ties by row and then column and assigned each pixel to one bin. A partial run, stopped before a decision, showed that about half the conditional mass sits on exact ties. That is the nearest-upsampling plateau, not a second density. The raster rule was replaced by the fractional rule above before any fractional bin count was used for the letter. The discarded partial files are not a result.

Masks with fewer than four pixels, or with region mass at or below 1e-8, are skipped and counted. A fixation outside the mask does not enter the histogram.

The primary contrast, named before the counts, is the image-equal mean of (share of in-mask fixations in the highest-density bin, minus that bin's predicted mass). The comparison with one quarter is reported beside it. Bin 0 is the highest-density bin.

## What has to match before the bins are interpreted

The stored face, text, and object-union region-mass rows must reproduce from `region_events/` within 1e-8. Each new log density must match `outputs/osie_deepgaze3_v2` within 1e-3 nats. Each recomputed region mass must match the stored `p_model` within 1e-4. The face-lift on the images being scored must stay at or above 2. A failure stops the run.

Full per-fixation maps were not cached. Only nine example maps exist, and those are means or a single example. This run rebuilds each map and discards it after the bin assignment.

## Sample and uncertainty

The pilot is every 20th sorted image name, plus image 1001. That set was fixed from the file names. The full run is every annotated image, and only a manifest with `"complete": true` is the measurement.

Intervals resample images, subjects, or both, with 1,000 draws and seed 20261115. Fixations inside a mask are not treated as independent people.

Sensitivity rows, also fixed here, are mask area of at least 100 and at least 1,000 pixels, fixations off the one-pixel rim, face/text overlap versus one label only, and target index 4–6 versus 7 and later. Shifted masks use the same 80-pixel, eight-shift, 80%-inside rule and the same per-image seed as the region-mass audit. The shift histogram uses fixations that land in the shifted mask, whether or not they land in the original mask. The shift result does not change the letter. If the shifted-mask highest-bin interval excludes zero on the same side as the real mask, the imbalance is also present in location-matched regions and is not read as a property of the face or text outline alone.

## Decision rule

For each of face and text, the stored image-equal region-mass interval is the total-mass evidence. The new image-equal highest-bin interval is the within-mask evidence. An interval excludes zero when both ends lie on the same side of zero.

- A: total-mass interval excludes zero, and the within-mask interval includes zero.
- B: the within-mask interval excludes zero, and the total-mass interval includes zero.
- C: both exclude zero.
- D: neither excludes zero.

If face and text receive the same letter, that letter is the overall call. If they differ, the call is the split, and the two letters are both reported. The journal gate stays NO GO unless a later note changes it. These 700 images were already inspected, so the call is exploratory.

## Command

```bash
PYTHONPATH=src TORCH_HOME=/tmp/torch SEMANTIC_SPATIAL_STAGE=pilot .venv/bin/python scripts/run_semantic_spatial_v1.py
PYTHONPATH=src TORCH_HOME=/tmp/torch SEMANTIC_SPATIAL_STAGE=all .venv/bin/python scripts/run_semantic_spatial_v1.py
```

The second command resumes from `completed_images.txt`. It needs the gitignored OSIE files and the DeepGaze checkout.
