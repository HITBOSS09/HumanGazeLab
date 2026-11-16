# Publication readiness audit

This note is a review of the repository as it stands. It is not a manuscript, and it does not assign a journal-acceptance probability.

## A. Novelty

DeepGaze III (Kümmerer, Bethge, and Wallis, Journal of Vision, April 2022) already predicts the next fixation from image features and scanpath history, scores conditional information gain against uniform and center-bias baselines, withholds the initial fixation from evaluation, and compares the contribution of scene content with the contribution of history. A 2022 Journal of Vision abstract, read here only as a search snippet, already reports that predictability decays over the scanpath and that the decay is carried by scene content while history stays roughly constant.

This repository does not fit that model and does not evaluate those datasets. It measures binned densities and leave-one-observer histograms on the public COCO-FreeView train/validation release, and it repeats the density protocol inside OSIE. The confirmed difference is the dataset plus the nonparametric baselines. That difference is a replication-style measurement, not a new account of fixation selection.

## B. Scientific contribution

The tables answer a descriptive question: on these releases, how many bits per fixation each declared information source carries, at each index, under a leakage rule that was tested. They do not test an investigator-approved hypothesis. The worksheet is unfilled. A descriptive measurement can still be useful. It is not yet a claim.

## C. Experimental validity

The leakage checks that were implemented do what they say: validation images do not enter the fit, the held-out observer is subtracted from the same-image count, bandwidths are chosen on train folds, and the order-shuffle and image-shuffle controls use the declared alternative information. The subject bootstrap matches the repeated-observer structure better than a fixation-level interval would.

The validity problems that remain are in `docs/LIMITATIONS.md`. The checkpoint reproduced the frozen means, tested the B3 density against an explicit leave-one-observer map, compared subject, image, and crossed intervals, and scored bandwidths below 24 px on training folds only. The history train-fold maximum moved to 16 px and was not scored on held-out images. The B3 decline remains inside scanpaths that reach index 15. Those checks support the implementation. They do not identify a mechanism.

## D. Generalization

OSIE was scored as a second dataset with its own split, its own canvas, and its own index definitions. Center bias, same-image agreement, and previous-fixation information are all above the uniform reference there, and the wrong-image control is below it. B3 does not keep the COCO-FreeView sign. The early-minus-late pattern for history does not match across the two protocols. Nine OSIE stimuli were overlaid in the 800 by 600 frame; the monitor scaling was not reconstructed. The OSIE numbers are dataset-adapted fits, not a zero-shot transfer of a COCO density. MIT1003 was not converted from raw samples. External validation remains partial: one extra dataset, no author split, no shared time base.

## E. Reproducibility

Another person with the public COCO-FreeView JSON can rerun `scripts/run_temporal_v1.py`. Another person with the OSIE `fixations.mat` can rerun `scripts/run_osie_v1.py`. Seeds, hashes, and the role rule are in the configs and in `docs/REPRODUCIBILITY.md`. The original descriptive outputs are still in `outputs/`. The MATLAB eye-sample archive for MIT1003 was intentionally not turned into fixations here, so that path is not reproducible from this repository because it was not run.

## F. Claims that are not justified

- A new computational model of gaze.
- A neural mechanism, including any V1 or feature-integration claim. Features were not computed.
- A population estimate beyond these participants.
- A causal effect of time, salience, or task.
- A leaderboard number on the COCO-FreeView test set.
- A statement that OSIE confirms the COCO-FreeView history result. The history early-minus-late intervals do not agree in whether they include zero, and the indices are not aligned.
- Publishability in a specific quartile of journals.

## G. Pathway

The honest label is an exploratory measurement report with a technically specified protocol and a partial external repeat. It is also a usable portfolio piece if it is described that way. It is not a completed replication of DeepGaze III, because that model was not run. It is not manuscript-ready.

## Three threats a reviewer would raise

1. The scientific question has already been attacked with a higher-capacity conditional model. A KDE decomposition needs a reason to exist beyond a change of dataset, and that reason is not yet a hypothesis the investigator has adopted.
2. Ten observers, shared across train and validation images. The crossed resample does not shrink the COCO-FreeView subject-equal interval. The tighter image intervals overstate precision if a reader treats them as the uncertainty that matters.
3. The late-index and index-conditioned results use one fixation per observer per index, and the frozen bandwidth sat on the edge of a pre-specified list. The decline remains inside long scanpaths, so it is not only dropout of short trials, and the OSIE sign for B3 still does not match COCO-FreeView. The train-fold history maximum of 16 px has no held-out score.

## Assessment for beginning a manuscript

**NO GO.**

The tables can support an internal exploratory report and an application portfolio that shows a leakage-controlled measurement. They cannot yet support a manuscript that claims a result beyond those measurements. The missing pieces are an investigator-written hypothesis, a bandwidth decision that is not stuck on the boundary of the candidate list or is explicitly bounded by it, a pre-registered contrast, and an external check whose time base and display geometry are actually aligned. No acceptance probability is stated.
