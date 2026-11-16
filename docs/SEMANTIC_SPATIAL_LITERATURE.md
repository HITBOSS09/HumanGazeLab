# Literature comparison for the within-mask bins

This note says what the published record already contains. It does not claim a new method, and it does not use the bin counts to argue for one. The counts, once the run manifest says `"complete": true`, are in `docs/SEMANTIC_SPATIAL_RESULTS.md`. The letter is in `docs/SEMANTIC_SPATIAL_DECISION.md`. The journal gate stays **NO GO**.

The measurements are computed from the frozen DeepGaze III checkpoint and the released OSIE masks. The interpretations are marked as such.

## Equal-mass spatial calibration already exists

Linardos, Kümmerer, Wallis, and Bethge, ICCV 2021, arXiv:2105.12441, "Calibrated prediction in and out-of-domain for state-of-the-art saliency modeling," test whether a predicted fixation density is overconfident. They sort locations by predicted probability, cut the map into bins of equal probability mass, and compare the share of fixations in each bin with that mass. On DeepGaze IIE, individual backbones were overconfident out of domain. The ensemble was closer to equal counts. That is calibration of a spatial density over the whole image.

The check in this folder is that comparison with three restrictions that the 2021 paper is not: the density is DeepGaze III's scanpath-conditional map, the bins are computed only inside an OSIE face mask or an OSIE text mask, and the fixation has to have landed inside that mask. The conditional density is the map divided by the mask's own probability mass. A uniform histogram here would mean the model places mass well inside the mask. It would not mean the mask received the right total mass. That total is the separate measurement in `docs/SEMANTIC_RESULTS.md`.

Splitting exact ties by raster order was tried and dropped. The frozen finalizer upsamples with nearest neighbors, so many pixels share one density. Those pixels are one plateau. The recorded rule gives every pixel in a plateau the same fractional weights. That is a way to apply an equal-mass cut to a piecewise-constant map. It is not a new calibrator.

## Faces, text, and these masks are already measured

Xu, Jiang, Wang, Kankanhalli, and Zhao, Journal of Vision 14(1):28, 2014, DOI 10.1167/14.1.28, released the 700 images, the 15 observers, the 5,551 object masks, and the 12 attributes. They reported that object-level and semantic-level information matters for where people look, including faces. de Haas, Iakovidis, Schwarzkopf, and Gegenfurtner, PNAS 2019, measured stable observer differences in how much fixation time falls on these attributes, including faces and text. Those papers count fixations inside the masks. They do not score a DeepGaze density.

Kümmerer, Wallis, Gatys, and Bethge, ICCV 2017, showed that DeepGaze II's high-level features predict fixations on text and faces that an intensity-contrast model misses, and also showed cases where that prediction does not match the fixations. DeepGaze III (Kümmerer, Bethge, and Wallis, Journal of Vision, 2022, DOI 10.1167/jov.22.5.7) is a conditional density along a scanpath. Its reported metric is spatial likelihood, not a Brier score on an attribute mask and not an equal-mass histogram inside a mask.

## What a completed table can and cannot add

A completed histogram can say whether, on these 700 images, the frozen conditional density is uniformly used inside faces and inside text after the total mass has been divided out. These images were already scored for information gain and for region-mass calibration. The histogram is a description of that same set. It is not a confirmation on images this project has not inspected, and it is not a model.

No claim is made that this combination of DeepGaze III, OSIE masks, and within-mask equal-mass bins was absent from every paper. The sources read for the earlier audit did not contain that exact table. Absence from that reading is not a discovery.
