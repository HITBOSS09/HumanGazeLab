# Overlap with published work

This note records what the sources already contain. It does not claim a new method. The journal gate stays **NO GO**. Whether a small event-level calibrator is worth fitting is a separate question and is answered only in `docs/SEMANTIC_NEXT_DECISION.md`, after the descriptive tables.

The measurements, once written, are computed. The interpretations are not a paper.

## What OSIE already measured

Xu, Jiang, Wang, Kankanhalli, and Zhao, Journal of Vision 14(1):28, 2014, DOI 10.1167/14.1.28, released the 700 images, the 15 observers, the 5,551 object masks, and the 12 semantic attributes. They report 86,768 fixations on labeled objects, 17.53% of those on objects with no semantic label, and more than a quarter of those on faces. Their pixel, object, and semantic maps were fit with a linear SVM on a 500/200 image split and compared with Itti and Koch, GBVS, GBVS plus a face detector, Image Signature, AIM, SUN, and the Judd MIT model. The paper's conclusion is that object-level and semantic-level information matters for where people look. Face attention and text as gaze targets are not open discoveries.

The attribute order printed in a comment in `computeSemanticMaps.m` is not the order stored in `attrNames`. That is an implementation fact about the file, not a scientific finding.

## Semantic occupancy, without a deep model

de Haas, Iakovidis, Schwarzkopf, and Gegenfurtner, PNAS 2019, "Individual differences in visual salience vary along semantic dimensions," used these OSIE attributes. They measured how much fixation time each observer spends on each attribute and found stable observer differences, including for faces and text. A later short-set paper (OSIEshort) repeated that occupancy question on fewer images. Those papers count fixations inside the masks. They do not score a DeepGaze density.

## Object features inside gaze models

Kümmerer, Wallis, Gatys, and Bethge, ICCV 2017, "Understanding Low- and High-Level Contributions to Fixation Prediction," compare DeepGaze II (a readout on VGG-19 object-recognition features) with an isotropic intensity-contrast model that uses the same readout. High-level features explain more of MIT300. The low-level model is better on 94 of 1,003 MIT1003 images. Their figures include cases where DeepGaze II predicts text or a face and the contrast model does not, and cases where that prediction does not match the fixations. DeepGaze II itself is Kümmerer, Wallis, and Bethge, arXiv:1610.01563.

Zabihi, Mansoori, and Yazdi, Journal of Visual Communication and Image Representation 73:102931, 2020, DOI 10.1016/j.jvcir.2020.102931, add segmented object identity, location, and appearance to a deep gaze model.

## Calibration of a spatial density

Linardos et al., ICCV 2021, arXiv:2105.12441, "Calibrated prediction in and out-of-domain for state-of-the-art saliency modeling" (DeepGaze IIE), already test whether a predicted fixation density is overconfident. They sort pixels by probability, cut the map into bins of equal probability mass, and compare the fixation counts. Individual backbones were overconfident out of domain. The ensemble was closer to uniform counts. That is calibration of the spatial density. It is the comparison this protocol explicitly does not treat as the same question as a binary semantic-region event.

Kümmerer, Wallis, and Bethge, ECCV 2018, "Saliency Benchmarking Made Easy," define a saliency model as a probability density over fixations and derive the map that a given metric wants. Kümmerer, Theis, and Bethge, 2015, "Information-theoretic model comparison unifies saliency metrics," put model comparison on likelihood. DeepGaze III (Kümmerer, Bethge, and Wallis, Journal of Vision, 2022) is a conditional density along a scanpath. Its reported metric is spatial likelihood, not a Brier score on an OSIE attribute mask.

## What this audit can add, and what it cannot

The narrow question left is whether the frozen DeepGaze III conditional density, summed inside these published masks, is a calibrated probability for the binary event "this fixation is inside the mask," after area and position controls. That question can be answered with a measurement. It does not become a new model of gaze, a new semantic feature, or a new fact about faces and text.

If the measurement agrees with the area and center-bias references, there is nothing further to fit. If it disagrees only before those controls, the disagreement is the spatial confound, which the density-calibration literature already treats as part of the probability map. If a disagreement remains, it is still an event-level residual on a dataset whose model performance has already been inspected. Fitting a scalar correction to that residual would not be a normalized spatial model, and it would not be a confirmation on held-out images this project has not scored.

No claim is made that this combination of region event, DeepGaze III, and OSIE masks was absent from every paper. The search that was done did not find that exact comparison. Absence from that search is not a discovery.
