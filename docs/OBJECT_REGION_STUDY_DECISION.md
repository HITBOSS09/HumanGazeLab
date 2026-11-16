# Decision on an object-region follow-on

This gate is separate from the journal publication gate. The journal gate stays **NO GO**. No manuscript is drafted, and no hypothesis is adopted.

The question considered here is whether to begin a study of where the frozen DeepGaze III density fails inside the object contours and semantic attributes released with OSIE.

## Decision

**NO GO.** Do not begin that study.

The expansion does not change the decision. On the 691 images that were not in the nine-image run, the image-equal mean of true-image minus wrong-image information gain is 4.438 bits, with an image bootstrap interval of 4.357 to 4.530. The crossed subject-and-image interval is 4.269 to 4.595. Both exclude zero. That is the published scanpath model assigning higher likelihood to the fixations on the photograph it was shown. It is not evidence of an unexamined object-contour failure, and it was not used to choose a region analysis. The contours and the semantic labels were published so that this question could be asked, and it was asked in the same paper. Later fixation models already compared object-recognition features with low-level features, and at least one later paper already put explicit object identity into a deep gaze model. A new pass over the same masks would repeat that literature. It would not be an original study. `attrs.mat` was not opened, and no object mask was scored. The measured expansion is in `docs/OSIE_DEEPGAZE3_V2.md`.

## What OSIE already contains

Xu, Jiang, Wang, Kankanhalli, and Zhao, Journal of Vision 14(1):28, 2014, DOI 10.1167/14.1.28, built the 700-image, 15-viewer set for object and semantic analysis. The paper reports 5,551 segmented objects with fine contours and 12 semantic attributes: face, emotion, touched, gazed, motion, sound, smell, taste, touch, text, watchability, and operability. Object-level attributes in their model are size, complexity, convexity, solidity, and eccentricity. Segmentation used graph cuts. Regions that are too small or too blurry to recognize were left out. Large regions such as sky, ocean, ground, and wall were treated as background and not extracted. Clustered objects of the same type were grouped. Ten paid annotators labeled disjoint subsets of 70 images each.

They report 86,768 fixations on the labeled objects. Of those, 17.53% fall on objects with no semantic label, and more than a quarter fall on faces. Their three-layer model (pixel, object, semantic) was fit on a 500/200 image split and compared with Itti and Koch, GBVS, GBVS plus a face detector, Image Signature, AIM, SUN, and the Judd et al. MIT model. The paper's own conclusion is that object- and semantic-level information matters for predicting where people look. The author page and the GitHub release distribute the images, the eye data, the labels, and the Matlab code with the paper. The repository license text is an MIT license for the software. That license is not, by itself, a claim that the photographs are MIT-licensed.

## What later work already did

Kümmerer, Wallis, Gatys, and Bethge, ICCV 2017, "Understanding Low- and High-Level Contributions to Fixation Prediction," compare DeepGaze II, which reads VGG-19 features trained for object recognition, with an isotropic intensity-contrast model that uses the same readout. They report that high-level features, including objects, faces, and text, explain more of MIT300, and that the low-level model is still better on 94 of 1,003 MIT1003 images. DeepGaze II is described in Kümmerer, Wallis, and Bethge, arXiv:1610.01563, as a readout on frozen object-recognition features plus an explicit center-bias prior.

Zabihi, Mansoori, and Yazdi, Journal of Visual Communication and Image Representation 73:102931, 2020, DOI 10.1016/j.jvcir.2020.102931, "Exploiting object features in deep gaze prediction models," add object identity, location, and appearance, from segmentation, to a deep gaze model. Their statement that state-of-the-art deep gaze models omit object information sits next to DeepGaze II, which already used object-recognition features. Either reading leaves the proposed OSIE-contour study occupied: the 2014 paper analyzed these masks directly, and the deep-feature papers analyzed object information in fixation models.

DeepGaze III, the model scored in this repository, is a scanpath mixture on DenseNet-201 features with a four-fixation history. The README trains the pre-MSDB models on MIT1003. It is not an object-contour model, and running it on OSIE photographs does not make a contour analysis new.

## What was not done

Object masks were not aligned to the DeepGaze III density. Failures were not counted inside faces, text, or background. No claim is made that DeepGaze III misses a particular attribute. That count is the study this memo declines to start.
