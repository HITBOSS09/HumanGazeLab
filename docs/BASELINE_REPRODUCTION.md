# Baseline reproduction

Parent commit `fd68b38e82626d50e6a6a9d4779d02f5056e0e1d`. This pass writes `outputs/baseline_v1/` and does not rewrite `outputs/temporal_v1/`, `outputs/osie_v1/`, `outputs/audit_checkpoint/`, `outputs/temporal_v2/`, `outputs/protocolshift_b3_v1/`, `outputs/protocolshift_prep_v1/`, `outputs/figures/`, or `outputs/tables/`.

`pytest` collected 47 tests and reported 47 passed: 15 core, 19 temporal, 3 protocol-shift, 3 preparation, 4 baseline, 2 real-data, 1 zip-remote.

No new network was trained. The reserved MIT1003 final partition was not opened.

## Geometry

These statements are not one pixels-per-degree value.

| Dataset | What the source states | What was done with it |
| --- | --- | --- |
| OSIE | Xu, Jiang, Wang, Kankanhalli, and Zhao, Journal of Vision, 2014. The released grid is 800 by 600. The paper states about 24 pixels per degree in that image grid, from stimuli scaled to the full height of a 1680 by 1050 display. Side padding was not reconstructed in this repository. | DeepGaze III was shown the nine local JPEGs rescaled by 35/24, to 1167 by 875, with PIL bilinear. Coordinates were scaled by the same width and height ratios. |
| COCO-FreeView | Chen et al., the dataset paper, does not state pixels per degree or viewing distance. The local readme gives 1680 by 1050 zero-padding and a 5-second trial. A later preprint states 54 by 35 degrees at 47 cm for both COCO-Search18 and COCO-FreeView. That is not the dataset paper, and it is not isotropic (1680/54 and 1050/35). | Not used as a model scale. DeepGaze III was not scored on COCO-FreeView. |
| MIT1003 | Three statements remain side by side. Judd et al., ICCV 2009: about two feet from a 19-inch 1280 by 1024 screen, longest side 1024, majority height 768. `checkFixations.m` header: 34 px/deg, with a comment that 1024 by 768 px equals 40.5 by 30 cm at 75 cm. DeepGaze README: 35 px/deg, long side mostly 1024. | No one of these was selected. Feature windows are 16 pixels on the JPEG. They are not degrees. |

On the 117 tuning JPEGs, 56 are 1024 by 768 and 14 are 768 by 1024. The rest are other heights at a long side of 1024. That matches the paper's "longest side 1024, majority height 768" description. It does not decide the pixels-per-degree conflict.

## DeepGaze III

Official code at commit `c7db17e2d1d7ea6468ffdee2cfaddf141095dcff`. Weights `deepgaze3.pth` from release v1.1.0, loaded with 0 missing and 0 unexpected keys. DenseNet-201 is the torchvision ImageNet-1K backbone loaded by the official class. The center-bias file is the MIT1003 1024 by 1024 log-density template from release v1.0.0, zoomed with nearest neighbors and renormalized by log-sum-exp, as in the README.

The forward used here matches `DeepGazeIII.forward` with a maximum absolute difference of 0.0 on the checked fixation. Backbone features are computed once per image. The mixture, the scanpath encoding, and the finalizer are the official modules, called again for each target.

History is the last four fixations strictly before the target, in the order `[-1, -2, -3, -4]`. Targets before index 4 are not scored. The README's dataset class uses that rule when missing fixations are not allowed. The scored value is the natural-log density at the rounded resized pixel. Information gain is that value minus the center-bias log density, divided by ln(2). It is not the uniform-bin information gain in the frozen tables.

The nine local OSIE JPEGs were already used in earlier inspected analyses. Two of them fall in this repository's hash holdout. The table is exploratory.

| Quantity | Value |
| --- | --- |
| Fixations scored | 657 |
| Images, subjects | 9, 15 |
| Fixation-mean information gain | 1.951 bits |
| Image-mean, image SD, SE of the image-mean | 1.930 bits, 0.557, 0.186 |
| Subject-mean, subject SD | 1.970 bits, 0.410 |
| Wrong-image control, image-mean and image SD | −1.983 bits, 0.868, on 535 paired targets |
| Runtime | 1125 seconds on CPU |

The wrong-image control keeps this image and this scanpath history, and reads the density at the same subject's fixation of the same index on the next image. Every one of the nine image-means on true targets is positive. Every one of the nine wrong-image image-means is negative.

Per image, true-target means in bits: 1001 train 1.530, 1009 holdout 1.143, 1116 train 2.666, 1138 train 1.750, 1189 train 1.290, 1275 holdout 2.100, 1521 train 2.689, 1650 train 2.270, 1660 train 1.934. The two holdout images are not a confirmation set.

COCO-FreeView was not scored. The blocker is the missing pixels-per-degree statement in the dataset paper. A long-side-1024 resize would be an input convention, not a matched comparison.

MIT1003 was not scored with DeepGaze III. The README says every DeepGaze model before MSDB was trained on MIT1003. A local final split of those images is still inside that training set.

## DeepGaze MSDB

Not executed. The v1.2.0 file `deepgazemsdb.pth` is 114,331 bytes. The official constructor loads CLIP ResNet-50x64 and DINOv2 ViT-B/14 before that head is used. CLIP is not installed. Stub modules were inserted only so the package could import DeepGaze III. No MSDB log density was computed.

The README names the built-in slots MIT1003, CAT2000, COCO-FreeView, DAEMONS, and FIGRIM. OSIE is not one of those names. COCO-FreeView is. An MSDB score on COCO-FreeView would be an in-training evaluation even if the weights had been loaded.

## Frozen scores that are not this metric

These rows are copied from the frozen checksums. They were not recomputed. They use uniform-bin information gain. They are not comparable to the DeepGaze III center-bias information gain above.

| Score | Estimate | Subject interval | Fixations | What it is |
| --- | --- | --- | --- | --- |
| COCO-FreeView B3 | −0.31382255826016203 | −0.476 to −0.169 | 86,811 validation | Leave-one-observer density, uniform bins, sigma 40 px, index from 1 |
| OSIE B3 | 1.8482047247030444 | 1.705 to 1.991 | 20,685 held-out | Same estimator adapted to OSIE, sigma 24 px, index from 0 |

## MIT1003 fixation conversion

The converter calls `checkFixations.m`, SHA-256 `c3721a971a754a149109e751d4739ef0e989983961a82b1a70d0b91454545664`, under Octave 8.4.0. The m-file was not edited and is not in this repository. Octave statistics 1.6.3 does not implement `nanmedian` or `nanvar`, which the no-saccade branch calls. Two local functions of those names, median and variance of the non-NaN values, are placed on the Octave path. They are used only when that branch is reached.

An independent Octave session, before the wrapper existed, produced the first fixation of user `ajs` on three training images. The wrapper matches those three medians and durations with maximum absolute error 3.3e-9.

Some files store a third eye-position column whose values are 0, 1, 2, or 3, the labels the m-file writes, and a `fixationData` struct with `medianXY` and `duration`. The detector is given the first two columns. On 2,060 retried trials that contained a stored struct (users `tmj` 878, `ya` 746, `emb` 436), the fresh medians match the stored medians: 0 count mismatches, maximum absolute error about 5e-11 in x, y, and duration.

MATLAB variable names for seven stems are shorter than the filename. The reader uses the single non-header variable in those files. After that correction, and after accepting a third column, all 13,170 train and tune trials convert. The first pass had left 2,207 of them unread. None of the 125 final-partition stems were opened.

| Partition | Trials | Mean fixations | SD | Mean duration |
| --- | --- | --- | --- | --- |
| Train | 11,415 | 8.381 | 2.557 | 362.2 ms |
| Tune | 1,755 | 8.392 | 2.570 | 367.1 ms |

The readme describes negative samples as blinks. The m-file special-cases NaN, not negatives. Negatives were left in the official call. On 1,949 trials that both the official call and a negatives-as-NaN call could finish, the mean fixation count is 8.560 official and 7.862 with negatives set to NaN. The mean absolute difference is 0.707 fixations. That second call is a sensitivity. It is not the authors' procedure.

Of 14,728 tuning fixations, 13,874 fall inside the JPEG and 854 do not. Of those 854, 814 have a negative x and 800 have a negative y. One fixation has x past the JPEG width and 26 have y past the JPEG height. The positive coordinates sit in the JPEG frame. The outside medians are mostly blink samples that the official median did not remove. Feature means use the inside points only.

## Tuning-image features

117 tuning images, 15 observers, window radius 16 pixels, seed 20261114 for the radius-matched points. The center is `(width/2, height/2)` of the JPEG, not a reconstructed screen center. The antipode is the point opposite the fixation through that center, so it has the same radius. No object masks or boxes are in the local ALLSTIMULI release. No object descriptor was computed.

The table is the mean, across images, of the fixation value minus the control value. The image SD is the sample standard deviation of the 117 image means. The observer SD is the standard deviation of the 15 observer means. Standard errors of the image-mean use SD divided by sqrt(117).

| Feature | Radius-matched difference | Image SD (SE) | Antipode difference | Image SD | Observer SD, radius-matched |
| --- | --- | --- | --- | --- | --- |
| Luminance | −0.0220 | 0.0782 (0.0072) | −0.0306 | 0.1100 | 0.0103 |
| RMS contrast | 0.0218 | 0.0222 (0.0021) | 0.0236 | 0.0295 | 0.0053 |
| Gradient energy | 0.0075 | 0.0102 (0.0009) | 0.0082 | 0.0133 | 0.0013 |
| Orientation energy | 0.0072 | 0.0103 (0.0009) | 0.0079 | 0.0136 | 0.0012 |
| Orientation anisotropy | 0.0107 | 0.0151 (0.0014) | 0.0114 | 0.0194 | 0.0018 |

Luminance at the fixation is slightly lower than at the same radius. RMS contrast, gradient energy, and the orientation summaries are slightly higher. The antipode control has the same signs and similar sizes, so the contrast excess is not an accident of one random-angle draw. The image SD of the contrast excess is about as large as the mean. The 15 observers agree more tightly than the 117 images. The figure is `outputs/baseline_v1/figures/visual_feature_controls.png`.

## Leakage and licensing

| Item | Finding |
| --- | --- |
| DeepGaze III readout | README: trained on MIT1003. Scoring any MIT1003 split, including the local final split, is an in-training evaluation. It was not done. |
| DeepGaze III backbone | Torchvision DenseNet-201, ImageNet-1K. Overlap between the nine OSIE photographs and ImageNet was not tested. |
| DeepGaze MSDB | Named training slots include MIT1003 and COCO-FreeView. Not run. OSIE is not a named slot. |
| DeepGaze3.5-VL | Previously read, not run. The released training list includes MIT, CAT, COCO, Daemons, and Figrim. OSIE is not in that list. |
| Local MIT1003 final split | Not read. Untouched membership does not make the images unseen by a model trained on MIT1003. |
| OSIE nine JPEGs | Already used in earlier scored analyses in this repository. The DeepGaze III number is exploratory. |
| COCO-FreeView validation and the OSIE hash holdout | Closed in `configs/evaluation_boundaries.yaml`. Not reused as confirmation. |
| MIT1003 archive | Readme says the database is publicly available with the ICCV 2009 paper. No license file was in the archive. The m-file has an author comment and no SPDX identifier. It was not committed. |
| OSIE and COCO stimuli | Used under the terms recorded in the earlier dataset notes. Image files stay gitignored. |
| Weights | Downloaded for this run and not committed. |

The publication gate remains NO GO. The decision and the one recommended experiment are in `docs/RESEARCH_DECISION_MEMO.md`.
