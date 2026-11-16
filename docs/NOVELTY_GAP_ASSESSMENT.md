# Novelty-gap assessment

Decision for a protocol-aware temporal gaze model: **NO GO**.

This is not a publication decision in the other direction. `docs/PUBLICATION_GATE.md` remains NO GO. No manuscript was drafted. No journal quartile is claimed. No research hypothesis was adopted for the investigator. The proposal below is a candidate that would still need the investigator's own wording and an explicit approval before any model is trained or any final partition is scored.

Parent commit `7ca61a5947be0ac0599617d19d96d1ee2945e7bb` is unchanged. Frozen output directories were not rewritten.

## What the official implementations already do

Read in this pass: the DeepGaze README and `deepgaze_pytorch/deepgaze3.py` at https://github.com/matthias-k/DeepGaze; the DeepGaze3.5-VL README on branch `master` at https://github.com/Susmit-A/DeepGaze3.5-VL; the HAT README at https://github.com/cvlab-stonybrook/HAT; the ICCV 2025 open-access PDF of Modeling Saliency Dataset Bias; the arXiv HTML of DeepGaze3.5-VL; `checkFixations.m`; and the MIT1003 database readme. None of these models was executed.

| Method | Protocol | Time | Observers | What it does not do |
| --- | --- | --- | --- | --- |
| DeepGaze III, official class `DeepGazeIII` | The README says models before MSDB were trained at MIT1003 geometry, 35 px/deg, long side mostly 1024, and that other images may need to be rescaled by the caller. The class itself has no `pixel_per_dva` or dataset argument. | `included_fixations=[-1, -2, -3, -4]`: the prediction uses the last four history locations. Ten mixture components. A learnable finalizer starts at sigma 8. | No subject token. History is the same observer's previous fixations. | It is not a same-index map of other observers. It was not run on this repository's files. |
| DeepGaze MSDB, the official code for Modeling Saliency Dataset Bias | `pixel_per_dva` is a required argument. Named slots include `MIT1003`, `CAT2000`, `COCO_FREEVIEW`, `DAEMONS`, and `FIGRIM`. The README's adaptation path freezes the backbone and trains 13 dataset scalars, initialized from the average of the built-in datasets. The paper separates absolute px/deg scales from relative image scales, and reports that presentation time tracks a priority-scaling parameter with leftover differences, including CAT2000 versus COCO-Freeview. | Spatial saliency. The paper says dynamic eye movements are future work. | Inter-observer consistency is a benchmark, not a subject-conditioned predictor. The paper says subject-cohort preferences are not in the bias set. | It does not predict the next fixation from scanpath history. Its COCO-Freeview slot is the benchmark protocol, not a demonstration that this repository's public train/validation JSON and OSIE share one density. |
| DeepGaze3.5-VL | The released combined adapter was trained jointly on MIT, CAT, COCO, Daemons, and Figrim. Coordinates are integers on a 100 by 100 grid. Information gain uses a center-bias prior. The README does not take pixels per degree. | The model emits a coordinate sequence. The paper, not the bundled README, also reports duration conditioning and a history-invariant ordinal model with a gap of about 0.3 to 0.6 bits over a spatial-only baseline. | The paper conditions on subject identifiers with leave-subject-out cross-validation on MIT1003. The bundled README describes the combined free-viewing adapter and a COCO-Search18 search adapter, not that subject-token experiment. | OSIE is not in the released training list. The 8B model was not run. Its information gain is relative to center bias, not to this repository's uniform bin model. |
| HAT | Loaders exist for COCO-FreeView, OSIE, and MIT1003. The README does not mention pixels per degree. Subject ids are required to be consecutive integers from 1 if subject experiments are run. | Sequential dense fixation heatmaps and a termination probability. | The paper uses another viewer's scanpath as a human-consistency baseline. The README does not describe a subject-conditioned density. | Sequence score is not this repository's bin information gain. The code was not run. |

A method that is aware of recording protocol, of fixation order, and of observer differences therefore already exists as this published set. MSDB carries the protocol parameters and says it is not a scanpath model. DeepGaze III and DeepGaze3.5-VL carry temporal prediction, and the latter's paper also carries observer identity and cross-dataset training. HAT already trains free-viewing scanpaths on COCO-FreeView, OSIE, and MIT1003. A new network that repeated those ingredients would not be a distinct contribution on the sources read here.

## Does a narrower claim already exist?

The 1.080-bit difference between the OSIE 10-observer joint mean and the COCO-FreeView joint point estimate, in `docs/RESULTS_AND_ABLATIONS.md`, is not reported in those papers. That fact does not make it a contribution. The splits had already been inspected, degrees of visual angle were not aligned, and no official model was scored on the same fixations. The pages read here do not contain this estimator's table. Absence from those pages is not evidence that the scientific question is open.

## Falsifiable proposal, not adopted

Candidate, for the investigator to accept, rewrite, or discard:

> After bandwidth is expressed in verified degrees of visual angle, and after a center-distance-matched low-level feature comparison, the joint-control B3 difference between the public COCO-FreeView train/validation release and OSIE `fixations.mat` is statistically compatible with zero on a partition that has not yet been scored.

Competing explanations, all still open:

1. The COCO-FreeView sign already moves when the index window, the pixel bandwidth, the probability floor, or index 0 changes. Part of the original sign difference is the discrete kernel.
2. Degrees of visual angle were not measured. Canvas fractions are not degrees. For MIT1003 alone, the paper, `checkFixations.m`, and the DeepGaze README disagree: about two feet from a 19-inch 1280 by 1024 screen; 34 px/deg with a 1024 by 768, 40.5 by 30 cm, 75 cm note; and 35 px/deg.
3. Radius matching removes distance from the canvas center. It does not remove objects that sit in the middle of the scene.
4. A 5-second session and a 3-second session are different clocks. MSDB already relates presentation time to a spatial parameter and still finds leftover dataset differences.
5. The observers are different people. A subset of 10 OSIE file positions matches a count, not a population.
6. Fixation extraction differs, and MIT1003 raw samples are not fixations until `checkFixations.m` is applied.
7. DeepGaze MSDB's dataset slot and DeepGaze3.5-VL's leave-one-dataset-out model may already absorb the predictable part of the difference. That was not tested, because those models were not run.

The candidate is falsified if the degree-aligned difference remains after the center-distance control and after an official MSDB or DeepGaze III likelihood on the same untouched partition. It is also falsified if the only way to make the difference disappear is to change a kernel setting that was chosen after seeing the sign. It is not established by another score on the COCO-FreeView validation split or the OSIE hash holdout.

## Experimental design that is written and not run

`configs/evaluation_boundaries.yaml` closes the inspected COCO-FreeView validation split and the inspected OSIE hash holdout. They stay available as exploratory appendices. They are not a final test, including after a reshuffle.

The only unscored boundary prepared here is MIT1003. Image stems from the raw archive are assigned by the first byte of SHA-256(stem): bytes 0–31 tune (117 images), bytes 32–223 train (761), bytes 224–255 final (125). The assignment is in `outputs/protocolshift_prep_v1/mit1003_split_manifest.csv`. Every row is marked unscored, and no fixation was extracted. The final slice stays unused until the investigator approves a hypothesis and a conversion has been checked against `checkFixations.m`. Tuning any bandwidth or feature weight on the closed splits, or on the MIT1003 final slice, is outside this design.

## Baseline plan, not executed

If the investigator later approves a comparison, the baselines are the official code, on that code's own metric, after the inputs match:

1. DeepGaze MSDB with an explicit `pixel_per_dva`, the `COCO_FREEVIEW` slot where the benchmark protocol applies, and the documented 13-scalar adaptation only on a train slice. This is a spatial baseline. It is not a scanpath score.
2. DeepGaze III with the caller's image rescaled as the README requires, and history limited to the four preceding fixations the class requests. Compare likelihood only with a model that sees the same history.
3. DeepGaze3.5-VL's released combined adapter, information gain against its center-bias prior on the 100 by 100 grid. Do not compare that number with this repository's uniform-bin information gain.
4. HAT only if the sequence metric and the COCO-FreeView test split are the ones in the paper. Its COCO-FreeView images are the COCO-Search18 images, which is not automatically this repository's 4,317-image public JSON.

No new network is part of the plan. Training one would repeat published ingredients without a distinct claim.

## Center-distance check on the images already extracted

`outputs/protocolshift_prep_v1/center_distance_summary.csv` compares each fixation's window with a pixel on the same circle about the canvas center, and with an unmatched random pixel. Eighteen COCO-FreeView images and nine OSIE images. Image-level means:

| Dataset | Luminance, unmatched | Luminance, radius-matched | Gradient, unmatched | Gradient, radius-matched |
| --- | ---: | ---: | ---: | ---: |
| COCO-FreeView | 0.074 (image sd 0.100) | 0.025 (0.055) | 0.013 (0.008) | 0.008 (0.006) |
| OSIE | 0.005 (0.094) | −0.003 (0.063) | 0.021 (0.022) | 0.017 (0.019) |

On these files, matching distance from the center removes most of the COCO-FreeView luminance gap. A smaller positive gradient gap remains, and on OSIE the gradient image mean is about the same size as the image standard deviation. This is not evidence that edges predict gaze beyond center bias, and it is not a test of the 1.080-bit B3 residual.

## MIT1003 ingestion

`DATA.zip` was downloaded to `data/raw/external/mit1003_DATA.zip` and left gitignored. SHA-256 `a4aa26a535c90325f9acdbe125ab44aae6a9a216563b8ce9fa57d8963d8db346`. It contains the 15 readme users, 1,003 raw `.mat` files each. One inspected file, `DATA/ajs/i05june05_static_street_boston_p1010764.mat`, has `eyeData` of shape (725, 2) and no fixation label column. Fixation extraction raises `FixationExtractionRefused`. Stimulus JPEGs were not downloaded. The readme does not include a license file; the authors describe the database as publicly available with the paper.

## Resources a later approved run would consume

DeepGaze III and MSDB load published PyTorch weights, and the README examples use CUDA. MSDB adaptation updates 13 scalars with the backbone frozen. DeepGaze3.5-VL's README requires one GPU, downloads `OpenGVLab/InternVL3_5-8B-HF`, and sets GPU memory utilization to 0.90. HAT requires Detectron2 and a compiled deformable-attention operator. None of that was started. The local preparation that did run is the radius-matched window summary on 27 images and a zip inventory of about 47 MB.

## Unresolved

- No investigator hypothesis.
- No verified pixels per degree for the COCO-FreeView release or the reconstructed OSIE monitor.
- Three disagreeing MIT1003 geometry statements.
- No MATLAB check of `checkFixations.m`, so MIT1003 final images have names and no fixations.
- Official models were read and not run.
- The 27-image feature check is not independent of the earlier image selection.
