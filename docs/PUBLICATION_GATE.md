# Publication gate

Assessment for beginning a manuscript on the proposed ProtocolShift-Gaze track. No journal quartile and no acceptance probability is stated. Journal scope was not scored.

## 1. Exact claimed contribution

No contribution is claimed. The proposal was to ask whether the frozen temporal measurements support a new result about cross-dataset temporal uncertainty, observer variability, measurement protocol, and visual-feature dependence. The work that was run is a sensitivity analysis of one nonparametric index-conditioned density, plus a unit-square history-kernel transfer and a 27-image contrast description.

## 2. Evidence the contribution is not already published

`docs/NOVELTY_AUDIT.md` records the sources that were opened. DeepGaze III and its 2022 temporal abstract already decompose scanpath predictability over time. HAT (CVPR 2024) already evaluates free-viewing scanpaths on COCO-FreeView, MIT1003, and OSIE. The ICCV 2025 dataset-bias paper already treats cross-dataset spatial prediction, blur in degrees, and few-shot adaptation, and it names dynamic eye movements as work it did not do. DeepGaze3.5-VL already reports COCO-FreeView information gain, leave-one-dataset-out transfer, subject identity, and an index-pooled population fixation model. ScanDiff already models scanpath diversity. The pages read here do not contain this repository's B3 table. That is not evidence of an unsolved problem.

## 3. Appropriate prior-method baselines

Not run. The official repositories named in the audit were not executed. Comparing this repository's uniform-bin information gain with a published center-bias information gain, or with HAT's sequence scores, would be a comparison of different metrics. It was not done.

## 4. Independent validation

Not available. The COCO-FreeView validation split and the OSIE hash holdout were inspected before these controls. MIT1003 was not ingested. No new confirmatory image set was scored. The unit-square history transfer uses the same inspected holdouts.

## 5. Statistical design

The frozen subject bootstrap and the crossed resample remain the uncertainty statements for the original models. The new subject intervals are exploratory. The range across 20 OSIE subsets is not a confidence interval for a population of observers. No confirmatory contrast was predeclared by the investigator. No multiplicity correction is claimed, because no confirmatory family was tested.

## 6. Robustness

OSIE B3 stays positive across the bandwidths, floors, index rules, and 10-observer subsets that were run. COCO-FreeView B3 changes sign when the index set, the bandwidth, the floor, or index 0 is changed. The joint control leaves about 1.08 bits between the OSIE subset mean and the COCO point estimate, with non-overlapping descriptive ranges of different kinds. Canvas-fraction coordinates do not close that gap. Degrees of visual angle were not tested. The result is robust as a warning that the COCO sign is estimator-dependent, and it is not robust as a cross-dataset constant.

## 7. Limitation severity

Severe for a manuscript. Ten recurring COCO-FreeView observers, no degree conversion, no untouched partition, no official baseline, no approved hypothesis, and a literature that already covers the surrounding questions. The residual is unexplained by the controls that were run. Unexplained is not the same as a discovered protocol effect.

## 8. Reproducibility

The control script, the pilot script, the tests, the config implicit in the script, and the CSV outputs are in the branch. Frozen directories were not rewritten. Raw data remain gitignored. `docs/REPRODUCIBILITY.md` records the commands. Another person still needs the public COCO-FreeView JSON and the OSIE mat file, which are not in git.

## 9. Alternative explanations

The residual can be a difference in image content, in unmeasured visual angle, in session length, in how fixations were extracted, or in the particular observers. The controls do not rank those explanations. A wider kernel or a higher floor can also move the COCO mean across zero, so part of the original sign difference is a property of the discrete kernel.

## 10. What would falsify a proposed contribution

The following claims are already inconsistent with the tables in this branch:

- That the sign difference is produced by having 15 rather than 10 observers. The 10-observer OSIE scores stay above 1.3 bits in the native setting and above 1.1 bits in the joint setting.
- That the COCO-FreeView sign is a stable property of the fixations, independent of bandwidth, floor, and which indices are averaged. Sigma 64 px, floor 0.10, index 0, and indices 1 through 8 move that sign or put the interval across zero.
- That canvas-fraction normalization removes the discrepancy. It does not.

A future claim that the residual is a causal effect of viewing protocol would be unsupported by observational controls of this kind even if it were repeated on a new dataset. A future claim of a new scanpath model would be unsupported until an official baseline is run on a matched split and metric.

## Decision

**NO GO.**

A manuscript outline is not written. The measurements can sit beside the frozen temporal report as a sensitivity analysis. They do not justify a paper, and they do not justify starting visual-feature training or a MIT1003 fixation conversion.

## Addendum after the official-code reading

A second pass read the DeepGaze, DeepGaze3.5-VL, and HAT repositories and the MIT1003 fixation code. The decision is unchanged. A protocol-aware temporal model is not a distinct unreported method: protocol parameters are in DeepGaze MSDB, fixation history is in DeepGaze III, and cross-dataset scanpaths plus observer tokens are in the DeepGaze3.5-VL paper. Details, the unadopted falsifiable candidate, and the unscored MIT1003 boundary are in `docs/NOVELTY_GAP_ASSESSMENT.md`. No network was trained. The publication gate remains **NO GO**.

## Addendum after baseline reproduction

Official DeepGaze III weights were then run on the nine local OSIE images, and MIT1003 train and tune samples were converted with `checkFixations.m`. The measured numbers, the blockers, and the leakage audit are in `docs/BASELINE_REPRODUCTION.md`. The decision memo is `docs/RESEARCH_DECISION_MEMO.md`. No new network was trained. The publication gate remains **NO GO**.

## Addendum after the OSIE expansion

Official DeepGaze III was then scored on all 700 released OSIE images. The nine-image table was kept as a reproduction check and was not overwritten. The new result is exploratory. Measured tables are in `docs/OSIE_DEEPGAZE3_V2.md`. No network was trained. The publication gate remains **NO GO**. A separate decision, not to begin a study of failures inside the published object contours, is in `docs/OBJECT_REGION_STUDY_DECISION.md`.

## Addendum after the semantic-region audit

The same frozen density was then summed inside the official OSIE masks. The measured region-event gaps, the controls, and the small out-of-fold recalibration are in `docs/SEMANTIC_RESULTS.md`. The decision not to propose a spatial correction, and the journal gate, are in `docs/SEMANTIC_NEXT_DECISION.md`. No new network was trained. The publication gate remains **NO GO**.
