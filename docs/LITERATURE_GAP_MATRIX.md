# Literature gap matrix

Sources below were opened or retrieved in this audit, except where a row says the paper was only cited by a source that was opened. Numerical results from a paper are included only when they appeared in the retrieved text. Nothing here is a claim that a gap is worth a paper.

## Confirmed overlap

| Source | What overlaps this repository | What does not |
| --- | --- | --- |
| Kümmerer, Bethge, and Wallis, Journal of Vision, April 2022, DeepGaze III. https://jov.arvojournals.org/article.aspx?articleid=2778776 | Conditional fixation densities; information gain against a uniform model and against a center-bias model; the initial fixation used as history and not as an evaluation target; scanpath history compared with spatial prediction; free viewing of static images. | They fit a deep network on image features. Their main reported datasets are MIT1003, MIT300, and CAT2000. The retrieved text does not say they evaluated COCO-FreeView. |
| Journal of Vision abstract, "Using the DeepGaze III model to decompose spatial and dynamic contributions to fixation placement over time." https://jov.arvojournals.org/article.aspx?articleid=2784505 | The retrieved snippet says fixation predictability decays over the scanpath and attributes that decay to scene content, while the history effect stays nearly constant. Only the search snippet was read, not a full paper. | The snippet does not mention COCO-FreeView or a leave-one-observer KDE. |
| Kümmerer, Wallis, and Bethge, ECCV 2018, Saliency Benchmarking Made Easy; MIT/Tübingen evaluation page https://saliency.tuebingen.ai/evaluation.html | Information gain, NSS, and the idea that a probabilistic model should be scored as a density. The benchmark lists COCO-FreeView as 1,000 test images, 10 observers, 5 seconds. | This repository does not submit to that benchmark and does not use the held-out test fixations. |
| Chen et al. 2022, as cited by the COCO-FreeView readme | The dataset. The readme cites "Characterizing Target-Absent Human Attention," Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 5031–5040. | This audit did not re-read that PDF, so it does not attribute findings to the paper. |

## Paper notes

### DeepGaze III (2022)

- Dataset: MIT1003 for the main analyses, MIT300 as a held-out benchmark set, CAT2000 as a repeat. The retrieved text says MIT1003 has 15 subjects and 3-second presentation, and that the initial central fixation is not evaluated.
- Question: where the next fixation is, given the image and the fixation history.
- Method: a modular deep network with a spatial pathway, a scanpath pathway, and a center bias.
- Validation: conditional log-likelihood, information gain relative to center bias, AUC, and NSS. The paper argues against scanpath-similarity metrics as the main comparison and points to Kümmerer and Bethge (2021) for that argument.
- Finding, from the retrieved text: on their main dataset, scene content has a stronger effect than scanpath history, and the history effect comes mostly from the current fixation rather than earlier ones. The retrieved text also gives overall likelihoods against a uniform baseline of 2.442 bit/fixation on MIT1003, 2.248 on MIT300, and 3.064 on CAT2000.
- Limitation they state: high-capacity models can capture many patterns; that does not by itself identify a mechanism.
- Overlap with this repository: the metric family and the history-versus-spatial contrast.
- Confirmed difference: this repository's phase-1 models are normalized bin densities and leave-one-observer histograms, on the public COCO-FreeView train/validation file. DeepGaze III was not run. No number in `outputs/temporal_v1/` or `outputs/osie_v1/` is a DeepGaze III number.
- The COCO-FreeView index curves decline for the same-image maps (B2 and B3). The previous-fixation early-minus-late interval on the balanced subset includes values below zero. The 2022 temporal abstract's retrieved snippet says predictability decays and attributes that decay to scene content, while history stays nearly constant. The qualitative shape is the overlap. This repository did not compute image features, so it does not test that attribution. A dataset change and a binned KDE are not a new mechanism and are not a contradiction of DeepGaze III.

### DeepGaze III temporal abstract (snippet only)

- The snippet says overall predictability decays, the decay is attributed to scene content, and the history contribution stays nearly constant.
- Confirmed: the question of how spatial and history contributions change with fixation index has already been posed with DeepGaze III.
- Not confirmed: any numerical value from that abstract beyond the snippet's qualitative statement. Volume, issue, and a full methods section were not retrieved.

### Kümmerer and Bethge (2021), arXiv:2102.12239

- Cited by the 2022 DeepGaze III paper as the scanpath-model review. This audit did not recompute that review's tables.
- The 2022 paper says the review found that few scanpath models beat a strong spatial model when scored by conditional prediction, and that scanpath similarity can rank a wrong model above ground truth.

### Kümmerer, Wallis, and Bethge (2015), PNAS 112(52), 16054–16059

- Cited by DeepGaze III as the information-gain reference. This audit did not re-read the 2015 paper. The metric implemented here is the bin log-likelihood ratio described in the protocol, not a reimplementation of their software.

### MIT/Tübingen benchmark and ECCV 2018

- The evaluation page lists IG, AUC, shuffled AUC, NSS, CC, KL, and SIM, implemented in pysaliency.
- COCO-FreeView on that page is the 1,000-image test set. The public JSON in this repository is the train/validation release of 4,317 images. Those are not the same split, and this repository does not score the benchmark test set.

### COCO-FreeView readme and project page

- 10 participants, 5 seconds, no target, 1680 by 1050 zero-padded images.
- The readme's headline fixation count is 822,602. The public JSON has 665,005. The difference was not checked against a test file because the test URL returned 404.

### Judd, Ehinger, Durand, and Torralba (2009), MIT1003

- Official page and database readme were retrieved. 15 named viewers, 1,003 images. The readme says the downloadable eye data are raw samples, with negative coordinates for blinks, and that fixations are derived by `checkFixations.m`.
- A header request on 14 November 2026 returned HTTP 200 for `DATA.zip`, Content-Length 47,024,024, Last-Modified 10 September 2009.
- This repository did not download that zip and did not derive fixations from raw samples.

### Xu, Jiang, Wang, Kankanhalli, and Zhao, Journal of Vision, January 2014

- Retrieved journal text: 15 participants, 700 images, 3-second free viewing, 800 by 600 stimuli scaled to the full height of a 1680 by 1050 monitor, EyeLink 1000.
- The same text says fixations shorter than 100 ms were discarded. The MATLAB file parsed in this repository has durations as small as 20. That disagreement is unresolved. Durations were not used as time.
- The fixation file used here is `fixations.mat` from `NUS-VIP/predicting-human-gaze-beyond-pixels`, SHA-256 `2aab9133a407e4eff3f57ab5799872b162dbc77ca462c0391762eb5e67deadb9`. The code repository states an MIT license for the software. A separate data-license sentence was not found in the files read here.
- Nine stimulus JPEGs from that repository, including `1001.jpg`, are 800 by 600. Released coordinates drawn with y increasing downward fall on scene content in `outputs/audit_checkpoint/figures/osie_fixation_overlay.png`. The paper's monitor scaling was not reconstructed. OSIE scores in this repository are densities refit on OSIE train images. A COCO-FreeView density was not evaluated on OSIE pixels.

## Confirmed gaps

- The public COCO-FreeView train/validation JSON does not, in the documents read here, come with a published leave-one-observer information-gain curve by fixation index using the baselines in `docs/EXPERIMENT_PROTOCOL.md`.
- DeepGaze III's published numbers are not COCO-FreeView numbers. Copying them across datasets would be an error.
- The benchmark test fixations are not in the public JSON. A leaderboard comparison is not available from this file.

## Speculation, not a gap that has been shown to matter

- That a binned density on COCO-FreeView would change a conclusion from DeepGaze III. The checkpoint does not make that comparison. The models were not run on the same images, and no scene features were computed here.
- That low-level image features would add information beyond center bias on this dataset. Those features were not computed.
- That the OSIE and COCO-FreeView early-versus-late patterns are the same phenomenon. The protocols differ, and the analyses were not pooled.
