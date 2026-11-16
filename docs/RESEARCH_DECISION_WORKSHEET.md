# Research decision worksheet

Nothing in this file is a preregistered hypothesis. The investigator has not approved a hypothesis. The agent does not write conclusions in the investigator's name and does not invent originality.

## Literature-derived questions

These questions are already posed by the sources in `docs/NOVELTY_AUDIT.md`.

- How much of the next fixation is predictable from the image, from scanpath history, and from center bias, and how those shares change with fixation index. DeepGaze III and the 2022 temporal abstract.
- How much of a cross-dataset drop in spatial fixation prediction is absorbed by a few interpretable parameters, including blur in degrees of visual angle. ICCV 2025, Modeling Saliency Dataset Bias.
- Whether one trained scanpath model can cover free viewing and visual search on COCO-FreeView, MIT1003, and OSIE. HAT, CVPR 2024.
- Whether observer identity and fixation index carry additional likelihood in a token model. DeepGaze3.5-VL.

## Exploratory tests implemented by the agent

These tests were run after the COCO-FreeView validation scores and the OSIE held-out scores had already been inspected. They are not confirmatory. The configuration choices are recorded in `outputs/protocolshift_b3_v1/run_manifest.json`. They were not locked before the original discrepancy was known.

- Recompute B3 and compare it with the frozen tables.
- Include or drop released index 0.
- Restrict both datasets to indices 1 through 8.
- Rescore B3 at 16, 24, 40, and 64 px and at floors 0.01 and 0.10.
- Repeat OSIE with seeded subsets of 10 of the 15 file-position subject slots, and with file positions 0 through 9.
- Repeat both datasets on the unit square with bandwidths equal to 24/800 and 40/1680 of canvas width.
- A joint setting: indices 1 through 8, unit-square bandwidth 24/800 of width, and 10 OSIE observers.
- Transfer a unit-square history kernel from each training split to the other validation split.
- Summarize predictive entropy, half-mass coverage, and low-entropy misses.
- On the 18 already extracted COCO-FreeView images and the 9 OSIE overlay images, compare local luminance and gradient energy at fixations with random pixels.

## Investigator-formulated hypotheses

| ID | Hypothesis, in the investigator's words | Date | Status |
| --- | --- | --- | --- |
| | | | Not adopted. This row is intentionally blank. |

An agent-written candidate is in `docs/NOVELTY_GAP_ASSESSMENT.md`. It is not this row and it is not preregistered. The model-development decision on that candidate is NO GO.

Options, if the investigator later chooses one. None is recommended here as the investigator's idea.

- Stop at the measurement report. The literature overlap and the lack of a confirmatory partition are the reasons already written in `docs/PUBLICATION_GATE.md`.
- Treat the B3 residual as a methods note about a nonparametric estimator, with the controls as sensitivity analyses and without a causal protocol claim.
- Run an official scanpath model on a split that model already defines, and treat that as a replication rather than a new contribution.
- Reconstruct degrees of visual angle for both releases before any further cross-dataset density comparison. That requires display geometry this repository does not yet have for both datasets.

## AI-implemented software

The agent wrote `src/humangazelab/protocolshift.py`, `scripts/run_protocolshift_b3.py`, `scripts/run_stimulus_pilot.py`, and `tests/test_protocolshift.py`. The agent also wrote this worksheet and the other ProtocolShift documents. The earlier temporal code remains the frozen implementation.

## Verified measurements

The numbers live in `outputs/protocolshift_b3_v1/` and are summarized in `docs/RESULTS_AND_ABLATIONS.md`. They are measurements of the code that was run. They are not interpretations.

## Interpretations that need the investigator

- Whether a remaining difference of about one bit under the joint control is worth any further study.
- Whether canvas-fraction coordinates are an acceptable sensitivity or whether the work should wait for degrees of visual angle.
- Whether any sentence about protocol, observers, or image content should appear in a report under the investigator's name.

No row above is preregistered. A later approval would need to name the hypothesis, the confirmatory images, and the rule for stopping, before those images are scored.
