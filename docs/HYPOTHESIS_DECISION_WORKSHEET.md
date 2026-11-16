# Hypothesis decision worksheet

No hypothesis in this file has been adopted. The temporal experiment is a measurement protocol. It was not used to choose a hypothesis after seeing the validation scores, and it should not be read as a hypothesis written in the investigator's voice.

The investigator marks one row, or writes a different hypothesis, before any sentence in a manuscript treats a contrast as a test.

| Decision | Mark | Candidate statement the measurements could later be compared with |
| --- | --- | --- |
| Not adopted |  | Cross-observer information gain on a held-out observer declines from the early fixation-index group to the late group, on scanpaths long enough to contribute to both groups. |
| Not adopted |  | A previous-fixation density adds held-out information beyond a train-fitted center-bias density, and that added information is not the same in the early and late groups. |
| Not adopted |  | Conditioning the same-image reference on fixation index adds information beyond a reference that pools indices. |
| Not adopted |  | None of the above. The tables stay descriptive until a hypothesis is written independently of these outputs. |

## What was pre-specified as measurement, not as a hypothesis

- Primary COCO-FreeView targets are released fixation indices at or above 1. Index 0 remains available as history for index 1.
- Information gain is the base-2 log ratio of the model bin probability to the uniform bin probability.
- Bandwidths and mixture weights are chosen on train-image folds. Validation images are not used for that choice.
- Same-image maps omit the target observer.
- Early indices are 1–3, middle indices are 4–8, and late indices are 9 and later, on COCO-FreeView only.
- The balanced COCO-FreeView subset is released length at least 12.
- Uncertainty is a percentile bootstrap of subject means, with an image-cluster bootstrap reported separately.
- OSIE is analyzed inside its own protocol. Its indices are not treated as the same events as COCO-FreeView indices.

## What this worksheet does not do

It does not interpret a bootstrap interval that excludes zero as a causal result. It does not treat ten COCO-FreeView participants as a population estimate. It does not claim that a same-image leave-one-observer map is an image-only gaze predictor.
