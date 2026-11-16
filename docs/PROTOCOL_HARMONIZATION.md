# Protocol harmonization

COCO-FreeView and OSIE were not pooled. The controls in `outputs/protocolshift_b3_v1/` change one analysis choice at a time, and one joint setting changes several at once. A shared number after a control is a sensitivity of the estimator. It is not evidence that the acquisition protocols are the same, and it is not a causal effect of a protocol.

## What the releases actually specify

| Item | COCO-FreeView public train/validation JSON | OSIE `fixations.mat` |
| --- | --- | --- |
| Images | 4,317 in this file. The MIT/Tübingen benchmark description of 1,000 test images is a different release. | 700 |
| Observers | 10 ids. Subjects 106 and 111–115 are absent. The same ids appear in train and validation. | 15 file positions. Original participant codes are not in the file. |
| Canvas in the file | 1680 by 1050, including the readme's zero padding. | Coordinates fall in 800 by 600. Nine JPEGs are 800 by 600. |
| Display | Readme does not give viewing distance or pixels per degree. `docs/DATASET.md` records that no degree conversion was applied. | The 2014 paper describes 800 by 600 stimuli scaled to the full height of a 1680 by 1050 monitor. That scaling, including side padding, was not reconstructed. |
| Duration of the trial | Readme: 5 seconds. | Paper: 3 seconds. |
| Index 0 | Readme includes an initial center fixation. Primary targets start at index 1. | Not labeled as a forced fixation. Primary targets include index 0. About 20% of first fixations are within 50 px of the canvas center. |
| Time base | `T` is stored and was not used. Its unit is not named in the readme. | Durations are stored. The file minimum is 20 and the paper says fixations under 100 ms were removed. The column was not used. |
| Split | Author field `train` or `valid`. | `crc32(name) % 5 == 4` is this repository's holdout, not the authors' 500/200 split. |

## What was aligned in the exploratory controls

- Index window: both datasets scored on released indices 1 through 8.
- Observer count: OSIE rescored on subsets of 10 file positions. COCO-FreeView already has 10.
- Bandwidth in pixels: 16, 24, 40, and 64 on each native grid.
- Probability floor: 0.01, 0.05, and 0.10.
- Canvas fraction: both datasets mapped to the unit square by dividing x by canvas width and y by canvas height, then scored at 24/800 and 40/1680 of canvas width. The grid is 20 by 15 bins on that square.

## What was not aligned

- Degrees of visual angle. Dividing by canvas width is not a retinal metric. The ICCV 2025 dataset-bias paper treats pixels-per-degree scale and relative image scale as different parameters. This repository does not have a verified pixels-per-degree value for the COCO-FreeView release, and it has not rebuilt the OSIE monitor mapping.
- Clock time. Index 8 is not the same moment in a 5-second session and a 3-second session.
- Subject identity across datasets. The people are different. A 10-observer OSIE subset is a count match, not a population match.
- Image content, padding, and the benchmark test split.
- The author splits.

## Joint setting

Declared in the run script before the control table was interpreted, and after the original sign difference was already known: indices 1 through 8, unit-square bandwidth equal to 0.03 of canvas width, OSIE restricted to 10 observers. Results are in `docs/RESULTS_AND_ABLATIONS.md`. The joint setting does not establish a causal protocol effect.
