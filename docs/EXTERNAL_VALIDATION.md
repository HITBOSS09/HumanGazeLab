# External validation

## What was attempted

MIT1003 and OSIE were both checked as independent free-viewing datasets with sequential fixations. Only OSIE was scored.

### MIT1003

- Official page: https://people.csail.mit.edu/tjudd/WherePeopleLook/
- On 14 November 2026 a header request to `DATA.zip` returned HTTP 200, Content-Length 47,024,024, Last-Modified Thursday 10 September 2009.
- The publisher's readme says that zip contains raw eye samples and that fixations are produced by their MATLAB function `checkFixations.m`.
- This run did not download the zip and did not write a substitute fixation table. A Python reimplementation of their fixation detector would have been an unvalidated method.

### OSIE

- Paper text retrieved: Xu, Jiang, Wang, Kankanhalli, and Zhao, Journal of Vision, January 2014. Fifteen participants, 700 images, 3 seconds, 800 by 600 stimuli scaled to a 1680 by 1050 display.
- Fixation file: `fixations.mat` from https://github.com/NUS-VIP/predicting-human-gaze-beyond-pixels , SHA-256 `2aab9133a407e4eff3f57ab5799872b162dbc77ca462c0391762eb5e67deadb9`.
- The struct has one entry per image, 15 subject slots, and ordered `fix_x`, `fix_y`, and `fix_duration`. Subject codes are not in the file. Tables use slot index 0–14.
- Nine stimulus JPEGs were downloaded from the same repository and were not committed. Each is 800 by 600. All fixations on those images fall inside the file, and the overlay in `outputs/audit_checkpoint/figures/osie_fixation_overlay.png` places them on scene content with y increasing downward. The paper's scaling onto a 1680 by 1050 display was not reconstructed. The scores below are dataset-adapted: densities were refit on OSIE train images. They are not a zero-shot transfer of a COCO-FreeView density.
- The authors' 500/200 split is not a field in the file. This run held out images with `crc32(name) % 5 == 4` (148 images) and selected bandwidths on the remaining images. That split is documented in `configs/osie_v1.yaml`. It is not their split.

## What transferred

"Transferred" here means: the same density definitions, refit on OSIE train images.

Adapted OSIE bandwidths, held-out images, subject-equal means, 15 subject slots, 20,685 fixations, 2,220 scanpaths:

| Model | Bits per fixation | Subject interval |
| --- | ---: | --- |
| B0 uniform | 0 | — |
| B1 center | 0.565 | 0.505 to 0.622 |
| B2 same-image leave-one-observer | 2.498 | 2.382 to 2.613 |
| B3 same index | 1.848 | 1.705 to 1.991 |
| B4 previous fixation | 1.430 | 1.243 to 1.608 |
| B6 wrong image | -1.277 | -1.363 to -1.190 |

B4 is undefined at released index 0. Its interval is the subject mean after those undefined rows are dropped inside each scanpath. The paired B4-minus-B1 contrast uses 18,465 finite fixations.

Frozen COCO-FreeView pixel bandwidths, still refit on OSIE train images:

- B1 stayed 0.565 because OSIE also selected 24 pixels.
- B4 stayed 1.430 for the same reason.
- B2 fell from 2.498 to 2.166 when the leave-one-observer bandwidth was kept at the COCO-selected 40 pixels instead of the OSIE-selected 24 pixels.

On the OSIE balanced subset (length at least 9; 1,512 scanpaths), early indices 0–2 minus late indices 6 and later:

| Model | Difference (bits) | Subject interval |
| --- | ---: | --- |
| B1 | 0.480 | 0.393 to 0.585 |
| B2 | 0.787 | 0.588 to 0.977 |
| B4 | 0.415 | 0.245 to 0.582 |

History at the early end of that contrast is carried by indices 1 and 2, because index 0 has no previous fixation.

## What did not transfer

- B3 is positive on OSIE (1.848 bits) and negative on the COCO-FreeView primary table (−0.314 bits). Both values were recomputed from the frozen bandwidths and match the published tables. The sign is not shared. The checkpoint measurements, in `outputs/audit_checkpoint/`, are: COCO-FreeView index 5 is already negative while the median number of other observers is still 9 and the median nearest observer is 97 px; OSIE index 0 has 14 other observers, a median nearest distance of 15 px, and 3.455 bits. Keeping the COCO leave-one-observer bandwidth of 40 px on OSIE, with densities still refit on OSIE, left B3 at 1.897 bits in the frozen OSIE table. No single cause is assigned.
- The COCO-FreeView history early-minus-late interval included values below zero. The OSIE history interval did not. The index definitions are not the same events.
- No COCO density was applied to OSIE pixels. A zero-shot density transfer across canvases was not claimed.
- Display geometry beyond the 800 by 600 JPEG frame was not checked. The overlay covers nine images.

## Protocol differences that block a pooled conclusion

- COCO-FreeView readme: 5 seconds, 10 participants, 1680 by 1050 canvas, initial center fixation included in the release.
- OSIE paper: 3 seconds, 15 participants, 800 by 600 stimulus scaled onto a larger monitor, drift correction between trials.
- Released index *k* is not a shared time stamp. The analyses stay inside each dataset.
