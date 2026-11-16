# Dataset audit

Counts below were recomputed from the local files during this experiment, not copied from an earlier narrative as the only check. The descriptive audit remains in `docs/DATASET.md` and `docs/ASSUMPTIONS.md`.

## COCO-FreeView public JSON

- Path: `data/raw/COCOFreeView_fixations_trainval.json`
- SHA-256: `771167f4b5de5b9b58fb07c40d6e285d38b3b2ab9e4c6267df8a454a909da730`
- 43,048 scanpaths, of which 37,038 are train and 6,010 are valid.
- 621,957 primary targets at released index 1 or later. 86,811 of those are on validation images, from 6,001 scanpaths on 603 images. Nine validation scanpaths have length 1 and contribute no primary target.
- 10 subject ids. 603 validation images include 29 images with 9 subjects rather than 10.
- Scanpaths still observed at each index are in `outputs/temporal_v1/sample_counts_by_index.csv`. At index 9 there are 5,781 validation scanpaths. At index 15 there are 3,144. Later indices are a survivor subset.
- The existing 18 tests in the descriptive suite passed again before this experiment was added (18 passed). After the temporal tests, `pytest` reported 34 passed.
- The descriptive output directory was not rewritten. `git status` at the start of the temporal work was clean at commit `2b90f8fd774bd0ad4b9946487d0e6af7f9141c6d`.

## OSIE fixation file

- Path: `data/raw/external/osie_fixations.mat` (not committed)
- SHA-256: `2aab9133a407e4eff3f57ab5799872b162dbc77ca462c0391762eb5e67deadb9`
- 1,077,850 bytes
- 700 images, 15 subject slots on every image, 10,500 scanpaths, 98,321 fixations
- Coordinates from 1.2 to 799.2 horizontally and 1.2 to 599.2 vertically. None fell outside 800 by 600.
- Length minimum 1, median 9, maximum 22
- Held-out hash bucket: 148 images, 2,220 scanpaths, 20,685 fixations in the adapted evaluation table
- First-fixation median distance to (400, 300) was 113.3 pixels. The share within 50 pixels was 0.203.

## What was checked and still disagrees with a written source

- COCO-FreeView readme headline 822,602 fixations versus 665,005 in the public JSON. The test URL returned 404 in the earlier audit. Not re-resolved.
- OSIE paper statement that fixations under 100 ms were removed, versus a minimum parsed duration of 20. Not used, not resolved.
- COCO-Search18 is described elsewhere as 6,202 images, while this JSON has 4,317 unique names. Not resolved. The temporal experiment uses the names that are in the JSON.
