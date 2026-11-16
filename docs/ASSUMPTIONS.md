# Assumptions and open decisions

These are the places where the files do not force a single analysis. The pipeline still produces a number for each of them, and it labels which rule it used. The decision of which rule belongs in a description of the study is open.

## 1. Order is the array order

The readme says `X`, `Y`, and `T` run from the first fixation to the last. The pipeline uses array position as that order. The release does not include the EyeLink sample stream, so fixation order was not rebuilt from raw gaze samples.

`T` is not a timestamp. 42,902 of 43,048 sequences are not strictly increasing. That matches the readme's statement that `T` holds durations.

## 2. The unit of `T` is treated as milliseconds

The readme says "fixation durations" and does not name the unit. Values such as 316, 148, and 123, and a median within-scanpath sum of 4,161, are the magnitudes expected for milliseconds in a trial the authors describe as 5 seconds long. Ordinal analyses do not use this unit. Temporal analyses do. If the unit is something else, the temporal rows move and the ordinal rows stay.

51 individual `T` values are above 5,000, and 325 scanpaths have a `T` sum above 5,000. One scanpath sums to 10,170. Those records are kept. A filter was not applied.

## 3. Index 0 is the initial center fixation named in the readme

The readme says the sequences include the initial center fixation. The median distance from index 0 to the canvas center is 22.1 px. The share within 50 px is 0.867, and the share within 100 px is 0.958. Index 0 is therefore near the center on most trials and not a single pixel on every trial.

Primary rows drop index 0. Rows marked `_all_` keep it. Both are in `metrics_summary.csv`.

## 4. The canvas is 1680 × 1050

That size is the readme's statement. It matches the coordinate range of every fixation and the pixel size of all 18 extracted images. Image size was not checked for the other 4,299 images, because those files were not downloaded.

## 5. Y increases downward

The six overlay panels were drawn with y increasing downward. The marks fall on scene content. That check covers six images, not the full set.

## 6. Cutoffs were set in the config, then the table was computed

`k = 3`, thresholds 1,500 / 1,000 / 2,000, and radii 200 / 100 / 400 are in `configs/analysis.yaml`. They were written after the schema pass showed that the median length is 16, so a prefix of 3 is defined for most sequences, and before the contrast table was used to keep or drop a cutoff. The sensitivity rows are the other values. They are all reported. None of them was deleted after seeing the interval.

`k = 3` is an analysis choice. The file does not contain an early/late label.

## 7. The shuffle keeps positions and breaks order

The baseline permutes fixation order inside a scanpath. It does not move fixations to new locations, and it does not shuffle across images or subjects. A different baseline would answer a different question. This is the one that was implemented.

## 8. Subject ids are the grouping variable the file provides

There are 10 ids: 101, 102, 103, 104, 105, 107, 108, 109, 110, 116. The integers 106 and 111–115 do not appear. The file does not say why. The ids were not renumbered to 1–10.

The bootstrap resamples these 10 ids. With 10 clusters the interval is coarse. The table reports the interval it computed. It is a description of resampling this sample.

Image-level intervals are also reported. They do not replace the subject intervals.

## 9. `condition`, `correct`, `fixOnTarget`, and `task` are not used as measures

`condition` is `freeview` on every record, which is not the target-present / target-absent label described in readme v1.0. `correct` is 1 on every record. `fixOnTarget` is false on every record. `task` and `trial_ind` are absent. Category folders in the image zips repeat identical bytes under more than one folder, so the folder name was not used as a task label.

## 10. The public file is the train+validation release only

There is no test split inside the JSON. The test URL returned 404. No test-set score is computed, and no test labels were created.

## 11. Zero-padding is measured only on the 18-image subset

The border count uses pixels that are exactly black. A real black pixel in a photograph would be counted as padding. On this subset, 1 post-initial fixation falls in that border.

## 12. No degree-of-visual-angle conversion

Viewing distance and screen size are not columns in the release. Positions and distances stay in pixels.
