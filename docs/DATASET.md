# Dataset record

This note describes the files that were actually retrieved. It is the source for the counts in the tests.

## Fixation release

| Item | Value |
| --- | --- |
| Project page | https://sites.google.com/view/cocosearch/coco-freeview |
| File URL | http://vision.cs.stonybrook.edu/~cvlab_download/COCOFreeView_fixations_trainval.json |
| HTTP last-modified | Fri, 23 Jun 2023 05:25:00 GMT |
| Bytes | 50,110,884 |
| SHA-256 | `771167f4b5de5b9b58fb07c40d6e285d38b3b2ab9e4c6267df8a454a909da730` |
| Local path | `data/raw/COCOFreeView_fixations_trainval.json` (gitignored) |

The file is a JSON list of 43,048 objects. Required keys in every object are `name`, `subject`, `split`, `X`, `Y`, `T`, and `length`. `X`, `Y`, and `T` are equal-length lists. `length` equals that length in every record. There are no duplicate `(name, subject, split)` triples.

Counted from this file:

| Quantity | Value |
| --- | --- |
| Fixations | 665,005 |
| Images | 4,317 |
| Train scanpaths / images | 37,038 / 3,714 |
| Validation scanpaths / images | 6,010 / 603 |
| Subject ids | 101, 102, 103, 104, 105, 107, 108, 109, 110, 116 |
| `condition` | `freeview` on all 43,048 records |
| `correct` | 1 on all records |
| `fixOnTarget` | false on all records |
| Scanpath length | minimum 1, median 16, maximum 38 |
| Length-1 scanpaths | 51 |
| Coordinate range | x in [8.5, 1679.9], y in [0.0, 1049.7] |
| Fixations outside [0, 1680] x [0, 1050] | 0 |
| Released `T` range | 1 to 5,270 |
| Sum of `T` within a scanpath | minimum 274, median 4,161, maximum 10,170 |
| Scanpaths whose `T` sums to more than 5,000 | 325 |
| Individual `T` values above 5,000 | 51 |
| Sequences in which `T` is not strictly increasing | 42,902 |
| Sequences in which `T` is non-decreasing | 147 |

Subject scanpath counts are in `outputs/tables/subject_coverage.csv`. Subject 108 has 4,221 scanpaths. The others have 4,293 to 4,317. 4,195 images have all 10 subject ids. 122 images have 9.

The readme headline says the dataset contains 822,602 fixations. This file contains 665,005. The difference is 157,597. The URL `http://vision.cs.stonybrook.edu/~cvlab_download/COCOFreeView_fixations_test.json` returned HTTP 404 on 14 November 2026, so that difference was not checked against a test file. The project page says the testing data were taken down pending an online evaluation service.

COCO-Search18 is described by its authors as 6,202 images. This fixation file names 4,317 images. The MIT/Tübingen saliency benchmark page describes 1,000 COCO-FreeView test images. 6,202 minus 4,317 is 1,885, which is not 1,000. This repository does not resolve that discrepancy.

## Readme

| Item | Value |
| --- | --- |
| Page | https://drive.google.com/file/d/1Hj_jyK8Ml27Ge_5sogEtyI7XOjyad4aj/view |
| Download URL used | https://drive.google.com/uc?export=download&id=1Hj_jyK8Ml27Ge_5sogEtyI7XOjyad4aj |
| HTTP last-modified | Mon, 20 Jun 2022 22:40:34 GMT |
| Bytes | 2,222 |
| SHA-256 | `9812cd3b36c98d08de0bfccf472b18e801e5a9d824c5aa347b5fb5fd71b0122e` |
| Committed copy | `docs/sources/COCO-FreeView-readme.txt` |

The readme identifies itself as v1.0, dated 06/2022. It says images were resized to 1680×1050 with zero padding and the aspect ratio kept, that `X` and `Y` run from the first fixation to the last and include the initial center fixation, and that `T` holds fixation durations in that same order. It does not name the unit of `T`.

Fields named in that readme and absent from the JSON: `trial_ind`, `task`.

Fields present in the JSON and absent from that readme: `correct`, `fixOnTarget`.

The readme describes `condition` as the target-present or target-absent label of the same image in COCO-Search18. In this JSON the value is the constant string `freeview`.

The readme's BibTeX places Chen et al. 2022 in CVPR. The project page places the same paper in the CVPR workshops (CVPRW), pages 5031–5040. The open-access PDF linked from the project page is the workshop version: https://www3.cs.stonybrook.edu/~zhibyang/papers/cvpr22_gaze_workshop.pdf. A second citation on the project page is Yang et al. 2023, arXiv:2303.09383.

## Stimuli

The FreeView page says the images are the COCO-Search18 images:

- http://vision.cs.stonybrook.edu/~cvlab_download/COCOSearch18-images-TP.zip (1,134,881,023 bytes, last-modified 15 June 2020)
- http://vision.cs.stonybrook.edu/~cvlab_download/COCOSearch18-images-TA.zip (1,139,771,786 bytes, last-modified 2 July 2022)

`scripts/prepare_images.py` reads each zip's central directory with an HTTP range request and extracts selected JPEGs. It does not store the zip. The index built on 14 November 2026 reported:

| Check | Result |
| --- | --- |
| JPEG members in each zip | 3,101 |
| Unique JPEG basenames, TP / TA | 2,779 / 2,561 |
| FreeView names missing from both zips | 0 |
| FreeView names only in TP / only in TA / in both | 2,221 / 2,074 / 22 |
| Both-archive basenames whose CRC-32 disagrees | 0 |
| Within-archive duplicate basenames with the same CRC-32 | 305 (TP), 471 (TA) |
| Within-archive duplicate basenames with different CRC-32 | 0 |

Duplicate basenames are the same bytes stored under more than one category folder. The folder name is therefore not a unique task label, and the fixation file does not contain `task`. The loader does not invent one.

The seeded subset is 12 train images and 6 validation images, seed `20261114`, restricted to images with 10 subject ids. All 18 extracted files are 1680×1050. Their names, archive paths, CRC-32, SHA-256, and border measurements are in `outputs/tables/image_subset_manifest.csv`.

On those 18 images, a border of all-zero pixels is present on the left and right, or on the top and bottom, consistent with the readme's zero-padding statement. Of the post-initial fixations on those images, 1 falls inside that all-zero border. The border rule treats a pixel as padding only when R, G, and B are all 0, and it was applied only to this subset.

## Coordinate check on the six overlay panels

`outputs/figures/scanpath_examples.png` draws released coordinates on six subset images with y increasing downward, extent `[0, width] × [0, height]`. The marks fall on scene content (food, utensils, furniture, objects on a table), and the open red circle (index 0) sits near the middle of the canvas. That check supports using y-down image coordinates for those six images. It was not repeated for the other 4,311 images.

## Papers consulted for experimental context

These statements are taken from the papers and the project page. They were not re-measured from the files.

- Ten university students, 5 seconds per image, free viewing in anticipation of a memory test, EyeLink 1000 at 1000 Hz, fixations extracted with the tracker's default settings. Source: Chen et al. 2022, and the project page.
- The procedure was modified from COCO-Search18 by removing the target cue, removing the response, and fixing the viewing time at 5 seconds. Source: the project page and readme v1.0.
- Stimulus photographs are MS COCO images prepared for COCO-Search18. Source: Chen et al., Scientific Reports, 2021, and the COCO-Search18 page.

No pixels-per-degree conversion is applied. A degree conversion would need a viewing distance and screen size, and those quantities are not fields in the release.
