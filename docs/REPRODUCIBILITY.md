# Reproducibility

## Environment used for the committed outputs

- Python 3.12.3
- Packages pinned in `requirements.txt` and `requirements-lock.txt`
- Commands were run from the repository root on 14 November 2026
- The current tables and figures come from `scripts/run_analysis.py` at `generated_at_utc` `2026-11-14T05:07:54.837138+00:00`
- `pytest` reported 18 passed after that run

## Reproduce

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python scripts/run_all.py
.venv/bin/pytest
```

This environment had no `ensurepip`, so the environment was created with `python3 -m venv --without-pip .venv` and pip was installed from `https://bootstrap.pypa.io/get-pip.py`.

`run_all.py` calls download, image preparation, and analysis with the committed `configs/analysis.yaml`.

Expected fixation SHA-256:

```text
771167f4b5de5b9b58fb07c40d6e285d38b3b2ab9e4c6267df8a454a909da730
```

`tests/test_real_data.py` checks that hash and the counts listed in [DATASET.md](DATASET.md). If the upstream file changes, the test fails on purpose.

Seeds that affect the numbers:

| Use | Seed | Where |
| --- | --- | --- |
| Within-scanpath shuffles | 20261114 | `analysis.shuffle_seed` |
| Histogram shuffle (one permutation) | 20261115 | `shuffle_seed + 1` |
| Subject and image bootstrap | 20261114, mixed per row | `analysis.bootstrap_seed` |
| Monte Carlo cross-check | 20261114 | `analysis.uniform_mc_seed` |
| Image subset | 20261114 | `images.seed` |

The radius comparison in the table is a 1 px grid, not the Monte Carlo draw. Re-running the grid gives the same value. Re-running the Monte Carlo with the same seed gives the same cross-check, including the about 1 px gap recorded in `outputs/uniform_null.json`.

## If a download is blocked

The fixation URL, the readme URL, and the two image-zip URLs are in `configs/analysis.yaml`.

1. Download `COCOFreeView_fixations_trainval.json` from the COCO-FreeView page and save it as `data/raw/COCOFreeView_fixations_trainval.json`.
2. Download the readme from the Google Drive link on that page and save it as `data/raw/COCOFreeView_readme.txt`. The committed copy is `docs/sources/COCO-FreeView-readme.txt`.
3. Either let `scripts/prepare_images.py` range-request the official zip URLs, or download both zips from the COCO-Search18 page and run:

```bash
.venv/bin/python scripts/prepare_images.py \
  --tp-zip /path/to/COCOSearch18-images-TP.zip \
  --ta-zip /path/to/COCOSearch18-images-TA.zip
```

4. Run `scripts/run_analysis.py`. Position statistics need the JSON only. Overlays need the 18 images. If the image manifest is missing, the script still writes the tables and records `scanpath_figures: skipped`.

`scripts/download_data.py` refuses an HTML body in place of the JSON or the readme and prints these steps.

## What is committed

Committed: code, tests, config, the readme text, small figures, aggregate tables, the download verification, and the resolved config.

Not committed: `data/raw/` (the JSON, the readme download, and the 18 images), `.venv/`, and `outputs/tables/scanpath_level.csv`.

Disk free when the fixation file was verified: 263,465,545,728 bytes, on a volume of 271,966,183,424 bytes.

## Tests

Synthetic tests cover successive distance, the epoch boundary, shuffle determinism, the temporal clock, the disk-area formula, bootstrap weighting when group sizes differ, zip range extraction, image-subset seeding, and archive CRC matching.

`tests/test_real_data.py` loads the public JSON when it is present and checks the hash, subject ids, split counts, coordinate bounds, and the duration-versus-timestamp count. It does not recompute the full bootstrap.

## Temporal experiment (`temporal_v1`)

The descriptive outputs above are left in place. The temporal experiment writes only under `outputs/temporal_v1/`.

```bash
.venv/bin/python scripts/run_temporal_v1.py
.venv/bin/pytest
```

`pytest` reported 34 passed after the temporal and OSIE tests were added. The temporal command recomputes the validation information-gain tables from `data/raw/COCOFreeView_fixations_trainval.json` and `configs/temporal_v1.yaml`. It does not download data and it does not rewrite `outputs/figures/` or `outputs/tables/`.

Seeds in `configs/temporal_v1.yaml`: bootstrap `20261114`, image-shuffle `20261114`. Train-image roles use `zlib.crc32(image_name) % 5`, not a second random draw. Bucket 0 selects bandwidths, bucket 1 selects mixture weights, and the other train buckets fit densities. Validation images are never given a fitting role.

The OSIE run is separate:

```bash
.venv/bin/python scripts/run_osie_v1.py
```

It reads `data/raw/external/osie_fixations.mat` and writes `outputs/osie_v1/`. The copy used for the committed OSIE tables has SHA-256 `2aab9133a407e4eff3f57ab5799872b162dbc77ca462c0391762eb5e67deadb9` and was downloaded from the raw URL of `NUS-VIP/predicting-human-gaze-beyond-pixels` `data/eye/fixations.mat`. It is gitignored with the rest of `data/raw/`. OSIE held-out images are hash bucket 4. That split is not the authors' 500/200 split.

## Validation checkpoint

`outputs/temporal_v1/` and `outputs/osie_v1/` are frozen. The checkpoint writes `outputs/audit_checkpoint/` and `outputs/temporal_v2/` only.

```bash
PYTHONPATH=src .venv/bin/python scripts/run_audit_checkpoint.py
.venv/bin/pytest
```

After the B3 density tests and the crossed-bootstrap test were added, `pytest` reported 37 passed. The audit command recomputes the frozen subject-equal means and compares them with the published tables. `temporal_v2` scores extra bandwidths on training selection folds and does not read held-out information gain. Nine OSIE JPEGs used for the overlay stay under `data/raw/` and are gitignored.

## Protocol-shift sensitivity (`protocolshift_b3_v1`)

This run starts from commit `6b84f1ccb301f948641cebf54257b07bae379eba` and writes only `outputs/protocolshift_b3_v1/`. It does not rewrite `outputs/temporal_v1/`, `outputs/osie_v1/`, `outputs/audit_checkpoint/`, or `outputs/temporal_v2/`.

```bash
PYTHONPATH=src .venv/bin/python scripts/run_protocolshift_b3.py
PYTHONPATH=src .venv/bin/python scripts/run_stimulus_pilot.py
.venv/bin/pytest
```

The first command recomputes B3 and stops if the subject-equal means differ from the frozen tables by 1e-9 or more. On the run recorded in `outputs/protocolshift_b3_v1/checksum.csv`, both datasets matched. The stimulus pilot uses the 18 extracted COCO-FreeView images and the 9 extracted OSIE images. It does not fit a feature model. After `tests/test_protocolshift.py` was added, `pytest` reported 40 passed. The literature sources for the novelty audit are listed in `docs/NOVELTY_AUDIT.md`. The publication decision is NO GO, in `docs/PUBLICATION_GATE.md`.

## Preparation without a new model (`protocolshift_prep_v1`)

```bash
PYTHONPATH=src .venv/bin/python scripts/run_center_control.py
PYTHONPATH=src .venv/bin/python scripts/run_mit1003_inventory.py
.venv/bin/pytest
```

The center-distance script reads the 27 extracted images and writes `outputs/protocolshift_prep_v1/center_distance_summary.csv`. The inventory script reads the gitignored MIT1003 `DATA.zip` and writes an unscored split manifest. It refuses to extract fixations. The boundary is `configs/evaluation_boundaries.yaml`, with status `not_approved`. The model-development decision is NO GO, in `docs/NOVELTY_GAP_ASSESSMENT.md`. After `tests/test_preparation.py` was added, `pytest` reported 43 passed.
