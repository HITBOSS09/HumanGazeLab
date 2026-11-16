# Baseline audit

Checked against the checkout, not against the earlier narrative alone.

- Branch at the start of this work: `main`
- Commit: `2b90f8fd774bd0ad4b9946487d0e6af7f9141c6d`
- Working tree: clean
- `pytest`: 18 passed before new tests were added
- Fixation SHA-256 matched `771167f4b5de5b9b58fb07c40d6e285d38b3b2ab9e4c6267df8a454a909da730`
- Committed readme SHA-256 matched the copy under `docs/sources/`
- `data/raw/images_subset` contained 18 images
- `outputs/figures` and `outputs/tables` contained the descriptive products listed in `docs/INVENTORY.md`
- The temporal experiment writes to `outputs/temporal_v1/` and `outputs/osie_v1/`. A later `git diff` of the original figure and table directories was empty when those new files were still untracked.

The descriptive primary estimates were not recomputed. Their recorded subject-equal post-initial radius, shuffle contrast, and early/late rows remain the files already in `outputs/tables/metrics_summary.csv`.
