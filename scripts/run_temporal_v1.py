"""Run the versioned temporal predictability experiment.

Writes outputs/temporal_v1/. Does not rewrite outputs/figures or outputs/tables.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from humangazelab.temporal_run import run


def main() -> None:
    manifest = run(ROOT, "configs/temporal_v1.yaml")
    keys = (
        "experiment_id",
        "n_scanpaths",
        "n_primary_fixations",
        "n_valid_primary_fixations",
        "selected_sigma_center_px",
        "selected_sigma_history_px",
        "selected_sigma_loo_px",
        "selected_lambda_center_history",
        "selected_lambda_loo_history",
        "fixation_sha256",
        "hypothesis_adopted",
    )
    print(json.dumps({key: manifest[key] for key in keys}, indent=2))


if __name__ == "__main__":
    main()
