"""Write a MIT1003 file inventory and an unscored split manifest.

Does not extract fixations and does not compute a gaze score.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from humangazelab.mit1003 import image_stems, inventory_zip, partition_stems
from humangazelab.protocolshift import assert_new_output

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "protocolshift_prep_v1"
ZIP = ROOT / "data" / "raw" / "external" / "mit1003_DATA.zip"


def main() -> None:
    assert_new_output(ROOT, OUT)
    if not ZIP.exists():
        raise SystemExit(f"missing {ZIP}; the archive is gitignored and must be downloaded separately")
    OUT.mkdir(parents=True, exist_ok=True)
    inventory = inventory_zip(ZIP)
    inventory["approved_for_confirmation"] = False
    inventory["geometry_conflict"] = [
        "ICCV 2009 paper: about two feet from a 19-inch 1280 by 1024 screen, 3 seconds.",
        "checkFixations.m header: 34 px/deg, 1024 by 768 pixels described as 40.5 by 30 cm at 75 cm.",
        "DeepGaze README: MIT1003 collected at 35 px/deg.",
    ]
    (OUT / "mit1003_inventory.json").write_text(json.dumps(inventory, indent=2) + "\n")
    rows = partition_stems(image_stems(ZIP))
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "mit1003_split_manifest.csv", index=False)
    counts = frame["partition"].value_counts().to_dict()
    print(json.dumps({"users_match_readme": inventory["users_match_readme"], "n_image_stems": inventory["n_image_stems"], "partitions": counts}, indent=2))


if __name__ == "__main__":
    main()
