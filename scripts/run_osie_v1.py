"""Run the within-OSIE temporal evaluation. Does not modify COCO-FreeView outputs."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from humangazelab.osie_run import run


def main() -> None:
    manifest = run(ROOT)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
