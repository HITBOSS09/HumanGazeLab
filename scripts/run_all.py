#!/usr/bin/env python3
"""Download the public release, extract the image subset, and run the analysis."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from humangazelab.download import main as download_main
from humangazelab.pipeline import run
from humangazelab.prepare import main as prepare_main


def main():
    download_main([])
    prepare_main([])
    run(".", "configs/analysis.yaml")


if __name__ == "__main__":
    main()
