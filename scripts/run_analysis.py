#!/usr/bin/env python3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from humangazelab.pipeline import run


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Compute COCO-FreeView descriptive statistics, tables, and figures.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--config", default="configs/analysis.yaml")
    args = parser.parse_args()
    run(args.root, args.config)


if __name__ == "__main__":
    main()
