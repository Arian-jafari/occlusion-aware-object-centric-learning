#!/usr/bin/env python3
"""Print aggregate metrics and held-out calibration for an experiment folder."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment_directory", type=Path)
    args = parser.parse_args()
    directory = args.experiment_directory.expanduser()
    metrics_path = directory / "metrics.csv"
    calibration_path = directory / "calibration_metrics.json"
    if not metrics_path.is_file():
        print(
            f"No metrics.csv found in {directory}. Run scripts/run_experiments.py first."
        )
        return 2
    metrics = pd.read_csv(metrics_path)
    columns = [
        name
        for name in (
            "method",
            "identity_switches",
            "track_fragmentations",
            "association_precision",
            "association_recall",
            "association_coverage",
            "false_association_rate",
            "brier_score",
            "ece",
        )
        if name in metrics.columns
    ]
    print(metrics[columns].to_string(index=False))
    if calibration_path.is_file():
        report = json.loads(calibration_path.read_text(encoding="utf-8"))
        print(
            "\nHeld-out calibrated candidate pairs: "
            f"n={report.get('n_candidates')}, Brier={report.get('brier_score')}, "
            f"ECE={report.get('ece')}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
