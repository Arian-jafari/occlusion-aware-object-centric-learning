#!/usr/bin/env python3
"""Print a structured inspection report for DAVIS 2017 TrainVal."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from occlusion_association.config import ConfigError, load_config  # noqa: E402
from occlusion_association.dataset import DAVISDataset, DatasetLayoutError  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect DAVIS 2017 frame/annotation pairs and instance masks."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=REPOSITORY_ROOT / "configs" / "default.yaml",
        help="Central YAML configuration file.",
    )
    parser.add_argument("--root", type=Path, help="Override dataset.root for this run.")
    parser.add_argument(
        "--split",
        choices=("train", "val", "all"),
        default="all",
        help="Official split to inspect (default: all).",
    )
    parser.add_argument(
        "--json", action="store_true", help="Print the full report as JSON."
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        overrides = {"dataset": {"root": str(args.root)}} if args.root else None
        config = load_config(args.config, overrides=overrides)
        dataset = DAVISDataset(
            root=config.dataset_root,
            split=args.split,
            train_split=str(config.dataset["train_split"]),
            validation_split=str(config.dataset["validation_split"]),
        )
        report = dataset.inspect()
    except (ConfigError, DatasetLayoutError, OSError, ValueError) as exc:
        print(f"Dataset inspection failed: {exc}", file=sys.stderr)
        return 2

    payload = report.to_dict()
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(f"DAVIS root: {report.dataset_root}")
        print(f"Requested split: {report.split}")
        print(
            "Official sequences: "
            f"train={report.train_sequences}, val={report.validation_sequences}, "
            f"requested={report.requested_sequences}"
        )
        for sequence in report.inspected_sequences:
            status = "SKIPPED" if sequence.skipped else "ok"
            print(
                f"{sequence.sequence}: {status}; images={sequence.image_frames}, "
                f"annotations={sequence.annotation_frames}, matched={sequence.matched_frames}, "
                f"valid={sequence.valid_frames}, objects={sequence.unique_objects}, "
                f"image_dims={sequence.image_dimensions}, "
                f"annotation_dims={sequence.annotation_dimensions}"
            )
            for lifecycle in sequence.object_lifecycles:
                print(
                    "  object {object_id}: first={first_observed_frame}, "
                    "last={last_observed_frame}, observations={observed_frames}, "
                    "valid-frame gaps={gaps}".format(
                        object_id=lifecycle["object_id"],
                        first_observed_frame=lifecycle["first_observed_frame"],
                        last_observed_frame=lifecycle["last_observed_frame"],
                        observed_frames=lifecycle["observed_frames"],
                        gaps=len(lifecycle["missing_gaps"]),
                    )
                )
        print(f"Dataset loading errors: {len(report.loading_errors)}")
        for issue in report.loading_errors:
            location = "/".join(part for part in (issue.sequence, issue.frame) if part)
            print(f"  [{issue.severity}:{issue.code}] {location}: {issue.message}")

    return 1 if report.loading_errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
