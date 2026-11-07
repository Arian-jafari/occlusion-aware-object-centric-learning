#!/usr/bin/env python3
"""Run the main DAVIS comparison, ablations, metrics, and plots."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from occlusion_association.config import ConfigError, load_config  # noqa: E402
from occlusion_association.dataset import DAVISDataset, DatasetLayoutError  # noqa: E402
from occlusion_association.experiments import run_experiment_suite  # noqa: E402
from occlusion_association.feature_cache import FeatureCache  # noqa: E402
from occlusion_association.result_io import export_experiment_results  # noqa: E402
from occlusion_association.utils import (  # noqa: E402
    create_experiment_directory,
    runtime_metadata,
    select_smoke_sequence_names,
    set_random_seeds,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=REPOSITORY_ROOT / "configs" / "default.yaml")
    parser.add_argument("--root", type=Path, help="Override dataset.root for this run.")
    parser.add_argument("--smoke-test", action="store_true", help="Limit sequences and frames for a quick run.")
    parser.add_argument("--experiment-name", help="Optional fixed result directory name.")
    parser.add_argument("--overwrite", action="store_true", help="Replace the named experiment directory if it exists.")
    parser.add_argument("--no-plots", action="store_true", help="Export result tables without generating figures.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    overrides = {"dataset": {"root": str(args.root)}} if args.root else None
    try:
        config = load_config(args.config, overrides=overrides)
        dataset = DAVISDataset(
            config.dataset_root,
            split="all",
            train_split=str(config.dataset["train_split"]),
            validation_split=str(config.dataset["validation_split"]),
        )
    except (ConfigError, DatasetLayoutError, OSError, ValueError) as exc:
        print(f"Experiment setup failed: {exc}", file=sys.stderr)
        return 2

    smoke = args.smoke_test or bool(config.experiment["smoke_test"])
    train_names = dataset.train_sequences
    validation_names = dataset.validation_sequences
    frame_limit = None
    if smoke:
        sequence_limit = int(config.experiment["smoke_test_sequences"])
        train_names = select_smoke_sequence_names(dataset, train_names, sequence_limit)
        validation_names = select_smoke_sequence_names(dataset, validation_names, sequence_limit)
        frame_limit = int(config.experiment["smoke_test_frames"])
        print(
            f"SMOKE TEST ACTIVE: train={len(train_names)} sequences, "
            f"validation={len(validation_names)} sequences, max {frame_limit} frames each"
        )
    else:
        print(
            f"FULL EVALUATION ACTIVE: train={len(train_names)} sequences for calibration, "
            f"validation={len(validation_names)} held-out sequences"
        )

    set_random_seeds(int(config.experiment["random_seed"]))
    try:
        experiment_dir = create_experiment_directory(
            config.experiment["output_dir"],
            name=args.experiment_name,
            overwrite=args.overwrite,
        )
    except (OSError, ValueError, FileExistsError) as exc:
        print(f"Could not create experiment directory: {exc}", file=sys.stderr)
        return 2
    (experiment_dir / "logs").mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=experiment_dir / "logs" / "run.log",
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    metadata = runtime_metadata(REPOSITORY_ROOT)
    metadata.update(
        {
            "config_path": str(config.source_path),
            "dataset_root": str(dataset.root),
            "dataset_split": {"train": train_names, "validation": validation_names},
            "backbone": f"{config.features['torch_hub_repository']}:{config.features['backbone']}",
            "random_seed": int(config.experiment["random_seed"]),
            "smoke_test": smoke,
            "smoke_test_frame_limit": frame_limit,
            "feature_cache_directory": str(config.features["cache_dir"]),
            "experiment_parameters": config.to_dict(),
        }
    )
    write_json(experiment_dir / "metadata.json", metadata)

    try:
        results = run_experiment_suite(
            dataset,
            FeatureCache(config.features["cache_dir"]),
            config.to_dict(),
            train_sequence_names=train_names,
            validation_sequence_names=validation_names,
            frame_limit=frame_limit,
        )
        results["smoke_test"] = smoke
        export_experiment_results(
            results,
            experiment_dir,
            config=config.to_dict(),
            metadata=metadata,
            make_plots=not args.no_plots,
        )
        logging.info("Experiment completed in %s", experiment_dir)
    except Exception as exc:
        logging.exception("Experiment failed: %s", exc)
        print(f"Experiment failed: {exc}. See {experiment_dir / 'logs' / 'run.log'}", file=sys.stderr)
        return 1

    print(f"Experiment results saved to: {experiment_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
