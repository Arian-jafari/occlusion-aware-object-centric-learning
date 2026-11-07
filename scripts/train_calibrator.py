#!/usr/bin/env python3
"""Generate training candidates and fit the train-only logistic calibrator."""

from __future__ import annotations

import argparse
import json
import sys
from itertools import islice
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from occlusion_association.calibration import (  # noqa: E402
    CalibrationDataError,
    fit_logistic_calibrator,
    generate_calibration_candidates,
)
from occlusion_association.config import ConfigError, load_config  # noqa: E402
from occlusion_association.dataset import DAVISDataset, DatasetLayoutError  # noqa: E402
from occlusion_association.feature_cache import FeatureCache  # noqa: E402
from occlusion_association.features import build_preprocessing_descriptor  # noqa: E402
from occlusion_association.utils import (  # noqa: E402
    select_smoke_sequence_names,
    set_random_seeds,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=REPOSITORY_ROOT / "configs" / "default.yaml")
    parser.add_argument("--root", type=Path, help="Override dataset.root for this run.")
    parser.add_argument("--output", type=Path, help="Calibrator artifact path (default: configured output directory).")
    parser.add_argument("--smoke-test", action="store_true", help="Limit train and validation sequences/frames.")
    parser.add_argument("--overwrite", action="store_true", help="Replace the explicitly selected calibrator files.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    overrides = {"dataset": {"root": str(args.root)}} if args.root else None
    try:
        config = load_config(args.config, overrides=overrides)
        set_random_seeds(int(config.experiment["random_seed"]))
        dataset = DAVISDataset(
            config.dataset_root,
            split="all",
            train_split=str(config.dataset["train_split"]),
            validation_split=str(config.dataset["validation_split"]),
        )
    except (ConfigError, DatasetLayoutError, OSError, ValueError) as exc:
        print(f"Could not prepare calibration data: {exc}", file=sys.stderr)
        return 2

    smoke = args.smoke_test or bool(config.experiment["smoke_test"])
    train_names = dataset.train_sequences
    validation_names = dataset.validation_sequences
    frame_limit = None
    if smoke:
        count = int(config.experiment["smoke_test_sequences"])
        train_names = select_smoke_sequence_names(dataset, train_names, count)
        validation_names = select_smoke_sequence_names(dataset, validation_names, count)
        frame_limit = int(config.experiment["smoke_test_frames"])
        print(
            f"SMOKE TEST active: train={train_names}, validation={validation_names}, "
            f"max_frames={frame_limit}"
        )
    else:
        print(
            f"FULL CALIBRATION active: train sequences={len(train_names)}, "
            f"held-out validation sequences={len(validation_names)}"
        )

    cache = FeatureCache(config.features["cache_dir"])

    def training_streams():
        for sequence_name in train_names:
            frames = dataset.iter_frames(sequence_name)
            if frame_limit is not None:
                frames = islice(frames, frame_limit)
            yield sequence_name, frames

    try:
        candidates = generate_calibration_candidates(
            training_streams(),
            cache,
            backbone_name=(
                f"{config.features['torch_hub_repository']}:{config.features['backbone']}"
            ),
            preprocessing=build_preprocessing_descriptor(
                int(config.features["input_size"]),
                device=str(config.features["device"]),
                use_mixed_precision=bool(config.features["use_mixed_precision"]),
            ),
            max_gap=int(config.memory["max_gap"]),
            appearance_history_size=int(config.memory["appearance_history_size"]),
        )
        calibrator = fit_logistic_calibrator(
            candidates,
            training_sequences=train_names,
            validation_sequences=validation_names,
            class_weight=config.calibration["class_weight"],
            max_iter=int(config.calibration["max_iter"]),
            random_state=int(config.calibration["random_state"]),
        )
    except (CalibrationDataError, OSError, ValueError) as exc:
        print(f"Calibration failed: {exc}", file=sys.stderr)
        return 2

    calibrator.metadata["validation_sequences_checked"] = list(validation_names)
    calibrator.metadata["max_memory_gap"] = int(config.memory["max_gap"])
    calibrator.metadata["smoke_test"] = smoke
    output = args.output or (
        Path(config.experiment["output_dir"])
        / "calibrators"
        / f"logistic_gap_{int(config.memory['max_gap'])}.joblib"
    )
    model_path = output if output.suffix == ".joblib" else output.with_suffix(".joblib")
    if (model_path.exists() or model_path.with_suffix(".json").exists()) and not args.overwrite:
        print(
            f"Calibrator output already exists: {model_path}; choose another --output or pass --overwrite",
            file=sys.stderr,
        )
        return 2
    try:
        saved_model, saved_metadata = calibrator.save(model_path)
    except OSError as exc:
        print(f"Could not save calibrator: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "model_path": str(saved_model),
                "metadata_path": str(saved_metadata),
                "training_candidates": int(candidates.labels.size),
                "positive_candidates": int((candidates.labels == 1).sum()),
                "negative_candidates": int((candidates.labels == 0).sum()),
                "training_sequences": train_names,
                "validation_sequences_used_for_fit": [],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
