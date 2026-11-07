#!/usr/bin/env python3
"""Extract frozen DINOv2 mask-pooled features and persist them in the cache."""

from __future__ import annotations

import argparse
import json
import sys
from itertools import islice
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from occlusion_association.config import ConfigError, load_config  # noqa: E402
from occlusion_association.dataset import DAVISDataset, DatasetLayoutError, SequenceData  # noqa: E402
from occlusion_association.feature_cache import FeatureCache  # noqa: E402
from occlusion_association.features import (  # noqa: E402
    DINOv2FeatureExtractor,
    extract_sequence_features,
)
from occlusion_association.utils import select_smoke_sequence_names  # noqa: E402
from occlusion_association.utils import set_random_seeds  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=REPOSITORY_ROOT / "configs" / "default.yaml")
    parser.add_argument("--root", type=Path, help="Override dataset.root for this run.")
    parser.add_argument("--split", choices=("train", "val", "all"), default="all")
    parser.add_argument("--batch-size", type=int, help="Override feature extraction batch size.")
    parser.add_argument("--device", help="Torch device, for example auto, cuda, or cpu.")
    parser.add_argument("--smoke-test", action="store_true", help="Use configured sequence/frame limits.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    overrides = {}
    if args.root:
        overrides["dataset"] = {"root": str(args.root)}
    if args.batch_size:
        overrides.setdefault("features", {})["batch_size"] = args.batch_size
    if args.device:
        overrides.setdefault("features", {})["device"] = args.device

    try:
        config = load_config(args.config, overrides=overrides or None)
        set_random_seeds(int(config.experiment["random_seed"]))
        dataset = DAVISDataset(
            config.dataset_root,
            split=args.split,
            train_split=str(config.dataset["train_split"]),
            validation_split=str(config.dataset["validation_split"]),
        )
        extractor = DINOv2FeatureExtractor(
            backbone_name=str(config.features["backbone"]),
            repository=str(config.features["torch_hub_repository"]),
            input_size=int(config.features["input_size"]),
            device=str(config.features["device"]),
            use_mixed_precision=bool(config.features["use_mixed_precision"]),
        )
    except (ConfigError, DatasetLayoutError, OSError, ValueError, RuntimeError) as exc:
        print(f"Feature extraction could not start: {exc}", file=sys.stderr)
        return 2

    cache = FeatureCache(config.features["cache_dir"])
    selected_names = dataset.sequence_names
    frame_limit = None
    if args.smoke_test or bool(config.experiment["smoke_test"]):
        per_split = int(config.experiment["smoke_test_sequences"])
        if args.split == "all":
            selected_names = select_smoke_sequence_names(
                dataset, dataset.train_sequences, per_split
            ) + select_smoke_sequence_names(
                dataset, dataset.validation_sequences, per_split
            )
        else:
            selected_names = select_smoke_sequence_names(
                dataset, selected_names, per_split
            )
        frame_limit = int(config.experiment["smoke_test_frames"])
        print(f"SMOKE TEST active: sequences={len(selected_names)}, max_frames={frame_limit}")
    else:
        print(f"FULL EXTRACTION active: split={args.split}, sequences={len(selected_names)}")

    failed_sequences = 0
    for name in selected_names:
        issues = []
        frame_iterator = dataset.iter_frames(name, issue_sink=issues)
        if frame_limit is not None:
            frame_iterator = islice(frame_iterator, frame_limit)
        batch = []
        summary = {
            "sequence": name,
            "objects_requested": 0,
            "cache_hits": 0,
            "features_written": 0,
            "invalid_objects": [],
        }
        sequence_failed = False
        try:
            for frame in frame_iterator:
                batch.append(frame)
                if len(batch) >= int(config.features["batch_size"]):
                    chunk_summary = extract_sequence_features(
                        SequenceData(name=name, frames=batch),
                        extractor,
                        cache,
                        batch_size=int(config.features["batch_size"]),
                    )
                    summary["objects_requested"] += chunk_summary.objects_requested
                    summary["cache_hits"] += chunk_summary.cache_hits
                    summary["features_written"] += chunk_summary.features_written
                    summary["invalid_objects"].extend(chunk_summary.invalid_objects)
                    batch = []
            if batch:
                chunk_summary = extract_sequence_features(
                    SequenceData(name=name, frames=batch),
                    extractor,
                    cache,
                    batch_size=int(config.features["batch_size"]),
                )
                summary["objects_requested"] += chunk_summary.objects_requested
                summary["cache_hits"] += chunk_summary.cache_hits
                summary["features_written"] += chunk_summary.features_written
                summary["invalid_objects"].extend(chunk_summary.invalid_objects)
        except (RuntimeError, ValueError, OSError) as exc:
            failed_sequences += 1
            sequence_failed = True
            print(f"ERROR {name}: {exc}", file=sys.stderr)
        if not sequence_failed:
            print(json.dumps(summary, sort_keys=True))
        for issue in issues:
            print(f"[{issue.severity}:{issue.code}] {issue.sequence}: {issue.message}")
    print(
        json.dumps(
            {
                "cache_directory": str(cache.directory),
                "sequences_processed": len(selected_names) - failed_sequences,
                "sequences_failed": failed_sequences,
                "backbone": extractor.backbone_name,
                "device": str(extractor.device),
            },
            indent=2,
        )
    )
    return 1 if failed_sequences else 0


if __name__ == "__main__":
    raise SystemExit(main())
