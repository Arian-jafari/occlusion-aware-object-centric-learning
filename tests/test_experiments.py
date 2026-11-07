from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from occlusion_association.calibration import (
    fit_logistic_calibrator,
    generate_calibration_candidates,
)
from occlusion_association.config import load_config
from occlusion_association.dataset import DAVISDataset
from occlusion_association.experiments import run_experiment_suite
from occlusion_association.feature_cache import FeatureCache
from occlusion_association.features import build_preprocessing_descriptor, source_content_digests
from occlusion_association.result_io import export_experiment_results


def create_synthetic_davis(root: Path) -> DAVISDataset:
    split_dir = root / "ImageSets" / "2017"
    split_dir.mkdir(parents=True)
    (split_dir / "train.txt").write_text("train_sequence\n", encoding="utf-8")
    (split_dir / "val.txt").write_text("validation_sequence\n", encoding="utf-8")
    for sequence_name in ("train_sequence", "validation_sequence"):
        image_dir = root / "JPEGImages" / "480p" / sequence_name
        annotation_dir = root / "Annotations" / "480p" / sequence_name
        image_dir.mkdir(parents=True)
        annotation_dir.mkdir(parents=True)
        for frame_index in range(2):
            mask = np.zeros((14, 28), dtype=np.uint8)
            mask[:, :14] = 1
            mask[:, 14:] = 2
            frame_name = f"{frame_index:05d}"
            Image.new("RGB", (28, 14), (30, 60, 90)).save(image_dir / f"{frame_name}.jpg")
            Image.fromarray(mask).save(annotation_dir / f"{frame_name}.png")
    return DAVISDataset(root, split="all")


def test_tiny_synthetic_end_to_end_experiment_runs_without_dataset_or_gpu(
    tmp_path: Path,
) -> None:
    repository_root = Path(__file__).resolve().parents[1]
    config = load_config(repository_root / "configs" / "default.yaml").to_dict()
    config["features"]["cache_dir"] = str(tmp_path / "feature_cache")
    config["evaluation"]["memory_gaps"] = [5, 30]
    config["evaluation"]["confidence_thresholds"] = [0.25, 0.5, 0.75]
    dataset = create_synthetic_davis(tmp_path / "DAVIS")
    cache = FeatureCache(config["features"]["cache_dir"])
    backbone = f"{config['features']['torch_hub_repository']}:{config['features']['backbone']}"
    model_revision = "synthetic-checkpoint-v1"
    cache.record_model_revision(backbone, model_revision)
    preprocessing = build_preprocessing_descriptor(
        int(config["features"]["input_size"]),
        device=str(config["features"]["device"]),
        use_mixed_precision=bool(config["features"]["use_mixed_precision"]),
    )
    for sequence_name in dataset.sequence_names:
        for frame in dataset.iter_frames(sequence_name):
            image_digest, mask_digest = source_content_digests(frame.image, frame.mask)
            for object_id, vector in (
                (1, np.array([1.0, 0.0], dtype=np.float32)),
                (2, np.array([0.0, 1.0], dtype=np.float32)),
            ):
                key = cache.make_key(
                    sequence=sequence_name,
                    frame=frame.frame_name,
                    object_id=object_id,
                    backbone=backbone,
                    preprocessing=preprocessing,
                    model_revision=model_revision,
                    source_image_digest=image_digest,
                    source_mask_digest=mask_digest,
                )
                cache.put(key, vector)

    train_candidates = generate_calibration_candidates(
        [("train_sequence", dataset.iter_frames("train_sequence"))],
        cache,
        backbone_name=backbone,
        preprocessing=preprocessing,
        max_gap=int(config["memory"]["max_gap"]),
        appearance_history_size=int(config["memory"]["appearance_history_size"]),
    )
    calibrator = fit_logistic_calibrator(
        train_candidates,
        training_sequences=["train_sequence"],
        validation_sequences=["validation_sequence"],
        random_state=int(config["calibration"]["random_state"]),
    )
    calibrator.metadata["validation_sequences_checked"] = ["validation_sequence"]
    calibrator.metadata["max_memory_gap"] = int(config["memory"]["max_gap"])
    result = run_experiment_suite(
        dataset,
        cache,
        config,
        train_sequence_names=["train_sequence"],
        validation_sequence_names=["validation_sequence"],
        training_candidates=train_candidates,
        fitted_calibrator=calibrator,
    )

    assert {row["method"] for row in result["metrics"]} == {
        "cosine_forced",
        "cosine_threshold",
        "motion_appearance",
        "calibrated_selective",
    }
    assert len(result["appearance_ablation"]) == 3
    assert [row["max_memory_gap"] for row in result["memory_duration_ablation"]] == [5, 30]
    assert result["calibration_metrics"]["n_candidates"] == 4
    assert result["calibration_metrics"]["evaluation_sequences"] == ["validation_sequence"]
    assert "brier_score" in result["calibration_metrics"]
    assert "ece" in result["calibration_metrics"]
    assert result["precision_coverage_curve_level"] == "candidate_pair"
    assert result["assignment_precision_coverage_curve"]
    assert any(
        row["outcome"] == "no_candidate" for row in result["detection_decisions"]
    )
    assert all("pair_outcome" in row for row in result["association_predictions"])
    calibrated_rows = [
        row for row in result["per_sequence_metrics"]
        if row["method"] == "calibrated_selective"
    ]
    assert calibrated_rows[0]["calibration_candidate_pairs"] == 4
    assert "brier_score" in calibrated_rows[0]
    assert result["sequence_split"] == {
        "train": ["train_sequence"],
        "validation": ["validation_sequence"],
    }
    assert result["calibrator"].metadata["validation_sequences_used_for_fit"] == []
    assert result["calibrator"].metadata["candidate_generation_policy"].startswith(
        "cosine_threshold;"
    )

    output = tmp_path / "synthetic_experiment"
    export_experiment_results(result, output, make_plots=True)

    assert (output / "metrics.csv").is_file()
    assert (output / "per_sequence_metrics.csv").is_file()
    assert (output / "association_predictions.csv").is_file()
    assert (output / "detection_decisions.csv").is_file()
    assert (output / "training_calibration_candidates.csv").is_file()
    assert (output / "assignment_precision_coverage_curve.csv").is_file()
    assert (output / "calibrator.joblib").is_file()
    assert (output / "experiment_summary.md").is_file()
    assert (output / "figures" / "reliability_diagram.png").is_file()
    assert (output / "figures" / "identity_switches.pdf").is_file()


def test_experiment_runner_rejects_validation_sequences_as_calibration_training(
    tmp_path: Path,
) -> None:
    repository_root = Path(__file__).resolve().parents[1]
    config = load_config(repository_root / "configs" / "default.yaml").to_dict()
    dataset = create_synthetic_davis(tmp_path / "DAVIS")

    import pytest

    with pytest.raises(ValueError, match="official training split"):
        run_experiment_suite(
            dataset,
            FeatureCache(tmp_path / "unused_cache"),
            config,
            train_sequence_names=["validation_sequence"],
            validation_sequence_names=[],
        )
