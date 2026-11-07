from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from occlusion_association.calibration import (
    CalibrationCandidates,
    CalibrationDataError,
    fit_logistic_calibrator,
    generate_calibration_candidates,
)
from occlusion_association.dataset import FrameData, SequenceData
from occlusion_association.feature_cache import FeatureCache
from occlusion_association.features import source_content_digests


def make_training_sequence(tmp_path: Path) -> tuple[SequenceData, FeatureCache]:
    cache = FeatureCache(tmp_path / "features")
    sequence = SequenceData(name="train_sequence")
    preprocessing = {"input_size": 14}
    backbone = "test:patch-model"
    cache.record_model_revision(backbone, "test-checkpoint-v1")
    for frame_index in range(2):
        mask = np.zeros((14, 28), dtype=np.uint8)
        mask[:, :14] = 1
        mask[:, 14:] = 2
        frame_name = f"{frame_index:05d}"
        frame = FrameData(
            sequence="train_sequence",
            frame_name=frame_name,
            image=Image.new("RGB", (28, 14)),
            mask=mask,
            object_ids=(1, 2),
            image_path=tmp_path / f"{frame_name}.jpg",
            annotation_path=tmp_path / f"{frame_name}.png",
        )
        sequence.frames.append(frame)
        image_digest, mask_digest = source_content_digests(frame.image, frame.mask)
        for object_id, embedding in (
            (1, np.array([1.0, 0.0], dtype=np.float32)),
            (2, np.array([0.0, 1.0], dtype=np.float32)),
        ):
            key = cache.make_key(
                sequence="train_sequence",
                frame=frame_name,
                object_id=object_id,
                backbone=backbone,
                preprocessing=preprocessing,
                model_revision="test-checkpoint-v1",
                source_image_digest=image_digest,
                source_mask_digest=mask_digest,
            )
            cache.put(key, embedding)
    return sequence, cache


def test_training_candidate_generation_uses_only_training_cache_and_binary_targets(
    tmp_path: Path,
) -> None:
    sequence, cache = make_training_sequence(tmp_path)

    candidates = generate_calibration_candidates(
        [sequence],
        cache,
        backbone_name="test:patch-model",
        preprocessing={"input_size": 14},
        max_gap=30,
    )

    assert candidates.features.shape == (4, 6)
    assert sorted(candidates.labels.tolist()) == [0, 0, 1, 1]
    assert candidates.sequence_names == {"train_sequence"}
    assert candidates.generation_policy == "cosine_threshold;cosine_threshold=0.5"
    assert "ground_truth_id" not in candidates.metadata[0]


def test_logistic_calibrator_fits_and_returns_probability_rows(tmp_path: Path) -> None:
    sequence, cache = make_training_sequence(tmp_path)
    candidates = generate_calibration_candidates(
        [sequence],
        cache,
        backbone_name="test:patch-model",
        preprocessing={"input_size": 14},
    )

    calibrator = fit_logistic_calibrator(
        candidates,
        training_sequences=["train_sequence"],
        validation_sequences=["held_out"],
        random_state=4,
    )
    probabilities = calibrator.predict_proba(candidates.features)

    assert probabilities.shape == (4, 2)
    assert np.all((probabilities >= 0.0) & (probabilities <= 1.0))
    assert calibrator.metadata["validation_sequences_used_for_fit"] == []
    assert calibrator.metadata["n_positive_candidates"] == 2
    assert calibrator.metadata["feature_model_revision"] == "test-checkpoint-v1"
    assert calibrator.metadata["class_weight"] is None
    assert calibrator.estimator.named_steps["logistic_regression"].class_weight is None
    assert calibrator.metadata["candidate_generation_policy"] == (
        "cosine_threshold;cosine_threshold=0.5"
    )


def test_calibrator_rejects_train_validation_overlap_and_foreign_examples() -> None:
    candidates = CalibrationCandidates(
        features=np.ones((2, 6), dtype=np.float32),
        labels=np.array([0, 1]),
        metadata=[{"sequence": "train"}, {"sequence": "train"}],
    )
    with pytest.raises(CalibrationDataError, match="overlap"):
        fit_logistic_calibrator(
            candidates,
            training_sequences=["train"],
            validation_sequences=["train"],
        )
    with pytest.raises(CalibrationDataError, match="outside the training split"):
        fit_logistic_calibrator(
            candidates,
            training_sequences=["other"],
            validation_sequences=["held_out"],
        )


def test_calibrator_rejects_test_sequence_overlap_with_training() -> None:
    candidates = CalibrationCandidates(
        features=np.ones((2, 6), dtype=np.float32),
        labels=np.array([0, 1]),
        metadata=[{"sequence": "train"}, {"sequence": "train"}],
    )

    with pytest.raises(CalibrationDataError, match="pairwise disjoint"):
        fit_logistic_calibrator(
            candidates,
            training_sequences=["train", "reserved_test"],
            validation_sequences=["validation"],
            test_sequences=["reserved_test"],
        )
