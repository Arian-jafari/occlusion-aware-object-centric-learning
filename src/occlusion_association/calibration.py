"""Training-only candidate generation and logistic association calibration."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .association import OnlineAssociator
from .association_features import FEATURE_NAMES
from .dataset import FrameData, SequenceData
from .feature_cache import FeatureCache
from .features import load_cached_frame_embeddings
from .memory import ObjectMemory
from .object_representation import build_object_detections


class CalibrationDataError(ValueError):
    """Raised when candidate data cannot support leakage-safe calibration."""


@dataclass
class CalibrationCandidates:
    """Pair-level feature matrix, correctness targets, and audit metadata."""

    features: np.ndarray
    labels: np.ndarray
    metadata: list[dict[str, Any]]

    @property
    def sequence_names(self) -> set[str]:
        return {str(item["sequence"]) for item in self.metadata}


@dataclass
class LogisticAssociationCalibrator:
    """Standardized logistic regression estimating pair correctness."""

    estimator: Pipeline
    metadata: dict[str, Any]

    @property
    def classes_(self):
        return self.estimator.classes_

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """Estimate class probabilities from association feature rows."""

        values = np.asarray(features, dtype=np.float32)
        if values.ndim == 1:
            values = values.reshape(1, -1)
        if values.ndim != 2 or values.shape[1] != len(FEATURE_NAMES):
            raise ValueError(
                f"Expected a 2D matrix with {len(FEATURE_NAMES)} features in {FEATURE_NAMES} order"
            )
        return self.estimator.predict_proba(values)

    def save(self, path: str | Path) -> tuple[Path, Path]:
        """Save estimator and human-readable audit metadata side by side."""

        import joblib

        model_path = Path(path).expanduser()
        if model_path.suffix != ".joblib":
            model_path = model_path.with_suffix(".joblib")
        model_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path = model_path.with_suffix(".json")
        joblib.dump(self.estimator, model_path)
        metadata_path.write_text(json.dumps(self.metadata, indent=2, sort_keys=True), encoding="utf-8")
        return model_path, metadata_path

    @classmethod
    def load(cls, path: str | Path) -> "LogisticAssociationCalibrator":
        """Load a previously saved calibrator and its audit metadata."""

        import joblib

        model_path = Path(path).expanduser()
        if model_path.suffix != ".joblib":
            model_path = model_path.with_suffix(".joblib")
        metadata_path = model_path.with_suffix(".json")
        if not model_path.is_file() or not metadata_path.is_file():
            raise FileNotFoundError(f"Calibrator files are missing: {model_path} and/or {metadata_path}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if tuple(metadata.get("feature_names", ())) != FEATURE_NAMES:
            raise CalibrationDataError("Saved calibrator feature schema does not match this project")
        return cls(joblib.load(model_path), metadata)


def _frame_index(frame_name: str, ordinal: int) -> int:
    """Use the DAVIS numeric frame stem when available, else sequence order."""

    return int(frame_name) if frame_name.isdigit() else ordinal


def generate_calibration_candidates(
    sequences: Iterable[SequenceData | tuple[str, Iterable[FrameData]]],
    cache: FeatureCache,
    *,
    backbone_name: str,
    preprocessing: dict[str, Any],
    max_gap: int = 30,
    appearance_history_size: int = 5,
) -> CalibrationCandidates:
    """Generate pair examples by replaying training sequences with forced cosine.

    Each training sequence is replayed with forced-cosine association to create
    realistic predicted-memory candidate states without a learned calibrator.
    The binary target follows the stated task definition: a candidate is
    positive exactly when the memory's prior ground-truth owner is the current
    detection's ground-truth object. Ground-truth labels are used only for that
    target and external ownership bookkeeping; they are not pair features or
    inference-time decisions.
    """

    feature_rows: list[np.ndarray] = []
    labels: list[int] = []
    metadata: list[dict[str, Any]] = []
    for sequence_item in sequences:
        if isinstance(sequence_item, SequenceData):
            sequence_name, sequence_frames = sequence_item.name, iter(sequence_item.frames)
        else:
            sequence_name, frame_iterable = sequence_item
            sequence_frames = iter(frame_iterable)
        memory = ObjectMemory(max_gap=max_gap, appearance_history_size=appearance_history_size)
        online = OnlineAssociator(memory, "cosine_forced")
        ground_truth_owner: dict[int, int] = {}
        for ordinal, frame in enumerate(sequence_frames):
            embeddings, missing = load_cached_frame_embeddings(
                frame,
                cache,
                backbone_name=backbone_name,
                preprocessing=preprocessing,
            )
            if missing:
                raise CalibrationDataError(
                    f"Missing cached features for {sequence_name}/{frame.frame_name}, "
                    f"objects {missing}; run feature extraction first"
                )
            detections, gt_by_detection, invalid = build_object_detections(
                frame.mask, embeddings
            )
            if invalid:
                raise CalibrationDataError(
                    f"Invalid object embeddings in {sequence_name}/{frame.frame_name}: {invalid}"
                )
            result = online.step(
                detections,
                image_size=frame.image.size,
                frame_index=_frame_index(frame.frame_name, ordinal),
            )
            for (predicted_id, detection_id), feature in result.pair_features.items():
                gt_id = gt_by_detection[detection_id]
                owner = ground_truth_owner.get(predicted_id)
                feature_rows.append(feature.to_array())
                labels.append(int(owner is not None and owner == gt_id))
                metadata.append(
                    {
                        "sequence": sequence_name,
                        "frame": frame.frame_name,
                        "predicted_identity_id": predicted_id,
                        "detection_id": detection_id,
                    }
                )
            for expired_id in result.expired_identity_ids:
                ground_truth_owner.pop(expired_id, None)
            for detection_id, predicted_id in result.predicted_identity_by_detection.items():
                ground_truth_owner[predicted_id] = gt_by_detection[detection_id]

    matrix = (
        np.stack(feature_rows).astype(np.float32)
        if feature_rows
        else np.empty((0, len(FEATURE_NAMES)), dtype=np.float32)
    )
    return CalibrationCandidates(matrix, np.asarray(labels, dtype=np.int64), metadata)


def fit_logistic_calibrator(
    candidates: CalibrationCandidates,
    *,
    training_sequences: Sequence[str],
    validation_sequences: Sequence[str] = (),
    class_weight: str | dict[int, float] | None = "balanced",
    max_iter: int = 1000,
    random_state: int = 17,
) -> LogisticAssociationCalibrator:
    """Fit a probability model only after checking train/validation isolation."""

    train_names = set(training_sequences)
    validation_names = set(validation_sequences)
    overlap = train_names & validation_names
    if overlap:
        raise CalibrationDataError(
            f"Training and validation sequence splits overlap: {sorted(overlap)}"
        )
    unknown = candidates.sequence_names - train_names
    if unknown:
        raise CalibrationDataError(
            "Calibration examples contain sequences outside the training split: "
            f"{sorted(unknown)}"
        )
    if candidates.features.ndim != 2 or candidates.features.shape[1] != len(FEATURE_NAMES):
        raise CalibrationDataError("Candidate feature matrix has an invalid shape")
    if candidates.features.shape[0] != candidates.labels.shape[0]:
        raise CalibrationDataError("Candidate feature and label counts do not match")
    classes = np.unique(candidates.labels)
    if not np.array_equal(classes, np.array([0, 1])):
        raise CalibrationDataError(
            f"Calibration requires both positive and negative candidates; got classes {classes.tolist()}"
        )

    estimator = Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "logistic_regression",
                LogisticRegression(
                    class_weight=class_weight,
                    max_iter=max_iter,
                    random_state=random_state,
                ),
            ),
        ]
    )
    estimator.fit(candidates.features, candidates.labels)
    metadata = {
        "model": "standardized_logistic_regression",
        "feature_names": list(FEATURE_NAMES),
        "training_sequences": sorted(train_names),
        "validation_sequences_used_for_fit": [],
        "n_training_candidates": int(candidates.labels.size),
        "n_positive_candidates": int(np.sum(candidates.labels == 1)),
        "n_negative_candidates": int(np.sum(candidates.labels == 0)),
        "class_weight": class_weight,
        "max_iter": int(max_iter),
        "random_state": int(random_state),
    }
    return LogisticAssociationCalibrator(estimator, metadata)
