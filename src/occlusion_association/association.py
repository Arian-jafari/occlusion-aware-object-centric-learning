"""Hungarian association baselines and calibrated selective decisions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping

import numpy as np
from scipy.optimize import linear_sum_assignment

from .association_features import (
    FEATURE_NAMES,
    AssociationFeatureVector,
    compute_association_features,
)
from .memory import ObjectMemory
from .object_representation import ObjectDetection


AssociationMethod = Literal[
    "cosine_forced",
    "cosine_threshold",
    "motion_appearance",
    "calibrated_selective",
]


@dataclass(frozen=True)
class AssignedPair:
    """One Hungarian identity/detection assignment and its acceptance status."""

    predicted_identity_id: int
    detection_id: int
    score: float
    accepted: bool


@dataclass
class AssociationDecision:
    """Selected pair assignments plus explicit rejections and unmatched items."""

    method: str
    assignments: list[AssignedPair]
    accepted: list[AssignedPair]
    rejected: list[AssignedPair]
    unmatched_detection_ids: list[int]
    unmatched_identity_ids: list[int]
    candidate_scores: dict[tuple[int, int], float]


@dataclass
class FrameAssociationResult:
    """Online association outcome, including new predicted IDs per detection."""

    frame_index: int
    decision: AssociationDecision
    predicted_identity_by_detection: dict[int, int]
    new_identity_by_detection: dict[int, int]
    expired_identity_ids: tuple[int, ...]
    pair_features: dict[tuple[int, int], AssociationFeatureVector]


def _hungarian_assign(
    score_map: Mapping[tuple[int, int], float],
    identity_ids: list[int],
    detection_ids: list[int],
) -> list[tuple[int, int]]:
    """Maximize a rectangular pair-score matrix with one-to-one constraints."""

    if not identity_ids or not detection_ids:
        return []
    matrix = np.asarray(
        [[score_map[(identity_id, detection_id)] for detection_id in detection_ids]
         for identity_id in identity_ids],
        dtype=np.float64,
    )
    row_indices, column_indices = linear_sum_assignment(-matrix)
    return [
        (identity_ids[row], detection_ids[column])
        for row, column in zip(row_indices.tolist(), column_indices.tolist())
    ]


def _calibrated_probabilities(
    calibrator: Any,
    feature_map: Mapping[tuple[int, int], AssociationFeatureVector],
) -> dict[tuple[int, int], float]:
    if calibrator is None:
        raise ValueError("calibrated_selective requires a fitted calibrator")
    pairs = list(feature_map)
    if not pairs:
        return {}
    matrix = np.stack([feature_map[pair].to_array() for pair in pairs])
    probabilities = np.asarray(calibrator.predict_proba(matrix), dtype=np.float64)
    if probabilities.ndim != 2 or probabilities.shape[0] != len(pairs):
        raise ValueError("calibrator.predict_proba returned an invalid probability matrix")
    classes = list(getattr(calibrator, "classes_", [0, 1]))
    if 1 not in classes:
        raise ValueError("Calibrator has no positive class (1)")
    positive_column = classes.index(1)
    if probabilities.shape[1] <= positive_column:
        raise ValueError("Calibrator probability matrix does not contain positive class 1")
    values = np.clip(probabilities[:, positive_column], 0.0, 1.0)
    return {pair: float(value) for pair, value in zip(pairs, values.tolist())}


def decide_associations(
    method: AssociationMethod,
    feature_map: Mapping[tuple[int, int], AssociationFeatureVector],
    identity_ids: list[int],
    detection_ids: list[int],
    *,
    cosine_threshold: float = 0.5,
    fixed_score_threshold: float = 0.5,
    confidence_threshold: float = 0.5,
    motion_weights: Mapping[str, float] | None = None,
    calibrator: Any | None = None,
) -> AssociationDecision:
    """Select pairs by Hungarian assignment, then apply explicit rejection.

    The feature calculation itself lives in ``association_features`` and is
    independent of these decision rules.
    """

    if method not in {
        "cosine_forced",
        "cosine_threshold",
        "motion_appearance",
        "calibrated_selective",
    }:
        raise ValueError(f"Unknown association method: {method}")
    expected_pairs = [(identity_id, detection_id) for identity_id in identity_ids for detection_id in detection_ids]
    absent = [pair for pair in expected_pairs if pair not in feature_map]
    if absent:
        raise ValueError(f"Missing association features for candidate pair {absent[0]}")

    if method in {"cosine_forced", "cosine_threshold"}:
        candidate_scores = {
            pair: feature.cosine_similarity for pair, feature in feature_map.items()
        }
    elif method == "motion_appearance":
        weights = dict(motion_weights or {})
        appearance_weight = float(weights.get("appearance", 1.0))
        center_weight = float(weights.get("center_displacement", 0.25))
        iou_weight = float(weights.get("bbox_iou", 0.25))
        area_weight = float(weights.get("area_ratio", 0.0))
        weight_sum = appearance_weight + center_weight + iou_weight + area_weight
        if weight_sum <= 0:
            raise ValueError("At least one motion/appearance weight must be positive")
        candidate_scores = {
            pair: (
                appearance_weight * ((feature.cosine_similarity + 1.0) / 2.0)
                + center_weight * (1.0 - feature.center_displacement)
                + iou_weight * feature.bbox_iou
                + area_weight * feature.area_ratio
            )
            / weight_sum
            for pair, feature in feature_map.items()
        }
    else:
        candidate_scores = _calibrated_probabilities(calibrator, feature_map)

    selected_pairs = _hungarian_assign(candidate_scores, identity_ids, detection_ids)
    assignments: list[AssignedPair] = []
    for identity_id, detection_id in selected_pairs:
        score = candidate_scores[(identity_id, detection_id)]
        if method == "cosine_forced":
            accepted = True
        elif method == "cosine_threshold":
            accepted = score >= cosine_threshold
        elif method == "motion_appearance":
            accepted = score >= fixed_score_threshold
        else:
            accepted = score >= confidence_threshold
        assignments.append(AssignedPair(identity_id, detection_id, score, accepted))

    accepted = [assignment for assignment in assignments if assignment.accepted]
    rejected = [assignment for assignment in assignments if not assignment.accepted]
    accepted_detections = {assignment.detection_id for assignment in accepted}
    accepted_identities = {assignment.predicted_identity_id for assignment in accepted}
    return AssociationDecision(
        method=method,
        assignments=assignments,
        accepted=accepted,
        rejected=rejected,
        unmatched_detection_ids=sorted(set(detection_ids) - accepted_detections),
        unmatched_identity_ids=sorted(set(identity_ids) - accepted_identities),
        candidate_scores=candidate_scores,
    )


class OnlineAssociator:
    """Apply an association method and update an ``ObjectMemory`` per frame."""

    def __init__(
        self,
        memory: ObjectMemory,
        method: AssociationMethod,
        *,
        cosine_threshold: float = 0.5,
        fixed_score_threshold: float = 0.5,
        confidence_threshold: float = 0.5,
        motion_weights: Mapping[str, float] | None = None,
        calibrator: Any | None = None,
    ) -> None:
        self.memory = memory
        self.method = method
        self.cosine_threshold = cosine_threshold
        self.fixed_score_threshold = fixed_score_threshold
        self.confidence_threshold = confidence_threshold
        self.motion_weights = motion_weights
        self.calibrator = calibrator

    def step(
        self,
        detections: list[ObjectDetection],
        image_size: tuple[int, int],
        frame_index: int,
    ) -> FrameAssociationResult:
        """Associate detections, reject weak matches, and update identity memory."""

        expired_ids = self.memory.advance_to(frame_index)
        identities = list(self.memory.active_identities())
        identity_ids = [identity.predicted_identity_id for identity in identities]
        detection_ids = [detection.detection_id for detection in detections]
        pair_features = compute_association_features(
            identities,
            detections,
            image_size=image_size,
            max_gap=self.memory.max_gap,
        )
        decision = decide_associations(
            self.method,
            pair_features,
            identity_ids,
            detection_ids,
            cosine_threshold=self.cosine_threshold,
            fixed_score_threshold=self.fixed_score_threshold,
            confidence_threshold=self.confidence_threshold,
            motion_weights=self.motion_weights,
            calibrator=self.calibrator,
        )
        detection_by_id = {detection.detection_id: detection for detection in detections}
        predicted_by_detection: dict[int, int] = {}
        new_by_detection: dict[int, int] = {}
        for assignment in decision.accepted:
            self.memory.update_identity(
                assignment.predicted_identity_id,
                detection_by_id[assignment.detection_id],
                frame_index,
            )
            predicted_by_detection[assignment.detection_id] = assignment.predicted_identity_id
        for detection_id in decision.unmatched_detection_ids:
            new_identity_id = self.memory.create_identity(
                detection_by_id[detection_id], frame_index
            )
            predicted_by_detection[detection_id] = new_identity_id
            new_by_detection[detection_id] = new_identity_id
        return FrameAssociationResult(
            frame_index=int(frame_index),
            decision=decision,
            predicted_identity_by_detection=predicted_by_detection,
            new_identity_by_detection=new_by_detection,
            expired_identity_ids=expired_ids,
            pair_features=pair_features,
        )
