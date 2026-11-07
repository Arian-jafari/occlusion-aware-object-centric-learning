"""Hungarian association baselines and calibrated selective decisions."""

from __future__ import annotations

from dataclasses import dataclass, field
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
PairOutcome = Literal["accepted", "below_threshold", "assignment_conflict"]
DetectionOutcome = Literal[
    "accepted", "confidence_rejection", "assignment_conflict", "no_candidate"
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
    """Pair-level and detection-level outcomes for one association decision.

    ``rejected`` is retained for compatibility with the former post-assignment
    threshold API; threshold-aware assignment does not select rejected pairs.
    Detection rejection IDs contain confidence rejections only. Assignment
    conflicts and detections without a candidate have separate fields.
    """

    method: str
    assignments: list[AssignedPair]
    accepted: list[AssignedPair]
    rejected: list[AssignedPair]
    unmatched_detection_ids: list[int]
    unmatched_identity_ids: list[int]
    candidate_scores: dict[tuple[int, int], float]
    rejected_detection_ids: list[int] = field(default_factory=list)
    assignment_conflict_detection_ids: list[int] = field(default_factory=list)
    no_candidate_detection_ids: list[int] = field(default_factory=list)
    pair_outcomes: dict[tuple[int, int], PairOutcome] = field(default_factory=dict)
    detection_outcomes: dict[int, DetectionOutcome] = field(default_factory=dict)
    score_kind: Literal["probability", "association_score"] = "association_score"


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


def _hungarian_assign_with_unmatched(
    score_map: Mapping[tuple[int, int], float],
    identity_ids: list[int],
    detection_ids: list[int],
    threshold: float,
) -> list[tuple[int, int]]:
    """Maximize score gain over the unmatched option under one-to-one matching.

    The augmented square matrix has one dummy column per identity and one
    dummy row per detection. Scores below the acceptance threshold are made
    infeasible. Each unmatched choice has utility zero, and an eligible edge
    has utility ``score - threshold``. Thus the threshold is the explicit
    unmatched alternative: an edge is preferred only to the extent that it
    exceeds that alternative. Shifting all scores and the threshold by the
    same constant leaves the objective unchanged.
    """

    if not identity_ids or not detection_ids:
        return []
    identity_count, detection_count = len(identity_ids), len(detection_ids)
    matrix = np.zeros(
        (identity_count + detection_count, detection_count + identity_count),
        dtype=np.float64,
    )
    matrix[:identity_count, :detection_count] = -1e12
    for row, identity_id in enumerate(identity_ids):
        for column, detection_id in enumerate(detection_ids):
            score = float(score_map[(identity_id, detection_id)])
            if score >= threshold:
                matrix[row, column] = score - threshold

    row_indices, column_indices = linear_sum_assignment(-matrix)
    return [
        (identity_ids[row], detection_ids[column])
        for row, column in zip(row_indices.tolist(), column_indices.tolist())
        if row < identity_count
        and column < detection_count
        and score_map[(identity_ids[row], detection_ids[column])] >= threshold
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
    """Select eligible pairs with optional unmatched identity/detection choices.

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

    if method == "cosine_forced":
        selected_pairs = _hungarian_assign(candidate_scores, identity_ids, detection_ids)
    else:
        threshold = {
            "cosine_threshold": cosine_threshold,
            "motion_appearance": fixed_score_threshold,
            "calibrated_selective": confidence_threshold,
        }[method]
        selected_pairs = _hungarian_assign_with_unmatched(
            candidate_scores, identity_ids, detection_ids, threshold
        )
    assignments: list[AssignedPair] = []
    for identity_id, detection_id in selected_pairs:
        score = candidate_scores[(identity_id, detection_id)]
        assignments.append(AssignedPair(identity_id, detection_id, score, True))

    accepted = assignments
    rejected: list[AssignedPair] = []
    accepted_detections = {assignment.detection_id for assignment in accepted}
    accepted_identities = {assignment.predicted_identity_id for assignment in accepted}
    threshold = None if method == "cosine_forced" else {
        "cosine_threshold": cosine_threshold,
        "motion_appearance": fixed_score_threshold,
        "calibrated_selective": confidence_threshold,
    }[method]
    selected_set = set(selected_pairs)
    pair_outcomes: dict[tuple[int, int], PairOutcome] = {}
    for pair, score in candidate_scores.items():
        eligible = threshold is None or score >= threshold
        if pair in selected_set:
            pair_outcomes[pair] = "accepted"
        elif eligible:
            pair_outcomes[pair] = "assignment_conflict"
        else:
            pair_outcomes[pair] = "below_threshold"

    candidate_detections = {detection_id for _, detection_id in candidate_scores}
    eligible_detections = {
        detection_id
        for (_, detection_id), outcome in pair_outcomes.items()
        if outcome != "below_threshold"
    }
    detection_outcomes: dict[int, DetectionOutcome] = {}
    for detection_id in detection_ids:
        if detection_id in accepted_detections:
            outcome: DetectionOutcome = "accepted"
        elif detection_id not in candidate_detections:
            outcome = "no_candidate"
        elif detection_id in eligible_detections:
            outcome = "assignment_conflict"
        else:
            outcome = "confidence_rejection"
        detection_outcomes[detection_id] = outcome
    rejected_detection_ids = sorted(
        detection_id
        for detection_id, outcome in detection_outcomes.items()
        if outcome == "confidence_rejection"
    )
    assignment_conflict_detection_ids = sorted(
        detection_id
        for detection_id, outcome in detection_outcomes.items()
        if outcome == "assignment_conflict"
    )
    no_candidate_detection_ids = sorted(
        detection_id
        for detection_id, outcome in detection_outcomes.items()
        if outcome == "no_candidate"
    )
    return AssociationDecision(
        method=method,
        assignments=assignments,
        accepted=accepted,
        rejected=rejected,
        rejected_detection_ids=rejected_detection_ids,
        assignment_conflict_detection_ids=assignment_conflict_detection_ids,
        no_candidate_detection_ids=no_candidate_detection_ids,
        pair_outcomes=pair_outcomes,
        detection_outcomes=detection_outcomes,
        score_kind=("probability" if method == "calibrated_selective" else "association_score"),
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
