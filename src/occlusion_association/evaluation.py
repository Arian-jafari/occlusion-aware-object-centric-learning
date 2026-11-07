"""Identity, association, and held-out calibration evaluation metrics."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np

from .association import FrameAssociationResult
from .association_features import FEATURE_NAMES


@dataclass
class FrameEvaluationLog:
    """Evaluation-only labels and decisions for one processed frame.

    ``identity_gt_owner_before`` is captured by the experiment runner before
    the frame's association updates. Ground-truth labels are therefore used
    only to label/evaluate pair decisions, never to produce model features.
    """

    sequence: str
    frame_name: str
    frame_index: int
    gt_id_by_detection: dict[int, int]
    identity_gt_owner_before: dict[int, int]
    result: FrameAssociationResult


def _ratio(numerator: int | float, denominator: int | float) -> float:
    return float(numerator / denominator) if denominator else float("nan")


def _counts_for_sequence(sequence: str, logs: Sequence[FrameEvaluationLog]) -> dict[str, Any]:
    identity_observations: dict[int, list[tuple[int, int]]] = defaultdict(list)
    correct_associations = 0
    incorrect_associations = 0
    eligible_gt_transitions = 0
    accepted_decisions = 0
    rejected_decisions = 0
    assignment_decisions = 0
    new_predicted_identities = 0
    candidate_labels: list[int] = []
    candidate_probabilities: list[float] = []
    occlusion_gaps: list[dict[str, Any]] = []
    valid_frame_indices = {log.frame_index for log in logs}

    for log in logs:
        result = log.result
        gt_by_detection = log.gt_id_by_detection
        owner_before = log.identity_gt_owner_before
        for detection_id, ground_truth_id in gt_by_detection.items():
            identity_observations[ground_truth_id].append(
                (log.frame_index, result.predicted_identity_by_detection[detection_id])
            )
            if any(
                owner_before.get(predicted_id) == ground_truth_id
                for predicted_id, candidate_detection_id in result.pair_features
                if candidate_detection_id == detection_id
            ):
                eligible_gt_transitions += 1

        for assignment in result.decision.accepted:
            accepted_decisions += 1
            current_gt_id = gt_by_detection[assignment.detection_id]
            previous_owner = owner_before.get(assignment.predicted_identity_id)
            if previous_owner == current_gt_id:
                correct_associations += 1
            else:
                incorrect_associations += 1
        rejected_decisions += len(result.decision.rejected)
        assignment_decisions += len(result.decision.assignments)
        new_predicted_identities += len(result.new_identity_by_detection)

        for pair, feature in result.pair_features.items():
            predicted_id, detection_id = pair
            target = int(owner_before.get(predicted_id) == gt_by_detection[detection_id])
            candidate_labels.append(target)
            if pair in result.decision.candidate_scores:
                candidate_probabilities.append(float(result.decision.candidate_scores[pair]))

    identity_switches = 0
    track_fragmentations = 0
    correctly_associated_identity_transitions = 0
    for gt_id, observations in identity_observations.items():
        observations.sort()
        for (previous_frame, previous_predicted_id), (current_frame, current_predicted_id) in zip(
            observations, observations[1:]
        ):
            if previous_predicted_id == current_predicted_id:
                correctly_associated_identity_transitions += 1
            else:
                identity_switches += 1
                intervening = set(range(previous_frame + 1, current_frame))
                complete_interval = intervening.issubset(valid_frame_indices)
                if intervening and complete_interval:
                    track_fragmentations += 1
            missing_gap = current_frame - previous_frame - 1
            intervening = set(range(previous_frame + 1, current_frame))
            if missing_gap > 0 and intervening.issubset(valid_frame_indices):
                occlusion_gaps.append(
                    {
                        "ground_truth_id": gt_id,
                        "gap": missing_gap,
                        "same_predicted_identity": previous_predicted_id == current_predicted_id,
                    }
                )

    accepted = correct_associations + incorrect_associations
    precision = _ratio(correct_associations, accepted)
    recall = _ratio(correct_associations, eligible_gt_transitions)
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if np.isfinite(precision) and np.isfinite(recall) and precision + recall > 0
        else float("nan")
    )
    return {
        "sequence": sequence,
        "identity_switches": identity_switches,
        "track_fragmentations": track_fragmentations,
        "correctly_associated_identity_transitions": correctly_associated_identity_transitions,
        "correct_associations": correct_associations,
        "incorrect_associations": incorrect_associations,
        "rejected_associations": rejected_decisions,
        "new_predicted_identities": new_predicted_identities,
        "eligible_gt_transitions": eligible_gt_transitions,
        "association_assignment_decisions": assignment_decisions,
        "accepted_associations": accepted_decisions,
        "association_precision": precision,
        "association_recall": recall,
        "association_f1": f1,
        "association_coverage": _ratio(accepted_decisions, assignment_decisions),
        "false_association_rate": _ratio(incorrect_associations, accepted),
        "occlusion_gaps": occlusion_gaps,
        "candidate_labels": candidate_labels,
        "candidate_probabilities": candidate_probabilities,
    }


def evaluate_sequences(
    frame_logs: Iterable[FrameEvaluationLog],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Return per-sequence and pooled aggregate identity/association metrics.

    Identity switches count every change in a GT object's predicted identity
    between consecutive observed frames, including reappearance after absence;
    the first observation is never a switch. Fragmentation counts only changes
    after at least one intervening frame with a valid image/annotation pair in
    which the object is absent. Gaps containing missing or invalid dataset
    frames are excluded from natural-occlusion fragmentation analysis.

    Association recall's denominator is the number of current object
    observations for which a retained, previously associated memory identity
    of that same GT object was available at frame start. Coverage is accepted
    Hungarian assignments divided by all Hungarian assignment decisions before
    threshold rejection. Rejected assignment decisions are reported separately.
    """

    grouped: dict[str, list[FrameEvaluationLog]] = defaultdict(list)
    for log in frame_logs:
        grouped[log.sequence].append(log)
    per_sequence = [
        _counts_for_sequence(name, sorted(logs, key=lambda log: log.frame_index))
        for name, logs in sorted(grouped.items())
    ]
    count_names = (
        "identity_switches",
        "track_fragmentations",
        "correctly_associated_identity_transitions",
        "correct_associations",
        "incorrect_associations",
        "rejected_associations",
        "new_predicted_identities",
        "eligible_gt_transitions",
        "association_assignment_decisions",
        "accepted_associations",
    )
    totals = {name: sum(int(row[name]) for row in per_sequence) for name in count_names}
    accepted = totals["correct_associations"] + totals["incorrect_associations"]
    precision = _ratio(totals["correct_associations"], accepted)
    recall = _ratio(totals["correct_associations"], totals["eligible_gt_transitions"])
    totals.update(
        {
            "sequence_count": len(per_sequence),
            "association_precision": precision,
            "association_recall": recall,
            "association_f1": (
                2.0 * precision * recall / (precision + recall)
                if np.isfinite(precision) and np.isfinite(recall) and precision + recall > 0
                else float("nan")
            ),
            "association_coverage": _ratio(
                totals["accepted_associations"], totals["association_assignment_decisions"]
            ),
            "false_association_rate": _ratio(totals["incorrect_associations"], accepted),
        }
    )
    return per_sequence, totals


def candidate_targets_and_probabilities(
    frame_logs: Iterable[FrameEvaluationLog],
) -> tuple[np.ndarray, np.ndarray]:
    """Collect pair correctness targets and model probabilities from logs."""

    labels: list[int] = []
    probabilities: list[float] = []
    for log in frame_logs:
        for pair in log.result.pair_features:
            predicted_id, detection_id = pair
            labels.append(
                int(
                    log.identity_gt_owner_before.get(predicted_id)
                    == log.gt_id_by_detection[detection_id]
                )
            )
            probabilities.append(float(log.result.decision.candidate_scores[pair]))
    return np.asarray(labels, dtype=np.int64), np.asarray(probabilities, dtype=np.float64)


def calibration_metrics(
    labels: Sequence[int] | np.ndarray,
    probabilities: Sequence[float] | np.ndarray,
    ece_bins: int = 10,
) -> dict[str, Any]:
    """Compute Brier score, uniform-bin ECE, and reliability-bin statistics."""

    y = np.asarray(labels, dtype=np.int64)
    p = np.asarray(probabilities, dtype=np.float64)
    if y.shape != p.shape or y.ndim != 1:
        raise ValueError("labels and probabilities must be one-dimensional arrays of equal length")
    if ece_bins < 1:
        raise ValueError("ece_bins must be positive")
    if y.size and (not np.isin(y, [0, 1]).all() or not np.isfinite(p).all() or np.any((p < 0) | (p > 1))):
        raise ValueError("Calibration inputs must be binary labels and finite probabilities in [0, 1]")
    if not y.size:
        return {
            "brier_score": float("nan"),
            "ece": float("nan"),
            "n_candidates": 0,
            "positive_rate": float("nan"),
            "ece_bins": int(ece_bins),
            "reliability": [],
        }

    brier = float(np.mean(np.square(p - y)))
    bin_ids = np.minimum((p * ece_bins).astype(int), ece_bins - 1)
    reliability = []
    ece = 0.0
    for bin_index in range(ece_bins):
        selected = bin_ids == bin_index
        count = int(selected.sum())
        lower, upper = bin_index / ece_bins, (bin_index + 1) / ece_bins
        if count:
            confidence = float(p[selected].mean())
            accuracy = float(y[selected].mean())
            ece += (count / y.size) * abs(confidence - accuracy)
        else:
            confidence = float("nan")
            accuracy = float("nan")
        reliability.append(
            {
                "bin": bin_index,
                "lower": lower,
                "upper": upper,
                "count": count,
                "mean_confidence": confidence,
                "empirical_accuracy": accuracy,
            }
        )
    return {
        "brier_score": brier,
        "ece": float(ece),
        "n_candidates": int(y.size),
        "positive_rate": float(y.mean()),
        "ece_bins": int(ece_bins),
        "reliability": reliability,
    }


def precision_coverage_curve(
    labels: Sequence[int] | np.ndarray,
    probabilities: Sequence[float] | np.ndarray,
    thresholds: Sequence[float],
) -> list[dict[str, float | int]]:
    """Compute candidate-pair precision and coverage at each confidence cutoff."""

    y = np.asarray(labels, dtype=np.int64)
    p = np.asarray(probabilities, dtype=np.float64)
    if y.shape != p.shape or y.ndim != 1:
        raise ValueError("labels and probabilities must be one-dimensional arrays of equal length")
    points = []
    for threshold in thresholds:
        accepted = p >= threshold
        accepted_count = int(accepted.sum())
        true_positive = int(np.sum(y[accepted] == 1)) if accepted_count else 0
        points.append(
            {
                "threshold": float(threshold),
                "accepted_candidates": accepted_count,
                "total_candidates": int(y.size),
                "precision": _ratio(true_positive, accepted_count),
                "coverage": _ratio(accepted_count, y.size),
            }
        )
    return points


def occlusion_gap_metrics(
    per_sequence: Sequence[dict[str, Any]],
    gap_bins: Sequence[tuple[str, int, int | None]] = (
        ("1-5", 1, 5),
        ("6-15", 6, 15),
        ("16-30", 16, 30),
        ("31+", 31, None),
    ),
) -> list[dict[str, Any]]:
    """Aggregate natural GT absence/reappearance identity consistency by gap."""

    all_gaps = [gap for sequence in per_sequence for gap in sequence["occlusion_gaps"]]
    result = []
    for name, lower, upper in gap_bins:
        selected = [
            gap
            for gap in all_gaps
            if gap["gap"] >= lower and (upper is None or gap["gap"] <= upper)
        ]
        same = sum(bool(gap["same_predicted_identity"]) for gap in selected)
        result.append(
            {
                "gap_range": name,
                "reappearances": len(selected),
                "same_identity": same,
                "identity_changes": len(selected) - same,
                "identity_consistency": _ratio(same, len(selected)),
            }
        )
    return result
