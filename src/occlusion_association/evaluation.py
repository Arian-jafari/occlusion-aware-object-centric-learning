"""Identity, association, and held-out calibration evaluation metrics."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np

from .association import FrameAssociationResult, _hungarian_assign_with_unmatched
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
    assignment_conflicts = 0
    no_candidate_detections = 0
    assignment_decisions = 0
    new_predicted_identities = 0
    candidate_labels: list[int] = []
    candidate_probabilities: list[float] = []
    association_candidate_pairs = 0
    has_probability_predictions = False
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
        rejected_decisions += len(result.decision.rejected_detection_ids)
        assignment_conflicts += len(result.decision.assignment_conflict_detection_ids)
        no_candidate_detections += len(result.decision.no_candidate_detection_ids)
        assignment_decisions += (
            len(result.decision.accepted)
            + len(result.decision.rejected_detection_ids)
            + len(result.decision.assignment_conflict_detection_ids)
        )
        new_predicted_identities += len(result.new_identity_by_detection)
        association_candidate_pairs += len(result.pair_features)

        if result.decision.score_kind == "probability":
            has_probability_predictions = True
            for pair in result.pair_features:
                predicted_id, detection_id = pair
                candidate_labels.append(
                    int(owner_before.get(predicted_id) == gt_by_detection[detection_id])
                )
                candidate_probabilities.append(
                    float(result.decision.candidate_scores[pair])
                )

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
    calibration_report = calibration_metrics(candidate_labels, candidate_probabilities)
    calibration_status = (
        calibration_report["calibration_status"]
        if has_probability_predictions
        else "not_applicable_non_probability_scores"
    )
    return {
        "sequence": sequence,
        "identity_switches": identity_switches,
        "track_fragmentations": track_fragmentations,
        "correctly_associated_identity_transitions": correctly_associated_identity_transitions,
        "correct_associations": correct_associations,
        "incorrect_associations": incorrect_associations,
        "rejected_associations": rejected_decisions,
        "confidence_rejections": rejected_decisions,
        "assignment_conflicts": assignment_conflicts,
        "no_candidate_detections": no_candidate_detections,
        "new_predicted_identities": new_predicted_identities,
        "eligible_gt_transitions": eligible_gt_transitions,
        "association_assignment_decisions": assignment_decisions,
        "accepted_associations": accepted_decisions,
        "association_precision": precision,
        "association_recall": recall,
        "association_f1": f1,
        "association_coverage": _ratio(accepted_decisions, assignment_decisions),
        "false_association_rate": _ratio(incorrect_associations, accepted),
        "brier_score": calibration_report["brier_score"],
        "ece": calibration_report["ece"],
        "calibration_candidate_pairs": calibration_report["n_candidates"],
        "association_candidate_pairs": association_candidate_pairs,
        "calibration_positive_candidate_pairs": calibration_report["n_positive_candidates"],
        "calibration_negative_candidate_pairs": calibration_report["n_negative_candidates"],
        "calibration_status": calibration_status,
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
    frames are excluded from mask-defined absence fragmentation analysis.

    Association recall's denominator is the number of current object
    observations for which a retained, previously associated memory identity
    of that same GT object was available at frame start. Coverage is accepted
    matches divided by detection-level decisions that had at least one memory
    candidate. Confidence rejections and assignment conflicts are separate
    counts; detections with no candidate are excluded from coverage.
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
        "confidence_rejections",
        "assignment_conflicts",
        "no_candidate_detections",
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
    pooled_labels = [label for row in per_sequence for label in row["candidate_labels"]]
    pooled_probabilities = [
        probability for row in per_sequence for probability in row["candidate_probabilities"]
    ]
    has_probability_predictions = any(
        row["calibration_status"] != "not_applicable_non_probability_scores"
        for row in per_sequence
    )
    pooled_calibration = calibration_metrics(pooled_labels, pooled_probabilities)
    totals["brier_score"] = pooled_calibration["brier_score"]
    totals["ece"] = pooled_calibration["ece"]
    totals["calibration_candidate_pairs"] = pooled_calibration["n_candidates"]
    totals["association_candidate_pairs"] = sum(
        row["association_candidate_pairs"] for row in per_sequence
    )
    totals["calibration_positive_candidate_pairs"] = pooled_calibration[
        "n_positive_candidates"
    ]
    totals["calibration_negative_candidate_pairs"] = pooled_calibration[
        "n_negative_candidates"
    ]
    totals["calibration_status"] = (
        pooled_calibration["calibration_status"]
        if has_probability_predictions
        else "not_applicable_non_probability_scores"
    )
    return per_sequence, totals


def candidate_targets_and_probabilities(
    frame_logs: Iterable[FrameEvaluationLog],
) -> tuple[np.ndarray, np.ndarray]:
    """Collect targets and probabilities; reject raw association scores."""

    labels: list[int] = []
    probabilities: list[float] = []
    for log in frame_logs:
        if log.result.decision.score_kind != "probability":
            raise ValueError(
                "Calibration metrics require probability predictions; "
                f"{log.result.decision.method!r} produces association scores"
            )
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
    """Compute Brier/ECE for probability predictions and report class support.

    Empty data have undefined Brier/ECE (NaN). Brier and ECE remain defined for
    a single-class set, but the report marks its limited class support.
    """

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
            "n_positive_candidates": 0,
            "n_negative_candidates": 0,
            "positive_rate": float("nan"),
            "ece_bins": int(ece_bins),
            "reliability": [],
            "calibration_status": "undefined_no_candidates",
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
        "n_positive_candidates": int(np.sum(y == 1)),
        "n_negative_candidates": int(np.sum(y == 0)),
        "positive_rate": float(y.mean()),
        "ece_bins": int(ece_bins),
        "reliability": reliability,
        "calibration_status": (
            "computed_both_classes"
            if np.unique(y).size == 2
            else "computed_single_class"
        ),
    }


def precision_coverage_curve(
    labels: Sequence[int] | np.ndarray,
    probabilities: Sequence[float] | np.ndarray,
    thresholds: Sequence[float],
) -> list[dict[str, float | int | str]]:
    """Compute candidate-pair-level precision and coverage at each cutoff.

    This curve thresholds pairs independently; it is not a curve of final
    Hungarian assignments.
    """

    y = np.asarray(labels, dtype=np.int64)
    p = np.asarray(probabilities, dtype=np.float64)
    if y.shape != p.shape or y.ndim != 1:
        raise ValueError("labels and probabilities must be one-dimensional arrays of equal length")
    if y.size and (
        not np.isin(y, [0, 1]).all()
        or not np.isfinite(p).all()
        or np.any((p < 0) | (p > 1))
    ):
        raise ValueError(
            "Candidate-pair precision/coverage requires binary labels and probabilities in [0, 1]"
        )
    points = []
    for threshold in thresholds:
        accepted = p >= threshold
        accepted_count = int(accepted.sum())
        true_positive = int(np.sum(y[accepted] == 1)) if accepted_count else 0
        points.append(
            {
                "threshold": float(threshold),
                "analysis_unit": "candidate_pair",
                "accepted_candidates": accepted_count,
                "total_candidates": int(y.size),
                "precision": _ratio(true_positive, accepted_count),
                "coverage": _ratio(accepted_count, y.size),
            }
        )
    return points


def assignment_precision_coverage_curve(
    frame_logs: Iterable[FrameEvaluationLog],
    thresholds: Sequence[float],
) -> list[dict[str, Any]]:
    """Re-run one-to-one assignment at each cutoff and score selected pairs.

    Coverage uses detections with at least one memory candidate as its fixed
    denominator. Candidate detections that fail the confidence cutoff and
    eligible detections lost to one-to-one assignment are reported separately.
    """

    logs = list(frame_logs)
    if any(log.result.decision.score_kind != "probability" for log in logs):
        raise ValueError("Assignment precision/coverage requires probability predictions")
    points: list[dict[str, Any]] = []
    for threshold in thresholds:
        accepted = correct = candidate_detections = 0
        confidence_rejections = assignment_conflicts = 0
        for log in logs:
            score_map = log.result.decision.candidate_scores
            identity_ids = sorted({identity_id for identity_id, _ in score_map})
            detection_ids = list(log.gt_id_by_detection)
            candidate_by_detection = {detection_id: False for detection_id in detection_ids}
            eligible_by_detection = {detection_id: False for detection_id in detection_ids}
            for (identity_id, detection_id), score in score_map.items():
                candidate_by_detection[detection_id] = True
                if score >= threshold:
                    eligible_by_detection[detection_id] = True
            selected = _hungarian_assign_with_unmatched(
                score_map, identity_ids, detection_ids, float(threshold)
            )
            selected_detections = {detection_id for _, detection_id in selected}
            candidate_detections += sum(candidate_by_detection.values())
            confidence_rejections += sum(
                has_candidate and not eligible_by_detection[detection_id]
                for detection_id, has_candidate in candidate_by_detection.items()
            )
            assignment_conflicts += sum(
                eligible_by_detection[detection_id]
                and detection_id not in selected_detections
                for detection_id in detection_ids
            )
            for identity_id, detection_id in selected:
                accepted += 1
                correct += int(
                    log.identity_gt_owner_before.get(identity_id)
                    == log.gt_id_by_detection[detection_id]
                )
        points.append(
            {
                "threshold": float(threshold),
                "analysis_unit": "hungarian_assignment",
                "accepted_assignments": accepted,
                "total_candidate_detections": candidate_detections,
                "confidence_rejections": confidence_rejections,
                "assignment_conflicts": assignment_conflicts,
                "precision": _ratio(correct, accepted),
                "coverage": _ratio(accepted, candidate_detections),
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
    """Aggregate mask-defined GT absence/reappearance identity consistency by gap."""

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
