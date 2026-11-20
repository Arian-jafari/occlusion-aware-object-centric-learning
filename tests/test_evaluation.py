from __future__ import annotations

import numpy as np

from occlusion_association.association import (
    FrameAssociationResult,
    OnlineAssociator,
    decide_associations,
)
from occlusion_association.association_features import AssociationFeatureVector
from occlusion_association.evaluation import (
    FrameEvaluationLog,
    assignment_precision_coverage_curve,
    calibration_metrics,
    candidate_targets_and_probabilities,
    evaluate_sequences,
    occlusion_gap_metrics,
    precision_coverage_curve,
)
from occlusion_association.memory import ObjectMemory
from occlusion_association.object_representation import ObjectDetection


def detection(vector: tuple[float, float]) -> ObjectDetection:
    return ObjectDetection(
        detection_id=0,
        embedding=np.asarray(vector, dtype=np.float32),
        center=(5.0, 5.0),
        bbox=(0.0, 0.0, 10.0, 10.0),
        area=100,
    )


def make_logs() -> list[FrameEvaluationLog]:
    online = OnlineAssociator(
        ObjectMemory(max_gap=5), "cosine_threshold", cosine_threshold=0.5
    )
    logs = []
    owner: dict[int, int] = {}
    frames = [
        (0, detection((1.0, 0.0))),
        (1, detection((1.0, 0.0))),
        (2, None),
        (3, None),
        (4, detection((0.0, 1.0))),
    ]
    for frame_index, current_detection in frames:
        before = owner.copy()
        detections = [] if current_detection is None else [current_detection]
        result = online.step(detections, image_size=(100, 100), frame_index=frame_index)
        gt_map = {} if current_detection is None else {0: 7}
        logs.append(
            FrameEvaluationLog(
                sequence="synthetic",
                frame_name=f"{frame_index:05d}",
                frame_index=frame_index,
                gt_id_by_detection=gt_map,
                identity_gt_owner_before=before,
                result=result,
            )
        )
        for detection_id, predicted_id in result.predicted_identity_by_detection.items():
            owner[predicted_id] = gt_map[detection_id]
    return logs


class ProbabilityFromFirstFeature:
    classes_ = np.array([0, 1])

    def predict_proba(self, matrix: np.ndarray) -> np.ndarray:
        positive = np.clip(matrix[:, 0], 0.0, 1.0)
        return np.column_stack((1.0 - positive, positive))


def make_probability_log(
    frame_index: int,
    scores: dict[tuple[int, int], float],
    identity_ids: list[int],
    gt_by_detection: dict[int, int],
    owner_before: dict[int, int],
) -> FrameEvaluationLog:
    pair_features = {
        pair: AssociationFeatureVector(
            predicted_identity_id=pair[0],
            detection_id=pair[1],
            cosine_similarity=score,
            center_displacement=0.0,
            bbox_iou=1.0,
            area_ratio=1.0,
            frames_since_last_observation=0.0,
            similarity_margin=0.0,
        )
        for pair, score in scores.items()
    }
    decision = decide_associations(
        "calibrated_selective",
        pair_features,
        identity_ids=identity_ids,
        detection_ids=list(gt_by_detection),
        confidence_threshold=0.5,
        calibrator=ProbabilityFromFirstFeature(),
    )
    predicted = {
        assignment.detection_id: assignment.predicted_identity_id
        for assignment in decision.accepted
    }
    new_identity_by_detection = {}
    for detection_id in decision.unmatched_detection_ids:
        new_id = 100 + detection_id
        predicted[detection_id] = new_id
        new_identity_by_detection[detection_id] = new_id
    result = FrameAssociationResult(
        frame_index=frame_index,
        decision=decision,
        predicted_identity_by_detection=predicted,
        new_identity_by_detection=new_identity_by_detection,
        expired_identity_ids=(),
        pair_features=pair_features,
    )
    return FrameEvaluationLog(
        sequence="synthetic",
        frame_name=f"{frame_index:05d}",
        frame_index=frame_index,
        gt_id_by_detection=gt_by_detection,
        identity_gt_owner_before=owner_before,
        result=result,
    )


def make_outcome_logs() -> list[FrameEvaluationLog]:
    return [
        make_probability_log(0, {}, [], {0: 1}, {}),
        make_probability_log(1, {(10, 1): 0.2}, [10], {1: 7}, {10: 7}),
        make_probability_log(
            2,
            {(20, 2): 0.9, (20, 3): 0.8},
            [20],
            {2: 8, 3: 9},
            {20: 8},
        ),
    ]


def test_identity_switch_and_fragmentation_count_reappearance_change() -> None:
    per_sequence, aggregate = evaluate_sequences(make_logs())

    assert per_sequence[0]["identity_switches"] == 1
    assert per_sequence[0]["track_fragmentations"] == 1
    assert per_sequence[0]["correctly_associated_identity_transitions"] == 1
    assert per_sequence[0]["correct_associations"] == 1
    assert per_sequence[0]["rejected_associations"] == 1
    assert per_sequence[0]["new_predicted_identities"] == 2
    assert per_sequence[0]["eligible_gt_transitions"] == 2
    assert np.isclose(per_sequence[0]["association_coverage"], 0.5)
    assert per_sequence[0]["association_candidate_pairs"] == 2
    assert per_sequence[0]["calibration_candidate_pairs"] == 0
    assert per_sequence[0]["calibration_status"] == "not_applicable_non_probability_scores"
    assert aggregate["calibration_status"] == "not_applicable_non_probability_scores"
    assert aggregate["sequence_count"] == 1
    assert aggregate["identity_switches"] == 1


def test_calibration_metrics_and_precision_coverage_curve() -> None:
    metrics = calibration_metrics([0, 1], [0.1, 0.9], ece_bins=10)
    curve = precision_coverage_curve(
        [1, 0, 1], [0.9, 0.8, 0.2], thresholds=[0.8, 0.9]
    )

    assert np.isclose(metrics["brier_score"], 0.01)
    assert np.isclose(metrics["ece"], 0.1)
    assert metrics["n_candidates"] == 2
    assert metrics["n_positive_candidates"] == 1
    assert metrics["n_negative_candidates"] == 1
    assert curve[0]["accepted_candidates"] == 2
    assert curve[0]["analysis_unit"] == "candidate_pair"
    assert np.isclose(curve[0]["precision"], 0.5)
    assert np.isclose(curve[0]["coverage"], 2 / 3)
    assert np.isclose(curve[1]["precision"], 1.0)


def test_held_out_candidate_targets_and_natural_gap_bins() -> None:
    logs = make_outcome_logs()
    labels, probabilities = candidate_targets_and_probabilities(logs)
    per_sequence, _ = evaluate_sequences(make_logs())
    gap_report = occlusion_gap_metrics(per_sequence)

    assert labels.shape == probabilities.shape
    assert labels.size > 0
    assert np.all((probabilities >= 0.0) & (probabilities <= 1.0))
    assert gap_report[0]["reappearances"] == 1
    assert gap_report[0]["identity_changes"] == 1


def test_detection_outcomes_and_aggregate_counts_keep_reasons_separate() -> None:
    logs = make_outcome_logs()
    per_sequence, aggregate = evaluate_sequences(logs)
    row = per_sequence[0]

    assert logs[0].result.decision.detection_outcomes[0] == "no_candidate"
    assert logs[1].result.decision.detection_outcomes[1] == "confidence_rejection"
    assert logs[2].result.decision.detection_outcomes[3] == "assignment_conflict"
    assert row["accepted_associations"] == 1
    assert row["confidence_rejections"] == row["rejected_associations"] == 1
    assert row["assignment_conflicts"] == 1
    assert row["no_candidate_detections"] == 1
    assert row["association_assignment_decisions"] == 3
    assert np.isclose(row["association_coverage"], 1 / 3)
    assert aggregate["confidence_rejections"] == 1
    assert aggregate["assignment_conflicts"] == 1
    assert aggregate["no_candidate_detections"] == 1
    assert row["calibration_candidate_pairs"] == 3
    assert row["association_candidate_pairs"] == 3


def test_candidate_pair_and_selected_assignment_precision_coverage_are_distinct() -> None:
    logs = make_outcome_logs()
    labels, probabilities = candidate_targets_and_probabilities(logs)
    pair_curve = precision_coverage_curve(labels, probabilities, [0.5])
    assignment_curve = assignment_precision_coverage_curve(logs, [0.5])

    assert pair_curve[0]["total_candidates"] == 3
    assert pair_curve[0]["accepted_candidates"] == 2
    assert assignment_curve[0]["total_candidate_detections"] == 3
    assert assignment_curve[0]["accepted_assignments"] == 1
    assert assignment_curve[0]["analysis_unit"] == "hungarian_assignment"
    assert assignment_curve[0]["confidence_rejections"] == 1
    assert assignment_curve[0]["assignment_conflicts"] == 1
    assert np.isclose(assignment_curve[0]["coverage"], 1 / 3)


def test_raw_association_scores_are_not_accepted_as_calibration_probabilities() -> None:
    import pytest

    with pytest.raises(ValueError, match="require probability predictions"):
        candidate_targets_and_probabilities(make_logs())
    with pytest.raises(ValueError, match="requires probability predictions"):
        assignment_precision_coverage_curve(make_logs(), [0.5])


def test_empty_and_single_class_calibration_evaluation_are_reported_safely() -> None:
    empty = calibration_metrics([], [])
    single_class = calibration_metrics([1, 1], [0.7, 0.9])

    assert np.isnan(empty["brier_score"])
    assert np.isnan(empty["ece"])
    assert empty["n_candidates"] == 0
    assert empty["calibration_status"] == "undefined_no_candidates"
    assert np.isfinite(single_class["brier_score"])
    assert np.isfinite(single_class["ece"])
    assert single_class["n_positive_candidates"] == 2
    assert single_class["n_negative_candidates"] == 0
    assert single_class["calibration_status"] == "computed_single_class"


def test_missing_dataset_frames_are_not_counted_as_natural_occlusion() -> None:
    logs = [log for log in make_logs() if log.frame_index in {0, 1, 4}]

    per_sequence, _ = evaluate_sequences(logs)
    gaps = occlusion_gap_metrics(per_sequence)

    assert per_sequence[0]["identity_switches"] == 1
    assert per_sequence[0]["track_fragmentations"] == 0
    assert gaps[0]["reappearances"] == 0
