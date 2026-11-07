from __future__ import annotations

import numpy as np

from occlusion_association.association import OnlineAssociator
from occlusion_association.evaluation import (
    FrameEvaluationLog,
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
    assert curve[0]["accepted_candidates"] == 2
    assert np.isclose(curve[0]["precision"], 0.5)
    assert np.isclose(curve[0]["coverage"], 2 / 3)
    assert np.isclose(curve[1]["precision"], 1.0)


def test_held_out_candidate_targets_and_natural_gap_bins() -> None:
    logs = make_logs()
    labels, probabilities = candidate_targets_and_probabilities(logs)
    per_sequence, _ = evaluate_sequences(logs)
    gap_report = occlusion_gap_metrics(per_sequence)

    assert labels.shape == probabilities.shape
    assert labels.size > 0
    assert gap_report[0]["reappearances"] == 1
    assert gap_report[0]["identity_changes"] == 1


def test_missing_dataset_frames_are_not_counted_as_natural_occlusion() -> None:
    logs = [log for log in make_logs() if log.frame_index in {0, 1, 4}]

    per_sequence, _ = evaluate_sequences(logs)
    gaps = occlusion_gap_metrics(per_sequence)

    assert per_sequence[0]["identity_switches"] == 1
    assert per_sequence[0]["track_fragmentations"] == 0
    assert gaps[0]["reappearances"] == 0
