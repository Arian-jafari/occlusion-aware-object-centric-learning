from __future__ import annotations

import numpy as np

from occlusion_association.association import OnlineAssociator, decide_associations
from occlusion_association.association_features import AssociationFeatureVector, compute_association_features
from occlusion_association.memory import ObjectMemory
from occlusion_association.object_representation import (
    ObjectDetection,
    build_object_detections,
)


def make_detection(
    detection_id: int,
    embedding: tuple[float, float],
    center: tuple[float, float],
    bbox: tuple[float, float, float, float],
    area: int = 100,
) -> ObjectDetection:
    return ObjectDetection(
        detection_id,
        np.asarray(embedding, dtype=np.float32),
        center,
        bbox,
        area,
    )


def features(pair_scores: dict[tuple[int, int], float]):
    return {
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
        for pair, score in pair_scores.items()
    }


def test_hungarian_matching_finds_global_one_to_one_optimum() -> None:
    pair_features = features(
        {
            (1, 0): 0.90,
            (1, 1): 0.80,
            (2, 0): 0.85,
            (2, 1): 0.10,
        }
    )

    decision = decide_associations(
        "cosine_threshold",
        pair_features,
        identity_ids=[1, 2],
        detection_ids=[0, 1],
        cosine_threshold=0.5,
    )

    assert {(pair.predicted_identity_id, pair.detection_id) for pair in decision.accepted} == {
        (1, 1),
        (2, 0),
    }
    assert not decision.rejected
    assert not decision.rejected_detection_ids
    assert decision.pair_outcomes[(2, 1)] == "below_threshold"
    assert decision.detection_outcomes[1] == "accepted"


def test_low_similarity_detection_is_rejected_as_an_unmatched_choice() -> None:
    memory = ObjectMemory(max_gap=5)
    online = OnlineAssociator(memory, "cosine_threshold", cosine_threshold=0.5)
    first = make_detection(0, (1.0, 0.0), (5.0, 5.0), (0.0, 0.0, 10.0, 10.0))
    second = make_detection(0, (0.0, 1.0), (5.0, 5.0), (0.0, 0.0, 10.0, 10.0))

    initial = online.step([first], (100, 100), frame_index=0)
    reappearance = online.step([second], (100, 100), frame_index=1)

    original_id = initial.predicted_identity_by_detection[0]
    new_id = reappearance.predicted_identity_by_detection[0]
    assert reappearance.decision.rejected == []
    assert reappearance.decision.rejected_detection_ids == [0]
    assert reappearance.decision.unmatched_identity_ids == [original_id]
    assert new_id != original_id
    assert reappearance.new_identity_by_detection == {0: new_id}
    assert original_id in memory.active_identity_ids  # Rejection does not erase memory.


def test_unmatched_alternative_remains_in_optimization_for_eligible_edges() -> None:
    pair_features = features(
        {
            (1, 0): 0.99,
            (1, 1): 0.75,
            (2, 0): 0.72,
            (2, 1): 0.49,
        }
    )

    decision = decide_associations(
        "cosine_threshold",
        pair_features,
        identity_ids=[1, 2],
        detection_ids=[0, 1],
        cosine_threshold=0.5,
    )

    # The diagonal high score has utility 0.99 - 0.5 = 0.49. The two cross
    # matches together have lower utility (0.75 - 0.5) + (0.72 - 0.5) = 0.47,
    # so the optimizer leaves detection 1 unmatched even though its edge is
    # individually eligible.
    assert {
        (pair.predicted_identity_id, pair.detection_id) for pair in decision.accepted
    } == {(1, 0)}
    assert decision.unmatched_detection_ids == [1]
    assert decision.assignment_conflict_detection_ids == [1]
    assert decision.rejected_detection_ids == []


def test_negative_eligible_score_is_left_unmatched_when_another_detection_is_better() -> None:
    decision = decide_associations(
        "cosine_threshold",
        features({(1, 0): -0.2, (1, 1): -0.4}),
        identity_ids=[1],
        detection_ids=[0, 1],
        cosine_threshold=-0.5,
    )

    assert [(pair.predicted_identity_id, pair.detection_id) for pair in decision.accepted] == [
        (1, 0)
    ]
    assert decision.detection_outcomes == {
        0: "accepted",
        1: "assignment_conflict",
    }
    assert decision.pair_outcomes[(1, 1)] == "assignment_conflict"


def test_negative_eligible_score_can_win_against_unmatched_threshold_and_is_shift_invariant() -> None:
    scores = {(1, 0): -0.2}
    decision = decide_associations(
        "cosine_threshold",
        features(scores),
        identity_ids=[1],
        detection_ids=[0],
        cosine_threshold=-0.5,
    )
    shifted = decide_associations(
        "cosine_threshold",
        features({pair: score + 1.0 for pair, score in scores.items()}),
        identity_ids=[1],
        detection_ids=[0],
        cosine_threshold=0.5,
    )

    assert decision.accepted[0].score == -0.2
    assert decision.detection_outcomes[0] == "accepted"
    assert [pair.detection_id for pair in shifted.accepted] == [0]


def test_no_eligible_edges_and_no_candidate_have_distinct_detection_outcomes() -> None:
    rejected = decide_associations(
        "cosine_threshold",
        features({(1, 0): -0.8}),
        identity_ids=[1],
        detection_ids=[0],
        cosine_threshold=-0.5,
    )
    no_candidate = decide_associations(
        "cosine_threshold",
        {},
        identity_ids=[],
        detection_ids=[4],
        cosine_threshold=0.5,
    )

    assert rejected.detection_outcomes[0] == "confidence_rejection"
    assert rejected.pair_outcomes[(1, 0)] == "below_threshold"
    assert rejected.rejected_detection_ids == [0]
    assert rejected.assignment_conflict_detection_ids == []
    assert no_candidate.detection_outcomes[4] == "no_candidate"
    assert no_candidate.no_candidate_detection_ids == [4]


def test_thresholded_methods_keep_positive_score_assignment_behavior() -> None:
    decision = decide_associations(
        "cosine_threshold",
        features({(1, 0): 0.9, (1, 1): 0.6, (2, 0): 0.7, (2, 1): 0.8}),
        identity_ids=[1, 2],
        detection_ids=[0, 1],
        cosine_threshold=0.5,
    )

    assert {
        (pair.predicted_identity_id, pair.detection_id) for pair in decision.accepted
    } == {(1, 0), (2, 1)}


def test_forced_matching_keeps_low_score_assignment() -> None:
    pair_features = features({(1, 0): -0.8})
    decision = decide_associations(
        "cosine_forced", pair_features, identity_ids=[1], detection_ids=[0]
    )
    assert len(decision.accepted) == 1
    assert not decision.rejected


def test_association_features_have_documented_ranges_and_geometry() -> None:
    memory = ObjectMemory(max_gap=9)
    original = make_detection(0, (1.0, 0.0), (5.0, 5.0), (0.0, 0.0, 10.0, 10.0))
    memory.advance_to(0)
    identity_id = memory.create_identity(original, 0)
    memory.advance_to(5)
    current = make_detection(1, (1.0, 0.0), (10.0, 5.0), (5.0, 0.0, 15.0, 10.0))

    feature = compute_association_features(
        memory.active_identities(), [current], image_size=(100, 100), max_gap=9
    )[(identity_id, 1)]

    assert feature.cosine_similarity == 1.0
    assert feature.center_displacement > 0.0
    assert np.isclose(feature.bbox_iou, 1 / 3)
    assert feature.area_ratio == 1.0
    assert np.isclose(feature.frames_since_last_observation, 0.5)
    assert feature.to_array().shape == (6,)


def test_appearance_feature_uses_historical_prototype_not_only_latest_embedding() -> None:
    memory = ObjectMemory(max_gap=5, appearance_history_size=2)
    memory.advance_to(0)
    identity_id = memory.create_identity(
        make_detection(0, (1.0, 0.0), (5.0, 5.0), (0.0, 0.0, 10.0, 10.0)), 0
    )
    memory.advance_to(1)
    memory.update_identity(
        identity_id,
        make_detection(0, (0.0, 1.0), (5.0, 5.0), (0.0, 0.0, 10.0, 10.0)),
        1,
    )
    current = make_detection(0, (1.0, 0.0), (5.0, 5.0), (0.0, 0.0, 10.0, 10.0))

    feature = compute_association_features(
        memory.active_identities(), [current], image_size=(100, 100), max_gap=5
    )[(identity_id, 0)]

    assert np.isclose(feature.cosine_similarity, 1.0 / np.sqrt(2.0))


def test_detection_order_is_geometry_based_and_gt_map_is_separate() -> None:
    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[0:5, 0:5] = 90
    mask[10:15, 10:15] = 2
    embeddings = {90: np.array([1.0, 0.0]), 2: np.array([0.0, 1.0])}

    detections, gt_map, invalid = build_object_detections(mask, embeddings)

    assert [detection.detection_id for detection in detections] == [0, 1]
    assert gt_map == {0: 90, 1: 2}
    assert not invalid
    assert not hasattr(detections[0], "ground_truth_id")
