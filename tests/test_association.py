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


def test_low_similarity_assignment_is_explicitly_rejected() -> None:
    memory = ObjectMemory(max_gap=5)
    online = OnlineAssociator(memory, "cosine_threshold", cosine_threshold=0.5)
    first = make_detection(0, (1.0, 0.0), (5.0, 5.0), (0.0, 0.0, 10.0, 10.0))
    second = make_detection(0, (0.0, 1.0), (5.0, 5.0), (0.0, 0.0, 10.0, 10.0))

    initial = online.step([first], (100, 100), frame_index=0)
    reappearance = online.step([second], (100, 100), frame_index=1)

    original_id = initial.predicted_identity_by_detection[0]
    new_id = reappearance.predicted_identity_by_detection[0]
    assert len(reappearance.decision.rejected) == 1
    assert new_id != original_id
    assert reappearance.new_identity_by_detection == {0: new_id}
    assert original_id in memory.active_identity_ids  # Rejection does not erase memory.


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
