from __future__ import annotations

import numpy as np
import pytest

from occlusion_association.memory import ObjectMemory
from occlusion_association.object_representation import ObjectDetection


def detection(
    detection_id: int = 0,
    embedding: tuple[float, ...] = (1.0, 0.0),
    *,
    center: tuple[float, float] = (5.0, 5.0),
    bbox: tuple[float, float, float, float] = (0.0, 0.0, 10.0, 10.0),
    area: int = 100,
) -> ObjectDetection:
    return ObjectDetection(detection_id, np.asarray(embedding, dtype=np.float32), center, bbox, area)


def test_memory_creates_updates_and_tracks_appearance_history() -> None:
    memory = ObjectMemory(max_gap=3, appearance_history_size=2)
    memory.advance_to(0)
    predicted_id = memory.create_identity(detection(), 0)

    assert predicted_id == 1
    assert memory.get(predicted_id).previous_observation_frame == 0
    memory.advance_to(1)
    memory.update_identity(
        predicted_id,
        detection(embedding=(0.8, 0.6), center=(6.0, 5.0)),
        1,
    )

    identity = memory.get(predicted_id)
    assert identity.frames_since_last_observation == 0
    assert identity.previous_center == (6.0, 5.0)
    assert len(identity.appearance_history) == 2
    np.testing.assert_allclose(np.linalg.norm(identity.appearance_embedding), 1.0)


def test_memory_retains_configured_missing_gap_then_expires() -> None:
    memory = ObjectMemory(max_gap=2)
    memory.advance_to(0)
    predicted_id = memory.create_identity(detection(), 0)

    assert memory.advance_to(3) == ()  # Frames 1 and 2 were absent.
    assert memory.get(predicted_id).frames_since_last_observation == 3
    assert memory.get(predicted_id).missing_frames == 2
    assert memory.advance_to(4) == (predicted_id,)  # A third missing frame exceeds the limit.
    assert memory.active_identity_ids == ()


def test_memory_rejects_non_monotonic_frame_order_and_zero_embedding() -> None:
    memory = ObjectMemory()
    memory.advance_to(2)
    with pytest.raises(ValueError, match="must increase"):
        memory.advance_to(2)
    with pytest.raises(ValueError, match="nonzero norm"):
        memory.create_identity(detection(embedding=(0.0, 0.0)), 2)
