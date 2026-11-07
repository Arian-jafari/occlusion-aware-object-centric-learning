"""Pairwise appearance, motion, geometry, and memory-age features."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .memory import IdentityMemory
from .object_representation import ObjectDetection


FEATURE_NAMES = (
    "cosine_similarity",
    "center_displacement",
    "bbox_iou",
    "area_ratio",
    "frames_since_last_observation",
    "similarity_margin",
)


@dataclass(frozen=True)
class AssociationFeatureVector:
    """Features for one memory-identity/detection candidate pair.

    Ranges: cosine similarity ``[-1, 1]``; normalized center displacement
    ``[0, 1]`` (Euclidean distance divided by image diagonal and clipped);
    previous/current box IoU ``[0, 1]``; symmetric area ratio ``[0, 1]``;
    normalized observation age ``[0, 1]`` (age divided by max-gap-plus-one,
    clipped); similarity margin ``[-1, 1]`` (pair cosine minus the best
    competing identity cosine, divided by two; zero if no competitor exists).
    All geometry uses pixel coordinates and half-open ``(x1,y1,x2,y2)`` boxes.
    """

    predicted_identity_id: int
    detection_id: int
    cosine_similarity: float
    center_displacement: float
    bbox_iou: float
    area_ratio: float
    frames_since_last_observation: float
    similarity_margin: float

    def to_array(self) -> np.ndarray:
        """Return a finite float32 vector in :data:`FEATURE_NAMES` order."""

        return np.asarray([getattr(self, name) for name in FEATURE_NAMES], dtype=np.float32)

    def to_dict(self) -> dict[str, float]:
        """Return feature values without identity bookkeeping fields."""

        return {name: float(getattr(self, name)) for name in FEATURE_NAMES}


def cosine_similarity(first: np.ndarray, second: np.ndarray) -> float:
    """Cosine similarity with safe behavior for invalid or zero vectors."""

    a = np.asarray(first, dtype=np.float32)
    b = np.asarray(second, dtype=np.float32)
    if a.ndim != 1 or b.ndim != 1 or a.shape != b.shape:
        return -1.0
    norm_a, norm_b = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if norm_a <= 1e-12 or norm_b <= 1e-12 or not np.isfinite(norm_a + norm_b):
        return -1.0
    return float(np.clip(np.dot(a, b) / (norm_a * norm_b), -1.0, 1.0))


def _box_iou(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> float:
    x1 = max(first[0], second[0])
    y1 = max(first[1], second[1])
    x2 = min(first[2], second[2])
    y2 = min(first[3], second[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_first = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    area_second = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = area_first + area_second - intersection
    return float(intersection / union) if union > 0 else 0.0


def compute_association_features(
    identities: Iterable[IdentityMemory],
    detections: Iterable[ObjectDetection],
    image_size: tuple[int, int],
    max_gap: int,
) -> dict[tuple[int, int], AssociationFeatureVector]:
    """Calculate all pair features independently of any match decision.

    Args:
        identities: Active predicted identities retained by memory.
        detections: Current frame's local detections; these contain no GT IDs.
        image_size: ``(width, height)`` of the original frame.
        max_gap: Memory retention in completely missing frames.
    """

    identity_list = list(identities)
    detection_list = list(detections)
    width, height = image_size
    if width <= 0 or height <= 0:
        raise ValueError("image_size dimensions must be positive")
    if max_gap < 0:
        raise ValueError("max_gap must be nonnegative")
    diagonal = max(float(np.hypot(width, height)), 1.0)
    similarities = {
        (identity.predicted_identity_id, detection.detection_id): cosine_similarity(
            identity.appearance_embedding, detection.embedding
        )
        for identity in identity_list
        for detection in detection_list
    }
    result: dict[tuple[int, int], AssociationFeatureVector] = {}
    for identity in identity_list:
        for detection in detection_list:
            pair = (identity.predicted_identity_id, detection.detection_id)
            cosine = similarities[pair]
            competitor_scores = [
                similarities[(other.predicted_identity_id, detection.detection_id)]
                for other in identity_list
                if other.predicted_identity_id != identity.predicted_identity_id
            ]
            margin = (
                float(np.clip((cosine - max(competitor_scores)) / 2.0, -1.0, 1.0))
                if competitor_scores
                else 0.0
            )
            center_distance = float(
                np.hypot(
                    identity.previous_center[0] - detection.center[0],
                    identity.previous_center[1] - detection.center[1],
                )
            )
            area_ratio = (
                min(identity.previous_area, detection.area)
                / max(identity.previous_area, detection.area)
                if max(identity.previous_area, detection.area) > 0
                else 0.0
            )
            age = min(
                max(identity.frames_since_last_observation, 0) / max(max_gap + 1, 1),
                1.0,
            )
            result[pair] = AssociationFeatureVector(
                predicted_identity_id=identity.predicted_identity_id,
                detection_id=detection.detection_id,
                cosine_similarity=cosine,
                center_displacement=float(np.clip(center_distance / diagonal, 0.0, 1.0)),
                bbox_iou=_box_iou(identity.previous_bbox, detection.bbox),
                area_ratio=float(area_ratio),
                frames_since_last_observation=float(age),
                similarity_margin=margin,
            )
    return result
