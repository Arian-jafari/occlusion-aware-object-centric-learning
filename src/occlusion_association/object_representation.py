"""Mask-aware pooling of spatial visual tokens into object embeddings."""

from __future__ import annotations

import numpy as np
from PIL import Image
from dataclasses import dataclass


@dataclass(frozen=True)
class ObjectDetection:
    """One frame-local object detection, without any ground-truth identity."""

    detection_id: int
    embedding: np.ndarray
    center: tuple[float, float]
    bbox: tuple[float, float, float, float]
    area: int


def pool_patch_features(
    patch_tokens: np.ndarray,
    object_mask: np.ndarray,
    resized_size: tuple[int, int],
    patch_size: int = 14,
) -> np.ndarray | None:
    """Area-pool spatial patch tokens over one resized object mask.

    Each patch contributes in proportion to the fraction of its image region
    covered by the object. This preserves partial coverage for tiny objects
    that overlap only a few patches. The pooled vector is L2-normalized.

    Args:
        patch_tokens: Spatial features shaped ``(grid_height, grid_width, dim)``.
            The global CLS token must not be passed here.
        object_mask: A two-dimensional binary or label mask at original image
            resolution. Every nonzero pixel is treated as part of this object.
        resized_size: ``(height, width)`` used to preprocess the image for the
            feature backbone. Both dimensions must be multiples of
            ``patch_size``.
        patch_size: Backbone patch width/height in pixels (14 for DINOv2 ViT-S/14).

    Returns:
        Unit-length float32 embedding, or ``None`` if the mask is empty at the
        resized resolution, the feature tensor is invalid, or the pooled vector
        has zero/non-finite magnitude.
    """

    tokens = np.asarray(patch_tokens, dtype=np.float32)
    mask = np.asarray(object_mask)
    if tokens.ndim != 3:
        raise ValueError(f"patch_tokens must have shape (grid_h, grid_w, dim), got {tokens.shape}")
    if mask.ndim != 2:
        raise ValueError(f"object_mask must be two-dimensional, got {mask.shape}")
    if patch_size <= 0:
        raise ValueError("patch_size must be positive")
    height, width = resized_size
    if height <= 0 or width <= 0 or height % patch_size or width % patch_size:
        raise ValueError(
            f"resized_size {resized_size} must be positive and divisible by patch_size={patch_size}"
        )
    expected_grid = (height // patch_size, width // patch_size)
    if tokens.shape[:2] != expected_grid:
        raise ValueError(
            f"Patch grid {tokens.shape[:2]} does not match resized image grid {expected_grid}"
        )
    if tokens.shape[2] == 0 or not np.isfinite(tokens).all():
        return None
    if mask.size == 0 or not np.any(mask):
        return None

    binary = Image.fromarray((mask != 0).astype(np.uint8) * 255)
    resized = binary.resize((width, height), resample=Image.Resampling.BOX)
    occupancy = np.asarray(resized, dtype=np.float32) / 255.0
    occupancy = occupancy.reshape(
        height // patch_size,
        patch_size,
        width // patch_size,
        patch_size,
    ).mean(axis=(1, 3))
    total_weight = float(occupancy.sum())
    if total_weight <= 0.0 or not np.isfinite(total_weight):
        return None

    pooled = np.einsum("hw,hwd->d", occupancy, tokens, optimize=True) / total_weight
    norm = float(np.linalg.norm(pooled))
    if norm <= 1e-12 or not np.isfinite(norm):
        return None
    return (pooled / norm).astype(np.float32)


def object_ids_from_mask(instance_mask: np.ndarray) -> tuple[int, ...]:
    """Return positive instance labels, excluding background label zero."""

    mask = np.asarray(instance_mask)
    if mask.ndim != 2:
        raise ValueError(f"instance_mask must be two-dimensional, got {mask.shape}")
    if not np.issubdtype(mask.dtype, np.integer):
        raise ValueError(f"instance_mask must contain integer labels, got {mask.dtype}")
    return tuple(int(label) for label in np.unique(mask) if int(label) > 0)


def build_object_detections(
    instance_mask: np.ndarray,
    embeddings_by_object_id: dict[int, np.ndarray],
) -> tuple[list[ObjectDetection], dict[int, int], list[int]]:
    """Build geometry-sorted local detections and a separate target mapping.

    The returned detection objects contain no ground-truth identity labels.
    The mapping from local detection IDs to mask labels is returned separately
    for calibration-target generation and evaluation only. Association code
    should receive only the first tuple item.

    Objects without a valid embedding are excluded from the detections and
    listed in the third return value.
    """

    mask = np.asarray(instance_mask)
    object_records = []
    missing_embeddings: list[int] = []
    for object_id in object_ids_from_mask(mask):
        binary = mask == object_id
        embedding = embeddings_by_object_id.get(object_id)
        if embedding is None:
            missing_embeddings.append(object_id)
            continue
        embedding = np.asarray(embedding, dtype=np.float32)
        norm = float(np.linalg.norm(embedding)) if embedding.ndim == 1 else 0.0
        if embedding.ndim != 1 or embedding.size == 0 or not np.isfinite(embedding).all() or norm <= 1e-12:
            missing_embeddings.append(object_id)
            continue
        embedding = embedding / norm
        ys, xs = np.nonzero(binary)
        x1, x2 = int(xs.min()), int(xs.max()) + 1
        y1, y2 = int(ys.min()), int(ys.max()) + 1
        center = (float(xs.mean()), float(ys.mean()))
        # The bitmap bytes provide a deterministic geometric tie-breaker that
        # does not depend on the ground-truth object label.
        tie_break = np.packbits(binary.reshape(-1)).tobytes()
        sort_key = (center[1], center[0], y1, x1, y2, x2, int(binary.sum()), tie_break)
        object_records.append(
            (
                sort_key,
                int(object_id),
                embedding.astype(np.float32),
                center,
                (float(x1), float(y1), float(x2), float(y2)),
                int(binary.sum()),
            )
        )
    object_records.sort(key=lambda record: record[0])

    detections: list[ObjectDetection] = []
    ground_truth_id_by_detection: dict[int, int] = {}
    for detection_id, record in enumerate(object_records):
        _, object_id, embedding, center, bbox, area = record
        detections.append(ObjectDetection(detection_id, embedding, center, bbox, area))
        ground_truth_id_by_detection[detection_id] = object_id
    return detections, ground_truth_id_by_detection, missing_embeddings
