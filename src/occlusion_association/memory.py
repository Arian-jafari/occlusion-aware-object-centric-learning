"""Predicted-identity memory independent of features and match algorithms."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from .object_representation import ObjectDetection


@dataclass
class IdentityMemory:
    """Appearance and last observed geometry for one predicted identity."""

    predicted_identity_id: int
    appearance_embedding: np.ndarray
    previous_center: tuple[float, float]
    previous_bbox: tuple[float, float, float, float]
    previous_area: int
    previous_observation_frame: int
    frames_since_last_observation: int = 0
    missing_frames: int = 0
    appearance_history: deque[np.ndarray] = field(default_factory=deque)


class ObjectMemory:
    """Maintain active predicted identities for a configurable missing gap.

    A maximum gap of N retains an identity through N complete absent frames.
    On a later observed frame, ``age = current_frame - last_observation`` and
    ``missing_frames = age - 1``. The identity expires when missing_frames > N.
    Thus an object last seen at frame 0 remains matchable at frame 31 for a
    30-frame gap (frames 1–30 were absent), and expires before frame 32.
    """

    def __init__(self, max_gap: int = 30, appearance_history_size: int = 5) -> None:
        if max_gap < 0:
            raise ValueError("max_gap must be nonnegative")
        if appearance_history_size < 1:
            raise ValueError("appearance_history_size must be positive")
        self.max_gap = int(max_gap)
        self.appearance_history_size = int(appearance_history_size)
        self._identities: dict[int, IdentityMemory] = {}
        self._next_identity_id = 1
        self.current_frame_index: int | None = None

    @property
    def active_identity_ids(self) -> tuple[int, ...]:
        """Predicted identity IDs currently retained in memory."""

        return tuple(sorted(self._identities))

    def active_identities(self) -> tuple[IdentityMemory, ...]:
        """Retrieve current identity records in predicted-ID order."""

        return tuple(self._identities[key] for key in sorted(self._identities))

    def get(self, predicted_identity_id: int) -> IdentityMemory:
        """Retrieve one active predicted identity or raise ``KeyError``."""

        return self._identities[predicted_identity_id]

    def advance_to(self, frame_index: int) -> tuple[int, ...]:
        """Advance time and expire identities beyond the configured gap.

        Returns the predicted IDs that expired on this call. Frame indices
        must be strictly increasing.
        """

        frame_index = int(frame_index)
        if self.current_frame_index is not None and frame_index <= self.current_frame_index:
            raise ValueError(
                f"frame_index must increase ({frame_index} <= {self.current_frame_index})"
            )
        self.current_frame_index = frame_index
        expired = []
        for predicted_id, identity in self._identities.items():
            age = max(0, frame_index - identity.previous_observation_frame)
            identity.frames_since_last_observation = age
            identity.missing_frames = max(0, age - 1)
            if identity.missing_frames > self.max_gap:
                expired.append(predicted_id)
        for predicted_id in expired:
            del self._identities[predicted_id]
        return tuple(sorted(expired))

    def _normalized_embedding(self, detection: ObjectDetection) -> np.ndarray:
        embedding = np.asarray(detection.embedding, dtype=np.float32)
        if embedding.ndim != 1 or embedding.size == 0 or not np.isfinite(embedding).all():
            raise ValueError("Detection embedding must be a nonempty finite vector")
        norm = float(np.linalg.norm(embedding))
        if norm <= 1e-12:
            raise ValueError("Detection embedding must have nonzero norm")
        return (embedding / norm).astype(np.float32)

    def create_identity(self, detection: ObjectDetection, frame_index: int | None = None) -> int:
        """Create a new predicted identity for an unmatched detection."""

        frame = self.current_frame_index if frame_index is None else int(frame_index)
        if frame is None:
            raise ValueError("Call advance_to or provide frame_index before creating an identity")
        if self.current_frame_index != frame:
            raise ValueError("New identities must be created on the current memory frame")
        identity_id = self._next_identity_id
        self._next_identity_id += 1
        embedding = self._normalized_embedding(detection)
        history: deque[np.ndarray] = deque(maxlen=self.appearance_history_size)
        history.append(embedding.copy())
        self._identities[identity_id] = IdentityMemory(
            predicted_identity_id=identity_id,
            appearance_embedding=embedding,
            previous_center=tuple(map(float, detection.center)),
            previous_bbox=tuple(map(float, detection.bbox)),
            previous_area=int(detection.area),
            previous_observation_frame=frame,
            appearance_history=history,
        )
        return identity_id

    def update_identity(
        self,
        predicted_identity_id: int,
        detection: ObjectDetection,
        frame_index: int | None = None,
    ) -> None:
        """Update one retained identity from an accepted association."""

        if predicted_identity_id not in self._identities:
            raise KeyError(f"Predicted identity {predicted_identity_id} is not active")
        frame = self.current_frame_index if frame_index is None else int(frame_index)
        if frame is None:
            raise ValueError("Call advance_to or provide frame_index before updating memory")
        if self.current_frame_index != frame:
            raise ValueError("Identities must be updated on the current memory frame")
        identity = self._identities[predicted_identity_id]
        embedding = self._normalized_embedding(detection)
        identity.appearance_embedding = embedding
        identity.previous_center = tuple(map(float, detection.center))
        identity.previous_bbox = tuple(map(float, detection.bbox))
        identity.previous_area = int(detection.area)
        identity.previous_observation_frame = frame
        identity.frames_since_last_observation = 0
        identity.missing_frames = 0
        identity.appearance_history.append(embedding.copy())
