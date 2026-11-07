"""Atomic on-disk object feature cache with preprocessing-aware keys."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping

import numpy as np


class FeatureCache:
    """Store one object embedding per identity/frame/configuration key.

    Cache entries are NumPy arrays outside the source tree. A SHA-256 digest of
    canonical key metadata makes changes to the backbone or preprocessing
    configuration produce a distinct entry.
    """

    CACHE_SCHEMA = 1

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory).expanduser()

    @classmethod
    def make_key(
        cls,
        *,
        sequence: str,
        frame: str,
        object_id: int,
        backbone: str,
        preprocessing: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Build the complete cache key for one object embedding."""

        return {
            "schema": cls.CACHE_SCHEMA,
            "sequence": str(sequence),
            "frame": str(frame),
            "object_id": int(object_id),
            "backbone": str(backbone),
            "preprocessing": dict(preprocessing),
        }

    def key_digest(self, key: Mapping[str, Any]) -> str:
        """Return a stable digest for JSON-compatible key metadata."""

        canonical = json.dumps(key, sort_keys=True, separators=(",", ":"), default=str)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def path_for(self, key: Mapping[str, Any]) -> Path:
        """Map a cache key into a sharded ``.npy`` path."""

        digest = self.key_digest(key)
        return self.directory / digest[:2] / f"{digest}.npy"

    def get(self, key: Mapping[str, Any]) -> np.ndarray | None:
        """Load a valid cached vector, returning ``None`` for a cache miss."""

        path = self.path_for(key)
        if not path.is_file():
            return None
        try:
            embedding = np.load(path, allow_pickle=False)
        except (OSError, ValueError):
            return None
        if embedding.ndim != 1 or embedding.size == 0 or not np.isfinite(embedding).all():
            return None
        return np.asarray(embedding, dtype=np.float32)

    def put(self, key: Mapping[str, Any], embedding: np.ndarray) -> Path:
        """Atomically save a finite one-dimensional feature vector."""

        vector = np.asarray(embedding, dtype=np.float32)
        if vector.ndim != 1 or vector.size == 0 or not np.isfinite(vector).all():
            raise ValueError("Cached embedding must be a nonempty finite 1D array")
        path = self.path_for(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        file_descriptor, temp_name = tempfile.mkstemp(
            prefix=f".{path.stem}.", suffix=".tmp", dir=path.parent
        )
        try:
            with os.fdopen(file_descriptor, "wb") as stream:
                np.save(stream, vector, allow_pickle=False)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temp_name, path)
        except BaseException:
            try:
                os.unlink(temp_name)
            except FileNotFoundError:
                pass
            raise
        return path
