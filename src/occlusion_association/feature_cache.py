"""Atomic on-disk object feature cache with content- and model-aware keys."""

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
    canonical key metadata includes input content and model-state fingerprints,
    so changes to the source or checkpoint produce distinct entries.
    """

    CACHE_SCHEMA = 2

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
        model_revision: str | None = None,
        source_image_digest: str | None = None,
        source_mask_digest: str | None = None,
    ) -> dict[str, Any]:
        """Build the complete cache key for one object embedding."""

        required_digests = {
            "model_revision": model_revision,
            "source_image_digest": source_image_digest,
            "source_mask_digest": source_mask_digest,
        }
        missing = [name for name, value in required_digests.items() if not value]
        if missing:
            raise ValueError(
                "Feature cache keys require exact model and source fingerprints: "
                + ", ".join(missing)
            )

        return {
            "schema": cls.CACHE_SCHEMA,
            "sequence": str(sequence),
            "frame": str(frame),
            "object_id": int(object_id),
            "backbone": str(backbone),
            "preprocessing": dict(preprocessing),
            "model_revision": str(model_revision),
            "source_image_digest": str(source_image_digest),
            "source_mask_digest": str(source_mask_digest),
        }

    def _model_revision_path(self, backbone: str) -> Path:
        identity = hashlib.sha256(str(backbone).encode("utf-8")).hexdigest()
        return self.directory / ".model-revisions" / f"{identity}.json"

    def record_model_revision(self, backbone: str, revision: str) -> None:
        """Record the checkpoint fingerprint used by the current extraction."""

        path = self._model_revision_path(backbone)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(
            {"backbone": str(backbone), "model_revision": str(revision)},
            sort_keys=True,
        )
        file_descriptor, temp_name = tempfile.mkstemp(
            prefix=f".{path.stem}.", suffix=".tmp", dir=path.parent
        )
        try:
            with os.fdopen(file_descriptor, "w", encoding="utf-8") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temp_name, path)
        except BaseException:
            try:
                os.unlink(temp_name)
            except FileNotFoundError:
                pass
            raise

    def model_revision_for(self, backbone: str) -> str | None:
        """Return the checkpoint fingerprint recorded by feature extraction."""

        path = self._model_revision_path(backbone)
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        if data.get("backbone") != str(backbone):
            return None
        revision = data.get("model_revision")
        return str(revision) if revision else None

    def key_digest(self, key: Mapping[str, Any]) -> str:
        """Return a stable digest for JSON-compatible key metadata."""

        canonical = json.dumps(key, sort_keys=True, separators=(",", ":"), default=str)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def path_for(self, key: Mapping[str, Any]) -> Path:
        """Map a cache key into a sharded ``.npy`` path."""

        if key.get("schema") != self.CACHE_SCHEMA:
            raise ValueError(f"Feature cache key schema must be {self.CACHE_SCHEMA}")
        missing = [
            name
            for name in ("model_revision", "source_image_digest", "source_mask_digest")
            if not key.get(name)
        ]
        if missing:
            raise ValueError(
                "Feature cache keys require exact model and source fingerprints: "
                + ", ".join(missing)
            )
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
