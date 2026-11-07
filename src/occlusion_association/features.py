"""Frozen DINOv2 spatial feature extraction and object-feature caching."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass
from itertools import islice
from typing import Any, Protocol, Sequence

import numpy as np
from PIL import Image

from .dataset import DAVISDataset, DatasetIssue, FrameData, SequenceData
from .feature_cache import FeatureCache
from .object_representation import object_ids_from_mask, pool_patch_features


class SpatialFeatureExtractor(Protocol):
    """Minimal interface needed by the object feature extraction pipeline."""

    backbone_name: str
    model_revision: str
    patch_size: int
    preprocessing: dict[str, Any]

    def extract_patch_tokens(
        self, images: Sequence[Image.Image]
    ) -> list[tuple[np.ndarray, tuple[int, int]]]:
        """Return ``(Hpatch x Wpatch x D tokens, resized (H, W))`` per image."""


@dataclass
class FeatureExtractionSummary:
    """Counts returned by one sequence's cached feature extraction."""

    sequence: str
    objects_requested: int
    cache_hits: int
    features_written: int
    invalid_objects: list[dict[str, Any]]


def build_preprocessing_descriptor(
    input_size: int,
    device: str = "auto",
    use_mixed_precision: bool = True,
) -> dict[str, Any]:
    """Return every input/precision setting that can change cached features."""

    if device == "auto":
        try:
            import torch

            device_type = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device_type = "cpu"
    else:
        device_type = "cuda" if str(device).startswith("cuda") else "cpu"
    precision = "cuda_fp16" if device_type == "cuda" and use_mixed_precision else "fp32"
    return {
        "input_size": int(input_size),
        "resize_policy": "preserve_aspect_round_dimensions_to_patch_multiple",
        "interpolation": "bicubic",
        "patch_size": 14,
        "normalization_mean": [0.485, 0.456, 0.406],
        "normalization_std": [0.229, 0.224, 0.225],
        "device_type": device_type,
        "precision_mode": precision,
    }


class DINOv2FeatureExtractor:
    """Frozen DINOv2 ViT-S/14 returning spatial patch tokens only.

    The model is loaded from the official ``facebookresearch/dinov2`` Torch Hub
    repository, set to evaluation mode, and frozen. It never returns or pools
    the global CLS token.
    """

    def __init__(
        self,
        backbone_name: str = "dinov2_vits14",
        repository: str = "facebookresearch/dinov2",
        input_size: int = 518,
        device: str = "auto",
        use_mixed_precision: bool = True,
        model: Any | None = None,
    ) -> None:
        if input_size < 14:
            raise ValueError("input_size must be at least one 14-pixel patch")
        self.backbone_name = f"{repository}:{backbone_name}"
        self.repository = repository
        self.patch_size = 14
        self.input_size = input_size
        self.use_mixed_precision = use_mixed_precision

        import torch

        self.torch = torch
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but this PyTorch runtime has no available CUDA device")
        self.device = torch.device(device)
        if model is None:
            try:
                model = torch.hub.load(repository, backbone_name, trust_repo=True)
            except Exception as exc:
                raise RuntimeError(
                    f"Could not load official DINOv2 model {repository}:{backbone_name}. "
                    "Check Colab network access or the Torch Hub cache."
                ) from exc
        self.model_revision = _model_state_revision(model, torch)
        self.model = model.to(self.device)
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

        self.preprocessing = build_preprocessing_descriptor(
            input_size,
            device=str(self.device),
            use_mixed_precision=use_mixed_precision,
        )

    def _resize_dimensions(self, image: Image.Image) -> tuple[int, int]:
        width, height = image.size
        if width <= 0 or height <= 0:
            raise ValueError("Cannot extract features from an image with an empty dimension")
        scale = self.input_size / max(width, height)
        resized_width = max(self.patch_size, int(round(width * scale / self.patch_size)) * self.patch_size)
        resized_height = max(self.patch_size, int(round(height * scale / self.patch_size)) * self.patch_size)
        return resized_width, resized_height

    def _prepare(self, image: Image.Image):
        torch = self.torch
        # Keep the image coordinate system unchanged so the ground-truth mask
        # remains spatially aligned with the image pixels.
        image = image.convert("RGB")
        resized_width, resized_height = self._resize_dimensions(image)
        image = image.resize(
            (resized_width, resized_height), resample=Image.Resampling.BICUBIC
        )
        array = np.asarray(image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array).permute(2, 0, 1)
        mean = torch.tensor([0.485, 0.456, 0.406], dtype=tensor.dtype)[:, None, None]
        std = torch.tensor([0.229, 0.224, 0.225], dtype=tensor.dtype)[:, None, None]
        tensor = (tensor - mean) / std
        return tensor, (resized_height, resized_width)

    def extract_patch_tokens(
        self, images: Sequence[Image.Image]
    ) -> list[tuple[np.ndarray, tuple[int, int]]]:
        """Extract normalized spatial patch tokens, supporting shape grouping."""

        if not images:
            return []
        prepared = [self._prepare(image) for image in images]
        groups: dict[tuple[int, int], list[int]] = defaultdict(list)
        for index, (_, (height, width)) in enumerate(prepared):
            groups[(height, width)].append(index)
        results: list[tuple[np.ndarray, tuple[int, int]] | None] = [None] * len(images)
        torch = self.torch

        for (height, width), indices in groups.items():
            batch = torch.stack([prepared[index][0] for index in indices]).to(self.device)
            autocast = (
                torch.autocast(device_type="cuda", dtype=torch.float16)
                if self.device.type == "cuda" and self.use_mixed_precision
                else _NullContext()
            )
            with torch.inference_mode(), autocast:
                features = self.model.forward_features(batch)
            if not isinstance(features, dict) or "x_norm_patchtokens" not in features:
                raise RuntimeError(
                    "DINOv2 forward_features did not return x_norm_patchtokens; "
                    "a global-only representation cannot be used for this project"
                )
            tokens = features["x_norm_patchtokens"]
            grid_height, grid_width = height // self.patch_size, width // self.patch_size
            expected_tokens = grid_height * grid_width
            if tokens.ndim != 3 or tokens.shape[1] != expected_tokens:
                raise RuntimeError(
                    f"Expected {expected_tokens} spatial patch tokens for {(height, width)}, "
                    f"got shape {tuple(tokens.shape)}"
                )
            tokens = tokens.reshape(len(indices), grid_height, grid_width, -1)
            tokens = tokens.detach().float().cpu().numpy()
            for local_index, original_index in enumerate(indices):
                results[original_index] = (
                    tokens[local_index],
                    (height, width),
                )
        return [result for result in results if result is not None]


class _NullContext:
    """Small context manager used when CUDA mixed precision is disabled."""

    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc_value, traceback):
        return False


def _model_state_revision(model: Any, torch: Any) -> str:
    """Hash the exact model state used to produce cached embeddings."""

    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        tensor = value.detach().to(device="cpu").contiguous()
        descriptor = {
            "name": str(name),
            "dtype": str(tensor.dtype),
            "shape": list(tensor.shape),
        }
        digest.update(json.dumps(descriptor, sort_keys=True).encode("utf-8"))
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return f"sha256:{digest.hexdigest()}"


def source_content_digests(image: Image.Image, mask: np.ndarray) -> tuple[str, str]:
    """Hash the decoded RGB image and exact mask array independently."""

    rgb = image.convert("RGB")
    image_digest = hashlib.sha256()
    image_digest.update(
        json.dumps({"mode": "RGB", "size": list(rgb.size)}, sort_keys=True).encode("utf-8")
    )
    image_digest.update(rgb.tobytes())

    source_mask = np.ascontiguousarray(mask)
    mask_digest = hashlib.sha256()
    mask_digest.update(
        json.dumps(
            {"dtype": str(source_mask.dtype), "shape": list(source_mask.shape)},
            sort_keys=True,
        ).encode("utf-8")
    )
    mask_digest.update(source_mask.tobytes())
    return image_digest.hexdigest(), mask_digest.hexdigest()


def make_object_feature_cache_key(
    cache: FeatureCache,
    *,
    sequence: str,
    frame_name: str,
    object_id: int,
    backbone_name: str,
    preprocessing: dict[str, Any],
    model_revision: str | None = None,
    source_image_digest: str | None = None,
    source_mask_digest: str | None = None,
) -> dict[str, Any]:
    return cache.make_key(
        sequence=sequence,
        frame=frame_name,
        object_id=object_id,
        backbone=backbone_name,
        preprocessing=preprocessing,
        model_revision=model_revision,
        source_image_digest=source_image_digest,
        source_mask_digest=source_mask_digest,
    )


def load_cached_frame_embeddings(
    frame: FrameData,
    cache: FeatureCache,
    *,
    backbone_name: str,
    preprocessing: dict[str, Any],
    model_revision: str | None = None,
) -> tuple[dict[int, np.ndarray], list[int]]:
    """Load cached embeddings by mask label and return missing object labels."""

    embeddings: dict[int, np.ndarray] = {}
    missing: list[int] = []
    image_digest, mask_digest = source_content_digests(frame.image, frame.mask)
    revision = model_revision or cache.model_revision_for(backbone_name)
    if revision is None:
        raise ValueError(
            f"No checkpoint fingerprint is recorded for {backbone_name}; run feature extraction first"
        )
    for object_id in object_ids_from_mask(frame.mask):
        key = make_object_feature_cache_key(
            cache,
            sequence=frame.sequence,
            frame_name=frame.frame_name,
            object_id=object_id,
            backbone_name=backbone_name,
            preprocessing=preprocessing,
            model_revision=revision,
            source_image_digest=image_digest,
            source_mask_digest=mask_digest,
        )
        embedding = cache.get(key)
        if embedding is None:
            missing.append(object_id)
        else:
            embeddings[object_id] = embedding
    return embeddings, missing


def extract_sequence_features(
    sequence: SequenceData,
    extractor: SpatialFeatureExtractor,
    cache: FeatureCache,
    batch_size: int = 4,
) -> FeatureExtractionSummary:
    """Extract and persist every valid object's spatially pooled embedding.

    Cache lookup happens before image inference. Frames are batched together;
    if their preprocessed sizes differ, the extractor groups compatible shapes.
    No ground-truth ID is stored as a predicted identity; IDs here only select
    the mask pixels whose visual features are pooled.
    """

    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    jobs: list[tuple[FrameData, list[int], str, str]] = []
    requested = 0
    cache_hits = 0
    invalid: list[dict[str, Any]] = []
    model_revision = getattr(extractor, "model_revision", None)
    if not model_revision:
        raise ValueError("Feature extractor must expose an exact model_revision fingerprint")
    model_revision = str(model_revision)
    cache.record_model_revision(extractor.backbone_name, model_revision)
    for frame in sequence.frames:
        object_ids = object_ids_from_mask(frame.mask)
        requested += len(object_ids)
        missing = []
        image_digest, mask_digest = source_content_digests(frame.image, frame.mask)
        for object_id in object_ids:
            key = make_object_feature_cache_key(
                cache,
                sequence=frame.sequence,
                frame_name=frame.frame_name,
                object_id=object_id,
                backbone_name=extractor.backbone_name,
                preprocessing=extractor.preprocessing,
                model_revision=model_revision,
                source_image_digest=image_digest,
                source_mask_digest=mask_digest,
            )
            if cache.get(key) is None:
                missing.append(object_id)
            else:
                cache_hits += 1
        if missing:
            jobs.append((frame, missing, image_digest, mask_digest))

    features_written = 0
    for start in range(0, len(jobs), batch_size):
        batch_jobs = jobs[start : start + batch_size]
        extracted = extractor.extract_patch_tokens([frame.image for frame, _, _, _ in batch_jobs])
        if len(extracted) != len(batch_jobs):
            raise RuntimeError("Feature extractor returned a different number of results than inputs")
        for (frame, object_ids, image_digest, mask_digest), (tokens, resized_size) in zip(
            batch_jobs, extracted
        ):
            for object_id in object_ids:
                embedding = pool_patch_features(
                    tokens,
                    frame.mask == object_id,
                    resized_size,
                    patch_size=extractor.patch_size,
                )
                if embedding is None:
                    invalid.append({"frame": frame.frame_name, "object_id": object_id})
                    continue
                cache.put(
                    make_object_feature_cache_key(
                        cache,
                        sequence=frame.sequence,
                        frame_name=frame.frame_name,
                        object_id=object_id,
                        backbone_name=extractor.backbone_name,
                        preprocessing=extractor.preprocessing,
                        model_revision=model_revision,
                        source_image_digest=image_digest,
                        source_mask_digest=mask_digest,
                    ),
                    embedding,
                )
                features_written += 1
    return FeatureExtractionSummary(
        sequence=sequence.name,
        objects_requested=requested,
        cache_hits=cache_hits,
        features_written=features_written,
        invalid_objects=invalid,
    )


def extract_dataset_features(
    dataset: DAVISDataset,
    sequence_names: Sequence[str],
    extractor: SpatialFeatureExtractor,
    cache: FeatureCache,
    *,
    batch_size: int = 4,
    frame_limit: int | None = None,
) -> tuple[list[FeatureExtractionSummary], list[DatasetIssue]]:
    """Stream selected sequences through bounded frame batches into the cache."""

    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if frame_limit is not None and frame_limit < 1:
        raise ValueError("frame_limit must be positive when provided")
    summaries: list[FeatureExtractionSummary] = []
    issues: list[DatasetIssue] = []
    for sequence_name in sequence_names:
        sequence_issues: list[DatasetIssue] = []
        iterator = dataset.iter_frames(sequence_name, issue_sink=sequence_issues)
        if frame_limit is not None:
            iterator = islice(iterator, frame_limit)
        total = FeatureExtractionSummary(sequence_name, 0, 0, 0, [])
        batch: list[FrameData] = []
        for frame in iterator:
            batch.append(frame)
            if len(batch) >= batch_size:
                chunk = extract_sequence_features(
                    SequenceData(name=sequence_name, frames=batch),
                    extractor,
                    cache,
                    batch_size=batch_size,
                )
                total.objects_requested += chunk.objects_requested
                total.cache_hits += chunk.cache_hits
                total.features_written += chunk.features_written
                total.invalid_objects.extend(chunk.invalid_objects)
                batch = []
        if batch:
            chunk = extract_sequence_features(
                SequenceData(name=sequence_name, frames=batch),
                extractor,
                cache,
                batch_size=batch_size,
            )
            total.objects_requested += chunk.objects_requested
            total.cache_hits += chunk.cache_hits
            total.features_written += chunk.features_written
            total.invalid_objects.extend(chunk.invalid_objects)
        summaries.append(total)
        issues.extend(sequence_issues)
    return summaries, issues
