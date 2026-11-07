from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from occlusion_association.dataset import FrameData, SequenceData
from occlusion_association.feature_cache import FeatureCache
from occlusion_association.features import extract_sequence_features, source_content_digests
from occlusion_association.features import build_preprocessing_descriptor
from occlusion_association.object_representation import (
    object_ids_from_mask,
    pool_patch_features,
)


def test_pooling_uses_spatial_tokens_and_normalizes_embedding() -> None:
    tokens = np.array(
        [
            [[3.0, 0.0], [0.0, 1.0]],
            [[0.0, 1.0], [0.0, 1.0]],
        ],
        dtype=np.float32,
    )
    mask = np.zeros((28, 28), dtype=np.uint8)
    mask[:14, :14] = 1

    embedding = pool_patch_features(tokens, mask, resized_size=(28, 28))

    assert embedding is not None
    np.testing.assert_allclose(embedding, np.array([1.0, 0.0]), atol=1e-6)
    np.testing.assert_allclose(np.linalg.norm(embedding), 1.0, atol=1e-6)


def test_pooling_preserves_tiny_object_coverage_and_handles_empty_masks() -> None:
    tokens = np.array([[[0.0, 2.0]]], dtype=np.float32)
    tiny_mask = np.zeros((28, 28), dtype=np.uint8)
    tiny_mask[13, 13] = 1

    tiny_embedding = pool_patch_features(tokens, tiny_mask, resized_size=(14, 14))
    empty_embedding = pool_patch_features(
        tokens, np.zeros_like(tiny_mask), resized_size=(14, 14)
    )

    assert tiny_embedding is not None
    np.testing.assert_allclose(tiny_embedding, np.array([0.0, 1.0]), atol=1e-6)
    assert empty_embedding is None


def test_object_id_extraction_omits_background() -> None:
    mask = np.array([[0, 3], [9, 9]], dtype=np.uint8)
    assert object_ids_from_mask(mask) == (3, 9)


def test_cache_key_changes_with_preprocessing_and_round_trips(tmp_path: Path) -> None:
    cache = FeatureCache(tmp_path)
    key = cache.make_key(
        sequence="seq",
        frame="00001",
        object_id=2,
        backbone="repo:model",
        preprocessing={"input_size": 518},
        model_revision="checkpoint-a",
        source_image_digest="image-a",
        source_mask_digest="mask-a",
    )
    changed_key = cache.make_key(
        sequence="seq",
        frame="00001",
        object_id=2,
        backbone="repo:model",
        preprocessing={"input_size": 392},
        model_revision="checkpoint-a",
        source_image_digest="image-a",
        source_mask_digest="mask-a",
    )
    vector = np.array([0.6, 0.8], dtype=np.float32)

    assert cache.get(key) is None
    path = cache.put(key, vector)
    np.testing.assert_allclose(cache.get(key), vector)
    assert cache.get(changed_key) is None
    assert path.is_file()


def test_cache_keys_reject_missing_model_or_source_fingerprints(tmp_path: Path) -> None:
    cache = FeatureCache(tmp_path)
    with pytest.raises(ValueError, match="require exact model and source fingerprints"):
        cache.make_key(
            sequence="seq",
            frame="00001",
            object_id=2,
            backbone="repo:model",
            preprocessing={"input_size": 518},
        )


def test_preprocessing_descriptor_invalidates_device_precision_changes() -> None:
    mixed = build_preprocessing_descriptor(518, device="cuda:0", use_mixed_precision=True)
    full_precision = build_preprocessing_descriptor(
        518, device="cuda:0", use_mixed_precision=False
    )
    cpu = build_preprocessing_descriptor(518, device="cpu", use_mixed_precision=True)

    assert mixed["precision_mode"] == "cuda_fp16"
    assert full_precision["precision_mode"] == "fp32"
    assert cpu["device_type"] == "cpu"
    assert mixed != full_precision
    assert full_precision != cpu


class _FakeSpatialExtractor:
    backbone_name = "test:spatial"
    model_revision = "test-checkpoint-1"
    patch_size = 14
    preprocessing = {"test_resolution": 14}

    def __init__(self) -> None:
        self.calls = 0

    def extract_patch_tokens(self, images):
        self.calls += 1
        return [
            (np.array([[[float(index + 1), 1.0]]], dtype=np.float32), (14, 14))
            for index, _ in enumerate(images)
        ]


def test_sequence_extraction_caches_each_object_embedding(tmp_path: Path) -> None:
    image = Image.new("RGB", (14, 14), (40, 80, 120))
    mask = np.zeros((14, 14), dtype=np.uint8)
    mask[:, :7] = 1
    mask[:, 7:] = 2
    frame = FrameData(
        sequence="seq",
        frame_name="00000",
        image=image,
        mask=mask,
        object_ids=(1, 2),
        image_path=tmp_path / "00000.jpg",
        annotation_path=tmp_path / "00000.png",
    )
    sequence = SequenceData(name="seq", frames=[frame])
    extractor = _FakeSpatialExtractor()
    cache = FeatureCache(tmp_path / "features")

    first = extract_sequence_features(sequence, extractor, cache, batch_size=2)
    second = extract_sequence_features(sequence, extractor, cache, batch_size=2)

    assert first.objects_requested == 2
    assert first.features_written == 2
    assert first.cache_hits == 0
    assert second.cache_hits == 2
    assert second.features_written == 0
    assert extractor.calls == 1


def test_sequence_cache_invalidates_source_contents_and_model_revision(tmp_path: Path) -> None:
    image = Image.new("RGB", (14, 14), (40, 80, 120))
    mask = np.zeros((14, 14), dtype=np.uint8)
    mask[:, :7] = 1
    mask[:, 7:] = 2
    frame = FrameData(
        sequence="seq",
        frame_name="00000",
        image=image,
        mask=mask,
        object_ids=(1, 2),
        image_path=tmp_path / "00000.jpg",
        annotation_path=tmp_path / "00000.png",
    )
    sequence = SequenceData(name="seq", frames=[frame])
    extractor = _FakeSpatialExtractor()
    cache = FeatureCache(tmp_path / "features")

    assert extract_sequence_features(sequence, extractor, cache).features_written == 2
    assert extract_sequence_features(sequence, extractor, cache).cache_hits == 2

    frame.image.putpixel((0, 0), (200, 80, 120))
    changed_image = extract_sequence_features(sequence, extractor, cache)
    assert changed_image.features_written == 2
    assert changed_image.cache_hits == 0

    frame.mask = np.fliplr(frame.mask).copy()
    changed_mask = extract_sequence_features(sequence, extractor, cache)
    assert changed_mask.features_written == 2
    assert changed_mask.cache_hits == 0

    extractor.model_revision = "test-checkpoint-2"
    changed_checkpoint = extract_sequence_features(sequence, extractor, cache)
    assert changed_checkpoint.features_written == 2
    assert changed_checkpoint.cache_hits == 0
    assert cache.model_revision_for(extractor.backbone_name) == "test-checkpoint-2"


def test_source_digests_distinguish_image_and_mask_content() -> None:
    image = Image.new("RGB", (2, 2), (1, 2, 3))
    mask = np.zeros((2, 2), dtype=np.uint8)
    image_digest, mask_digest = source_content_digests(image, mask)

    image.putpixel((0, 0), (3, 2, 1))
    changed_image_digest, same_mask_digest = source_content_digests(image, mask)
    changed_mask_digest = source_content_digests(image, np.ones_like(mask))[1]

    assert changed_image_digest != image_digest
    assert same_mask_digest == mask_digest
    assert changed_mask_digest != mask_digest


def test_dinov2_wrapper_uses_frozen_spatial_tokens_on_cpu() -> None:
    import pytest

    torch = pytest.importorskip("torch")
    from occlusion_association.features import DINOv2FeatureExtractor

    class _TinySpatialModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.ones(1))

        def forward_features(self, batch):
            patches = batch.unfold(2, 14, 14).unfold(3, 14, 14)
            means = patches.mean(dim=(-1, -2)).permute(0, 2, 3, 1)
            batch_size = batch.shape[0]
            return {
                "x_norm_patchtokens": means.reshape(batch_size, -1, 3),
                "x_norm_clstoken": torch.full((batch_size, 3), 1000.0),
            }

    model = _TinySpatialModel()
    extractor = DINOv2FeatureExtractor(
        backbone_name="tiny-test-model",
        repository="local-test",
        input_size=28,
        device="cpu",
        model=model,
    )
    tokens, resized_size = extractor.extract_patch_tokens([Image.new("RGB", (28, 28))])[0]

    assert extractor.device.type == "cpu"
    assert extractor.model_revision.startswith("sha256:")
    assert model.training is False
    assert all(not parameter.requires_grad for parameter in model.parameters())
    assert tokens.shape == (2, 2, 3)
    assert resized_size == (28, 28)
    assert np.max(tokens) < 1000.0  # The deliberately large CLS token was not included.
