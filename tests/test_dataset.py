from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from occlusion_association.dataset import DAVISDataset, DatasetLayoutError


def make_davis(
    base: Path,
    *,
    nested: bool = False,
    train: tuple[str, ...] = ("sequence_a",),
    val: tuple[str, ...] = ("sequence_b",),
) -> Path:
    root = base / "DAVIS" if nested else base
    (root / "ImageSets" / "2017").mkdir(parents=True)
    (root / "ImageSets" / "2017" / "train.txt").write_text(
        "\n".join(train) + "\n", encoding="utf-8"
    )
    (root / "ImageSets" / "2017" / "val.txt").write_text(
        "\n".join(val) + "\n", encoding="utf-8"
    )
    return root


def add_frame(root: Path, sequence: str, frame: str, mask: np.ndarray) -> None:
    image_dir = root / "JPEGImages" / "480p" / sequence
    annotation_dir = root / "Annotations" / "480p" / sequence
    image_dir.mkdir(parents=True, exist_ok=True)
    annotation_dir.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (mask.shape[1], mask.shape[0]), (10, 20, 30)).save(
        image_dir / f"{frame}.jpg"
    )
    Image.fromarray(mask).save(annotation_dir / f"{frame}.png")


def test_nested_root_pairs_files_by_stem_and_excludes_background(tmp_path: Path) -> None:
    root = make_davis(tmp_path, nested=True)
    add_frame(root, "sequence_a", "00002", np.array([[0, 4], [4, 9]], dtype=np.uint8))
    add_frame(root, "sequence_a", "00001", np.array([[0, 0], [7, 7]], dtype=np.uint8))

    dataset = DAVISDataset(tmp_path, split="train")
    sequence = dataset.load_sequence("sequence_a")

    assert dataset.root == root
    assert [frame.frame_name for frame in sequence.frames] == ["00001", "00002"]
    assert sequence.frames[0].object_ids == (7,)
    assert sequence.frames[1].object_ids == (4, 9)
    assert not sequence.issues


def test_inspection_reports_missing_annotations_and_object_gaps(tmp_path: Path) -> None:
    root = make_davis(tmp_path)
    first = np.array([[0, 1], [1, 1]], dtype=np.uint8)
    absent = np.zeros((2, 2), dtype=np.uint8)
    add_frame(root, "sequence_a", "00000", first)
    add_frame(root, "sequence_a", "00001", absent)
    add_frame(root, "sequence_a", "00002", first)
    image_dir = root / "JPEGImages" / "480p" / "sequence_a"
    Image.new("RGB", (2, 2)).save(image_dir / "00003.jpg")

    report = DAVISDataset(root, split="train").inspect()
    sequence = report.inspected_sequences[0]

    assert sequence.image_frames == 4
    assert sequence.annotation_frames == 3
    assert sequence.matched_frames == 3
    assert sequence.unique_objects == 1
    assert sequence.object_lifecycles[0]["missing_gaps"] == [
        {
            "after_frame": "00000",
            "before_frame": "00002",
            "valid_annotation_frames_absent": 1,
        }
    ]
    assert any(issue.code == "missing_annotation" for issue in report.loading_errors)


def test_invalid_multichannel_mask_is_reported_and_frame_skipped(tmp_path: Path) -> None:
    root = make_davis(tmp_path)
    image_dir = root / "JPEGImages" / "480p" / "sequence_a"
    annotation_dir = root / "Annotations" / "480p" / "sequence_a"
    image_dir.mkdir(parents=True)
    annotation_dir.mkdir(parents=True)
    Image.new("RGB", (2, 2)).save(image_dir / "00000.jpg")
    Image.new("RGB", (2, 2), (1, 2, 3)).save(annotation_dir / "00000.png")

    dataset = DAVISDataset(root, split="train")
    sequence = dataset.load_sequence("sequence_a")

    assert sequence.frames == []
    assert [issue.code for issue in sequence.issues] == ["invalid_mask", "skipped_sequence"]


def test_missing_official_validation_split_fails_closed(tmp_path: Path) -> None:
    root = tmp_path / "DAVIS"
    (root / "JPEGImages" / "480p").mkdir(parents=True)
    (root / "Annotations" / "480p").mkdir(parents=True)
    (root / "ImageSets" / "2017").mkdir(parents=True)
    (root / "ImageSets" / "2017" / "train.txt").write_text("train_seq\n")

    with pytest.raises(DatasetLayoutError, match="Both official DAVIS train and validation"):
        DAVISDataset(root, split="train")


def test_all_split_retains_official_train_and_validation_counts(tmp_path: Path) -> None:
    root = make_davis(tmp_path, train=("sequence_a",), val=("sequence_b",))
    add_frame(root, "sequence_a", "00000", np.zeros((2, 2), dtype=np.uint8))
    add_frame(root, "sequence_b", "00000", np.zeros((2, 2), dtype=np.uint8))

    dataset = DAVISDataset(root, split="all")
    report = dataset.inspect()

    assert dataset.sequence_names == ["sequence_a", "sequence_b"]
    assert report.train_sequences == 1
    assert report.validation_sequences == 1
    assert report.requested_sequences == 2


def test_streaming_frame_iterator_matches_sequence_loader(tmp_path: Path) -> None:
    root = make_davis(tmp_path)
    add_frame(root, "sequence_a", "00000", np.zeros((2, 2), dtype=np.uint8))
    add_frame(root, "sequence_a", "00001", np.array([[0, 2], [2, 0]], dtype=np.uint8))
    dataset = DAVISDataset(root, split="train")
    issues = []

    streamed = list(dataset.iter_frames("sequence_a", issue_sink=issues))
    loaded = dataset.load_sequence("sequence_a")

    assert [frame.frame_name for frame in streamed] == [frame.frame_name for frame in loaded.frames]
    assert not issues
    assert not loaded.issues


def test_overlapping_official_splits_are_rejected_to_prevent_leakage(tmp_path: Path) -> None:
    root = make_davis(tmp_path, train=("shared",), val=("shared",))
    (root / "JPEGImages" / "480p").mkdir(parents=True)
    (root / "Annotations" / "480p").mkdir(parents=True)

    with pytest.raises(DatasetLayoutError, match="splits overlap"):
        DAVISDataset(root, split="all")
