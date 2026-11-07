"""DAVIS 2017 TrainVal dataset discovery, loading, and inspection.

Ground-truth instance masks are exposed as object detections by this module.
They are used to isolate identity association from segmentation quality.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass, field
from itertools import islice
from pathlib import Path
from typing import Any, Iterator, Literal

import numpy as np
from PIL import Image, UnidentifiedImageError


SplitName = Literal["train", "val", "all"]
_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


class DatasetLayoutError(RuntimeError):
    """Raised when the official DAVIS TrainVal layout cannot be validated."""


@dataclass(frozen=True)
class DatasetIssue:
    """A missing, invalid, or skipped dataset item found by the loader."""

    severity: str
    code: str
    message: str
    sequence: str | None = None
    frame: str | None = None
    path: str | None = None


@dataclass(frozen=True)
class FrameData:
    """Decoded RGB image and indexed instance mask for one matched frame."""

    sequence: str
    frame_name: str
    image: Image.Image
    mask: np.ndarray
    object_ids: tuple[int, ...]
    image_path: Path
    annotation_path: Path


@dataclass
class SequenceData:
    """Valid matched frames and frame-level errors for a DAVIS sequence."""

    name: str
    frames: list[FrameData] = field(default_factory=list)
    issues: list[DatasetIssue] = field(default_factory=list)


@dataclass
class SequenceInspection:
    """Counts and object lifecycle observations for one requested sequence."""

    sequence: str
    image_frames: int
    annotation_frames: int
    matched_frames: int
    valid_frames: int
    unique_objects: int
    objects_per_frame: dict[str, int]
    image_dimensions: list[tuple[int, int]]
    annotation_dimensions: list[tuple[int, int]]
    object_lifecycles: list[dict[str, Any]]
    skipped: bool


@dataclass
class DatasetInspection:
    """Serializable report produced by :meth:`DAVISDataset.inspect`."""

    dataset_root: str
    split: str
    train_sequences: int
    validation_sequences: int
    requested_sequences: int
    inspected_sequences: list[SequenceInspection]
    loading_errors: list[DatasetIssue]

    def to_dict(self) -> dict[str, Any]:
        """Convert the report to JSON-serializable Python values."""

        return asdict(self)


@dataclass(frozen=True)
class _SequenceEntry:
    name: str
    image_paths: dict[str, Path]
    annotation_paths: dict[str, Path]
    discovery_issues: tuple[DatasetIssue, ...] = ()


def resolve_davis_root(root: str | Path) -> Path:
    """Find a DAVIS root, including the common extra ``DAVIS/`` nesting."""

    base = Path(root).expanduser()
    candidates = [base, base / "DAVIS", base / "DAVIS" / "DAVIS"]
    if base.name.lower() in {"davis", "davis2017"}:
        candidates.extend([base.parent / "DAVIS", base.parent / "DAVIS" / "DAVIS"])

    seen: set[Path] = set()
    for candidate in candidates:
        try:
            candidate = candidate.resolve()
        except OSError:
            candidate = candidate.absolute()
        if candidate in seen:
            continue
        seen.add(candidate)
        if (candidate / "JPEGImages" / "480p").is_dir() and (
            candidate / "Annotations" / "480p"
        ).is_dir():
            return candidate
    expected = " or ".join(str(path) for path in candidates[:3])
    raise DatasetLayoutError(
        f"Could not find JPEGImages/480p and Annotations/480p below {base}. "
        f"Expected a DAVIS root at one of: {expected}"
    )


def _read_split_file(path: Path) -> list[str]:
    names = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not names:
        raise DatasetLayoutError(f"Official split file is empty: {path}")
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise DatasetLayoutError(
            f"Duplicate sequence names in {path}: {', '.join(duplicates)}"
        )
    unsafe_names = [
        name
        for name in names
        if Path(name).name != name
        or "/" in name
        or "\\" in name
        or name in {".", ".."}
    ]
    if unsafe_names:
        raise DatasetLayoutError(
            f"Invalid sequence names in {path}: {', '.join(unsafe_names)}"
        )
    return names


def _split_file_paths(root: Path, train_split: str, validation_split: str) -> tuple[Path, Path]:
    """Locate both official split files; never infer validation from training."""

    split_dirs = [root / "ImageSets" / "2017", root / "ImageSets" / "480p", root / "ImageSets"]
    for split_dir in split_dirs:
        train_path = split_dir / f"{train_split}.txt"
        validation_path = split_dir / f"{validation_split}.txt"
        if train_path.is_file() and validation_path.is_file():
            return train_path, validation_path
    searched = ", ".join(str(path) for path in split_dirs)
    raise DatasetLayoutError(
        "Both official DAVIS train and validation split files are required. "
        f"Looked for {train_split}.txt and {validation_split}.txt in: {searched}. "
        "Install the official TrainVal release; training sequences will not be "
        "reused for validation."
    )


def _files_by_stem(directory: Path, suffixes: set[str]) -> dict[str, Path]:
    if not directory.is_dir():
        return {}
    files: dict[str, Path] = {}
    for path in sorted(directory.iterdir(), key=lambda item: item.name):
        if path.is_file() and path.suffix.lower() in suffixes:
            if path.stem in files:
                # Deterministically retain the first filename; the duplicate is
                # reported by the directory scan in _sequence_entry.
                continue
            files[path.stem] = path
    return files


class DAVISDataset:
    """Read DAVIS 2017 TrainVal sequences and validate frame annotations.

    Args:
        root: DAVIS directory or its parent. A nested ``DAVIS/DAVIS`` layout is
            recognized.
        split: ``train``, ``val``, or ``all``. Both official split files are
            always required before any split is used.
        train_split: Official training split filename stem.
        validation_split: Official validation split filename stem.
    """

    def __init__(
        self,
        root: str | Path,
        split: SplitName = "train",
        train_split: str = "train",
        validation_split: str = "val",
    ) -> None:
        if split not in {"train", "val", "all"}:
            raise ValueError("split must be 'train', 'val', or 'all'")
        self.root = resolve_davis_root(root)
        self.split = split
        train_path, val_path = _split_file_paths(self.root, train_split, validation_split)
        self.train_sequences = _read_split_file(train_path)
        self.validation_sequences = _read_split_file(val_path)
        split_overlap = sorted(set(self.train_sequences) & set(self.validation_sequences))
        if split_overlap:
            raise DatasetLayoutError(
                "Official train and validation splits overlap, which would cause "
                f"data leakage: {', '.join(split_overlap)}"
            )
        if split == "train":
            names = self.train_sequences
        elif split == "val":
            names = self.validation_sequences
        else:
            names = list(dict.fromkeys(self.train_sequences + self.validation_sequences))
        self._entries = [self._make_entry(name) for name in names]

    def _make_entry(self, name: str) -> _SequenceEntry:
        image_dir = self.root / "JPEGImages" / "480p" / name
        annotation_dir = self.root / "Annotations" / "480p" / name
        issues: list[DatasetIssue] = []
        if not image_dir.is_dir():
            issues.append(
                DatasetIssue("error", "missing_image_directory", f"No image directory for {name}", name, path=str(image_dir))
            )
        if not annotation_dir.is_dir():
            issues.append(
                DatasetIssue("error", "missing_annotation_directory", f"No annotation directory for {name}", name, path=str(annotation_dir))
            )
        image_paths = _files_by_stem(image_dir, _IMAGE_SUFFIXES)
        annotation_paths = _files_by_stem(annotation_dir, {".png"})
        for directory, suffixes, label in (
            (image_dir, _IMAGE_SUFFIXES, "image"),
            (annotation_dir, {".png"}, "annotation"),
        ):
            if directory.is_dir():
                seen_stems: set[str] = set()
                for path in sorted(directory.iterdir(), key=lambda item: item.name):
                    if path.is_file() and path.suffix.lower() in suffixes:
                        if path.stem in seen_stems:
                            issues.append(
                                DatasetIssue("error", f"duplicate_{label}_stem", f"More than one {label} file has stem {path.stem}", name, path.stem, str(path))
                            )
                        seen_stems.add(path.stem)

        for frame in sorted(image_paths.keys() - annotation_paths.keys()):
            issues.append(
                DatasetIssue("error", "missing_annotation", f"Image frame {frame} has no matching annotation", name, frame, str(image_paths[frame]))
            )
        for frame in sorted(annotation_paths.keys() - image_paths.keys()):
            issues.append(
                DatasetIssue("error", "orphan_annotation", f"Annotation frame {frame} has no matching image", name, frame, str(annotation_paths[frame]))
            )
        matched = image_paths.keys() & annotation_paths.keys()
        if not matched:
            issues.append(
                DatasetIssue("error", "skipped_sequence", f"Sequence {name} has no filename-matched image/annotation pairs and will be skipped", name)
            )
        return _SequenceEntry(name, image_paths, annotation_paths, tuple(issues))

    @property
    def sequence_names(self) -> list[str]:
        """Sequence names selected by the configured official split."""

        return [entry.name for entry in self._entries]

    def _decode_frame(
        self, entry: _SequenceEntry, frame_name: str
    ) -> tuple[FrameData | None, list[DatasetIssue]]:
        image_path = entry.image_paths[frame_name]
        annotation_path = entry.annotation_paths[frame_name]
        issues: list[DatasetIssue] = []
        try:
            with Image.open(image_path) as image_file:
                image = image_file.convert("RGB")
                image.load()
        except (OSError, UnidentifiedImageError, ValueError) as exc:
            return None, [
                DatasetIssue("error", "invalid_image", f"Could not decode image: {exc}", entry.name, frame_name, str(image_path))
            ]
        try:
            with Image.open(annotation_path) as annotation_file:
                mask = np.array(annotation_file, copy=True)
        except (OSError, UnidentifiedImageError, ValueError) as exc:
            return None, [
                DatasetIssue("error", "invalid_mask", f"Could not decode annotation: {exc}", entry.name, frame_name, str(annotation_path))
            ]

        if mask.ndim != 2 or not np.issubdtype(mask.dtype, np.integer):
            return None, [
                DatasetIssue(
                    "error",
                    "invalid_mask",
                    f"Expected a single-channel integer instance mask; got shape={mask.shape}, dtype={mask.dtype}",
                    entry.name,
                    frame_name,
                    str(annotation_path),
                )
            ]
        if np.any(mask < 0):
            return None, [
                DatasetIssue(
                    "error",
                    "invalid_mask",
                    "Instance masks must use nonnegative labels with 0 reserved for background",
                    entry.name,
                    frame_name,
                    str(annotation_path),
                )
            ]
        if mask.shape[1] != image.width or mask.shape[0] != image.height:
            return None, [
                DatasetIssue(
                    "error",
                    "dimension_mismatch",
                    f"Image dimensions {(image.width, image.height)} do not match annotation dimensions {(mask.shape[1], mask.shape[0])}",
                    entry.name,
                    frame_name,
                    str(annotation_path),
                )
            ]
        object_ids = tuple(int(label) for label in np.unique(mask) if int(label) != 0)
        return (
            FrameData(
                entry.name,
                frame_name,
                image,
                mask,
                object_ids,
                image_path,
                annotation_path,
            ),
            issues,
        )

    def load_sequence(self, name: str) -> SequenceData:
        """Decode a full sequence into memory for small datasets and tests."""

        issues: list[DatasetIssue] = []
        frames = list(self.iter_frames(name, issue_sink=issues))
        return SequenceData(name=name, frames=frames, issues=issues)

    def iter_frames(
        self,
        name: str,
        issue_sink: list[DatasetIssue] | None = None,
    ) -> Iterator[FrameData]:
        """Decode matched frames one at a time for bounded-memory workflows.

        If provided, ``issue_sink`` is populated with discovery and decode
        issues as iteration proceeds. Invalid frames are skipped and reported;
        a sequence with no valid frames receives a ``skipped_sequence`` issue.
        """

        entry = next((candidate for candidate in self._entries if candidate.name == name), None)
        if entry is None:
            raise KeyError(f"Sequence {name!r} is not part of split {self.split!r}")
        if issue_sink is not None:
            issue_sink.extend(entry.discovery_issues)
        valid_count = 0
        for frame_name in sorted(entry.image_paths.keys() & entry.annotation_paths.keys()):
            frame, frame_issues = self._decode_frame(entry, frame_name)
            if issue_sink is not None:
                issue_sink.extend(frame_issues)
            if frame is not None:
                valid_count += 1
                yield frame
        if valid_count == 0 and issue_sink is not None and not any(
            issue.code == "skipped_sequence" and issue.sequence == name
            for issue in issue_sink
        ):
            issue_sink.append(
                DatasetIssue("error", "skipped_sequence", f"Sequence {name} has no valid matched frames and will be skipped", name)
            )

    def __iter__(self) -> Iterator[SequenceData]:
        """Iterate sequences in split-file order."""

        for name in self.sequence_names:
            yield self.load_sequence(name)

    def inspect(
        self,
        sequence_names: list[str] | None = None,
        frame_limit: int | None = None,
    ) -> DatasetInspection:
        """Inspect selected sequences without holding a sequence in memory.

        ``sequence_names`` and ``frame_limit`` support a small Colab smoke
        inspection; the default inspects every sequence in the selected split.
        """

        if frame_limit is not None and frame_limit < 1:
            raise ValueError("frame_limit must be positive when provided")
        if sequence_names is None:
            entries = self._entries
        else:
            lookup = {entry.name: entry for entry in self._entries}
            unknown = [name for name in sequence_names if name not in lookup]
            if unknown:
                raise KeyError(f"Sequences are not part of split {self.split!r}: {unknown}")
            entries = [lookup[name] for name in sequence_names]

        reports: list[SequenceInspection] = []
        all_issues: list[DatasetIssue] = []
        for entry in entries:
            sequence_issues: list[DatasetIssue] = []
            seen_frames: dict[int, list[str]] = defaultdict(list)
            objects_per_frame: dict[str, int] = {}
            image_dimensions: set[tuple[int, int]] = set()
            annotation_dimensions: set[tuple[int, int]] = set()
            ordered_frames: list[str] = []
            frame_iterator = self.iter_frames(entry.name, issue_sink=sequence_issues)
            if frame_limit is not None:
                frame_iterator = islice(frame_iterator, frame_limit)
            for frame in frame_iterator:
                ordered_frames.append(frame.frame_name)
                objects_per_frame[frame.frame_name] = len(frame.object_ids)
                image_dimensions.add(frame.image.size)
                annotation_dimensions.add((frame.mask.shape[1], frame.mask.shape[0]))
                for object_id in frame.object_ids:
                    seen_frames[object_id].append(frame.frame_name)
            all_issues.extend(sequence_issues)

            lifecycles: list[dict[str, Any]] = []
            frame_indices = {frame_name: index for index, frame_name in enumerate(ordered_frames)}
            for object_id, observed in sorted(seen_frames.items()):
                indices = [frame_indices[frame_name] for frame_name in observed]
                missing_gaps = []
                for left, right in zip(indices, indices[1:]):
                    if right - left > 1:
                        missing_gaps.append(
                            {
                                "after_frame": ordered_frames[left],
                                "before_frame": ordered_frames[right],
                                "valid_annotation_frames_absent": right - left - 1,
                            }
                        )
                lifecycles.append(
                    {
                        "object_id": object_id,
                        "first_observed_frame": observed[0],
                        "last_observed_frame": observed[-1],
                        "observed_frames": len(observed),
                        "missing_gaps": missing_gaps,
                    }
                )

            image_frames = len(entry.image_paths)
            annotation_frames = len(entry.annotation_paths)
            matched_frames = len(entry.image_paths.keys() & entry.annotation_paths.keys())
            reports.append(
                SequenceInspection(
                    sequence=entry.name,
                    image_frames=image_frames,
                    annotation_frames=annotation_frames,
                    matched_frames=matched_frames,
                    valid_frames=len(ordered_frames),
                    unique_objects=len(seen_frames),
                    objects_per_frame=objects_per_frame,
                    image_dimensions=sorted(image_dimensions),
                    annotation_dimensions=sorted(annotation_dimensions),
                    object_lifecycles=lifecycles,
                    skipped=not bool(ordered_frames),
                )
            )
        return DatasetInspection(
            dataset_root=str(self.root),
            split=self.split,
            train_sequences=len(self.train_sequences),
            validation_sequences=len(self.validation_sequences),
            requested_sequences=len(entries),
            inspected_sequences=reports,
            loading_errors=all_issues,
        )
