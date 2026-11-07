"""Small reproducibility, metadata, and result-directory helpers."""

from __future__ import annotations

import json
import math
import platform
import random
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .dataset import DAVISDataset


def set_random_seeds(seed: int) -> dict[str, Any]:
    """Seed Python, NumPy, PyTorch, CUDA, and scikit-learn's shared RNG inputs."""

    random.seed(seed)
    np.random.seed(seed)
    versions: dict[str, Any] = {"python": platform.python_version(), "numpy": np.__version__}
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        versions["pytorch"] = torch.__version__
        versions["cuda"] = torch.version.cuda
        versions["cuda_available"] = bool(torch.cuda.is_available())
        versions["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except ImportError:
        versions.update({"pytorch": None, "cuda": None, "cuda_available": False, "gpu": None})
    return versions


def git_commit_hash(repository_root: str | Path) -> str | None:
    """Return a repository commit hash when available, else ``None``."""

    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(repository_root),
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path: str | Path, value: Any) -> Path:
    """Write readable standards-compliant JSON, replacing NaNs with null."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(_json_safe(value), indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    return destination


def write_yaml(path: str | Path, value: dict[str, Any]) -> Path:
    """Write a YAML configuration/metadata mapping with stable key ordering."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(yaml.safe_dump(_json_safe(value), sort_keys=False), encoding="utf-8")
    return destination


def create_experiment_directory(
    output_root: str | Path,
    name: str | None = None,
    overwrite: bool = False,
) -> Path:
    """Create a unique experiment directory unless explicit overwrite is set.

    Timestamp names use UTC. If a named directory exists, it is preserved and
    an error is raised by default. Passing ``overwrite=True`` explicitly
    replaces that exact named result directory.
    """

    root = Path(output_root).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    if name:
        if Path(name).name != name or name in {".", ".."}:
            raise ValueError("Experiment name must be a single directory name")
        target = root / name
        if target.exists():
            if not overwrite:
                raise FileExistsError(
                    f"Experiment directory already exists: {target}. Choose another name or pass --overwrite."
                )
            shutil.rmtree(target)
        target.mkdir(parents=True)
        return target

    base_name = "experiment_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    target = root / base_name
    counter = 1
    while target.exists():
        target = root / f"{base_name}_{counter:02d}"
        counter += 1
    target.mkdir(parents=True)
    return target


def runtime_metadata(repository_root: str | Path) -> dict[str, Any]:
    """Collect software/GPU/git state for reproducibility records."""

    metadata: dict[str, Any] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit_hash(repository_root),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
    }
    try:
        import torch

        metadata.update(
            {
                "pytorch_version": torch.__version__,
                "cuda_version": torch.version.cuda,
                "cuda_available": bool(torch.cuda.is_available()),
                "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            }
        )
        if torch.cuda.is_available():
            free_bytes, total_bytes = torch.cuda.mem_get_info(0)
            metadata["gpu_memory_free_bytes_at_start"] = int(free_bytes)
            metadata["gpu_memory_total_bytes"] = int(total_bytes)
    except ImportError:
        metadata.update(
            {
                "pytorch_version": None,
                "cuda_version": None,
                "cuda_available": False,
                "gpu_name": None,
            }
        )
    return metadata


def select_smoke_sequence_names(
    dataset: DAVISDataset,
    sequence_names: list[str],
    max_sequences: int,
) -> list[str]:
    """Prefer multi-object sequences so CPU smoke runs include negatives.

    The selection inspects only the first valid annotation in each candidate
    sequence. It remains within the provided official split and never mixes
    training and validation sequences.
    """

    if max_sequences < 1:
        raise ValueError("max_sequences must be positive")
    multi_object: list[str] = []
    single_or_empty: list[str] = []
    for name in sequence_names:
        iterator = dataset.iter_frames(name)
        try:
            first_frame = next(iterator, None)
        finally:
            iterator.close()
        if first_frame is not None and len(first_frame.object_ids) > 1:
            multi_object.append(name)
        else:
            single_or_empty.append(name)
        if len(multi_object) >= max_sequences:
            break
    selected = multi_object + single_or_empty
    return selected[:max_sequences]
