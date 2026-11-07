"""Typed access to the project's central YAML configuration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml


class ConfigError(ValueError):
    """Raised when a configuration file is missing or malformed."""


@dataclass(frozen=True)
class ProjectConfig:
    """Configuration sections used across the project.

    Section contents remain dictionaries so experiments can add settings
    without making configuration loading dependent on a large schema library.
    """

    dataset: dict[str, Any]
    features: dict[str, Any]
    memory: dict[str, Any]
    association: dict[str, Any]
    calibration: dict[str, Any]
    evaluation: dict[str, Any]
    experiment: dict[str, Any]
    source_path: Path

    @property
    def dataset_root(self) -> Path:
        """Return the configured DAVIS root with a user's home expanded."""

        return Path(str(self.dataset["root"])).expanduser()

    def to_dict(self) -> dict[str, Any]:
        """Return a plain mapping suitable for metadata or YAML output."""

        return {
            "dataset": self.dataset.copy(),
            "features": self.features.copy(),
            "memory": self.memory.copy(),
            "association": self.association.copy(),
            "calibration": self.calibration.copy(),
            "evaluation": self.evaluation.copy(),
            "experiment": self.experiment.copy(),
        }


_SECTION_NAMES = (
    "dataset",
    "features",
    "memory",
    "association",
    "calibration",
    "evaluation",
    "experiment",
)


def _merge(base: dict[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge nested override values without mutating ``base``."""

    merged = base.copy()
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), dict):
            merged[key] = _merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(
    path: str | Path,
    overrides: Mapping[str, Any] | None = None,
) -> ProjectConfig:
    """Load YAML configuration and apply optional nested runtime overrides.

    Args:
        path: YAML configuration path.
        overrides: Optional mapping merged recursively over the file contents.
            For example, ``{"dataset": {"root": "/content/drive/MyDrive/DAVIS"}}``.

    Raises:
        ConfigError: If the file is missing, invalid YAML, or lacks required
            sections/settings.
    """

    config_path = Path(path).expanduser()
    if not config_path.is_file():
        raise ConfigError(f"Configuration file does not exist: {config_path}")
    try:
        loaded = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"Invalid YAML in {config_path}: {exc}") from exc
    if not isinstance(loaded, dict):
        raise ConfigError(f"Expected a YAML mapping in {config_path}")
    if overrides:
        loaded = _merge(loaded, overrides)

    missing = [name for name in _SECTION_NAMES if not isinstance(loaded.get(name), dict)]
    if missing:
        raise ConfigError(
            "Missing or invalid configuration sections: " + ", ".join(missing)
        )
    if not loaded["dataset"].get("root"):
        raise ConfigError("dataset.root must be set to the DAVIS directory")
    for field in ("train_split", "validation_split"):
        if not loaded["dataset"].get(field):
            raise ConfigError(f"dataset.{field} must name an official split file")
    if int(loaded["memory"].get("max_gap", 0)) < 1:
        raise ConfigError("memory.max_gap must be a positive integer")
    if int(loaded["evaluation"].get("ece_bins", 0)) < 1:
        raise ConfigError("evaluation.ece_bins must be a positive integer")

    return ProjectConfig(
        **{name: loaded[name] for name in _SECTION_NAMES},
        source_path=config_path.resolve(),
    )
