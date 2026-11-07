"""Tools for studying selective object identity association under occlusion."""

from .config import ProjectConfig, load_config
from .dataset import DAVISDataset, DatasetInspection, DatasetLayoutError

__all__ = [
    "DAVISDataset",
    "DatasetInspection",
    "DatasetLayoutError",
    "ProjectConfig",
    "load_config",
]

__version__ = "0.1.0"
