from pathlib import Path

import pytest

from occlusion_association.config import ConfigError, load_config


def test_load_config_applies_nested_runtime_override(tmp_path: Path) -> None:
    config_path = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"

    config = load_config(config_path, {"dataset": {"root": str(tmp_path)}})

    assert config.dataset_root == tmp_path
    assert config.memory["max_gap"] == 30
    assert config.evaluation["ece_bins"] == 10


def test_missing_config_has_actionable_error(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="does not exist"):
        load_config(tmp_path / "missing.yaml")
