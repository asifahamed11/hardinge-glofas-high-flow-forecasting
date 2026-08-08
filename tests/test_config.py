from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from hardinge_high_flow.config import load_config, resolve_project_path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "default.yaml"


def test_default_config_loads() -> None:
    config = load_config(DEFAULT_CONFIG)
    assert config["target"]["source"] == "glofas_proxy"
    assert config["experiment"]["horizons"] == [1, 3, 5, 7]
    assert resolve_project_path(config, "data/processed").is_absolute()


def test_absolute_project_paths_are_rejected(tmp_path: Path) -> None:
    config = yaml.safe_load(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    config["paths"]["master_dataset_csv"] = str(tmp_path / "outside.csv")
    path = tmp_path / "invalid.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    with pytest.raises(ValueError, match="must be relative"):
        load_config(path)


def test_project_path_traversal_is_rejected() -> None:
    config = load_config(DEFAULT_CONFIG)
    with pytest.raises(ValueError, match="cannot escape"):
        resolve_project_path(config, "../outside.csv")
