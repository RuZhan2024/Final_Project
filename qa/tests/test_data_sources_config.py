from __future__ import annotations

from pathlib import Path

from fall_detection.data.resolver import build_data_path_resolver


def test_data_sources_default_config_exists_and_loads() -> None:
    cfg = Path("configs/experiments/data_sources.yaml")
    assert cfg.is_file()
    resolver = build_data_path_resolver(cfg)
    assert resolver.layout.root.name == "data"

