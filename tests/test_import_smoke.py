from __future__ import annotations

import importlib


def test_import_refactor_modules() -> None:
    modules = [
        "fall_detection",
        "fall_detection.data.datamodule",
        "fall_detection.data.pipeline",
        "fall_detection.data.transforms",
    ]
    for mod in modules:
        importlib.import_module(mod)

