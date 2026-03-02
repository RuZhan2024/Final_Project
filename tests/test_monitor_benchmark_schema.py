from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_script_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, str(path))
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_benchmark_summary_schema() -> None:
    script = Path("scripts/benchmark_monitor_e2e.py").resolve()
    mod = _load_script_module(script, "benchmark_monitor_mod")
    summary = mod.summarize([10.0, 20.0, 30.0, 40.0, 50.0])
    for k in ("count", "mean", "p50", "p95", "p99", "min", "max"):
        assert k in summary
        assert isinstance(summary[k], float)
    assert summary["count"] == 5.0
    assert summary["min"] == 10.0
    assert summary["max"] == 50.0

