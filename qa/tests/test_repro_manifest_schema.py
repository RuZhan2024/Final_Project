from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def _load_script_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, str(path))
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_reproduce_claim_manifest_schema(tmp_path) -> None:
    script = Path("scripts/reproduce_claim.py").resolve()
    mod = _load_script_module(script, "reproduce_claim_mod")

    repo_root = tmp_path
    metrics_dir = repo_root / "outputs" / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    (repo_root / "outputs" / "le2i_tcn_W48S12").mkdir(parents=True, exist_ok=True)
    ckpt = repo_root / "outputs" / "le2i_tcn_W48S12" / "best.pt"
    ckpt.write_bytes(b"stub")

    metric_json = metrics_dir / "tcn_le2i.json"
    metric_json.write_text(
        json.dumps(
            {
                "ops": {
                    "op2": {
                        "f1": 0.8,
                        "recall": 0.9,
                        "fa24h": 0.0,
                        "ap": 0.95,
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    manifest = mod.build_manifest(
        repo_root=repo_root,
        dataset="le2i",
        model="tcn",
        op="op2",
        checkpoint=ckpt,
        metric_globs=["outputs/metrics/tcn_le2i*.json"],
        run_results=[{"cmd": "echo smoke", "returncode": 0, "duration_s": 0.1}],
    )

    assert manifest["schema_version"] == "1.0"
    assert manifest["claim"]["dataset"] == "le2i"
    assert manifest["checkpoint"]["exists"] is True
    assert isinstance(manifest["checkpoint"]["sha256"], str)
    assert isinstance(manifest["metrics"], list)
    assert manifest["metrics"][0]["op"] == "op2"

