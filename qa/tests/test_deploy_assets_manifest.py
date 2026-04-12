from __future__ import annotations

import hashlib
import json
from pathlib import Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_deploy_assets_manifest_contract() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    manifest_path = repo_root / "ops" / "deploy_assets" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["schema_version"] == "1.0"
    contract = manifest["promotion_contract"]
    assert contract["promoted_runtime_assets_root"] == "ops/deploy_assets"
    assert "outputs" in contract["experimental_output_roots"]

    assets = manifest["promoted_runtime_assets"]

    checkpoints = assets["checkpoints"]
    assert len(checkpoints) == 4
    for entry in checkpoints:
        path = repo_root / entry["path"]
        assert path.exists(), entry["path"]
        assert path.stat().st_size == entry["size_bytes"]
        assert _sha256(path) == entry["sha256"]

    op_profiles = assets["op_profiles"]
    assert len(op_profiles) == 4
    for entry in op_profiles:
        path = repo_root / entry["path"]
        assert path.exists(), entry["path"]

    replay_sets = assets["replay_sets"]
    assert len(replay_sets) == 4
    total_clips = 0
    for entry in replay_sets:
        assert entry["label"] in {"fall", "adl"}
        assert entry["environment"] in {"corridor", "kitchen"}
        for clip in entry["clips"]:
            path = repo_root / clip
            assert path.exists(), clip
            total_clips += 1

    assert total_clips == 24
