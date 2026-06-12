from __future__ import annotations

from applications.backend import deploy_runtime as dr


def test_discover_specs_reads_promoted_runtime_assets() -> None:
    old_specs = dr._SPECS
    try:
        dr._SPECS = None
        specs = dr.discover_specs()
        assert set(specs.keys()) == {"caucafall_tcn"}
        spec = specs["caucafall_tcn"]
        assert "/ops/configs/ops/" in spec.ops_path.replace("\\", "/")
        assert "/ops/deploy_assets/checkpoints/" in spec.ckpt.replace("\\", "/")
        assert spec.temperature > 0.0
        assert spec.ops
    finally:
        dr._SPECS = old_specs
