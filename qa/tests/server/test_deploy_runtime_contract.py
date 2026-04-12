from __future__ import annotations

from applications.backend import deploy_runtime as dr


def test_discover_specs_reads_promoted_runtime_assets() -> None:
    old_specs = dr._SPECS
    try:
        dr._SPECS = None
        specs = dr.discover_specs()
        expected = {"caucafall_tcn", "caucafall_gcn", "le2i_tcn", "le2i_gcn"}
        assert expected.issubset(specs.keys())
        for key in expected:
            spec = specs[key]
            assert "/ops/configs/ops/" in spec.ops_path
            assert "/ops/deploy_assets/checkpoints/" in spec.ckpt
            assert spec.ops
    finally:
        dr._SPECS = old_specs
