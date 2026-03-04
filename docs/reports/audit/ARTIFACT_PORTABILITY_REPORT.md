# ARTIFACT_PORTABILITY_REPORT

## Scope
Audited artifact and output references for portability across machines.

## Inventory
- Bundle descriptor: `artifacts/artifact_bundle.json`
- Repro manifest: `artifacts/repro/RESULTS_20260302_204401/manifest.json`
- Ops configs: `configs/ops/*.yaml`
- Metrics outputs: `outputs/metrics/*.json`

## Portability Checklist

| Check | Result | Evidence |
|---|---|---|
| `outputs/metrics` files exist for key runs | PASS | `gcn_caucafall.json`, `gcn_le2i.json`, `tcn_caucafall.json`, `tcn_le2i.json` present |
| `configs/ops/*.yaml` present and references valid checkpoints | PASS | `model.ckpt: ../../outputs/.../best.pt` in each ops yaml |
| Primary bundle uses relative paths | PASS | `artifacts/artifact_bundle.json` uses `../...` paths |
| Repro artifacts avoid absolute local paths | FAIL | `artifacts/repro/.../manifest.json` contains `/Users/ruzhan/.../best.pt` |
| ops generation default avoids absolute path embedding | PASS | `fit_ops.py` supports relative output by default (`emit_absolute_paths=0`) |

## Findings

### Good
- Main artifact bundle is machine-portable when interpreted relative to `artifacts/`.
- Ops YAML model references are relative and align with current output layout.

### Gap
- Repro manifest embeds an absolute macOS home path:
  - `"path": "/Users/ruzhan/.../outputs/caucafall_gcn_W48S12/best.pt"`
- This breaks portability for external examiners or CI runners.

## Minimal Patch Plan
1. Update repro manifest generator (`scripts/reproduce_claim.py`) to emit paths relative to repo root or manifest directory.
2. Add a guard in generation code to reject absolute paths unless explicitly requested by a flag.
3. Add a smoke check command in audit CI:
   - `python scripts/audit_artifact_bundle.py --bundle_json artifacts/artifact_bundle.json`
   - plus a grep check for `/Users/` inside `artifacts/repro/**/manifest.json`.

## Validation Commands
```bash
python scripts/audit_artifact_bundle.py --bundle_json artifacts/artifact_bundle.json
rg -n '/Users/' artifacts/repro/**/manifest.json
```

## Verdict
- Artifact portability: **Yellow** (core bundle is portable, repro manifest path leak remains).
