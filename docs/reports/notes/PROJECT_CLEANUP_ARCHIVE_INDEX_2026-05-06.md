# Project Cleanup Archive Index - 2026-05-06

This note records the cleanup boundary for the post-CTR-GCN project state.

## Runtime Keep Set

- `ops/deploy_assets/manifest.json`
- `ops/deploy_assets/checkpoints/caucafall_tcn_best.pt`
- `ops/configs/ops/tcn_caucafall.yaml`
- `ops/configs/delivery/tcn_mix25_s42_op2_motion018.yaml`
- `ops/deploy_assets/replay_clips/**`
- `applications/backend/**`
- `applications/frontend/**`
- `ml/src/fall_detection/**`
- `qa/tests/**`

Current runtime discovery is manifest-driven. The backend should not infer deploy specs by scanning every YAML under `ops/configs/ops`.

Monitor prediction is intentionally a single promoted TCN runtime path:
`request -> TCN inference -> decision -> persistence/response`. Retired GCN/HYBRID request modes are rejected instead of silently falling back.

## Report / Thesis Keep Set

- `docs/reports/**`
- `artifacts/figures/report/**`
- `ops/configs/labels/*_c2*.json`
- `ops/configs/splits/*_c2*`
- `ops/configs/ops/archive/historical_profiles_20260506/`
- `ops/configs/ops/archive/legacy_root_configs_20260427/`
- Local-only evidence archive: `artifacts/archive/report_evidence_20260506/`

## Local-Only Archives

- `artifacts/archive/generated_ops_specs_20260506/`
  - Generated sweep YAML/JSON removed from the runtime ops folder.
  - `*wrongbone*` files were deleted because they were known erroneous calibration artifacts.
- `artifacts/archive/report_evidence_20260506/`
  - 2026-05-06 replay/evaluation folders.
  - 2026-05-06 training/evaluation logs.
  - Custom replay trace PNGs.
- `ops/deploy_assets/archive/checkpoints_legacy_20260506/`
  - Old GCN/LE2i deploy checkpoints removed from the active runtime manifest.

These archives are retained on this workstation for audit/report traceability. Large generated artifacts remain ignored by git unless explicitly promoted.

## Deleted / Retired From Active Runtime

- Top-level `configs/` is no longer a canonical config root.
- `configs/ops/*` was moved to `ops/configs/ops/archive/legacy_root_configs_20260427/`.
- Historical top-level `ops/configs/ops/*.yaml` and `*.sweep.json` were moved to `ops/configs/ops/archive/historical_profiles_20260506/`.
- GCN/HYBRID are no longer active deploy model codes. The active app is locked to the TCN champion; CTR-GCN remains in the ML codebase for training/evaluation work, not as a promoted online runtime asset yet.
- The backend monitor services no longer pass `gcn_key`, `tri_gcn`, dual-policy config loaders, or hybrid-fusion helpers through the request/inference/decision pipeline.
- CTR-GCN shared training helpers were split into `ml/src/fall_detection/training/graph_training_utils.py`.
- The packaged `fd-train-gcn` entry point, `ops/scripts/train_gcn.py` wrapper, and `ml/src/fall_detection/training/train_gcn.py` trainer were removed.
- Legacy simple-GCN Makefile targets were removed from the active training/fit/eval/plot/repro pipelines. Active graph-model targets use `ctr-gcn`.
- The unused `ml/src/fall_detection/deploy/run_modes.py` dual-mode runner and old dual-model triage classes were removed.
- Stale alerting tests that targeted removed private fast-path helpers were rewritten against the current public alerting/metrics APIs.
- Stale server route tests were rewritten around current service/repository boundaries instead of removed private helpers.
- Monitor request validation now rejects malformed pose payloads before inference (`raw_*` shape/timestamp checks and direct `xy`/`conf` shape checks).
- `applications/backend/core.py` was removed; active routes/tests now depend directly on explicit modules such as `runtime_assets`, `runtime_state`, `db_schema`, and `inmemory_state`.
- Events summary responses now include `db_available=True` when DB-backed reads succeed, matching the existing fallback `db_available=False` contract.
- The QA suite was aligned with the cleaned Windows/project layout: package imports use `fall_detection.*`, top-level script/config assumptions now point at `ops/scripts` and `ops/configs`, and the manifest contract expects the single promoted TCN runtime asset.
