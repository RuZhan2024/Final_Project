# Reproducibility Checklist

Date: 2026-03-02

## Environment
- [x] `requirements.txt` exists at repo root.
- [x] `pyproject.toml` exists for editable install.
- [x] Locked, reproducible environment file (exact hashes/lockfile) is present and documented for examiner use.

Validation:
- `python3 -m compileall src server scripts`
- `python3 - <<'PY'\nimport importlib; [importlib.import_module(m) for m in ['fall_detection','server.app','server.deploy_runtime']]; print('ok')\nPY`

## Pipeline determinism
- [x] Global seed exposed in Makefile (`SPLIT_SEED=33724876`).
- [x] Split script is deterministic and avoids Python hash randomness (`make_splits.py` stable hash logic).
- [x] CAUCAFall subject-split guard present in Makefile.

Validation:
- `PYTHONPATH="$(pwd)/src:$(pwd)" python3 -m pytest -q tests/test_split_group_leakage.py`

## Data + split provenance
- [x] Split summaries exist in `configs/splits/*_split_summary.json`.
- [x] CAUCAFall summary indicates subject mode (`caucafall_subject`).
- [x] End-to-end data acquisition instructions are complete for all datasets (including annotation assumptions).

## Model run provenance
- [x] LE2i TCN has `train_config.json` captured.
- [x] CAUCAFall GCN run has local `train_config.json` artifact in `outputs/caucafall_gcn_W48S12/`.
- [x] Headline AP claim is tied to a stored command manifest with checkpoint hash.

Validation:
- `python3 scripts/reproduce_claim.py --dataset caucafall --model gcn --run 0`

## Artifact bundle integrity
- [x] `artifacts/artifact_bundle.json` passes integrity audit.
- [x] Baseline placeholder file exists.
- [x] Baseline targets are populated in captured baseline (`baselines/le2i/58813e8/performance_baseline.json`).

Validation:
- `PYTHONPATH="$(pwd)/src:$(pwd)" python3 scripts/audit_artifact_bundle.py --bundle_json artifacts/artifact_bundle.json`

## Evaluation reproducibility
- [x] Event metrics implementation is centralized and documented (`core/alerting.py`, `evaluation/metrics_eval.py`).
- [x] Top-level ops sweep artifacts pass sanity audit.

Validation:
- `PYTHONPATH="$(pwd)/src:$(pwd)" python3 scripts/audit_ops_sanity.py --ops_dir configs/ops`

## Required to reach Green
All critical reproducibility gates in this checklist are currently satisfied.
