# Deployment Lock (Current Recommended)

Date: 2026-03-06

## Purpose

Freeze one known-good profile and provide one-command reproducibility for:
- locked training params (TCN + GCN),
- locked fit/eval params,
- locked deployment ops promotion.

## Locked checkpoints (reference)

- TCN (caucafall): `outputs/caucafall_tcn_W48S12_r1_augreg/best.pt`
- GCN (caucafall): `outputs/caucafall_gcn_W48S12_r2_recallpush_b/best.pt`

## Locked one-command workflow (Makefile)

### 1) Reproduce locked training hyperparameters

- `make train-best-tcn-caucafall`
- `make train-best-gcn-caucafall`
- `make train-best-caucafall` (runs both)

Outputs:
- `outputs/repro/caucafall_tcn_r1_augreg/best.pt`
- `outputs/repro/caucafall_gcn_r2_recallpush_b/best.pt`

### 2) Reproduce locked fit+eval artifacts

- `make repro-best-tcn-caucafall`
- `make repro-best-gcn-caucafall`
- `make repro-best-caucafall` (runs both)

Outputs:
- `configs/ops/tcn_caucafall_locked.yaml`
- `configs/ops/gcn_caucafall_locked.yaml`
- `outputs/metrics/tcn_caucafall_locked.json`
- `outputs/metrics/gcn_caucafall_locked.json`

### 3) Promote locked ops to runtime canonical files

- `make apply-locked-ops-caucafall`

Copies to:
- `configs/ops/tcn_caucafall.yaml`
- `configs/ops/gcn_caucafall.yaml`

## Notes

- Locked targets are explicit and parameterized in `Makefile` variables:
  - `LOCK_TCN_CAUC_*`
  - `LOCK_GCN_CAUC_*`
- If lock changes, update:
  - this document,
  - locked variable values in `Makefile`,
  - evidence entries in `docs/project_targets/THESIS_EVIDENCE_MAP.md`.
