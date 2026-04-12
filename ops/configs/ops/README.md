# Ops Config Index

This directory stores operating-point YAMLs and sweep outputs used across deployment, experiments, and report evidence.

Use this index to distinguish active runtime profiles from retained experiment history.

## Active Runtime Profiles

These are the main runtime-facing configs used by the current application:

- `tcn_caucafall.yaml`
- `gcn_caucafall.yaml`
- `tcn_le2i.yaml`
- `gcn_le2i.yaml`

These are the first files to check when the frontend or backend online behavior changes.

## Delivery / Repro Profiles

These are kept because they support reproducibility for defended results:

- `tcn_caucafall_locked.yaml`
- `gcn_caucafall_locked.yaml`
- `tcn_caucafall_r2_train_hneg.yaml`
- `configs/delivery/`

These are not always the default live profile, but they may be required to reproduce reported numbers.

## Paper / Thesis / Diagnostic Profiles

These are still relevant as evidence or comparison points:

- `*_papertrack.yaml`
- `gcn_le2i_paper_profile.yaml`
- `diagnostic/`
- `dual_policy/`

Keep these for report/thesis traceability even if they are not active in the app.

## Historical Experiment Families

These are retained as experiment evidence and should be treated as archive material rather than active runtime configs:

- `archive/`
- `grid_midplateau/`
- `grid_midplateau_temporal/`
- `grid_startguard_midplateau/`
- `cross_*`
- per-run variants such as:
  - `*stb_*`
  - `*confirm*`
  - `*recallpush*`
  - `*ablate*`
  - `*tune_*`
  - `*opt_*`

## MUVIM Profiles

MUVIM configs are still project-related, but they are best treated as a separate experiment track:

- `tcn_muvim*.yaml`
- `gcn_muvim*.yaml`

Current convention:

- keep the base or labels-oriented MUVIM configs that still describe the main MUVIM track at the root of `configs/ops/`
- move quick-search and later experimental branches such as `*muvim_quick*` and `*muvim_r3*` into `configs/ops/archive/muvim/`

If MUVIM is not part of the final submitted scope, these should stay archived rather than removed.

## Practical Lookup

- Need the live app profile:
  check the four active runtime YAMLs.
- Need the defended 24-video delivery setup:
  check `configs/delivery/` and the related Caucafall TCN profiles.
- Need experiment history for report/thesis:
  search the historical families and match them with `artifacts/reports/`.
