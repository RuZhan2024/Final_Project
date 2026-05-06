# Ops Config Index

This directory stores the runtime operating-point YAML promoted for the current application.

Use this index to distinguish the active runtime profile from retained experiment history.

## Active Runtime Profiles

This is the runtime-facing config used by the current application:

- `tcn_caucafall.yaml`

The backend discovers runtime specs through `ops/deploy_assets/manifest.json`; it does not scan this directory broadly. Keep new sweeps out of this top-level folder unless they are explicitly promoted.

## Delivery / Repro Profiles

These are kept in archive folders because they support reproducibility for defended results:

- `archive/historical_profiles_20260506/`
- `archive/legacy_root_configs_20260427/`
- `configs/delivery/`

These are not always the default live profile, but they may be required to reproduce reported numbers.

## Paper / Thesis / Diagnostic Profiles

These are still relevant as evidence or comparison points:

- `archive/historical_profiles_20260506/*_papertrack.yaml`
- `archive/historical_profiles_20260506/gcn_le2i_paper_profile.yaml`
- `diagnostic/`

Keep these for report/thesis traceability even if they are not active in the app.

## Historical Experiment Families

These are retained as experiment evidence and should be treated as archive material rather than active runtime configs:

- `archive/`
- `archive/historical_profiles_20260506/`
- `archive/muvim/`
- per-run variants such as:
  - `*stb_*`
  - `*confirm*`
  - `*recallpush*`
  - `*ablate*`
  - `*tune_*`
  - `*opt_*`

## MUVIM Profiles

MUVIM configs are still project-related, but they are best treated as a separate experiment track:

- `archive/historical_profiles_20260506/tcn_muvim*.yaml`
- `archive/historical_profiles_20260506/gcn_muvim*.yaml`

Current convention:

- keep MUVIM configs archived unless MUVIM is explicitly restored to the final submitted scope

If MUVIM is not part of the final submitted scope, these should stay archived rather than removed.

## Practical Lookup

- Need the live app profile:
  check `tcn_caucafall.yaml` and `ops/deploy_assets/manifest.json`.
- Need the defended 24-video delivery setup:
  check `ops/configs/delivery/`.
- Need experiment history for report/thesis:
  search `archive/` and the 2026-05-06 report evidence archive.
