# Cleanup Scope Audit

This audit marks files and directories that are likely unrelated to the core project software, report evidence, or thesis evidence, so they can be cleaned later with lower risk.

This is a marking document only. It does not imply deletion by default.

## Keep: Core Project Software

These are part of the software product and should stay:

- `apps/`
- `server/`
- `src/`
- `tests/`
- `scripts/start_fullstack.sh`
- `scripts/bootstrap_dev.sh`
- `scripts/release_doctor.sh`
- `scripts/release_manifest.sh`
- `Makefile`
- `README.md`
- `requirements*.txt`
- `pyproject.toml`
- `configs/ops/*.yaml` that are current active profiles
- `configs/labels/`
- `configs/splits/`

## Keep: Report / Thesis Evidence

These are related to report or thesis evidence and should not be treated as unrelated cleanup:

- `docs/project_targets/`
- `docs/reports/`
- `docs/training/`
- `artifacts/reports/`
- `artifacts/baseline/`
- `artifacts/repro/`
- `artifacts/registry/`
- `artifacts/templates/`
- `artifacts/figures/cross_dataset/`
- `artifacts/figures/pr_curves/`
- `artifacts/figures/stability/`
- `baselines/`

## Marked For Later Cleanup: Clearly Not Core Project / Report / Thesis

These look unrelated to the deliverable software and unrelated to the report/thesis evidence trail.

### Teaching / Coursework Content

- `docs/course/`
- `docs/tutorial/`
- `teaching/`

Reason:

- These are separate teaching/tutorial materials, not part of the fall-detection product or the thesis/report evidence chain.
- They have now been consolidated under `docs/archive/tutorial_materials/`.

### Standalone Teaching Audit Note

- `docs/archive/audit_repo_for_teaching.md`

Reason:

- This is archive-style teaching/admin material, not core product or thesis evidence.
- It has now been moved under `docs/archive/tutorial_materials/audit_repo_for_teaching.md`.

## Marked For Later Cleanup: Low-Value Research Churn

These are project-adjacent, but they are not usually needed for a clean deliverable branch.

### Historical Ops Archives

- `configs/ops/archive/`
- `configs/ops/grid_midplateau/`
- `configs/ops/grid_midplateau_temporal/`
- `configs/ops/grid_startguard_midplateau/`
- `configs/ops/cross_*`
- old per-run files such as:
  - `configs/ops/*stb_*`
  - `configs/ops/*confirm*`
  - `configs/ops/*recallpush*`
  - `configs/ops/*ablate*`
  - `configs/ops/*tune_*`
  - `configs/ops/*opt_*`

Reason:

- These are useful as experiment history, but not required for normal product operation.

### MUVIM Experimental Sweep Configs

- `configs/ops/*muvim*`

Reason:

- They are active only if MUVIM remains in scope. If the final project/report does not rely on MUVIM, these are cleanup candidates.

### Ad-Hoc Plot Scripts

- `scripts/plot_confusion_matrix.py`
- `scripts/plot_dataset_balance.py`
- `scripts/plot_failure_scatter.py`

Reason:

- Useful for analysis, but not required for running the system or defending the currently documented deployment path.

### One-Off Delivery Evaluation Script

- `scripts/eval_delivery_videos.py`

Reason:

- Useful for reproducing the 24-video delivery test, but not required by the product runtime itself.
- Keep it if you still need viva/demo reproducibility. Otherwise it is optional.

## Marked For Later Cleanup: Generated Artifacts

These are generated outputs, not source.

### Generated Evaluation Bundles

- `artifacts/fall_side_corridor_eval_20260315/`
- `artifacts/fall_test_eval_20260315/`
- `artifacts/fall_test_eval_20260315_online_reverify_20260315/`
- `artifacts/fall_test_generalization_20260315/`
- `artifacts/online_ops_fit_20260315/`
- `artifacts/online_ops_fit_20260315_verify/`
- `artifacts/online_ops_fit_20260315_verify_le2i_bypass/`
- `artifacts/ops_delivery_verify_20260315/`
- `artifacts/ops_reverify_20260315/`
- `artifacts/ops_reverify_20260315_after_gatefix/`
- `artifacts/ops_reverify_20260315_after_motionfix/`

Reason:

- These are reproducible run outputs and should generally be archived outside the main software branch if repository cleanliness matters.

### Additional Plot Output Bundle

- `artifacts/figures/plots/`

Reason:

- Generated plot outputs, not source.

### Cache / Build / Environment Directories

- `.mypy_cache/`
- `.pytest_cache/`
- `.mplcache/`
- `.make/`
- `apps/build/`
- `apps/node_modules/`
- `.venv/`

Reason:

- Local environment and build caches only.

## Needs Human Decision: Large But Still Project-Related

These are not unrelated, but they are large and may need archiving strategy.

### Training Outputs

- `outputs/`

Reason:

- Clearly project-related, but too large for a clean software delivery branch. Archive or externalize rather than treating as unrelated trash.

### Raw / Processed Data

- `data/`

Reason:

- Clearly project-related and central to experiments, but not usually something to keep in a clean application-delivery repository state.

## Practical Cleanup Order

If you want to clean without risking project loss, do it in this order:

1. Caches and local build directories.
2. `docs/course/`, `docs/tutorial/`, `teaching/`.
3. Generated evaluation artifact bundles under `artifacts/`.
4. Historical `configs/ops/` experiment churn.
5. Optional analysis scripts.
6. Large project assets (`outputs/`, `data/`) only with an explicit archive plan.

## Current High-Confidence Unrelated Targets

If you want the shortest “likely unrelated” list, start with:

- `docs/archive/tutorial_materials/course/`
- `docs/archive/tutorial_materials/tutorial/`
- `docs/archive/tutorial_materials/audit_repo_for_teaching.md`

These are the strongest candidates for being outside both the software project and the report/thesis scope.
