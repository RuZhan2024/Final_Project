# Integrated Refactor Master Plan

Supersedes and merges:
- `MULTI_DATASET_INTEGRATION_MANIFEST.md`
- `Refactor-Safety-Manifest.md`

Baseline commit: `58813e8`  
Planning mode: **No file/class moves or import rewrites until approval**

## 1) Mission and Guardrails

Objective:
- Integrate LE2i, CAUCAFall, MUVIM, URFall through one adapter contract.
- Reorganize project into a modern package layout (`src/` + `scripts/`) without changing frozen model logic from `58813e8`.

Hard constraints:
1. Preserve canonical feature semantics in `build_canonical_input`.
2. Preserve model/loss behavior, window label semantics, and thresholds unless explicitly approved.
3. Structural changes are packaging and modularization first, behavior changes second.
4. LE2i parity is a gate before multi-dataset training expansion.

## 2) Unified Data Integration Plan

### 2.1 Canonical Adapter Interface

Each dataset adapter must emit:
- `joints_xy: float32 [T,17,2]`
- `motion_xy: float32 [T,17,2] | None`
- `conf: float32 [T,17] | None`
- `mask: bool [T,17] | None`
- `fps: float`
- `meta: {path, video_id, w_start, w_end, y, dataset}`

Downstream contract remains:
`build_canonical_input(joints_xy, motion_xy, conf, mask, fps, feat_cfg)`

### 2.2 Joint Mapping (All 4 Datasets)

Internal skeleton (COCO-17 order):
`[nose, left_eye, right_eye, left_ear, right_ear, left_shoulder, right_shoulder, left_elbow, right_elbow, left_wrist, right_wrist, left_hip, right_hip, left_knee, right_knee, left_ankle, right_ankle]`

MediaPipe-33 -> internal-17 index map:
`[0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]`

Datasets covered:
- LE2i
- CAUCAFall
- MUVIM
- URFall/URFD

### 2.3 Sampling Parity

URFall normalization policy:
- Resample to `25 FPS` pre-windowing.
- Use LE2i-aligned window params (`W=48`, `S=12`).
- Keep stable channel/mask/conf behavior unchanged.

Parity checks required:
- XY mean/std
- motion magnitude stats
- valid joint ratio
- per-joint missingness profile

### 2.4 Baseline Guarantee

1. Reproduce LE2i stable performance first.
2. Block MUVIM/URFall training until LE2i parity passes with adapter-enabled pipeline.
3. Any LE2i regression blocks rollout.

## 3) Structural Refactor Blueprint

### 3.1 Target File Tree

```text
.
├── pyproject.toml
├── README.md
├── configs/
│   ├── data/
│   │   ├── datasets.yaml
│   │   └── windows.yaml
│   ├── model/
│   │   ├── tcn.yaml
│   │   └── gcn.yaml
│   ├── labels/
│   ├── splits/
│   └── ops/
├── scripts/
│   ├── train_tcn.py
│   ├── train_gcn.py
│   ├── make_windows.py
│   ├── make_unlabeled_windows.py
│   ├── make_fa_windows.py
│   ├── fit_ops.py
│   ├── eval_metrics.py
│   ├── extract_pose_videos.py
│   ├── extract_pose_images.py
│   ├── preprocess_pose.py
│   ├── make_labels_*.py
│   └── make_splits.py
├── src/
│   └── fall_detection/
│       ├── __init__.py
│       ├── core/
│       ├── data/
│       │   ├── adapters/
│       │   ├── labels/
│       │   ├── splits/
│       │   └── windowing/
│       ├── training/
│       ├── evaluation/
│       ├── deploy/
│       └── pose/
└── tests/
```

### 3.2 Import Convention and Editable Install

Import standard:
- `from fall_detection.core.features import build_canonical_input`
- `from fall_detection.core.models import build_model`
- `from fall_detection.data.adapters.base import DatasetAdapter`

Rules:
- No final-state imports from legacy top-level modules (`core.*`, `models.*`, etc.).
- `pip install -e .` must work via `src/` layout and `pyproject.toml`.

### 3.3 Decoupling Constraint for Windowing

`scripts/make_windows.py` and `scripts/make_unlabeled_windows.py` must:
1. Resolve data via adapter factory.
2. Use only adapter outputs for sequence loading/normalization.
3. Contain no dataset-specific branching for LE2i/CAUCAFall/MUVIM/URFall.
4. Keep overlap/quality/window logic dataset-agnostic.

## 4) Legacy -> Target Mapping Table

### 4.1 Core

| Legacy | Target |
|---|---|
| `core/features.py` | `src/fall_detection/core/features.py` |
| `core/models.py` | `src/fall_detection/core/models.py` |
| `core/losses.py` | `src/fall_detection/core/losses.py` |
| `core/metrics.py` | `src/fall_detection/core/metrics.py` |
| `core/calibration.py` | `src/fall_detection/core/calibration.py` |
| `core/alerting.py` | `src/fall_detection/core/alerting.py` |
| `core/confirm.py` | `src/fall_detection/core/confirm.py` |
| `core/ckpt.py` | `src/fall_detection/core/ckpt.py` |
| `core/ema.py` | `src/fall_detection/core/ema.py` |
| `core/uncertainty.py` | `src/fall_detection/core/uncertainty.py` |
| `core/yamlio.py` | `src/fall_detection/core/yamlio.py` |
| `core/check_spans.py` | `src/fall_detection/data/labels/check_spans.py` |
| `core/adapters/dataset_adapter.py` | `src/fall_detection/data/adapters/base.py` |
| `core/adapters/__init__.py` | `src/fall_detection/data/adapters/__init__.py` |
| `core/__init__.py` | `src/fall_detection/core/__init__.py` |

### 4.2 Data Prep and Windowing

| Legacy | Target |
|---|---|
| `labels/make_le2i_labels.py` | `src/fall_detection/data/labels/make_le2i_labels.py` |
| `labels/make_caucafall_labels.py` | `src/fall_detection/data/labels/make_caucafall_labels.py` |
| `labels/make_muvim_labels.py` | `src/fall_detection/data/labels/make_muvim_labels.py` |
| `labels/make_urfd_labels.py` | `src/fall_detection/data/labels/make_urfall_labels.py` |
| `labels/make_unlabeled_test_list.py` | `src/fall_detection/data/labels/make_unlabeled_test_list.py` |
| `split/make_splits.py` | `src/fall_detection/data/splits/make_splits.py` |
| `windows/make_windows.py` | `src/fall_detection/data/windowing/make_windows_impl.py` |
| `windows/make_unlabeled_windows.py` | `src/fall_detection/data/windowing/make_unlabeled_windows_impl.py` |
| `windows/make_fa_windows.py` | `src/fall_detection/data/windowing/make_fa_windows_impl.py` |
| `windows/check_windows.py` | `src/fall_detection/data/windowing/check_windows.py` |

### 4.3 Training, Eval, Deploy, Pose

| Legacy | Target |
|---|---|
| `models/train_tcn.py` | `src/fall_detection/training/train_tcn.py` |
| `models/train_gcn.py` | `src/fall_detection/training/train_gcn.py` |
| `eval/metrics.py` | `src/fall_detection/evaluation/metrics_eval.py` |
| `eval/fit_ops.py` | `src/fall_detection/evaluation/fit_ops.py` |
| `eval/mine_hard_negatives.py` | `src/fall_detection/evaluation/mine_hard_negatives.py` |
| `eval/score_unlabeled_alert_rate.py` | `src/fall_detection/evaluation/score_unlabeled_alert_rate.py` |
| `eval/plot_f1_vs_tau.py` | `src/fall_detection/evaluation/plot_f1_vs_tau.py` |
| `eval/plot_fa_recall.py` | `src/fall_detection/evaluation/plot_fa_recall.py` |
| `deploy/common.py` | `src/fall_detection/deploy/common.py` |
| `deploy/run_modes.py` | `src/fall_detection/deploy/run_modes.py` |
| `deploy/run_alert_policy.py` | `src/fall_detection/deploy/run_alert_policy.py` |
| `pose/extract_2d.py` | `src/fall_detection/pose/extract_2d_videos.py` |
| `pose/extract_2d_from_images.py` | `src/fall_detection/pose/extract_2d_images.py` |
| `pose/preprocess_pose_npz.py` | `src/fall_detection/pose/preprocess_pose_npz.py` |
| `pose/parse_ntu_skeleton.py` | `src/fall_detection/pose/parse_ntu_skeleton.py` |
| `eval/__init__.py` | `src/fall_detection/evaluation/__init__.py` |
| `deploy/__init__.py` | `src/fall_detection/deploy/__init__.py` |

### 4.4 Scripts and Tooling Wrappers

| New wrapper | Entry target |
|---|---|
| `scripts/train_tcn.py` | `fall_detection.training.train_tcn:main` |
| `scripts/train_gcn.py` | `fall_detection.training.train_gcn:main` |
| `scripts/make_windows.py` | `fall_detection.data.windowing.make_windows_impl:main` |
| `scripts/make_unlabeled_windows.py` | `fall_detection.data.windowing.make_unlabeled_windows_impl:main` |
| `scripts/make_fa_windows.py` | `fall_detection.data.windowing.make_fa_windows_impl:main` |
| `scripts/fit_ops.py` | `fall_detection.evaluation.fit_ops:main` |
| `scripts/eval_metrics.py` | `fall_detection.evaluation.metrics_eval:main` |
| `scripts/make_splits.py` | `fall_detection.data.splits.make_splits:main` |
| `scripts/extract_pose_videos.py` | `fall_detection.pose.extract_2d_videos:main` |
| `scripts/extract_pose_images.py` | `fall_detection.pose.extract_2d_images:main` |
| `scripts/preprocess_pose.py` | `fall_detection.pose.preprocess_pose_npz:main` |

### 4.5 Sweeps and Data Helpers

| Legacy | Target |
|---|---|
| `tools/sweep_tcn_le2i.py` | `scripts/sweeps/sweep_tcn_le2i.py` |
| `tools/sweeps/sweep_tcn_caucafall.py` | `scripts/sweeps/sweep_tcn_caucafall.py` |
| `tools/sweeps/sweep_gcn_le2i.py` | `scripts/sweeps/sweep_gcn_le2i.py` |
| `tools/sweeps/sweep_gcn_caucafall.py` | `scripts/sweeps/sweep_gcn_caucafall.py` |
| `tools/sweeps/sweep_lib.py` | `src/fall_detection/training/sweeps/sweep_lib.py` |
| `tools/sweeps_simple/sweep_tcn_le2i.py` | `scripts/sweeps_simple/sweep_tcn_le2i.py` |
| `tools/sweeps_simple/sweep_tcn_caucafall.py` | `scripts/sweeps_simple/sweep_tcn_caucafall.py` |
| `tools/sweeps_simple/sweep_gcn_le2i.py` | `scripts/sweeps_simple/sweep_gcn_le2i.py` |
| `tools/sweeps_simple/sweep_gcn_caucafall.py` | `scripts/sweeps_simple/sweep_gcn_caucafall.py` |
| `tools/sweeps_simple/sweep_lib_min.py` | `src/fall_detection/training/sweeps/sweep_lib_min.py` |
| `tools/urfall_group_by_prefix.py` | `scripts/data_prep/urfall_group_by_prefix.py` |
| `tools/urfall_merge_seq_splits.py` | `scripts/data_prep/urfall_merge_seq_splits.py` |
| `tools/urfall_strip_rf.py` | `scripts/data_prep/urfall_strip_rf.py` |
| `tools/__init__.py` | `src/fall_detection/training/sweeps/__init__.py` |

### 4.6 Explicitly Unchanged in This Phase

- `apps/**`
- `server/**`
- `configs/labels/**`
- `configs/splits/**`
- `configs/ops/**`
- `Makefile` (repoint incrementally to wrappers)

## 5) Migration Phases and Gates

Approval gates (must pass before physical moves):
1. Approve this integrated plan.
2. Approve `fall_detection.*` import convention and editable-install strategy.
3. Approve wrappers-first migration.

Execution phases after approval:
1. Create `pyproject.toml`, package skeleton, and script wrappers.
2. Move modules with compatibility shims from legacy paths.
3. Switch imports to `fall_detection.*`.
4. Run parity validation (LE2i first, then multi-dataset).
5. Remove legacy shims and retired paths.

## 6) Validation and Exit Criteria

Functional parity:
- LE2i metrics match `58813e8` baseline within agreed tolerance.
- Adapter pipeline reproduces existing canonical tensor semantics.

Packaging parity:
- `pip install -e .` succeeds.
- All CLI entry points work through `scripts/` and/or `project.scripts`.

Code hygiene:
- No dataset-specific logic inside windowing core.
- Dataset-specific handling isolated to adapters.

## 7) Current Status

- Plan consolidated: complete.
- Data adapter baseline: defined and implemented in current tree.
- Structural migration phase 1: completed.
  - `pyproject.toml` added for `src/` package discovery.
  - `scripts/` thin wrappers added.
  - `src/fall_detection/*` scaffold modules added.
- Structural migration phase 2: completed.
  - `Makefile` Python entrypoints repointed to `scripts/*.py` wrappers.
  - CLI flags and output paths preserved.
- Structural migration phase 3: completed.
  - Real module implementations copied into `src/fall_detection/**` from legacy paths.
  - Internal imports switched to `fall_detection.*` across moved training/evaluation/deploy/windowing modules.
  - Adapter path in package windowing now resolves via `fall_detection.data.adapters`.
  - Editable package install validated (`pip install -e . --no-build-isolation`).
  - Console script entrypoint validated (`fd-make-windows --help`).
- Structural migration phase 4: completed.
  - Legacy roots (`core/`, `models/`, `eval/`, `deploy/`, `pose/`, `labels/`, `split/`, `windows/`) converted to compatibility delegates.
  - Duplicate heavy logic removed from legacy paths; canonical implementation now lives under `src/fall_detection/**`.
  - Backward compatibility retained for old script/module entrypoints.
- Structural migration phase 5: completed.
  - Compatibility delegates retired; legacy trees removed.
  - Runtime path now standardized on `src/fall_detection/**` + `scripts/**`.
  - Server runtime imports switched to `fall_detection.core.*`.
