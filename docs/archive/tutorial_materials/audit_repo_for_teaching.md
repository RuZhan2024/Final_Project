# Repository Audit For Teaching Reuse

## Scope and method

This audit is based on repository code and runnable checks, not assumptions. I traced the main orchestration from [Makefile](../Makefile), the thin CLI wrappers in [scripts/](../scripts), the package modules under [src/fall_detection/](../src/fall_detection), and the FastAPI deployment path under [server/](../server).

Verification performed:

- `make help` succeeded and confirms the intended orchestration surface in [Makefile](../Makefile).
- `python3 scripts/audit_smoke.py` succeeded.
- `PYTHONPATH="$(pwd)/src:$(pwd)" python3 scripts/smoke_api_contract.py` succeeded.
- `PYTHONPATH="$(pwd)/src:$(pwd)" python3 -m pytest tests/test_import_smoke.py tests/test_windows_contract.py tests/test_split_group_leakage.py -q` succeeded (`4 passed`).
- `pytest -q` failed during collection because many tests still import legacy top-level modules `core`, `pose`, and `server`.
- `PYTHONPATH="$(pwd)/src:$(pwd)" pytest -q` still failed during collection for tests that import `core.*` and `pose.*`, confirming the repo currently mixes two import conventions.

## Executive summary

The real system is a script-and-Makefile driven pipeline whose stable runtime spine is:

1. raw videos or raw pose NPZs
2. pose extraction / pose cleanup
3. dataset-specific label and span generation
4. group-aware split generation
5. window export to NPZ
6. TCN or GCN training
7. operating-point fitting and optional temperature calibration
8. evaluation on labeled and unlabeled windows
9. deployment through FastAPI using `configs/ops/*.yaml`

The repository also contains a second, partially integrated architecture under [src/fall_detection/data/](../src/fall_detection/data) built around `DataPathResolver`, `UnifiedWindowDataModule`, typed contracts, and a contract-first `pipeline.py`. That layer is not the main execution path used by [Makefile](../Makefile), most scripts, or the deployment stack. For teaching reuse, the repository should be taught as "one active pipeline plus one incomplete refactor scaffold", not as a single coherent architecture.

## The real end-to-end pipeline

### 1. Data ingestion

Primary orchestration lives in [Makefile](../Makefile):

- extraction targets call [scripts/extract_pose_videos.py](../scripts/extract_pose_videos.py) and [scripts/extract_pose_images.py](../scripts/extract_pose_images.py)
- those wrappers call [src/fall_detection/pose/extract_2d_videos.py](../src/fall_detection/pose/extract_2d_videos.py) and [src/fall_detection/pose/extract_2d_images.py](../src/fall_detection/pose/extract_2d_images.py)

Supported raw roots are configured by variables in [Makefile](../Makefile): `RAW_le2i`, `RAW_urfd`, `RAW_caucafall`, `RAW_muvim`.

Important observation:

- the Makefile supports `le2i`, `urfd`, `caucafall`, and `muvim`
- [configs/experiments/data_sources.yaml](../configs/experiments/data_sources.yaml) only defines `le2i` and `caucafall`, with empty split lists

This means the `data_sources.yaml` / resolver path is not the source of truth for the active 4-dataset workflow.

### 2. Pose / skeleton preprocessing

The preprocessing entrypoint is [scripts/preprocess_pose.py](../scripts/preprocess_pose.py), which calls [src/fall_detection/pose/preprocess_pose_npz.py](../src/fall_detection/pose/preprocess_pose_npz.py).

This module is the actual bridge between raw pose dumps and model-ready sequence NPZs. It defines and writes the sequence-level schema consumed later by windowing. The downstream windowing code expects:

- `xy`
- `conf`
- optional `mask`
- optional metadata such as `fps`, `seq_id`, `src`, `seq_stem`

That contract is enforced later by [src/fall_detection/data/windowing/make_windows_impl.py](../src/fall_detection/data/windowing/make_windows_impl.py) and [src/fall_detection/core/features.py](../src/fall_detection/core/features.py).

### 3. Labels and fall spans

Label creation is dataset-specific and script-driven:

- LE2i: [scripts/make_labels_le2i.py](../scripts/make_labels_le2i.py) -> [src/fall_detection/data/labels/make_le2i_labels.py](../src/fall_detection/data/labels/make_le2i_labels.py)
- URFD: [scripts/make_labels_urfall.py](../scripts/make_labels_urfall.py) -> [src/fall_detection/data/labels/make_urfall_labels.py](../src/fall_detection/data/labels/make_urfall_labels.py)
- CAUCAFall: [scripts/make_labels_caucafall.py](../scripts/make_labels_caucafall.py) -> [src/fall_detection/data/labels/make_caucafall_labels.py](../src/fall_detection/data/labels/make_caucafall_labels.py)
- MUVIM: [scripts/make_labels_muvim.py](../scripts/make_labels_muvim.py) -> [src/fall_detection/data/labels/make_muvim_labels.py](../src/fall_detection/data/labels/make_muvim_labels.py)

This is not a generic adapter layer. Each dataset has custom parsing assumptions:

- LE2i parses `Annotation_files` text files and extracts scene/video IDs from sequence stems in `stem_to_scene_and_vid()` and `find_annotation_file()` in [src/fall_detection/data/labels/make_le2i_labels.py](../src/fall_detection/data/labels/make_le2i_labels.py)
- CAUCAFall infers subject/action from NPZ `src` metadata in `_infer_subj_action_from_npz()` in [src/fall_detection/data/labels/make_caucafall_labels.py](../src/fall_detection/data/labels/make_caucafall_labels.py)
- MUVIM uses filename heuristics plus `ZED_RGB.csv` in `extract_video_id()` and `load_spans_from_csv()` in [src/fall_detection/data/labels/make_muvim_labels.py](../src/fall_detection/data/labels/make_muvim_labels.py)
- URFD intentionally disables per-frame txt use by default because YOLO bbox txt can be misread as action labels; see `--use_per_frame_action_txt` and the guard in [src/fall_detection/data/labels/make_urfall_labels.py](../src/fall_detection/data/labels/make_urfall_labels.py)

For teaching, this is an important distinction: label generation is dataset adapter code in practice, but it is implemented as custom scripts, not as a unified adapter abstraction.

### 4. Splits

Split generation is generic and lives in [scripts/make_splits.py](../scripts/make_splits.py) -> [src/fall_detection/data/splits/make_splits.py](../src/fall_detection/data/splits/make_splits.py).

This is one of the cleaner reusable pieces:

- `group_id_for()` supports `none`, `before_dunder`, `regex`, `caucafall_subject`, and `json`
- `split_groups_to_match_targets()` balances by stem counts or group counts
- `enforce_min_per_class()` tries to preserve class presence across splits

Leakage control is real here, not aspirational. The most important dataset-specific safeguard is `caucafall_subject`, used by Makefile variables `CAUCA_SPLIT_GROUP_MODE` and `CAUCA_SPLIT_BALANCE_BY` in [Makefile](../Makefile).

### 5. Windowing

Window generation is the center of the offline pipeline:

- labeled windows: [scripts/make_windows.py](../scripts/make_windows.py) -> [src/fall_detection/data/windowing/make_windows_impl.py](../src/fall_detection/data/windowing/make_windows_impl.py)
- unlabeled windows: [scripts/make_unlabeled_windows.py](../scripts/make_unlabeled_windows.py) -> [src/fall_detection/data/windowing/make_unlabeled_windows_impl.py](../src/fall_detection/data/windowing/make_unlabeled_windows_impl.py)
- false-alert windows: [scripts/make_fa_windows.py](../scripts/make_fa_windows.py) -> [src/fall_detection/data/windowing/make_fa_windows_impl.py](../src/fall_detection/data/windowing/make_fa_windows_impl.py)
- sanity check: [scripts/check_windows.py](../scripts/check_windows.py) -> [src/fall_detection/data/windowing/check_windows.py](../src/fall_detection/data/windowing/check_windows.py)

The active window schema is defined by what [src/fall_detection/data/windowing/make_windows_impl.py](../src/fall_detection/data/windowing/make_windows_impl.py) writes and what [src/fall_detection/core/features.py](../src/fall_detection/core/features.py) reads:

- legacy keys kept for compatibility: `xy`, `conf`, `y`, `label`
- canonical runtime keys: `joints`, `motion`, `mask`, `valid_frac`, `overlap_frames`, `overlap_frac`, `fps`, `video_id`, `seq_id`, `src`, `seq_stem`, `w_start`, `w_end`

The main logic is:

- positive windows are defined by overlap with fall spans in `choose_balanced_windows_for_fall_video()`
- negatives are sampled with `neg_ratio`, `max_neg_per_video`, and optional hard-negative sampling near spans
- if spans are missing, fallback behavior is dataset-sensitive through `fallback_if_no_span`

Important teaching point:

- this repository is not training directly on raw sequences
- the stable reusable contract is "window NPZ folder with train/val/test subdirectories"

### 6. Training

Training is model-specific but structurally consistent:

- TCN: [scripts/train_tcn.py](../scripts/train_tcn.py) -> [src/fall_detection/training/train_tcn.py](../src/fall_detection/training/train_tcn.py)
- GCN: [scripts/train_gcn.py](../scripts/train_gcn.py) -> [src/fall_detection/training/train_gcn.py](../src/fall_detection/training/train_gcn.py)

Shared training dependencies:

- checkpoint contract: [src/fall_detection/core/ckpt.py](../src/fall_detection/core/ckpt.py)
- feature construction: [src/fall_detection/core/features.py](../src/fall_detection/core/features.py)
- model builders: [src/fall_detection/core/models.py](../src/fall_detection/core/models.py)
- losses: [src/fall_detection/core/losses.py](../src/fall_detection/core/losses.py)
- metrics: [src/fall_detection/core/metrics.py](../src/fall_detection/core/metrics.py)
- EMA: [src/fall_detection/core/ema.py](../src/fall_detection/core/ema.py)

Both trainers build datasets from window NPZ files:

- `WindowDatasetTCN` in [src/fall_detection/training/train_tcn.py](../src/fall_detection/training/train_tcn.py)
- `WindowDatasetGCN` in [src/fall_detection/training/train_gcn.py](../src/fall_detection/training/train_gcn.py)

Both trainers also support:

- checkpoint resume
- hard-negative augmentation from external file lists
- automatic threshold selection by validation F1
- feature flags embedded into checkpoint bundles

### 7. Thresholding / calibration / operating-point fitting

Operating-point fitting is handled by [scripts/fit_ops.py](../scripts/fit_ops.py) -> [src/fall_detection/evaluation/fit_ops.py](../src/fall_detection/evaluation/fit_ops.py).

This stage is essential to understanding deployment. It does more than choose a threshold:

- loads the trained checkpoint through `load_ckpt()`
- reconstructs features through `FeatCfg` overrides in `_override_feat_cfg()`
- infers logits on validation windows through `WindowDirDataset` and `infer_logits()`
- optionally fits temperature calibration with `fit_temperature()` from [src/fall_detection/core/calibration.py](../src/fall_detection/core/calibration.py)
- sweeps alert-policy parameters using `sweep_alert_policy_from_windows()` from [src/fall_detection/core/alerting.py](../src/fall_detection/core/alerting.py)
- writes deployable YAML through `yaml_dump_simple()` to `configs/ops/*.yaml`

The output YAML contains the deployment contract:

- checkpoint path
- `feat_cfg`
- `alert_cfg`
- `ops` for OP-1 / OP-2 / OP-3

### 8. Evaluation

Primary labeled evaluation:

- [scripts/eval_metrics.py](../scripts/eval_metrics.py) -> [src/fall_detection/evaluation/metrics_eval.py](../src/fall_detection/evaluation/metrics_eval.py)

Optional supporting evaluation:

- unlabeled alert-rate scoring: [src/fall_detection/evaluation/score_unlabeled_alert_rate.py](../src/fall_detection/evaluation/score_unlabeled_alert_rate.py)
- hard-negative mining: [scripts/mine_hard_negatives.py](../scripts/mine_hard_negatives.py) -> [src/fall_detection/evaluation/mine_hard_negatives.py](../src/fall_detection/evaluation/mine_hard_negatives.py)
- plotting: [src/fall_detection/evaluation/plot_fa_recall.py](../src/fall_detection/evaluation/plot_fa_recall.py), [src/fall_detection/evaluation/plot_f1_vs_tau.py](../src/fall_detection/evaluation/plot_f1_vs_tau.py), and several `scripts/plot_*`

[src/fall_detection/evaluation/metrics_eval.py](../src/fall_detection/evaluation/metrics_eval.py) is event-oriented evaluation, not just classification accuracy. It:

- reconstructs model input from saved windows
- applies `AlertCfg` from YAML or CLI
- computes alert events using `classify_states()`, `detect_alert_events()`, and `times_from_windows()` from [src/fall_detection/core/alerting.py](../src/fall_detection/core/alerting.py)
- optionally uses confirm heuristics from [src/fall_detection/core/confirm.py](../src/fall_detection/core/confirm.py)

### 9. Deployment / inference

There are two deployment-related paths:

- offline deploy runners in [src/fall_detection/deploy/run_modes.py](../src/fall_detection/deploy/run_modes.py) and [src/fall_detection/deploy/run_alert_policy.py](../src/fall_detection/deploy/run_alert_policy.py)
- the actual live system in [server/](../server)

The active live deployment path is:

- API assembly in [server/main.py](../server/main.py)
- inference spec discovery in [server/deploy_runtime.py](../server/deploy_runtime.py)
- window inference endpoint in [server/routes/monitor.py](../server/routes/monitor.py)
- online state machine in [server/online_alert.py](../server/online_alert.py)

Critical point:

- the server uses `configs/ops/*.yaml` as the source of truth for checkpoint path, feature config, alert config, and operating points
- [server/README.md](../server/README.md) explicitly says [configs/deploy_modes.yaml](../configs/deploy_modes.yaml) is not the primary source for `server/routes/monitor.py`

That means the real deployment handoff is `fit_ops.py -> configs/ops/*.yaml -> server/deploy_runtime.py`.

## Dependency map in execution order

### Offline training and evaluation path

1. [Makefile](../Makefile)
2. data extraction wrappers in [scripts/extract_pose_videos.py](../scripts/extract_pose_videos.py) or [scripts/extract_pose_images.py](../scripts/extract_pose_images.py)
3. pose extractors in [src/fall_detection/pose/extract_2d_videos.py](../src/fall_detection/pose/extract_2d_videos.py) and [src/fall_detection/pose/extract_2d_images.py](../src/fall_detection/pose/extract_2d_images.py)
4. pose cleanup in [scripts/preprocess_pose.py](../scripts/preprocess_pose.py) -> [src/fall_detection/pose/preprocess_pose_npz.py](../src/fall_detection/pose/preprocess_pose_npz.py)
5. dataset-specific labels in [src/fall_detection/data/labels/](../src/fall_detection/data/labels)
6. split generation in [src/fall_detection/data/splits/make_splits.py](../src/fall_detection/data/splits/make_splits.py)
7. labeled and eval windows in [src/fall_detection/data/windowing/make_windows_impl.py](../src/fall_detection/data/windowing/make_windows_impl.py)
8. optional unlabeled windows in [src/fall_detection/data/windowing/make_unlabeled_windows_impl.py](../src/fall_detection/data/windowing/make_unlabeled_windows_impl.py)
9. optional FA windows in [src/fall_detection/data/windowing/make_fa_windows_impl.py](../src/fall_detection/data/windowing/make_fa_windows_impl.py)
10. training in [src/fall_detection/training/train_tcn.py](../src/fall_detection/training/train_tcn.py) or [src/fall_detection/training/train_gcn.py](../src/fall_detection/training/train_gcn.py)
11. checkpoint bundle and feature/model config persistence in [src/fall_detection/core/ckpt.py](../src/fall_detection/core/ckpt.py)
12. operating-point fit and optional calibration in [src/fall_detection/evaluation/fit_ops.py](../src/fall_detection/evaluation/fit_ops.py)
13. labeled evaluation in [src/fall_detection/evaluation/metrics_eval.py](../src/fall_detection/evaluation/metrics_eval.py)
14. optional unlabeled FA scoring in [src/fall_detection/evaluation/score_unlabeled_alert_rate.py](../src/fall_detection/evaluation/score_unlabeled_alert_rate.py)
15. plots and reports in [scripts/plot_*.py](../scripts)

### Live deployment path

1. fitted YAML in [configs/ops/](../configs/ops)
2. FastAPI boot in [server/app.py](../server/app.py) and [server/main.py](../server/main.py)
3. spec discovery in `discover_specs()` and `_discover_from_ops_yaml()` in [server/deploy_runtime.py](../server/deploy_runtime.py)
4. request handling in `predict_window()` and `monitor_ws()` in [server/routes/monitor.py](../server/routes/monitor.py)
5. model reconstruction and forward pass in `predict_spec()` in [server/deploy_runtime.py](../server/deploy_runtime.py)
6. online temporal state in `OnlineAlertTracker.step()` in [server/online_alert.py](../server/online_alert.py)

## Essential vs optional for teaching

### Essential

- [Makefile](../Makefile): actual orchestration contract for the repo
- [src/fall_detection/pose/preprocess_pose_npz.py](../src/fall_detection/pose/preprocess_pose_npz.py): establishes sequence-level pose contract
- [src/fall_detection/data/labels/](../src/fall_detection/data/labels): real dataset adaptation layer
- [src/fall_detection/data/splits/make_splits.py](../src/fall_detection/data/splits/make_splits.py): leakage control
- [src/fall_detection/data/windowing/make_windows_impl.py](../src/fall_detection/data/windowing/make_windows_impl.py): converts sequences to training/eval examples
- [src/fall_detection/core/features.py](../src/fall_detection/core/features.py): single most important feature contract
- [src/fall_detection/core/models.py](../src/fall_detection/core/models.py): model reconstruction contract
- [src/fall_detection/core/ckpt.py](../src/fall_detection/core/ckpt.py): checkpoint portability contract
- [src/fall_detection/training/train_tcn.py](../src/fall_detection/training/train_tcn.py) and [src/fall_detection/training/train_gcn.py](../src/fall_detection/training/train_gcn.py): actual training logic
- [src/fall_detection/core/alerting.py](../src/fall_detection/core/alerting.py): event logic used by fit/eval/deploy
- [src/fall_detection/evaluation/fit_ops.py](../src/fall_detection/evaluation/fit_ops.py): converts a trained classifier into a deployment profile
- [src/fall_detection/evaluation/metrics_eval.py](../src/fall_detection/evaluation/metrics_eval.py): event-level evaluation
- [server/deploy_runtime.py](../server/deploy_runtime.py), [server/routes/monitor.py](../server/routes/monitor.py), [server/online_alert.py](../server/online_alert.py): real inference path

### Optional but useful

- [src/fall_detection/data/adapters/](../src/fall_detection/data/adapters): useful for teaching canonicalization and dataset normalization, but not the primary path used by the offline pipeline
- [src/fall_detection/evaluation/mine_hard_negatives.py](../src/fall_detection/evaluation/mine_hard_negatives.py): useful for iterative training improvement
- [src/fall_detection/evaluation/score_unlabeled_alert_rate.py](../src/fall_detection/evaluation/score_unlabeled_alert_rate.py): useful for deployment realism
- [src/fall_detection/core/confirm.py](../src/fall_detection/core/confirm.py): useful for heuristic post-filtering
- [src/fall_detection/core/uncertainty.py](../src/fall_detection/core/uncertainty.py): optional MC-dropout path in deployment
- plotting scripts in [scripts/plot_*.py](../scripts): reporting, not core pipeline
- audit scripts in [scripts/audit_*.py](../scripts): repository quality gates, not model pipeline

### Optional / not central for first teaching pass

- React app in [apps/](../apps)
- most artifacts in [artifacts/](../artifacts)
- experiment sweeps in [tools/sweeps/](../tools/sweeps) and [tools/sweeps_simple/](../tools/sweeps_simple)
- report-heavy material in [docs/project_targets/](../docs/project_targets) and [docs/reports/](../docs/reports)

## Architectural issues and risks

### 1. Two architectures coexist, but only one is actually driving the system

Evidence:

- active orchestration points to `scripts/*` in [Makefile](../Makefile)
- the contract-first pipeline in [src/fall_detection/data/pipeline.py](../src/fall_detection/data/pipeline.py), resolver in [src/fall_detection/data/resolver.py](../src/fall_detection/data/resolver.py), and datamodule in [src/fall_detection/data/datamodule.py](../src/fall_detection/data/datamodule.py) are not referenced by [Makefile](../Makefile)
- repository search found no active call path from orchestration into `UnifiedWindowDataModule` or `rebuild_labels_and_splits_from_raw()`

Impact:

- new readers may mistake the refactor scaffold for the actual data path
- maintenance risk because abstractions and actual scripts can drift independently

Teaching classification:

- teach `src/fall_detection/data/*` as "secondary scaffold / partial refactor", not as the canonical runtime path

### 2. Import and packaging conventions are inconsistent

Evidence:

- working code mostly imports `fall_detection.*` under [src/fall_detection/](../src/fall_detection)
- many tests under [tests/server/](../tests/server) still import `core.*`, `pose.*`, and `server.*`
- `pytest -q` and `PYTHONPATH="$(pwd)/src:$(pwd)" pytest -q` both fail collection because those top-level aliases are missing
- [scripts/smoke_api_contract.py](../scripts/smoke_api_contract.py) only works with repo-root `PYTHONPATH`

Impact:

- default local developer experience is fragile
- testing does not reflect the installed package shape

Teaching classification:

- important example of partial migration debt

### 3. Deployment dataset support is narrower than training/data-prep support

Evidence:

- Makefile and README expose `le2i`, `urfd`, `caucafall`, `muvim`
- [server/deploy_runtime.py](../server/deploy_runtime.py) hardcodes `SUPPORTED_DATASETS = {"caucafall", "le2i"}`

Impact:

- MUVIM and URFD may be trainable/evaluable offline but are not first-class deploy targets in the FastAPI path

Teaching classification:

- real architectural boundary, not just documentation drift

### 4. There are two deployment config surfaces

Evidence:

- live server uses `configs/ops/*.yaml` through [server/deploy_runtime.py](../server/deploy_runtime.py)
- offline deploy runners use [configs/deploy_modes.yaml](../configs/deploy_modes.yaml) through [src/fall_detection/deploy/run_modes.py](../src/fall_detection/deploy/run_modes.py)
- [server/README.md](../server/README.md) explicitly says `configs/deploy_modes.yaml` is not the server source of truth

Impact:

- readers can easily choose the wrong config source
- future drift risk between offline demo runners and the real API server

### 5. Dataset-specific hacks are embedded in orchestration and policy

Examples:

- CAUCAFall subject-level split behavior in [Makefile](../Makefile) and `group_id_for()` in [src/fall_detection/data/splits/make_splits.py](../src/fall_detection/data/splits/make_splits.py)
- MUVIM fallback `skip_fall` injected in [Makefile](../Makefile) via `WIN_EXTRA_muvim` and `WIN_EVAL_EXTRA_muvim`
- LE2i unlabeled scene selection via `UNLABELED_SCENES_le2i` in [Makefile](../Makefile)
- live start-guard prefixes and scene-scoped logic in [src/fall_detection/core/alerting.py](../src/fall_detection/core/alerting.py) and [server/routes/monitor.py](../server/routes/monitor.py)

Impact:

- reproducibility depends on orchestration variables, not only on module defaults

### 6. Leakage protections exist, but can be bypassed

Evidence:

- split leakage prevention is explicit in [src/fall_detection/data/splits/make_splits.py](../src/fall_detection/data/splits/make_splits.py)
- hard-negative path validation exists in `_validate_hard_neg_paths()` inside [src/fall_detection/training/train_tcn.py](../src/fall_detection/training/train_tcn.py) and [src/fall_detection/training/train_gcn.py](../src/fall_detection/training/train_gcn.py)
- both trainers expose `--allow_hard_neg_nontrain`

Impact:

- the repository is aware of leakage risk
- but it still allows deliberate opt-out, which is appropriate for experiments but dangerous for teaching unless clearly labeled

### 7. Evaluation and live alerting are similar, but not identical

Evidence:

- offline eval uses [src/fall_detection/core/alerting.py](../src/fall_detection/core/alerting.py) and can apply confirmation heuristics from [src/fall_detection/core/confirm.py](../src/fall_detection/core/confirm.py)
- live API uses [server/online_alert.py](../server/online_alert.py), which implements EMA + k-of-n + hysteresis + cooldown, but explicitly does not apply confirmation heuristics

Impact:

- deployment is close to evaluation, but not perfectly identical
- event behavior can diverge from offline confirm-enabled metrics

Teaching classification:

- important to explain as "deployment approximation of offline policy"

### 8. Legacy and duplicate code remains in-place

Confirmed duplicates or likely legacy surfaces:

- [windows/make_fa_windows.py](../windows/make_fa_windows.py) duplicates the packaged FA-window logic in [src/fall_detection/data/windowing/make_fa_windows_impl.py](../src/fall_detection/data/windowing/make_fa_windows_impl.py)
- thin wrappers in [scripts/](../scripts) mostly forward directly into package modules
- many `__pycache__` and `.DS_Store` files are committed or present in the repository tree

Needs verification:

- whether any external workflow still calls [windows/make_fa_windows.py](../windows/make_fa_windows.py) directly

### 9. `data_sources.yaml` and resolver contracts are incomplete relative to the repo

Evidence:

- [configs/experiments/data_sources.yaml](../configs/experiments/data_sources.yaml) only defines two datasets and empty splits
- active Makefile workflows target four datasets and use text split files under [configs/splits/](../configs/splits)

Impact:

- the typed resolver stack is not a full replacement for the real repo configuration

## File and directory map by role

### Core pipeline

- [Makefile](../Makefile)
- [README.md](../README.md)
- [src/fall_detection/core/](../src/fall_detection/core)
- [src/fall_detection/pose/preprocess_pose_npz.py](../src/fall_detection/pose/preprocess_pose_npz.py)
- [src/fall_detection/data/splits/make_splits.py](../src/fall_detection/data/splits/make_splits.py)
- [src/fall_detection/data/windowing/](../src/fall_detection/data/windowing)
- [src/fall_detection/training/](../src/fall_detection/training)
- [src/fall_detection/evaluation/fit_ops.py](../src/fall_detection/evaluation/fit_ops.py)
- [src/fall_detection/evaluation/metrics_eval.py](../src/fall_detection/evaluation/metrics_eval.py)
- [configs/labels/](../configs/labels)
- [configs/splits/](../configs/splits)
- [configs/ops/](../configs/ops)

### Dataset adapters

- [src/fall_detection/data/labels/](../src/fall_detection/data/labels)
- [src/fall_detection/data/adapters/](../src/fall_detection/data/adapters)
- [src/fall_detection/pose/extract_2d_videos.py](../src/fall_detection/pose/extract_2d_videos.py)
- [src/fall_detection/pose/extract_2d_images.py](../src/fall_detection/pose/extract_2d_images.py)
- [src/fall_detection/pose/parse_ntu_skeleton.py](../src/fall_detection/pose/parse_ntu_skeleton.py)

### Training

- [src/fall_detection/training/train_tcn.py](../src/fall_detection/training/train_tcn.py)
- [src/fall_detection/training/train_gcn.py](../src/fall_detection/training/train_gcn.py)
- [scripts/train_tcn.py](../scripts/train_tcn.py)
- [scripts/train_gcn.py](../scripts/train_gcn.py)
- [src/fall_detection/evaluation/mine_hard_negatives.py](../src/fall_detection/evaluation/mine_hard_negatives.py)
- [scripts/mine_hard_negatives.py](../scripts/mine_hard_negatives.py)

### Evaluation

- [src/fall_detection/evaluation/metrics_eval.py](../src/fall_detection/evaluation/metrics_eval.py)
- [src/fall_detection/evaluation/fit_ops.py](../src/fall_detection/evaluation/fit_ops.py)
- [src/fall_detection/evaluation/score_unlabeled_alert_rate.py](../src/fall_detection/evaluation/score_unlabeled_alert_rate.py)
- [src/fall_detection/evaluation/plot_fa_recall.py](../src/fall_detection/evaluation/plot_fa_recall.py)
- [src/fall_detection/evaluation/plot_f1_vs_tau.py](../src/fall_detection/evaluation/plot_f1_vs_tau.py)
- [scripts/eval_metrics.py](../scripts/eval_metrics.py)
- [scripts/fit_ops.py](../scripts/fit_ops.py)
- [scripts/plot_*.py](../scripts)

### Configs

- [configs/labels/](../configs/labels)
- [configs/splits/](../configs/splits)
- [configs/ops/](../configs/ops)
- [configs/ops/dual_policy/](../configs/ops/dual_policy)
- [configs/deploy_modes.yaml](../configs/deploy_modes.yaml)
- [configs/experiments/data_sources.yaml](../configs/experiments/data_sources.yaml)
- [configs/audit_gates.json](../configs/audit_gates.json)

### Scripts

- [scripts/](../scripts): primary CLI wrappers and audit/report tools
- [tools/](../tools): experiment and release tooling, mostly not core runtime

### Experiments

- [tools/sweeps/](../tools/sweeps)
- [tools/sweeps_simple/](../tools/sweeps_simple)
- [artifacts/reports/](../artifacts/reports)
- [baselines/](../baselines)
- [outputs/](../outputs)

### Deployment

- [server/](../server)
- [src/fall_detection/deploy/](../src/fall_detection/deploy)
- [apps/](../apps)

### Legacy / unused / duplicate

- [src/fall_detection/data/pipeline.py](../src/fall_detection/data/pipeline.py): useful scaffold, not on the main Makefile path
- [src/fall_detection/data/datamodule.py](../src/fall_detection/data/datamodule.py): typed data layer, not the training path used by current trainers
- [src/fall_detection/data/contracts.py](../src/fall_detection/data/contracts.py), [src/fall_detection/data/schema.py](../src/fall_detection/data/schema.py), [src/fall_detection/data/transforms.py](../src/fall_detection/data/transforms.py): refactor infrastructure, lightly used
- [windows/make_fa_windows.py](../windows/make_fa_windows.py): duplicate legacy script surface
- [Makefile.txt](../Makefile.txt): appears legacy or reference-only
- committed cache/junk files such as `.DS_Store` and `__pycache__` entries across the tree

## Recommended teaching cut

If this repository is reused for teaching, the smallest coherent subset is:

1. [Makefile](../Makefile)
2. [src/fall_detection/pose/preprocess_pose_npz.py](../src/fall_detection/pose/preprocess_pose_npz.py)
3. one label builder, ideally [src/fall_detection/data/labels/make_caucafall_labels.py](../src/fall_detection/data/labels/make_caucafall_labels.py) plus [src/fall_detection/data/splits/make_splits.py](../src/fall_detection/data/splits/make_splits.py)
4. [src/fall_detection/data/windowing/make_windows_impl.py](../src/fall_detection/data/windowing/make_windows_impl.py)
5. [src/fall_detection/core/features.py](../src/fall_detection/core/features.py)
6. one trainer, ideally [src/fall_detection/training/train_tcn.py](../src/fall_detection/training/train_tcn.py)
7. [src/fall_detection/evaluation/fit_ops.py](../src/fall_detection/evaluation/fit_ops.py)
8. [src/fall_detection/evaluation/metrics_eval.py](../src/fall_detection/evaluation/metrics_eval.py)
9. [server/deploy_runtime.py](../server/deploy_runtime.py) and [server/routes/monitor.py](../server/routes/monitor.py)

Everything else should be framed as optional extensions, experiment tooling, or partial refactor scaffolding.

## Needs verification

- Whether [src/fall_detection/data/pipeline.py](../src/fall_detection/data/pipeline.py) is used outside the repository by unpublished tooling
- Whether [windows/make_fa_windows.py](../windows/make_fa_windows.py) is still called by any external workflow
- Whether `urfd` is intentionally named `urfall` in parts of the codebase or whether this is unresolved naming drift
- Whether deployment support for `muvim` and `urfd` is intentionally out of scope or simply unfinished
