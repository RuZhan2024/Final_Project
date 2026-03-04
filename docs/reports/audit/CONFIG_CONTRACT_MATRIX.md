# CONFIG_CONTRACT_MATRIX

Legend: PASS = flag exists and is consumed; WARN = accepted but no-op or drift; FAIL = mismatch/break.

## Makefile -> Script CLI Contract

| Source (Makefile) | Script + Arg | Default | Status | Evidence |
|---|---|---:|---|---|
| `FEAT_FLAGS_TCN --use_motion` | `train_tcn.py --use_motion` | `1` | PASS | Makefile:203, train_tcn.py:533 |
| `FEAT_FLAGS_TCN --use_conf_channel` | `train_tcn.py --use_conf_channel` | `1` | PASS | Makefile:204, train_tcn.py:534 |
| `FEAT_FLAGS_TCN --use_bone` | `train_tcn.py --use_bone` | `0` | PASS | Makefile:205, train_tcn.py:535 |
| `FEAT_FLAGS_TCN --use_bone_length` | `train_tcn.py --use_bone_length` | `0` | PASS | Makefile:206, train_tcn.py:536 |
| `FEAT_FLAGS_TCN --motion_scale_by_fps` | `train_tcn.py --motion_scale_by_fps` | `1` | PASS | Makefile:207, train_tcn.py:537 |
| `FEAT_FLAGS_TCN --conf_gate` | `train_tcn.py --conf_gate` | `0.20` | PASS | Makefile:208, train_tcn.py:538 |
| `FEAT_FLAGS_TCN --use_precomputed_mask` | `train_tcn.py --use_precomputed_mask` | `1` | PASS | Makefile:209, train_tcn.py:539 |
| `FEAT_FLAGS_GCN --use_motion` | `train_gcn.py --use_motion` | `1` | PASS | Makefile:217, train_gcn.py:574 |
| `FEAT_FLAGS_GCN --use_conf` | `train_gcn.py --use_conf` | `1` | PASS | Makefile:218, train_gcn.py:573 |
| `FEAT_FLAGS_GCN --use_bone` | `train_gcn.py --use_bone` | `0` | PASS | Makefile:219, train_gcn.py:575 |
| `FEAT_FLAGS_GCN --use_bonelen` | `train_gcn.py --use_bonelen` | `0` | PASS | Makefile:220, train_gcn.py:576 |
| `FEAT_FLAGS_GCN --normalize` | `train_gcn.py --normalize` | `torso` | WARN | Accepted but documented no-op in train_gcn.py:489 |
| `FEAT_FLAGS_GCN --include_abs` | `train_gcn.py --include_abs` | `0` | WARN | Accepted but documented no-op in train_gcn.py:489 |
| `FEAT_FLAGS_GCN --include_vel` | `train_gcn.py --include_vel` | `1` | WARN | Accepted but documented no-op in train_gcn.py:489 |
| `FITOPS_FEAT_FLAGS --use_bone_length` | `fit_ops.py --use_bone_length` | `None` | PASS | Makefile:232, fit_ops.py:550 |
| `FITOPS_POLICY_FLAGS --confirm_*` | `fit_ops.py --confirm_*` | mixed | PASS | Makefile:370-374, fit_ops.py:601-607 |
| `FITOPS_GUARD_FLAGS --allow_degenerate_sweep` | `fit_ops.py --allow_degenerate_sweep` | `0` | PASS | Makefile:383, fit_ops.py:576 |
| `METRICS_SWEEP_FLAGS --thr_*` | `metrics_eval.py --thr_*` | `None` | PASS | Makefile:421, metrics_eval.py:606-608 |
| `windows-% --adapter_dataset` | `make_windows_impl.py --adapter_dataset` | `""` | PASS | Makefile:750, make_windows_impl.py:417 |
| `windows-% --adapter_urfall_target_fps` | `make_windows_impl.py --adapter_urfall_target_fps` | `25.0` | PASS | Makefile:175, make_windows_impl.py:423 |

## Pipeline DAG Contract

| Target | Expected | Actual | Status | Evidence |
|---|---|---|---|---|
| `pipeline-<ds>` | train + fit-ops + eval + plot | via dependency to `plot-%`, includes fit/eval chain | PASS | Makefile:1092 + target graph lines 911/874 |
| `pipeline-gcn-<ds>` | train + fit-ops + eval + plot | via dependency to `plot-gcn-%`, includes fit/eval chain | PASS | Makefile:1096 + target graph lines 921/886 |
| `pipeline-auto-tcn-<ds>` | comment says includes fit/eval | runs windows/eval-windows, optional FA, train, plot only | FAIL | Makefile:535 (doc) vs 1111-1118 (implementation) |
| `pipeline-auto-gcn-<ds>` | comment says includes fit/eval | runs windows/eval-windows, optional FA, train, plot only | FAIL | Makefile:536 (doc) vs 1119-1125 (implementation) |

## `configs/ops/*.yaml` Contract

| Key | fit_ops writer | metrics_eval reader | Status | Evidence |
|---|---|---|---|---|
| `ops.OP1/OP2/OP3` | emitted | parsed case-insensitive via `_extract_policy_and_ops` | PASS | fit_ops.py writes ops rows; metrics_eval.py:160-208 |
| `alert_cfg` | emitted | accepted via policy fallback (`alert_cfg`/`alert_base`) | PASS | metrics_eval.py:176-182 |
| `sweep_cfg` | emitted | not required in evaluator core | PASS | configs/ops/*.yaml + metrics_eval behavior |
| `model.ckpt` relative path | emitted relative by default | used by humans/deploy docs | PASS | fit_ops.py:116-123, ops yaml `../../outputs/...` |

## Training Metric/Scheduler Consistency

| Item | Observation | Status | Evidence |
|---|---|---|---|
| checkpoint monitor | `--monitor {f1,ap}` | PASS | train_tcn.py:514, train_gcn.py:547 |
| scheduler metric source | selectable `val_loss|val_ap|val_f1` | PASS | train_tcn.py:491, train_gcn.py:532 |
| scheduler smoothing | optional EMA beta | PASS | train_tcn.py:492, train_gcn.py:533 |
| val transform determinism | train-only masking via split guard | PASS | train_tcn dataset uses `split` gating; train_gcn similar |
| class imbalance strategy | balanced sampler xor pos_weight warning | PASS | train scripts enforce and log strategy |

## Contract Verdict
- Core flag wiring: **Mostly PASS**.
- Important drift: auto-pipeline target semantics and runtime two-stream feature slicing (documented in PATCH_PLAN P0).
