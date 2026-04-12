# Experiment Recording Protocol (Required)

This protocol is mandatory for all overfitting experiments on LE2i/CAUCAFall.
Every parameter change and result must be recorded so final report tables are reproducible.

## Registry file

- Primary registry:
  - `artifacts/registry/overfit_experiment_registry.csv`
- Legacy tuning changelog (already used):
  - `artifacts/reports/tuning/PARAM_CHANGELOG.csv`

Use the new registry as source-of-truth for this branch.

## Recording tool

- Script: `tools/track_experiment.py`
- Auto-captures:
  - `git_branch`
  - `git_commit`
  - `git_dirty`

## Required workflow per experiment

1. Record `planned` before running.
2. Record `running` when job starts.
3. Record `done` or `failed` immediately after run.
4. `done` rows must include:
   - `metrics_json`
   - `ops_yaml` (if applicable)
   - checkpoint/output path in `artifacts`

## Command templates

### 1) planned

```bash
python tools/track_experiment.py \
  --exp_id "exp_tcn_caucafall_aug_v1_seed33724876" \
  --phase train \
  --dataset caucafall \
  --arch tcn \
  --seed 33724876 \
  --status planned \
  --changed_params "flip=1,noise_std=0.01,joint_dropout=0.1,time_jitter=1" \
  --command "python scripts/train_tcn.py ... --save_dir outputs/caucafall_tcn_W48S12_aug_v1" \
  --notes "baseline overfit mitigation trial"
```

### 2) running

```bash
python tools/track_experiment.py \
  --exp_id "exp_tcn_caucafall_aug_v1_seed33724876" \
  --phase train \
  --dataset caucafall \
  --arch tcn \
  --seed 33724876 \
  --status running \
  --changed_params "flip=1,noise_std=0.01,joint_dropout=0.1,time_jitter=1" \
  --command "python scripts/train_tcn.py ... --save_dir outputs/caucafall_tcn_W48S12_aug_v1"
```

### 3) done

```bash
python tools/track_experiment.py \
  --exp_id "exp_tcn_caucafall_aug_v1_seed33724876" \
  --phase eval \
  --dataset caucafall \
  --arch tcn \
  --seed 33724876 \
  --status done \
  --changed_params "flip=1,noise_std=0.01,joint_dropout=0.1,time_jitter=1" \
  --command "make fit-ops-caucafall ... && make eval-caucafall ..." \
  --artifacts "outputs/caucafall_tcn_W48S12_aug_v1/best.pt;outputs/metrics/tcn_caucafall_aug_v1.json" \
  --metrics_json "outputs/metrics/tcn_caucafall_aug_v1.json" \
  --ops_yaml "configs/ops/tcn_caucafall_aug_v1.yaml" \
  --notes "compare to baseline OP-2"
```

### 4) failed

```bash
python tools/track_experiment.py \
  --exp_id "exp_tcn_caucafall_aug_v1_seed33724876" \
  --phase train \
  --dataset caucafall \
  --arch tcn \
  --seed 33724876 \
  --status failed \
  --changed_params "flip=1,noise_std=0.01,joint_dropout=0.1,time_jitter=1" \
  --command "python scripts/train_tcn.py ... --save_dir outputs/caucafall_tcn_W48S12_aug_v1" \
  --notes "OOM on epoch 2; reduce batch from 128 to 64"
```

## Report integration

Before writing final report tables:

1. Filter `status=done`.
2. Verify each row has existing `metrics_json`.
3. Build summary table grouped by:
   - dataset
   - arch
   - changed_params
4. Keep failed/rejected rows in appendix for auditability.

