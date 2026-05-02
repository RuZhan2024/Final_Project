# CTR-GCN Baseline Preparation

Date: 2026-05-02  
Branch: `feat/ctr-gcn-upgrade`

## 1. Purpose

This note splits `Task G` into:

- preparation work that can be completed on the current Intel/macOS development machine
- formal baseline work that should be run on a stronger NVIDIA training machine

The goal is to keep the `CTR-GCN` branch moving without pretending that the current machine is suitable for long formal training runs.

## 2. Current Machine Position

Current development machine:

- Intel 9th-gen i9
- 64 GB system RAM
- AMD Radeon Pro 5500M

Practical implication:

- good enough for code development
- good enough for smoke runs
- good enough for trainer / checkpoint / `fit_ops` / alert-policy validation
- **not** the preferred platform for full `CAUCAFall` baseline training

## 3. Task G Split

### 3.1 Work that is already done on the current machine

The following is already complete:

- architecture contract
- legacy regression protection
- canonical input/forward validation
- standalone trainer smoke run
- checkpoint reload validation
- `fit_ops` compatibility validation
- event-level alert-policy smoke validation

Reference:

- [CTR_GCN_TASK_A_B0_B_VALIDATION_2026-05-02.md](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/docs/reports/notes/CTR_GCN_TASK_A_B0_B_VALIDATION_2026-05-02.md:1)

### 3.2 Work that can still be done on the current machine

Before moving to the NVIDIA box, the branch can still safely do:

1. lock one official baseline command
2. lock one official output directory naming scheme
3. lock one official `fit_ops` command
4. lock one official replay/runtime validation command
5. keep the branch code in a clean commit state

### 3.3 Work deferred to the NVIDIA training machine

These should be run on the NVIDIA machine:

1. full `CAUCAFall` baseline training
2. longer-epoch training
3. multi-seed confirmation
4. tuning of batch size / learning rate / dropout / width
5. later `LE2i` transfer checks

## 4. Official First-Baseline Command

This is the first intended `CAUCAFall` baseline command for the NVIDIA machine.

```bash
source .venv/bin/activate
PYTHONPATH="$(pwd)/ml/src:$(pwd)" \
python3 ml/src/fall_detection/training/train_ctr_gcn.py \
  --train_dir data/processed/caucafall/windows_eval_W48_S12/train \
  --val_dir data/processed/caucafall/windows_eval_W48_S12/val \
  --save_dir outputs/caucafall_ctr_gcn_W48S12_baseline \
  --epochs 180 \
  --min_epochs 20 \
  --batch 64 \
  --lr 1e-3 \
  --weight_decay 1e-4 \
  --seed 33724876 \
  --patience 25 \
  --grad_clip 1.0 \
  --fps_default 23 \
  --center pelvis \
  --loss bce \
  --monitor ap \
  --pos_weight auto \
  --channel_schedule 64,64,64,128 \
  --rel_channels 8 \
  --ctr_rank 8 \
  --temporal_kernel 9 \
  --dropout 0.30 \
  --use_conf_channel 1 \
  --use_motion 1 \
  --use_bone 0 \
  --use_bone_length 0 \
  --motion_scale_by_fps 1 \
  --conf_gate 0.2 \
  --use_precomputed_mask 1 \
  --thr_min 0.05 \
  --thr_max 0.95 \
  --thr_step 0.01 \
  --num_workers 4 \
  --deterministic 1 \
  --amp 1
```

## 5. Official Follow-Up `fit_ops` Command

After the baseline training completes:

```bash
source .venv/bin/activate
PYTHONPATH="$(pwd)/ml/src:$(pwd)" \
python3 ml/src/fall_detection/evaluation/fit_ops.py \
  --arch ctr_gcn \
  --val_dir data/processed/caucafall/windows_eval_W48_S12/val \
  --ckpt outputs/caucafall_ctr_gcn_W48S12_baseline/best.pt \
  --out configs/ops/ctr_gcn_caucafall_baseline.yaml \
  --fps_default 23 \
  --center pelvis \
  --use_motion 1 \
  --use_conf_channel 1 \
  --use_bone 0 \
  --use_bone_length 0 \
  --ema_alpha 0.20 \
  --k 2 \
  --n 3 \
  --cooldown_s 30 \
  --tau_low_ratio 0.78 \
  --confirm 0 \
  --confirm_s 2.0 \
  --confirm_min_lying 0.65 \
  --confirm_max_motion 0.08 \
  --confirm_require_low 1 \
  --thr_min 0.01 \
  --thr_max 0.95 \
  --thr_step 0.01 \
  --time_mode center \
  --merge_gap_s 1.0 \
  --overlap_slack_s 0.5 \
  --op1_recall 0.95 \
  --op3_fa24h 1.0 \
  --ops_picker conservative \
  --op_tie_break max_thr \
  --tie_eps 1e-3 \
  --save_sweep_json 1 \
  --allow_degenerate_sweep 0 \
  --emit_absolute_paths 0 \
  --min_tau_high 0.20
```

## 6. Official Event-Level Smoke / Replay Command

After `fit_ops` completes:

```bash
source .venv/bin/activate
PYTHONPATH="$(pwd)/ml/src:$(pwd)" \
python3 ml/src/fall_detection/deploy/run_alert_policy.py \
  --arch ctr_gcn \
  --win_dir data/processed/caucafall/windows_eval_W48_S12/val \
  --ckpt outputs/caucafall_ctr_gcn_W48S12_baseline/best.pt \
  --alert_cfg configs/ops/ctr_gcn_caucafall_baseline.yaml \
  --time_mode center \
  --out_json outputs/caucafall_ctr_gcn_W48S12_baseline/alert_policy_val.json
```

## 7. Required Baseline Evidence

The first official baseline should not be treated as complete unless it leaves:

- `outputs/caucafall_ctr_gcn_W48S12_baseline/best.pt`
- `outputs/caucafall_ctr_gcn_W48S12_baseline/last.pt`
- `outputs/caucafall_ctr_gcn_W48S12_baseline/train_config.json`
- `configs/ops/ctr_gcn_caucafall_baseline.yaml`
- `configs/ops/ctr_gcn_caucafall_baseline.sweep.json`
- `outputs/caucafall_ctr_gcn_W48S12_baseline/alert_policy_val.json`

## 8. Promotion Rule

After the first official baseline, only then decide whether to continue to:

- multi-seed confirmation
- batch-size tuning
- learning-rate tuning
- width/depth tuning
- later `LE2i` transfer checks

No stronger claim should be made before the first official baseline exists.
