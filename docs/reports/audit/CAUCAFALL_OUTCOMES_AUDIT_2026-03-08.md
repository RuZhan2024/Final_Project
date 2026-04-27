# CAUCAFall Outcomes Audit (2026-03-08)

## Scope
- Verify CAUCAFall outcomes are metric-correct (not due to feature/code mismatch).
- Verify locked outcomes are reproducible from current repository artifacts.
- Check split leakage and model/window feature contract consistency.

## Commands Executed
```bash
make repro-best-caucafall ADAPTER_USE=1
python3 scripts/eval_metrics.py --win_dir data/processed/caucafall/windows_eval_W48_S12/test --ckpt outputs/caucafall_tcn_W48S12_r1_ctrl/best.pt --ops_yaml configs/ops/tcn_caucafall_r1_ctrl.yaml --out_json outputs/metrics/tcn_caucafall_locked.recheck.json --fps_default 23 --thr_min 0.001 --thr_max 0.95 --thr_step 0.01
python3 scripts/eval_metrics.py --win_dir data/processed/caucafall/windows_eval_W48_S12/test --ckpt outputs/caucafall_gcn_W48S12_r2_recallpush_b/best.pt --ops_yaml configs/ops/gcn_caucafall_locked.yaml --out_json outputs/metrics/gcn_caucafall_locked.recheck.json --fps_default 23 --thr_min 0.001 --thr_max 0.95 --thr_step 0.01
python3 scripts/fit_ops.py --arch tcn --val_dir data/processed/caucafall/windows_eval_W48_S12/val --ckpt outputs/caucafall_tcn_W48S12_r1_ctrl/best.pt --out /tmp/tcn_caucafall_fitops_recheck.yaml ...
python3 scripts/fit_ops.py --arch gcn --val_dir data/processed/caucafall/windows_eval_W48_S12/val --ckpt outputs/caucafall_gcn_W48S12_r2_recallpush_b/best.pt --out /tmp/gcn_caucafall_fitops_recheck_lockedparams.yaml ...
```

## Core Findings

### 1) Split leakage check: PASS
- `configs/splits/caucafall_train.txt` (80), `val.txt` (10), `test.txt` (10) are pairwise disjoint.

### 2) Feature/contract check: PASS
- Windows shape: `[T=48, V=33, XY=2]`, conf `[48,33]`.
- TCN ckpt: `num_joints=33`, `in_ch=264`, feature flags match YAML.
- GCN ckpt: `num_joints=33`, `in_feats_j=6`, `in_feats_m=2`, feature flags match YAML.

### 3) Locked metrics reproducibility: PASS
- Re-running eval with locked ckpt + locked ops reproduces exact metrics:
  - TCN (`outputs/metrics/tcn_caucafall_locked*.json`): AP `0.9795`, Recall `1.0000`, Precision `1.0000`, F1 `1.0000`, FA24h `0.0000`.
  - GCN (`outputs/metrics/gcn_caucafall_locked*.json`): AP `0.9683`, Recall `1.0000`, Precision `1.0000`, F1 `1.0000`, FA24h `0.0000`.

### 4) fit_ops drift risk: PARTIAL / IMPORTANT
- **GCN**: Using the same locked policy parameters (`op2_objective=cost_sensitive`, `cost_fp=10`, `min_tau_high=0.30`) reproduces locked OP thresholds exactly.
- **TCN**: Re-running current `fit_ops` with same nominal args as locked profile selects different OP2/OP3 thresholds than `configs/ops/tcn_caucafall_r1_ctrl.yaml`:
  - locked OP2: `tau_high=0.74` -> test Recall `1.0`
  - refit OP2: `tau_high=0.86` -> test Recall `0.6`
- Interpretation: TCN locked outcomes are reproducible **via locked YAML artifact**, but current auto-refit path does not recreate identical OP choice.

## Evidence Files
- Locked metrics:
  - `outputs/metrics/tcn_caucafall_locked.json`
  - `outputs/metrics/gcn_caucafall_locked.json`
- Recheck metrics:
  - `outputs/metrics/tcn_caucafall_locked.recheck.json`
  - `outputs/metrics/gcn_caucafall_locked.recheck.json`
- Refit ops (temporary):
  - `/tmp/tcn_caucafall_fitops_recheck.yaml`
  - `/tmp/gcn_caucafall_fitops_recheck_lockedparams.yaml`

## Final Assessment
- **Metric correctness (no feature mismatch): PASS**
- **Locked outcomes reproducibility: PASS (artifact-locked reproduction)**
- **Auto-refit reproducibility parity:**
  - GCN: PASS
  - TCN: FAIL (selection drift vs locked YAML)

## Minimal Fix Recommendation
1. Freeze one canonical TCN ops profile for thesis/deployment and reference it explicitly:
   - `configs/ops/tcn_caucafall_r1_ctrl.yaml`
2. Add a CI-style guard script to compare `fit_ops` regenerated OPs vs locked OPs and fail on drift.
3. In report/thesis, distinguish:
   - `artifact-locked reproducibility` (current PASS),
   - `regenerate-from-fit_ops reproducibility` (TCN currently drifted).
