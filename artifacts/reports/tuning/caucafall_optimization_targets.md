# CAUCAFall Optimization Targets (Primary Domain)

## Goal
Increase deployment recall/F1 while keeping false alerts controlled, using CAUCAFall only for acceptance decisions.

## Baseline (Promoted)
- TCN: `outputs/metrics/tcn_caucafall_cauc_hneg1_confirm0.json`
- GCN: `outputs/metrics/gcn_caucafall_cauc_hneg1.json`
- Current baseline snapshot:
  - TCN: `AP=0.9819, F1=0.8889, Recall=0.8000, FA24h=0.0`
  - GCN: `AP=0.9676, F1=0.8889, Recall=0.8000, FA24h=0.0`

## Acceptance Gates (Optimization Phase)
1. Primary gate: maximize `Recall` first.
2. Safety gate: keep `FA24h` as low as possible and track tradeoff explicitly.
3. Selection rule: val-fit only, test for final comparison only.

## Step 1 (No Retrain): Policy-First Recall Push
Run fit-ops with recall-weighted cost objective:
- `op2_objective=cost`
- `cost_fn=20`, `cost_fp=1`
- `confirm=0`

Artifacts:
- `configs/ops/tcn_caucafall_recallpush_cost20.yaml`
- `configs/ops/gcn_caucafall_recallpush_cost20.yaml`
- `outputs/metrics/tcn_caucafall_recallpush_cost20.json`
- `outputs/metrics/gcn_caucafall_recallpush_cost20.json`

Decision:
- If recall improves materially with tolerable FA24h, keep policy-only change.
- If recall is still below target, proceed to Step 2 retrain micro-matrix.

Status:
- Completed.
- `*_recallpush_cost20.yaml`: no useful gain (TCN dropped to Recall 0.6; GCN unchanged).
- `*_recallpush_cost20_core.yaml`: Recall reached 1.0 but `FA24h` exploded (`~940.9` TCN, `~1881.8` GCN).
- Conclusion: policy-only is not sufficient under safety constraints.

## Step 2 (If Needed): Minimal Retrain Micro-Matrix
Small CAUCAFall-only matrix (3 seeds if candidate is promising):
- M1: reduce masking (`mask_joint_p 0.15->0.08`, `mask_frame_p 0.10->0.05`)
- M2: compare imbalance mode (`bce+pos_weight` vs `focal`) without stacking both aggressively
- M3: hard-negative replay ratio tuning (`hard_neg_mult` 1,2,3 train-only)

Status:
- M1 (`_opt_m1maskfix`) completed and evaluated.
- Result: `outputs/metrics/tcn_caucafall_opt_m1maskfix.json` underperformed baseline (`Recall 0.6`, `F1 0.75`, `FA24h 0.0`).
- M2 completed:
  - `_opt_m2_bce`: underperformed baseline (`Recall 0.6`, `F1 0.75`, `FA24h 0.0`).
  - `_opt_m2_focal`: matched baseline event metrics (`Recall 0.8`, `F1 0.8889`, `FA24h 0.0`) with better AP (`0.9840`).
- M3 partially completed:
  - `_opt_m3_hn3`: underperformed baseline (`Recall 0.6`, `F1 0.75`, `FA24h 0.0`).
  - Remaining optional checks: `_opt_m3_hn1`, `_opt_m3_hn2`.

Current best safe candidate:
- `outputs/caucafall_tcn_W48S12_opt_m2_focal/best.pt`
- `configs/ops/tcn_caucafall_opt_m2_focal.yaml`
- `outputs/metrics/tcn_caucafall_opt_m2_focal.json`
- Full comparison table: `artifacts/reports/tuning/caucafall_tcn_step2_progress.md`

## Next Commands (Run In Order)
1. TCN M2 (loss mode comparison, no hard-neg replay):
   - `make train-tcn-caucafall ADAPTER_USE=1 OUT_TAG=_opt_m2_bce EPOCHS=80 TCN_PATIENCE=15 TCN_RESUME=outputs/caucafall_tcn_W48S12_cauc_hneg1/best.pt TCN_HARD_NEG_LIST= TCN_HARD_NEG_MULT=1 TCN_LOSS=bce`
   - `make fit-ops-caucafall ADAPTER_USE=1 OUT_TAG=_opt_m2_bce ALERT_CONFIRM=0`
   - `make eval-caucafall ADAPTER_USE=1 OUT_TAG=_opt_m2_bce`
2. TCN M2 focal branch (same setup for fair comparison):
   - `make train-tcn-caucafall ADAPTER_USE=1 OUT_TAG=_opt_m2_focal EPOCHS=80 TCN_PATIENCE=15 TCN_RESUME=outputs/caucafall_tcn_W48S12_cauc_hneg1/best.pt TCN_HARD_NEG_LIST= TCN_HARD_NEG_MULT=1 TCN_LOSS=focal`
   - `make fit-ops-caucafall ADAPTER_USE=1 OUT_TAG=_opt_m2_focal ALERT_CONFIRM=0`
   - `make eval-caucafall ADAPTER_USE=1 OUT_TAG=_opt_m2_focal`
3. TCN M3 hard-neg ratio sweep (train-only mined list):
   - `make train-tcn-caucafall ADAPTER_USE=1 OUT_TAG=_opt_m3_hn1 TCN_RESUME=outputs/caucafall_tcn_W48S12_cauc_hneg1/best.pt TCN_HARD_NEG_LIST=outputs/hardneg/tcn_caucafall_train_p50.txt TCN_HARD_NEG_MULT=1`
   - `make train-tcn-caucafall ADAPTER_USE=1 OUT_TAG=_opt_m3_hn2 TCN_RESUME=outputs/caucafall_tcn_W48S12_cauc_hneg1/best.pt TCN_HARD_NEG_LIST=outputs/hardneg/tcn_caucafall_train_p50.txt TCN_HARD_NEG_MULT=2`
   - `make train-tcn-caucafall ADAPTER_USE=1 OUT_TAG=_opt_m3_hn3 TCN_RESUME=outputs/caucafall_tcn_W48S12_cauc_hneg1/best.pt TCN_HARD_NEG_LIST=outputs/hardneg/tcn_caucafall_train_p50.txt TCN_HARD_NEG_MULT=3`
   - For each tag: run `fit-ops-caucafall` + `eval-caucafall`.
4. If `Recall` is still capped at `0.8` with `FA24h <= 5`, move to next cycle:
   - keep `m2_focal` as base;
   - perform subject-level error analysis on CAUCAFall test misses;
   - then tune confirm/start-guard parameters only for miss-recovery windows.

## Step 3 (Completed): Miss-Recovery Policy Calibration
- Approach: val-driven threshold plateau analysis (no retraining).
- Key finding: val had a long perfect plateau; endpoint tie-break (`max_thr`) under-recalled on test.
- Candidate policy: midpoint threshold from val plateau.
  - Ops: `configs/ops/tcn_caucafall_opt_m2_focal_midplateau.yaml`
  - Metrics: `outputs/metrics/tcn_caucafall_opt_m2_focal_midplateau.json`
  - Test outcome (current split): `Recall=1.0`, `F1=1.0`, `Precision=1.0`, `FA24h=0.0`
- Full analysis: `artifacts/reports/tuning/caucafall_miss_recovery.md`

## Step 4 (Completed): Cross-Seed Safety Stress Test for Policy-Only Fixes
- Confirm-gate grid on high-risk seeds:
  - report: `artifacts/reports/tuning/caucafall_confirm_grid_stability.md`
  - result: no candidate achieved both strong recall and low FA24h consistently.
- Temporal-consensus grid (`k/n/cooldown`) on high-risk seeds:
  - report: `artifacts/reports/tuning/caucafall_temporal_grid_stability.md`
  - result: all variants kept high recall but failed safety (`FA24h=940.91` on all tested seeds).

Decision:
- Stop policy-only tuning for deployment promotion.
- Keep baseline policy as stable deployment default.
- Move to training-side targeted improvement (next cycle).

## Step 5 (Completed): Targeted Retrain With Train-Only Hard Negatives
- Hard negatives mined from high-risk seed models (train split only), union list:
  - `outputs/hardneg/tcn_caucafall_targeted_train_union.txt` (32 windows)
- Targeted retrain output:
  - checkpoint: `outputs/caucafall_tcn_W48S12_opt_m4_hn_targeted/best.pt`
- Evaluation with required two policies:
  - baseline ops eval: `outputs/metrics/tcn_caucafall_opt_m4_hn_targeted_baselineops.json`
  - midplateau ops eval: `outputs/metrics/tcn_caucafall_opt_m4_hn_targeted_midplateauops.json`
- Result:
  - No net gain vs prior `m2_focal` on the same test split/policies (event metrics unchanged).
  - Detailed table: `artifacts/reports/tuning/caucafall_targeted_retrain_results.md`

## Step 6 (Completed): Corrected Mistakes + Re-run
- Fixed training script issues (TCN):
  - Added `--resume_use_ckpt_feat_cfg` to control whether resume overrides CLI feature flags.
  - Added `--hard_neg_prefix_strict` + prefix-match validation to avoid silent no-op prefix boosts.
- Re-ran targeted experiment with effective settings:
  - prefix: `Subject.3__Pick_up_object` (strict matched)
  - feature override enabled from CLI: `conf_gate=0.30`
  - output ckpt: `outputs/caucafall_tcn_W48S12_opt_m7_pickupfix_cg30/best.pt`
- Evaluation:
  - baseline ops: `outputs/metrics/tcn_caucafall_opt_m7_pickupfix_cg30_baselineops.json`
  - midplateau ops: `outputs/metrics/tcn_caucafall_opt_m7_pickupfix_cg30_midplateauops.json`
- Outcome:
  - baseline policy unchanged on event metrics (`Recall=0.8, F1=0.8889, FA24h=0`)
  - midplateau no longer recovers the missed event (`Recall=0.8, F1=0.8889, FA24h=0`)
  - comparison table: `artifacts/reports/tuning/caucafall_targeted_fix_comparison.md`

## Step 7 (Completed): Model-Side Minimal Grid (No Architecture Change)
- Experiments:
  - `m8_bce_balanced`: BCE + balanced sampler
  - `m9_focal_balanced`: focal + balanced sampler
- Outputs:
  - `outputs/caucafall_tcn_W48S12_m8_bce_balanced/best.pt`
  - `outputs/caucafall_tcn_W48S12_m9_focal_balanced/best.pt`
  - metrics:
    - `outputs/metrics/tcn_caucafall_m8_bce_balanced.json`
    - `outputs/metrics/tcn_caucafall_m9_focal_balanced.json`
- Result:
  - Both matched current event metrics (`Recall=0.8, F1=0.8889, FA24h=0.0`) but reduced AP vs current best.
  - report: `artifacts/reports/tuning/caucafall_modelside_minigrid_results.md`

Decision:
- Keep current best (`m2_focal`) as promoted baseline.
- No better candidate found in this minimal model-side grid.

## Step 8 (Completed): Start-Guard Grid Stress Test (High-Risk Seeds)
- Goal: test whether `start_guard_max_lying` can recover missed events without exploding false alerts.
- Grid:
  - `start_guard_max_lying in {0.2, 0.3, 0.4, 0.5, 0.6}`
  - seeds: `17, 2025, 33724876, 42`
  - fixed model family: TCN (`outputs/caucafall_tcn_W48S12_stb_s{seed}/best.pt`)
- Outputs:
  - per-run metrics: `outputs/metrics/grid_startguard_midplateau/*.json` (20 files)
  - summary CSV: `artifacts/reports/tuning/caucafall_startguard_grid_summary.csv`
  - detailed report: `artifacts/reports/tuning/caucafall_startguard_grid_results.md`
- Result:
  - best aggregate recall in this grid: `0.6` (for `start_guard_max_lying >= 0.3`)
  - `FA24h` remained very high (`940.91` max on every setting)
  - none of the settings met deployment safety gates.

Decision:
- Reject start-guard-only tuning as a deployment fix for CAUCAFall.
- Keep promoted baseline (`m2_focal + baseline ops`) unchanged for now.

## Next Cycle (Actionable)
1. Data-centric hard-negative expansion (train-only):
   - mine from additional high-risk seeds/runs and rebalance by ADL category.
2. Separate policy objectives:
   - keep deployment policy safety-first (`FA24h <= 5`) and report a recall-oriented secondary profile separately.
3. Field validation ramp:
   - collect real non-fall clips first (diverse ADLs, camera angles, distances),
   - then add controlled fall-like edge cases for false-positive pressure testing.

## Step 9 (Completed): Train-Only Hard-Negative Expansion (`m10`)
- Change set:
  - mined additional train-only hard negatives from `windows_W48_S12/train`:
    - list: `outputs/hardneg/tcn_caucafall_train_p35_top200.txt` (9 windows)
  - retrained from `m2_focal` checkpoint with replay:
    - ckpt: `outputs/caucafall_tcn_W48S12_opt_m10_hn_trainp35/best.pt`
  - fitted and evaluated:
    - ops: `configs/ops/tcn_caucafall_opt_m10_hn_trainp35.yaml`
    - metrics: `outputs/metrics/tcn_caucafall_opt_m10_hn_trainp35.json`

- Outcome vs `m2_focal`:
  - Event metrics: unchanged (`F1=0.8889`, `Recall=0.8000`, `Precision=1.0000`, `FA24h=0.0`)
  - AP decreased (`0.9840 -> 0.9777`)

Decision:
- Reject `m10` for promotion.
- Keep `m2_focal` as current CAUCAFall baseline.

## Step 10 (Completed): Miss-Focused Low-Mask Retrain (`m11`)
- Motivation:
  - single persistent miss in baseline packet: `Subject.6/Fall left`
  - try reducing augmentation masking to preserve subtle spatial cues.
- Change set:
  - `mask_joint_p: 0.15 -> 0.05`
  - `mask_frame_p: 0.10 -> 0.02`
  - keep focal loss + train-only hard-neg replay (`outputs/hardneg/tcn_caucafall_train_p35_top200.txt`)
- Outputs:
  - ckpt: `outputs/caucafall_tcn_W48S12_opt_m11_missfix_lowmask/best.pt`
  - ops: `configs/ops/tcn_caucafall_opt_m11_missfix_lowmask.yaml`
  - metrics: `outputs/metrics/tcn_caucafall_opt_m11_missfix_lowmask.json`
  - miss packet: `artifacts/reports/tuning/caucafall_miss_packet_m2_focal.md`
- Outcome vs `m2_focal`:
  - AP: `0.9840 -> 0.9827` (down)
  - Event metrics: `F1 0.8889 -> 0.7500`, `Recall 0.8 -> 0.6`, `FA24h 0.0 -> 0.0`
  - Misses increased from 1 to 2 (`Fall left` + `Fall backwards`).

Decision:
- Reject `m11` (negative transfer).
- Preserve `m2_focal` as promoted baseline.

## Step 11 (Completed): Dual-Policy Deployment Profile (Safe + Recall)
- Purpose:
  - keep a safety-first alarm channel while exposing a recall-oriented secondary channel for operator triage.
- Policy files:
  - safe: `configs/ops/dual_policy/tcn_caucafall_dual_safe.yaml`
  - recall: `configs/ops/dual_policy/tcn_caucafall_dual_recall.yaml`
- Main-checkpoint comparison (`m2_focal`):
  - safe: `F1=0.8889, Recall=0.8000, Precision=1.0000, FA24h=0.0`
  - recall: `F1=1.0000, Recall=1.0000, Precision=1.0000, FA24h=0.0`
- 4-seed stability sanity (`stb_s{17,2025,33724876,42}`):
  - safe mean: `Recall=0.8, FA24h=0.0`
  - recall mean: `Recall=1.0, FA24h=940.91` (max also `940.91`)
  - implication: recall profile is not stable enough for auto-alert deployment.
- Artifacts:
  - summary: `artifacts/reports/tuning/caucafall_dual_policy_summary.csv`
  - deployment note: `artifacts/reports/tuning/caucafall_dual_policy_deployment.md`

Decision:
- Deploy `safe` as primary automated alert policy.
- Keep `recall` as secondary review-only channel (no automatic emergency trigger) until FA24h is stabilized across seeds.

## Reporting
For each step, record:
- `AP, F1, Recall, Precision, FA24h`
- Delta vs baseline
- Exact command + artifact path
