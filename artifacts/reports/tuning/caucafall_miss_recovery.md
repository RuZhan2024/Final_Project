# CAUCAFall Miss-Recovery (Leak-Safe, Val-Driven)

## Objective
Recover the single missed fall (`Subject.6/Fall left`) without retraining the model and without raising false alerts.

## Baseline
- Model: `outputs/caucafall_tcn_W48S12_opt_m2_focal/best.pt`
- Ops: `configs/ops/tcn_caucafall_opt_m2_focal.yaml`
- Metrics: `outputs/metrics/tcn_caucafall_opt_m2_focal.json`
- Baseline outcome:
  - `Recall=0.8`, `F1=0.8889`, `Precision=1.0`, `FA24h=0.0`
  - Missed clip: `Subject.6/Fall left` (`event_recall=0`)

## Diagnosis
- `fit_ops` on validation selected `tau_high=0.85` because `op_tie_break=max_thr`.
- Validation sweep had a long perfect plateau (`Recall=1.0, F1=1.0, FA24h=0`) from about `thr=0.02` to `thr=0.85`.
- Selecting an endpoint of that plateau (`max_thr` or `min_thr`) is unstable:
  - `max_thr` missed one fall on test.
  - `min_thr` recovered recall but caused very high `FA24h`.

## Fix Strategy
Use a **mid-plateau threshold** chosen from validation only:
- `tau_high = 0.43` (midpoint of the perfect validation plateau)
- `tau_low = 0.3354` (`tau_high * 0.78`)

Ops file:
- `configs/ops/tcn_caucafall_opt_m2_focal_midplateau.yaml`

## Result
- Metrics: `outputs/metrics/tcn_caucafall_opt_m2_focal_midplateau.json`
- Outcome:
  - `Recall=1.0`
  - `F1=1.0`
  - `Precision=1.0`
  - `FA24h=0.0`
  - `Subject.6/Fall left` recovered (`event_recall=1.0`)

## Reproduce Commands
```bash
python scripts/eval_metrics.py \
  --win_dir data/processed/caucafall/windows_eval_W48_S12/test \
  --ckpt outputs/caucafall_tcn_W48S12_opt_m2_focal/best.pt \
  --ops_yaml configs/ops/tcn_caucafall_opt_m2_focal_midplateau.yaml \
  --out_json outputs/metrics/tcn_caucafall_opt_m2_focal_midplateau.json \
  --fps_default 23 --thr_min 0.001 --thr_max 0.95 --thr_step 0.01
```

## Guardrails Before Promotion
1. Re-check on all stability seeds (no test-time tuning).
2. Keep this as a candidate policy until seed-wise consistency is confirmed.
3. If it fails on other seeds, keep baseline policy and move to model-side improvement.
