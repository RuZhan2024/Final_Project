# MUVIM Metric Contract Fix (Event Recall/F1)

## Issue
- `outputs/metrics/*_muvim_quick.json` had impossible event metrics (`recall > 1`, `f1 > 1`).
- Root cause in `src/fall_detection/evaluation/metrics_eval.py`:
  - event-level aggregate recall was computed as:
    - `n_true_alerts / n_gt_events`
  - This is incorrect when multiple alert events overlap one GT event.

## Fix
- Updated `_aggregate_event_counts` to:
  - track `n_matched_gt` from per-video event metrics.
  - compute recall as:
    - `n_matched_gt / n_gt_events`
  - keep precision as:
    - `n_true_alerts / (n_true_alerts + n_false_alerts)`
  - compute F1 from corrected precision+recall.

## Regression test
- Added: `tests/test_metrics_eval_recall_contract.py`
- Test strategy:
  - monkeypatch per-video event metrics with:
    - `n_gt_events=1, n_matched_gt=1, n_true_alerts=3`
  - assert aggregate recall is `1.0` (not `3.0`).

## Validation commands run
```bash
python -m py_compile src/fall_detection/evaluation/metrics_eval.py tests/test_metrics_eval_recall_contract.py
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH="$(pwd)/src:$(pwd)" pytest -q tests/test_metrics_eval_recall_contract.py
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH="$(pwd)/src:$(pwd)" python3 scripts/eval_metrics.py --win_dir data/processed/muvim/windows_eval_W48_S12/test --ckpt outputs/muvim_tcn_W48S12_quick/best.pt --ops_yaml configs/ops/tcn_muvim_quick.yaml --out_json outputs/metrics/tcn_muvim_quick_fixed.json --fps_default 30
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH="$(pwd)/src:$(pwd)" python3 scripts/eval_metrics.py --win_dir data/processed/muvim/windows_eval_W48_S12/test --ckpt outputs/muvim_gcn_W48S12_quick/best.pt --ops_yaml configs/ops/gcn_muvim_quick.yaml --out_json outputs/metrics/gcn_muvim_quick_fixed.json --fps_default 30
```

## Post-fix sanity
- New files:
  - `outputs/metrics/tcn_muvim_quick_fixed.json`
  - `outputs/metrics/gcn_muvim_quick_fixed.json`
- In both files:
  - selected OP recall/F1 are now bounded (`<= 1.0`).
