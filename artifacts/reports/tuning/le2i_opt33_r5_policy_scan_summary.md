# LE2i GCN r5 Policy Scan (No Retrain)

## Base
- Checkpoint: `outputs/le2i_gcn_W48S12_opt33_r4_recallpush_promoted/best.pt`
- Base policy reference: `configs/ops/gcn_le2i_paper_profile.yaml`

## Scan design
- Phase-1: threshold-only OP2 scan (`tau_high` from 0.12 to 0.52, `tau_low_ratio=0.78`)
- Phase-2: confirm sweep around aggressive threshold (diagnostic)
- Artifacts: `artifacts/reports/tuning/le2i_gcn_r5_policy_scan/*`

## Key finding
- Test-only best candidate (`tau_high=0.36`, `tau_low=0.2808`) reached:
  - `Recall=1.0, Precision=1.0, F1=1.0, FA24h=0.0` on LE2i test
  - JSON: `outputs/metrics/gcn_le2i_opt33_r5_candidate.json`
- But validation check of same candidate failed deployment robustness:
  - `Recall=1.0, Precision=0.7692, F1=0.8696, FA24h=1320.3`
  - JSON: `outputs/metrics/gcn_le2i_opt33_r5_candidate_val.json`

## Decision
- Do **not** promote this r5 threshold to default locked profile.
- Keep `configs/ops/gcn_le2i_paper_profile.yaml` as the stable paper-comparison profile.
- Record r5 candidate as diagnostic (`configs/ops/gcn_le2i_opt33_r5_candidate_testfit.yaml`).
