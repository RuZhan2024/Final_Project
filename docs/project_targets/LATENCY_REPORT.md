# Latency Report

## Target Profiles
| Profile ID | Hardware | Mode | Batch | Target Gate |
|---|---|---|---|---|
| CPU-local | Laptop CPU | Real-time | 1 | `latency_p95 <= 200ms` |
| GPU-server | Cloud GPU | Real-time | 1 | `latency_p95 <= 120ms` |
| Offline-batch | CPU/GPU | Offline | 32 | throughput-only reference |

## Required Measurements
- Preprocessing latency
- Model forward latency
- End-to-end latency (I/O included)
- `p50`, `p95`, and worst-case tail

## Existing Artifacts
- `artifacts/reports/infer_profile_*.json`
- `artifacts/figures/latency/latency_profile_summary.png`

## Validation Commands
```bash
python scripts/profile_infer.py --win_dir data/processed/caucafall/windows_eval_W48_S12/val --ckpt outputs/caucafall_tcn_W48S12_cauc_hneg1/best.pt --profile cpu_local --out_json artifacts/reports/infer_profile_cpu_local_tcn_fc1.json
python scripts/profile_infer.py --win_dir data/processed/caucafall/windows_eval_W48_S12/val --ckpt outputs/caucafall_gcn_W48S12_cauc_hneg1/best.pt --profile cpu_local --out_json artifacts/reports/infer_profile_cpu_local_gcn_fc2.json
python scripts/plot_latency_profiles.py --reports artifacts/reports/infer_profile_*.json --out_fig artifacts/figures/latency/latency_profile_summary.png
```

## Status
- Plot generation: Done.
- Profile coverage completeness (all required profiles): Pending.
