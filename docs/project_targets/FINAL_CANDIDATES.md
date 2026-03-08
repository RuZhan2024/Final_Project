# Final Candidates

Dataset role policy:
- Primary deployment target: `CAUCAFall`
- Comparative benchmark: `LE2i`

Decision lock (2026-03-03):
- Dataset role is frozen for the final cycle.
- Do not switch to dual-primary or LE2i-primary acceptance unless this file and `PROJECT_FINAL_YEAR_EXECUTION_PLAN.md` are explicitly revised together with rationale and new gates.

Canonical id format:
`EXP={arch}_{dataset}_{W}W{S}S_seed{seed}_feat{...}_confirm{...}_op{...}`

| Candidate ID | EXP | Arch | Dataset | W | S | Seed | Feature Set | Confirm Policy | OP Policy | Selection Reason | Metrics Snapshot (AP/F1/Recall/FA24h) | Artifact Root | Reproduce Command | Config Hash | Status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| FC1 | `tcn_caucafall_48W12S_seed33724876_featM1C1B1BL1_confirm0_op2_tuned` | TCN | CAUCAFall | 48 | 12 | 33724876 | motion=1, conf=1, bone=1, bone_len=1, center=pelvis | `confirm=0` (promoted) | `OP2` from `configs/ops/tcn_caucafall_cauc_hneg1_confirm0.yaml` | Promoted primary deployment profile on CAUCAFall (best tie with zero false alerts and high recall). | `AP=0.9819 / F1=0.8889 / R=0.8000 / FA24h=0.0` | `outputs/caucafall_tcn_W48S12_cauc_hneg1` | `python scripts/eval_metrics.py --win_dir data/processed/caucafall/windows_eval_W48_S12/test --ckpt outputs/caucafall_tcn_W48S12_cauc_hneg1/best.pt --ops_yaml configs/ops/tcn_caucafall_cauc_hneg1_confirm0.yaml --out_json outputs/metrics/tcn_caucafall_cauc_hneg1_confirm0.json` | `03d007911ab5` | Promoted |
| FC2 | `gcn_caucafall_48W12S_seed33724876_featM1C1B1BL1TS1_confirm0_op2_guard40` | GCN | CAUCAFall | 48 | 12 | 33724876 | motion=1, conf=1, bone=1, bonelen=1, two_stream=1, fuse=concat | `confirm=0` (guardrail-calibrated) | `OP2` from `configs/ops/gcn_caucafall.yaml` | Promoted comparative CAUCAFall profile for architecture comparison; deploy-safe OP2 with threshold guardrail. | `AP=0.9640 / F1=0.8889 / R=0.8000 / FA24h=0.0` | `outputs/caucafall_gcn_W48S12` | `python scripts/eval_metrics.py --win_dir data/processed/caucafall/windows_eval_W48_S12/test --ckpt outputs/caucafall_gcn_W48S12/best.pt --ops_yaml configs/ops/gcn_caucafall.yaml --out_json outputs/metrics/gcn_caucafall.json` | `14edb07` | Promoted (Comparative) |
| FC3 | `tcn_le2i_48W12S_seed33724876_featM1C1B1BL1_confirm0_op2` | TCN | LE2i | 48 | 12 | 33724876 | motion=1, conf=1, bone=1, bone_len=1, center=pelvis | `confirm=0` | `OP2` from `configs/ops/tcn_le2i.yaml` | Strongest LE2i event metrics among current official outputs. | `AP=0.8541 / F1=0.8235 / R=0.7778 / FA24h=581.58` | `outputs/le2i_tcn_W48S12` | `python scripts/eval_metrics.py --win_dir data/processed/le2i/windows_eval_W48_S12/test --ckpt outputs/le2i_tcn_W48S12/best.pt --ops_yaml configs/ops/tcn_le2i.yaml --out_json outputs/metrics/tcn_le2i.json` | `2a1ff7fa3218` | Frozen |
| FC4 | `gcn_le2i_48W12S_seed33724876_featM1C1B1BL1TS1_confirm0_op2` | GCN | LE2i | 48 | 12 | 33724876 | motion=1, conf=1, bone=1, bonelen=1, two_stream=1, fuse=concat | `confirm=0` | `OP2` from `configs/ops/gcn_le2i.yaml` | Competitive baseline; needed for architecture comparison and significance tests. | `AP=0.7523 / F1=0.7500 / R=0.6667 / FA24h=581.58` | `outputs/le2i_gcn_W48S12` | `python scripts/eval_metrics.py --win_dir data/processed/le2i/windows_eval_W48_S12/test --ckpt outputs/le2i_gcn_W48S12/best.pt --ops_yaml configs/ops/gcn_le2i.yaml --out_json outputs/metrics/gcn_le2i.json` | `74738637031b` | Frozen |
| FC5 | `gcn_le2i_48W12S_seed33724876_featM1C1B1BL1TS1_confirm0_op2_startguard_coffee2_r8` | GCN | LE2i | 48 | 12 | 33724876 | motion=1, conf=1, bone=1, bonelen=1, two_stream=1, fuse=concat | `confirm=0` + `start_guard_max_lying=0.85` + `start_guard_prefixes=[Coffee_room_02]` | `OP2` from `configs/ops/gcn_le2i_paper_profile.yaml` | Locked LE2i GCN deployment profile: preserves zero false alerts while keeping high recall on current split. | `AP=0.8451 / F1=0.9412 / R=0.8889 / FA24h=0.0` | `outputs/le2i_gcn_W48S12_opt33_r8_dataside_noise` | `make repro-deploy-gcn-le2i ADAPTER_USE=1` | `n/a (ops-profile variant)` | Deployment-Locked (LE2i GCN) |

## Selection Rules
1. Selection is based on validation-fitted OP policy; no test tuning allowed.
2. Keep both architectures represented per dataset for fair comparison.
3. Deployment-oriented ranking prioritizes `Recall` + `FA24h` over AP-only ranking.
4. Final deployment acceptance gates apply to primary dataset (`CAUCAFall`) first; LE2i is used for comparative/generalization evidence.
5. `FC5` is the current LE2i GCN deployment lock. Keep `configs/ops/gcn_le2i_paper_profile.yaml` as deploy ops for LE2i GCN unless a new profile beats it on both recall/F1 and FA24h.
