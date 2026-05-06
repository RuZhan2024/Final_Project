# Report Core Experiment Inventory - 2026-05-06

## Decision

Yes: the report should include a complete TCN vs CTR-GCN matrix for CaucaFall-only, LE2i-only, and CaucaFall+LE2i. The depth should not be identical across all rows. CaucaFall/custom replay is the target deployment evidence; LE2i-only is source-domain completeness; LE2i pretrain and mix25 explain how source data was tested for transfer.

The locked primary result remains TCN + CaucaFall/LE2i controlled mix25 balanced, seed 42, OP2, delivery motion gate 0.18: 23/24 on the 24-video replay, TP=12, TN=11, FP=1, FN=0.

## Core Matrix

| Protocol | TCN | CTR-GCN | Report role |
|---|---:|---:|---|
| CaucaFall-only W48/S12 c2_bbox | AP 0.993, OP2 F1 1.000, recall 1.000; replay mean 16.6/24, max 20/24 | AP 0.996, OP2 F1 0.870, recall 0.800; replay mean 18.2/24, max 21/24 | Target-dataset baseline before mixed-data experiments. |
| LE2i-only W48/S12 c2 | AP 0.841, OP2 F1 0.947, recall 1.000 | AP 0.852, OP2 F1 0.947, recall 1.000 | Source-domain completeness row; not deployment replay evidence. |
| CaucaFall+LE2i controlled mix25 balanced | AP 0.998, OP2 F1 1.000, recall 1.000; replay mean 17.8/24, max 20/24, selected 20/24 | AP 0.994, OP2 F1 0.978, recall 0.960; replay mean 18.0/24, max 19/24, selected 18/24 | Main mixed-data protocol comparison. |
| Locked delivery profile | replay mean 23.0/24, max 23/24, selected 23/24 (TP=12 TN=11 FP=1 FN=0) | Not locked | Final deployed runtime result. |

## Transfer Bridge

| Protocol | TCN | CTR-GCN | Interpretation |
|---|---:|---:|---|
| LE2i supervised pretrain then CaucaFall fine-tune | AP 0.997, OP2 F1 1.000, recall 1.000; replay mean 12.6/24, max 17/24 | AP 0.995, OP2 F1 0.982, recall 1.000; replay mean 16.6/24, max 18/24 | Negative/weak transfer result; useful for defending why naive LE2i pretraining was not selected. |

## What Is Ready

- Ready for main report: CaucaFall-only TCN/CTR-GCN five-seed offline metrics and 24-clip replay metrics.
- Ready for source-domain completeness: LE2i-only TCN/CTR-GCN five-seed OP-fitted test metrics generated from current c2 LE2i windows.
- Ready for transfer narrative: LE2i pretrain then CaucaFall fine-tune, plus controlled mix25 balanced.
- Ready for deployment narrative: locked TCN mix25 seed42 OP2 motion-gated replay result, 23/24.

## Caveats

- LE2i-only test duration is short, so FA/24h is inflated by a single false alert. In the report, pair FA/24h with precision/FP counts rather than interpreting FA/24h alone.
- Custom replay is system/deployment evidence, not the same thing as offline test split evidence.
- Existing older cross-dataset metrics can be treated as appendix material, but they are not required for the core 3x2 matrix unless the report explicitly claims cross-domain generalisation.
- CTR-GCN is stronger than the earlier TCN CaucaFall-only custom replay baseline in several replay comparisons, but the final locked deployment champion is TCN mix25 with a delivery gate. The report should state that honestly.

## New Outputs Created

- `outputs/metrics/tcn_le2i_W48S12_c2_pretrain_policy_f1_minthr_k2n3_s*_test.json`
- `outputs/metrics/ctr_gcn_le2i_W48S12_c2_bone_twostream_sum_pretrain_policy_f1_minthr_k2n3_s*_test.json`
- `ops/configs/ops/tcn_le2i_W48S12_c2_pretrain_policy_f1_minthr_k2n3_s*.yaml`
- `ops/configs/ops/ctr_gcn_le2i_W48S12_c2_bone_twostream_sum_pretrain_policy_f1_minthr_k2n3_s*.yaml`
- `outputs/metrics/report_core_experiment_inventory_2026-05-06.csv`

## Source Artifacts

- `outputs/metrics/caucafall_W48S12_c2_bbox_fiveseed_summary_2026-05-06.json`
- `outputs/metrics/custom_replay_c2_bbox_f1_minthr_k2n3_fiveseed_summary_2026-05-06.json`
- `outputs/metrics/phase16_le2i_pretrain_finetune_fiveseed_summary_2026-05-06.json`
- `outputs/metrics/phase17_controlled_mix25_balanced_fiveseed_summary_2026-05-06.json`
- `outputs/metrics/phase17_controlled_mix25_balanced_validation_clean_candidates_2026-05-06.json`
- `outputs/metrics/phase18_tcn_mix25_s42_delivery_gate_sweep_2026-05-06.csv`
- `artifacts/fall_test_eval_20260506_phase18_tcn_mix25_s42_motion_gate/tcn_mix25_s42_op2_motion018_metrics.json`
