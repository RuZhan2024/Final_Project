# Claim Table

Date: 2026-03-22

Scope:
Current 33-joint profiles and artefacts only.

Protocol Freeze:
- `Paper Protocol Freeze v1`
- Candidate roots are locked by [FINAL_CANDIDATES.md](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/docs/project_targets/FINAL_CANDIDATES.md)
- Dataset roles are locked as:
  - `CAUCAFall`: primary benchmark and deployment target dataset
  - `LE2i`: comparative/generalization dataset
- Frozen seed set for current architecture-comparison evidence:
  - `1337`, `17`, `2025`, `33724876`, `42`

Rule:
No claim below should be read more broadly than its protocol and artefact coverage allow.

Paper-Ready Claim Set:
- Use `C1-C4` as the only paper-facing high-level claims under the current freeze.
- In the main manuscript, prefer short bounded wording over the full table phrasing.

## Claims

| Claim | Metric(s) | Threshold / Success Rule | Dataset / Scope | Protocol / Artifact | Reproduce Command | Failure Condition | Current Status |
|---|---|---|---|---|---|---|---|
| C1. The project delivers a working end-to-end replayable fall-detection system under a locked demo profile. | API health reachable, predict endpoint available, active model/dataset/op match lock, manual replay pass | `PASS` deployment lock validation with `TCN + caucafall + OP-2` and successful manual replay checks | Locked demo profile only | [DEPLOYMENT_LOCK.md](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/docs/project_targets/DEPLOYMENT_LOCK.md), [deployment_lock_validation.md](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/artifacts/reports/deployment_lock_validation.md) | `bash tools/run_deployment_lock_validation.sh` | Lock validation fails, predict route unavailable, or replay checks do not pass | Pass |
| C2. Under the frozen primary-dataset protocol, the final CAUCAFall TCN candidate trends stronger than the matched CAUCAFall GCN candidate while both retain zero false alerts in the current 5-seed artefacts. | Mean `F1`, `Recall`, `AP`, `FA24h` across the frozen seed set | Supported if TCN mean exceeds GCN mean on `F1` and `Recall`, and both remain at `FA24h = 0.0` in current stability artefacts; wording must remain cautious because the primary Wilcoxon result is still exploratory at `n=5` | CAUCAFall final-candidate comparison only under `Paper Protocol Freeze v1` | [FINAL_CANDIDATES.md](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/docs/project_targets/FINAL_CANDIDATES.md), [STABILITY_REPORT.md](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/docs/project_targets/STABILITY_REPORT.md), [SIGNIFICANCE_REPORT.md](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/docs/project_targets/SIGNIFICANCE_REPORT.md), [stability_summary.json](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/artifacts/reports/stability_summary.json), [significance_summary.json](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/artifacts/reports/significance_summary.json) | `python tools/run_stability_manifest.py --manifest artifacts/registry/stability_manifest.csv --start_status todo --stop_on_fail 1` | Updated stability artefacts no longer show TCN mean advantage, false-alert parity breaks, or the paper text upgrades the result to definitive superiority | Pass, with statistical caution |
| C3. Cross-dataset transfer is asymmetric and does not support a claim of robust universal generalization. | `delta_f1`, `delta_recall`, `delta_fa24h`, `delta_ap` | Supported if at least one transfer direction shows material degradation against in-domain performance | LE2i <-> CAUCAFall transfer only | [CROSS_DATASET_REPORT.md](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/docs/project_targets/CROSS_DATASET_REPORT.md), [cross_dataset_summary.csv](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/artifacts/reports/cross_dataset_summary.csv) | `python scripts/plot_cross_dataset_transfer.py --manifest artifacts/reports/cross_dataset_manifest.json --out_fig artifacts/figures/cross_dataset/cross_dataset_transfer_bars.png` | Both transfer directions become consistently close to in-domain performance without large drops | Pass |
| C4. The four-folder custom replay set should be evaluated under the same canonical online profile as the main monitor, and its result should be treated only as bounded supporting evidence. | `TP`, `TN`, `FP`, `FN` on the four-folder custom set under the canonical online profile | Supported only as a consistency check; current unified-profile result is not a strong success claim | Supporting custom replay evidence only, not a primary comparative or deployment-performance claim | [FOUR_VIDEO_DELIVERY_PROFILE.md](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/docs/reports/runbooks/FOUR_VIDEO_DELIVERY_PROFILE.md), [unified_tcn_caucafall_op2_metrics.json](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/artifacts/fall_test_eval_20260330/unified_tcn_caucafall_op2_metrics.json) | `python3 scripts/eval_delivery_videos.py --config_yaml configs/delivery/tcn_caucafall_r2_train_hneg_four_video.yaml` | Unified with main online profile; do not claim `24/24` | Supporting only |

## Non-Claims

These are intentionally not claimed from the current artefacts:

- strong real-world field-validation completion
- universal cross-dataset robustness
- formal statistical significance at `alpha=0.05` under the primary Wilcoxon test for TCN-vs-GCN
- fully closed publication-grade evidence outside the current repository artefacts
- low-alarm deployment readiness on `LE2i`

## Paper-Ready Short Form

Use this compressed wording in the paper when you need a short claim set:

1. The project delivers a working end-to-end pose-based fall-detection system under a locked replay/deployment profile.
2. Under the frozen primary-dataset protocol, TCN trends stronger than GCN on `CAUCAFall`, but this remains a directional rather than definitive superiority claim.
3. Cross-dataset transfer is asymmetric and should be treated as a limitation, not as evidence of universal robustness.
4. Replay and delivery-package validation support bounded deployment-oriented usefulness, while broader field validation remains incomplete.

## Notes for Dissertation / Viva

- Use C1 and C4 for the working-software and demo story.
- Use C2 for the main benchmark-comparison story, but keep the wording cautious because the significance report is exploratory at `n=5`.
- Use C3 to show honest evaluation of generalization limits rather than over-claiming.
- Treat `CAUCAFall` as the main result-bearing dataset and `LE2i` as comparative evidence throughout the paper.
- When space is tight, use the `Paper-Ready Short Form` section instead of paraphrasing claims ad hoc.
