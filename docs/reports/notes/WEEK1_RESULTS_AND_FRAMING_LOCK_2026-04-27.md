Date: 2026-04-27  
Purpose: provide the Week 1 framing lock and synthesis artifacts required by the publishing/elevation task plan.

# 1. Framing Lock

## 1.1 Project-Level Framing Sentence

This project is an end-to-end pose-based fall-detection and monitoring system study, supported by controlled model comparison and deployment-oriented evaluation.

## 1.2 Paper-Level Central Claim

The compact paper should be framed as a deployment-oriented applied AI systems study showing that a pose-based monitoring stack can be made operationally coherent under controlled offline comparison, validation-fitted alert policy, and bounded runtime evidence.

## 1.3 Research-Question-to-Evidence Mapping

| Research question | Evidence layer | Primary current branch artifacts | Boundary |
| --- | --- | --- | --- |
| `RQ1` comparative offline performance | frozen offline model evidence | `outputs/metrics/tcn_caucafall_stb_s*.json`, `outputs/metrics/gcn_caucafall_stb_s*.json`, `outputs/metrics/tcn_le2i_stb_s*.json`, `outputs/metrics/gcn_le2i_stb_s*.json` | do not answer with replay or delivery evidence |
| `RQ2` calibration and operational alerting | operating-point and policy evidence | `configs/ops/tcn_caucafall.yaml`, `configs/ops/gcn_caucafall.yaml`, `outputs/metrics/*locked.json`, `src/fall_detection/evaluation/fit_ops.py` | do not reduce to raw classifier scores alone |
| `RQ3` deployment feasibility and runtime limits | replay/runtime/deployment evidence | `artifacts/ops_delivery_verify_20260315/online_replay_summary.json`, `artifacts/fall_test_eval_20260315/summary_tcn_caucafall_locked_op2.csv`, `artifacts/fall_test_eval_20260315/summary_gcn_caucafall_locked_op2.csv`, `docs/reports/runbooks/FOUR_VIDEO_DELIVERY_PROFILE.md`, `artifacts/evidence/realtime/*.mp4` | bounded system evidence only; not broad field closure |

# 2. Main-Results Synthesis Table

The table below is intentionally compact. It should anchor both the full report and the paper after wording cleanup.

**Table W1. Main defended results synthesis**

| Dataset / setting | Model / profile | Main defended finding | Interpretation | Artifact anchor |
| --- | --- | --- | --- | --- |
| `CAUCAFall`, five-seed frozen stability summary | `TCN`, `OP2` across `s1337, s17, s2025, s33724876, s42` | mean `F1=0.8611`, mean recall `0.7600`, mean `AP=0.9819`, `FA24h=0.0` in all five locked summaries | strongest current offline model line on the primary dataset | `outputs/metrics/tcn_caucafall_stb_s*.json` |
| `CAUCAFall`, five-seed frozen stability summary | `GCN`, `OP2` across `s1337, s17, s2025, s33724876, s42` | mean `F1=0.5873`, mean recall `0.4400`, mean `AP=0.9706`, `FA24h=0.0` in all five locked summaries | supports a cautious directional TCN advantage on the primary dataset | `outputs/metrics/gcn_caucafall_stb_s*.json` |
| `LE2i`, five-seed frozen stability summary | `TCN`, `OP2` | `F1=0.8235`, recall `0.7778`, `FA24h=581.5843`, mean `AP=0.8389` | stronger in-domain `LE2i` result than matched GCN | `outputs/metrics/tcn_le2i_stb_s*.json` |
| `LE2i`, five-seed frozen stability summary | `GCN`, `OP2` | `F1=0.7500`, recall `0.6667`, `FA24h=581.5843`, mean `AP=0.7471` | confirms the TCN remains directionally stronger in the secondary in-domain comparison | `outputs/metrics/gcn_le2i_stb_s*.json` |
| Cross-dataset, `CAUCAFall -> LE2i` | `TCN`, frozen `OP2` | `F1=0.0`, recall `0.0`, `FA24h=1163.1686` | severe transfer collapse when the primary-dataset TCN profile is moved onto `LE2i` | `outputs/metrics/cross_tcn_caucafall_r2_train_hneg_to_le2i_frozen_20260409.json` |
| Cross-dataset, `CAUCAFall -> LE2i` | `GCN`, frozen `OP2` | `F1=0.7778`, recall `0.7778`, `FA24h=1163.1686` | transfer degradation is not uniform across models; cross-dataset interpretation must stay bounded and specific | `outputs/metrics/cross_gcn_caucafall_r2_recallpush_b_to_le2i_frozen_20260409.json` |
| Cross-dataset, `LE2i -> CAUCAFall` | `TCN`, frozen `OP2` | `F1=1.0`, recall `1.0`, `FA24h=0.0` | reverse transfer is materially less damaging than the `CAUCAFall -> LE2i` direction | `outputs/metrics/cross_tcn_le2i_opt33_r2_to_caucafall_frozen_20260409.json` |
| Cross-dataset, `LE2i -> CAUCAFall` | `GCN`, frozen `OP2` | `F1=1.0`, recall `1.0`, `FA24h=0.0` | further supports an asymmetric transfer boundary rather than a single generalisation story | `outputs/metrics/cross_gcn_le2i_opt33_r2_to_caucafall_frozen_20260409.json` |

# 3. Deployment-Evidence Synthesis Table

This table deliberately separates bounded runtime evidence from formal offline model evidence.

**Table W2. Deployment and runtime evidence synthesis**

| Scenario | Evidence type | Defended finding | Main limitation | Artifact anchor |
| --- | --- | --- | --- | --- |
| Canonical replay summary on `ops_delivery_verify_20260315` | replay/runtime summary | `caucafall_tcn OP-1` achieved `accuracy=1.0`, `recall=1.0`, `specificity=1.0` on the bounded `10`-video check | this is bounded replay evidence, not formal unseen-test evidence | `artifacts/ops_delivery_verify_20260315/online_replay_summary.json` |
| Same canonical replay summary | replay/runtime summary | `caucafall_tcn OP-2` fell to `accuracy=0.5`, `recall=0.0`, `specificity=1.0` on that same bounded replay check | shows replay/runtime behaviour is highly profile-sensitive and cannot be narrated as uniformly strong | `artifacts/ops_delivery_verify_20260315/online_replay_summary.json` |
| Same canonical replay summary | replay/runtime summary | `caucafall_gcn` remained weak across `OP-1` to `OP-3` with `accuracy=0.5`, `recall=0.0`, `specificity=1.0` | runtime replay evidence does not rescue the weaker model line | `artifacts/ops_delivery_verify_20260315/online_replay_summary.json` |
| Bounded 24-clip custom replay matrix | per-video runtime trace | `tcn_caucafall_locked_op2` produced `13/24` correct video-level outcomes with `TP=4`, `TN=9`, `FP=3`, `FN=8` | explicitly invalidates any stale `24/24` narrative on the current aligned path | `artifacts/fall_test_eval_20260315/summary_tcn_caucafall_locked_op2.csv` |
| Bounded 24-clip custom replay matrix | per-video runtime trace | `gcn_caucafall_locked_op2` produced `8/24` correct video-level outcomes with `TP=8`, `TN=0`, `FP=12`, `FN=4` | shows that higher fall capture can coexist with unusable non-fall control on the same bounded replay surface | `artifacts/fall_test_eval_20260315/summary_gcn_caucafall_locked_op2.csv` |
| Four-folder custom replay runbook | narrative boundary control | the current canonical four-folder profile should be described as a bounded custom replay check, not as a perfect delivery profile, and the aligned metrics there are `TP=3`, `TN=10`, `FP=2`, `FN=9` | this runbook now explicitly rejects the old `24/24` interpretation | `docs/reports/runbooks/FOUR_VIDEO_DELIVERY_PROFILE.md` |
| Realtime evidence pack | live demonstration evidence | the repository still contains live-run evidence showing monitor execution and event-history style behaviour under a bounded live path | this is demonstration evidence only and should not be promoted into broad field validation | `artifacts/evidence/realtime/realtime_fall_submission.mp4`, `artifacts/evidence/realtime/realtime_adl_submission.mp4` |

# 4. Immediate Implications for Writing

1. The strongest report story remains system-first, not model-paper-first.
2. The strongest current offline comparison story is a cautious TCN advantage under frozen in-domain stability evidence.
3. The strongest current deployment story is bounded coherence plus explicit runtime fragility, not uniformly strong replay success.
4. Any wording that still implies `24/24`, broad field closure, or uniformly strong `OP2` replay behaviour should be removed or downgraded.
