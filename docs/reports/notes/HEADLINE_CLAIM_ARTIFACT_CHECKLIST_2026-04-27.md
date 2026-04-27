Date: 2026-04-27  
Purpose: Week 1 headline claim-to-artifact checklist for the full report and compact paper.

| Claim text | Evidence layer | Current branch artifact anchor | Safe wording status | Action needed |
| --- | --- | --- | --- | --- |
| The project delivers an end-to-end pose-based fall-detection and monitoring system. | system artifact | `applications/frontend`, `applications/backend`, `artifacts/evidence/realtime/*.mp4`, `docs/reports/runbooks/USER_GUIDE.md` | safe if bounded as a working monitored system | keep system-first wording |
| The TCN is stronger than the matched GCN on the primary `CAUCAFall` protocol. | frozen offline comparison | `outputs/metrics/tcn_caucafall_stb_s*.json`, `outputs/metrics/gcn_caucafall_stb_s*.json` | safe only as cautious directional advantage | avoid “significantly outperforms” unless significance artifacts are restored and checked |
| The same directional preference appears on the in-domain `LE2i` comparison. | frozen offline comparison | `outputs/metrics/tcn_le2i_stb_s*.json`, `outputs/metrics/gcn_le2i_stb_s*.json` | safe | keep concise |
| Cross-dataset transfer is asymmetric. | cross-dataset limitation evidence | `outputs/metrics/cross_*_frozen_20260409.json` | safe | phrase as limitation boundary, not positive robustness claim |
| Validation-side operating-point fitting materially shapes deployable alert behaviour. | method / policy evidence | `configs/ops/tcn_caucafall.yaml`, `configs/ops/gcn_caucafall.yaml`, `outputs/metrics/*locked.json`, `src/fall_detection/evaluation/fit_ops.py` | safe | preserve as method/system claim |
| The preferred live demo preset is `CAUCAFall + TCN + OP-2`. | configuration / deployment path | `configs/ops/tcn_caucafall.yaml`, `README.md` | safe | keep as configuration fact, not result claim |
| Deployment evidence is strongest for bounded replay and delivery-style validation. | bounded runtime evidence | `docs/reports/runbooks/FOUR_VIDEO_DELIVERY_PROFILE.md`, `artifacts/ops_delivery_verify_20260315/online_replay_summary.json`, `artifacts/evidence/realtime/*.mp4` | safe if explicitly bounded | keep “bounded” in same sentence |
| `CAUCAFall + TCN + OP-2` is the strongest defended replay row. | bounded runtime evidence | not safely supported on the current branch by one clean artifact | unsafe in current form | downgrade or replace with profile-specific wording until one stable replay summary is selected |
| The current aligned four-folder replay path is `24/24`. | bounded runtime evidence | contradicted by `docs/reports/runbooks/FOUR_VIDEO_DELIVERY_PROFILE.md` and `artifacts/fall_test_eval_20260315/summary_tcn_caucafall_locked_op2.csv` | unsafe | remove everywhere |
| Realtime evidence proves broad field readiness. | live demonstration evidence | no current branch artifact supports this | unsafe | do not claim |
| Telegram / caregiver-facing delivery exists on the implemented path. | system path evidence | backend notification code, README, realtime evidence pack | safe as path-existence claim | do not use to strengthen model-quality claims |
| Statistical significance definitively proves TCN superiority. | inferential evidence | significance summary artifact not currently present on this branch | unsafe in current branch state | keep only directional wording unless significance artifacts are restored |
