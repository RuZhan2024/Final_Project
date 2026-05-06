# CTR-GCN Sweep and Stability Protocol

Date: 2026-05-05
Branch: `feat/ctr-gcn-upgrade`
Machine context: Windows NVIDIA laptop, RTX 5060 Laptop GPU, `.venv-win`

## 1. Purpose

This note locks the next CTR-GCN experiment sequence before more training runs
are launched.

The goal is to prevent the CTR-GCN upgrade from becoming an ad hoc best-run
search. The CTR-GCN line should be compared against the existing TCN/GCN
evidence using the same frozen protocol discipline already used in the report.

## 2. Existing Protocol Anchor

The defended project comparison already uses a frozen five-seed protocol.

Frozen seed set:

- `1337`
- `17`
- `2025`
- `33724876`
- `42`

Existing artifact anchors:

- `outputs/metrics/tcn_caucafall_stb_s*.json`
- `outputs/metrics/gcn_caucafall_stb_s*.json`
- `outputs/metrics/tcn_le2i_stb_s*.json`
- `outputs/metrics/gcn_le2i_stb_s*.json`

Therefore CTR-GCN must use the same seed set for any formal stability claim.

## 3. Current CTR-GCN Baseline State

Completed single-seed baseline:

- seed: `33724876`
- checkpoint: `outputs/ctr_gcn/caucafall_eval_W48_S12_baseline/best.pt`
- ops YAML: `ops/configs/ops/ctr_gcn_caucafall_eval_W48_S12_baseline.yaml`
- test metrics: `outputs/metrics/ctr_gcn_caucafall_eval_W48_S12_baseline.json`

Observed test summary:

- AP: `0.9674`
- AUC: `0.9775`
- OP1: recall `1.0000`, precision `1.0000`, F1 `1.0000`, FA/24h `0.0`, mean delay `1.46s`
- OP2/OP3: recall `0.8000`, precision `1.0000`, F1 `0.8889`, FA/24h `0.0`, mean delay `3.52s`

Important caveat:

- best checkpoint came from epoch `1`
- validation-side confirm gating caused a degenerate `fit_ops` sweep and was
  automatically disabled by the existing fallback path

This is a valid baseline, but not enough evidence for a final CTR-GCN claim.

## 4. Experiment Phases

### Phase A. Small CTR-GCN Hyperparameter Sweep

Run a bounded validation sweep on seed `33724876` only.

Allowed sweep dimensions:

| Knob | Values |
| --- | --- |
| learning rate | `1e-3`, `5e-4`, `3e-4` |
| dropout | `0.30`, `0.40` |
| ctr rank | `4`, `8` |

Fixed dimensions:

- `channel_schedule=64,64,64,128`
- `rel_channels=8`
- `temporal_kernel=9`
- `use_motion=1`
- `use_conf_channel=1`
- `use_bone=0`
- `use_bone_length=0`
- `motion_scale_by_fps=1`
- `conf_gate=0.2`
- `use_precomputed_mask=1`
- train/val/test split unchanged
- W/S unchanged: `W=48`, `S=12`

Total candidate count: `3 x 2 x 2 = 12`.

Phase A selection metric:

1. Prefer higher validation AP.
2. If validation AP is close, prefer lower validation NLL/ECE after calibration.
3. If still close, prefer deployable event behavior in `fit_ops`:
   - no false alerts on validation
   - higher recall
   - lower delay
4. Avoid selecting a configuration only because it has an unusually lucky test
   score. Test remains for final reporting only.

### Phase B. Freeze One CTR-GCN Candidate

After Phase A, freeze one CTR-GCN config.

The freeze must record:

- selected hyperparameters
- selection reason
- exact training command template
- exact `fit_ops` command template
- exact evaluation command template
- whether confirm fallback is expected or unexpected

No new hyperparameter dimensions should be added after this freeze unless the
task note is explicitly amended.

### Phase C. Five-Seed CTR-GCN Stability Run

Run the frozen CTR-GCN config on:

- `1337`
- `17`
- `2025`
- `33724876`
- `42`

Each seed must run:

1. train
2. fit_ops on validation
3. evaluate on test

Recommended artifact naming:

- checkpoints:
  - `outputs/ctr_gcn/caucafall_stb_s1337/`
  - `outputs/ctr_gcn/caucafall_stb_s17/`
  - `outputs/ctr_gcn/caucafall_stb_s2025/`
  - `outputs/ctr_gcn/caucafall_stb_s33724876/`
  - `outputs/ctr_gcn/caucafall_stb_s42/`
- ops:
  - `ops/configs/ops/ctr_gcn_caucafall_stb_s1337.yaml`
  - `ops/configs/ops/ctr_gcn_caucafall_stb_s17.yaml`
  - `ops/configs/ops/ctr_gcn_caucafall_stb_s2025.yaml`
  - `ops/configs/ops/ctr_gcn_caucafall_stb_s33724876.yaml`
  - `ops/configs/ops/ctr_gcn_caucafall_stb_s42.yaml`
- metrics:
  - `outputs/metrics/ctr_gcn_caucafall_stb_s1337.json`
  - `outputs/metrics/ctr_gcn_caucafall_stb_s17.json`
  - `outputs/metrics/ctr_gcn_caucafall_stb_s2025.json`
  - `outputs/metrics/ctr_gcn_caucafall_stb_s33724876.json`
  - `outputs/metrics/ctr_gcn_caucafall_stb_s42.json`

## 5. Formal Comparison Rules

CTR-GCN should be compared against existing TCN/GCN stability artifacts by:

- AP mean/std
- AUC mean/std
- OP2 F1 mean/std
- OP2 recall mean/std
- OP2 precision mean/std
- OP2 FA/24h mean/std
- OP2 mean delay mean/std

Primary comparison set:

- `outputs/metrics/tcn_caucafall_stb_s*.json`
- `outputs/metrics/gcn_caucafall_stb_s*.json`
- `outputs/metrics/ctr_gcn_caucafall_stb_s*.json`

Interpretation guardrail:

- If CTR-GCN beats one single seed but loses the five-seed mean, report it as a
  promising single-run result only.
- If CTR-GCN has similar AP/AUC but lower delay or better recall stability,
  report it as a deployment-behavior tradeoff rather than a universal accuracy
  improvement.
- If CTR-GCN has high variance, report it as an unstable prototype requiring
  further optimisation.

## 6. Non-Goals

Do not include the following in the first formal CTR-GCN sweep:

- channel schedule search
- temporal kernel search
- rel_channels search
- bone or bone-length feature expansion
- focal-loss search
- balanced-sampler search
- hard-negative mining changes
- different splits or window sizes

These can be follow-up ablations only after the first frozen CTR-GCN comparison
is complete.

## 7. Checklist

- [x] Windows CUDA environment validated
- [x] single-seed CTR-GCN baseline trained
- [x] single-seed CTR-GCN `fit_ops` completed
- [x] single-seed CTR-GCN test evaluation completed
- [x] Phase A bounded hyperparameter sweep completed
- [x] one CTR-GCN config frozen
- [x] five-seed CTR-GCN train/fit/eval completed
- [x] CTR-GCN stability summary generated
- [x] TCN/GCN/CTR-GCN paired comparison table generated

## 8. Phase A Result

Phase A completed on seed `33724876`.

Summary artifacts:

- `outputs/metrics/ctr_gcn_caucafall_sweep_summary_2026-05-05.json`
- `outputs/metrics/ctr_gcn_caucafall_sweep_summary_2026-05-05.csv`

Top validation-ranked candidates:

| Rank | Candidate | Val AP | Val AUC | Val OP2 recall | Val OP2 F1 | Val OP2 FA/24h | Val OP2 delay | ECE after |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `lr=3e-4`, `dropout=0.40`, `ctr_rank=8` | `0.9924` | `0.9942` | `0.8000` | `0.8889` | `0.0` | `3.26s` | `0.1194` |
| 2 | `lr=3e-4`, `dropout=0.40`, `ctr_rank=4` | `0.9919` | `0.9938` | `1.0000` | `1.0000` | `0.0` | `3.65s` | `0.0608` |
| 3 | `lr=3e-4`, `dropout=0.30`, `ctr_rank=8` | `0.9898` | `0.9928` | `1.0000` | `1.0000` | `0.0` | `3.55s` | `0.0799` |
| 4 | `lr=3e-4`, `dropout=0.30`, `ctr_rank=4` | `0.9896` | `0.9922` | `1.0000` | `1.0000` | `0.0` | `2.61s` | `0.0682` |

Recommended freeze candidate:

- `lr=3e-4`
- `dropout=0.40`
- `ctr_rank=4`

Reason:

- validation AP is within `0.0005` of the top AP configuration
- calibration is materially better than the top AP configuration
- validation OP2 reaches recall `1.0000`, F1 `1.0000`, and FA/24h `0.0`
- this choice follows the predeclared rule: prefer validation AP first, then
  calibration and deployable event behavior when AP is effectively tied

Deployment-prioritized alternate:

- `lr=3e-4`
- `dropout=0.30`
- `ctr_rank=4`

Reason:

- lower validation OP2 delay among the perfect-validation-OP2 candidates
- lower test OP2 delay and better test OP2 recall on this seed
- not selected as the primary freeze candidate because its validation AP is
  farther from the top candidate

## 9. Frozen CTR-GCN Stability Configuration

The frozen CTR-GCN configuration for the five-seed stability run is:

- `lr=3e-4`
- `dropout=0.40`
- `ctr_rank=4`
- `channel_schedule=64,64,64,128`
- `rel_channels=8`
- `temporal_kernel=9`
- `use_motion=1`
- `use_conf_channel=1`
- `use_bone=0`
- `use_bone_length=0`
- `motion_scale_by_fps=1`
- `conf_gate=0.2`
- `use_precomputed_mask=1`
- `mask_joint_p=0.05`
- `mask_frame_p=0.05`
- `batch=32`
- `epochs=180`
- `min_epochs=40`
- `patience=25`
- `monitor=ap`
- `deterministic=1`
- `amp=0`

Freeze rationale:

- `lr=3e-4`, `dropout=0.40`, `ctr_rank=4` was the best balance under the
  predeclared Phase A selection rule.
- It was effectively tied with the top validation-AP candidate while having
  materially better calibration and better validation OP2 behavior.
- The lower-rank refinement is also the more conservative model choice for the
  formal stability run.

Known issue to preserve in reporting:

- all Phase A candidates triggered validation-side confirm fallback in `fit_ops`
- the five-seed run should keep the same `fit_ops` command and record the same
  behavior if it recurs
- this should be reported as a deployment-policy interaction, not hidden as a
  training failure

Five-seed run command template:

```powershell
$env:PYTHONPATH="D:\goldsmiths\fall_detection_v2\ml\src;D:\goldsmiths\fall_detection_v2"

.\.venv-win\Scripts\python.exe .\ml\src\fall_detection\training\train_ctr_gcn.py `
  --train_dir data\processed\caucafall\windows_eval_W48_S12\train `
  --val_dir data\processed\caucafall\windows_eval_W48_S12\val `
  --save_dir outputs\ctr_gcn\caucafall_stb_s<SEED> `
  --epochs 180 `
  --min_epochs 40 `
  --batch 32 `
  --lr 3e-4 `
  --weight_decay 1e-4 `
  --seed <SEED> `
  --patience 25 `
  --fps_default 23 `
  --monitor ap `
  --channel_schedule 64,64,64,128 `
  --rel_channels 8 `
  --ctr_rank 4 `
  --temporal_kernel 9 `
  --dropout 0.40 `
  --use_motion 1 `
  --use_conf_channel 1 `
  --use_bone 0 `
  --use_bone_length 0 `
  --motion_scale_by_fps 1 `
  --conf_gate 0.2 `
  --use_precomputed_mask 1 `
  --mask_joint_p 0.05 `
  --mask_frame_p 0.05 `
  --num_workers 0 `
  --deterministic 1 `
  --amp 0
```

## 10. Five-Seed Stability Result

Five-seed CTR-GCN stability run completed on:

- `1337`
- `17`
- `2025`
- `33724876`
- `42`

CTR-GCN artifact anchors:

- checkpoints: `outputs/ctr_gcn/caucafall_stb_s*/`
- ops YAML: `ops/configs/ops/ctr_gcn_caucafall_stb_s*.yaml`
- metrics JSON: `outputs/metrics/ctr_gcn_caucafall_stb_s*.json`

Summary artifacts:

- `outputs/metrics/ctr_gcn_caucafall_stability_summary_2026-05-05.json`
- `outputs/metrics/ctr_gcn_caucafall_stability_summary_2026-05-05.csv`
- `outputs/metrics/caucafall_stability_comparison_tcn_gcn_ctr_gcn_2026-05-05.json`
- `outputs/metrics/caucafall_stability_comparison_tcn_gcn_ctr_gcn_2026-05-05.csv`

Five-seed comparison summary, OP2:

| Model | AP mean/std | AUC mean/std | OP2 F1 mean/std | OP2 recall mean/std | OP2 FA/24h mean/std | OP2 delay mean/std |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `TCN` | `0.9819 / 0.0021` | `0.9897 / 0.0020` | `0.8611 / 0.0621` | `0.7600 / 0.0894` | `0.0000 / 0.0000` | `5.0609s / 0.1091s` |
| `GCN` | `0.9706 / 0.0100` | `0.9798 / 0.0075` | `0.5873 / 0.1976` | `0.4400 / 0.2191` | `0.0000 / 0.0000` | `4.6174s / 0.5503s` |
| `CTR-GCN` | `0.9707 / 0.0046` | `0.9800 / 0.0043` | `0.7421 / 0.1128` | `0.6000 / 0.1414` | `0.0000 / 0.0000` | `4.4174s / 0.8026s` |

CTR-GCN per-seed OP2:

| Seed | AP | AUC | OP2 F1 | OP2 recall | OP2 FA/24h | OP2 delay |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `1337` | `0.9634` | `0.9728` | `0.7500` | `0.6000` | `0.0` | `5.2174s` |
| `17` | `0.9712` | `0.9806` | `0.7500` | `0.6000` | `0.0` | `3.4783s` |
| `2025` | `0.9721` | `0.9813` | `0.7500` | `0.6000` | `0.0` | `5.0435s` |
| `33724876` | `0.9710` | `0.9809` | `0.8889` | `0.8000` | `0.0` | `3.6522s` |
| `42` | `0.9759` | `0.9844` | `0.5714` | `0.4000` | `0.0` | `4.6957s` |

Interpretation:

- CTR-GCN improves substantially over the existing custom GCN on OP2 event
  metrics while keeping AP/AUC roughly tied with GCN.
- CTR-GCN does not beat the defended TCN five-seed line on AP, AUC, OP2 F1, or
  OP2 recall.
- CTR-GCN has lower mean OP2 delay than TCN, but this should be framed as a
  latency/recall tradeoff rather than a stronger overall model result.
- All CTR-GCN five-seed `fit_ops` runs again triggered confirm fallback. This
  is now a consistent deployment-policy interaction for the current CTR-GCN
  line and should be reported explicitly.

## 11. Replay Clips Check

Replay clips were evaluated after the five-seed stability run using the existing
delivery replay windows:

- windows: `artifacts/fall_test_eval_20260315/windows/unsplit`
- clips represented: 24 videos, 12 fall and 12 non-fall
- weak fall labels: `/corridor/`, `/kitchen/`
- weak non-fall labels: `/corridor_adl/`, `/kitchen_adl/`
- script: `ops/scripts/eval_delivery_videos.py`

The replay script was updated so project-adapted CTR-GCN checkpoints use the
single-stream `[B,T,V,F]` input path instead of the legacy two-stream GCN call.

CTR-GCN five-seed OP2 replay summary:

| Seed | OP2 tau_high | TP | TN | FP | FN | Precision | Recall | F1 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `1337` | `0.83` | `0` | `11` | `1` | `12` | `0.0000` | `0.0000` | `0.0000` |
| `17` | `0.59` | `3` | `11` | `1` | `9` | `0.7500` | `0.2500` | `0.3750` |
| `2025` | `0.84` | `1` | `11` | `1` | `11` | `0.5000` | `0.0833` | `0.1429` |
| `33724876` | `0.72` | `7` | `10` | `2` | `5` | `0.7778` | `0.5833` | `0.6667` |
| `42` | `0.79` | `0` | `11` | `1` | `12` | `0.0000` | `0.0000` | `0.0000` |

Additional seed `33724876` checks:

- OP1, ungated: `TP=12`, `TN=2`, `FP=10`, `FN=0`
- OP1, delivery-gated: `TP=11`, `TN=7`, `FP=5`, `FN=1`
- OP2, delivery-gated: `TP=6`, `TN=11`, `FP=1`, `FN=6`

CTR-GCN replay under strengthened TCN Candidate A/D parameters:

This test keeps the CTR-GCN checkpoints fixed but overrides the alert policy
with the strengthened TCN replay parameters:

- Candidate A parameters: `tau_high=0.86`, `tau_low=0.6708`, `ema_alpha=0.2`,
  `k=2`, `n=3`, `confirm=false`
- Candidate D parameters: `tau_high=0.82`, `tau_low=0.6396`, `ema_alpha=0.2`,
  `k=2`, `n=3`, `confirm=false`

| Parameter profile | Seed | TP | TN | FP | FN | Correct | Recall | F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Candidate A params | `1337` | `0` | `11` | `1` | `12` | `11/24` | `0.0000` | `0.0000` |
| Candidate A params | `17` | `0` | `12` | `0` | `12` | `12/24` | `0.0000` | `0.0000` |
| Candidate A params | `2025` | `0` | `11` | `1` | `12` | `11/24` | `0.0000` | `0.0000` |
| Candidate A params | `33724876` | `0` | `11` | `1` | `12` | `11/24` | `0.0000` | `0.0000` |
| Candidate A params | `42` | `0` | `12` | `0` | `12` | `12/24` | `0.0000` | `0.0000` |
| Candidate D params | `1337` | `0` | `11` | `1` | `12` | `11/24` | `0.0000` | `0.0000` |
| Candidate D params | `17` | `0` | `11` | `1` | `12` | `11/24` | `0.0000` | `0.0000` |
| Candidate D params | `2025` | `2` | `11` | `1` | `10` | `13/24` | `0.1667` | `0.2667` |
| Candidate D params | `33724876` | `3` | `11` | `1` | `9` | `14/24` | `0.2500` | `0.4000` |
| Candidate D params | `42` | `0` | `11` | `1` | `12` | `11/24` | `0.0000` | `0.0000` |

Replay artifacts:

- `artifacts/fall_test_eval_20260315/delivery_ctr_gcn_5seed_op2_summary.json`
- `artifacts/fall_test_eval_20260315/delivery_ctr_gcn_5seed_op2_summary.csv`
- `artifacts/fall_test_eval_20260315/delivery_ctr_gcn_5seed_tcn_candidate_params_summary.json`
- `artifacts/fall_test_eval_20260315/delivery_ctr_gcn_5seed_tcn_candidate_params_summary.csv`
- `artifacts/fall_test_eval_20260315/delivery_ctr_gcn_stb_s*_op2_metrics.json`

Report-branch replay anchors:

| Reference | TP | TN | FP | FN | Correct | Status |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `tcn_caucafall_locked_op2` | `5` | `10` | `2` | `7` | `15/24` | defended historical locked surface |
| `TCN Candidate A/D` | `6` | `10` | `2` | `6` | `16/24` | modest strengthened runtime line |
| old delivery-only `24/24` profile | `12` | `12` | `0` | `0` | `24/24` | excluded; special tuned profile, not a defended baseline |

Interpretation:

- The old `24/24` replay artifact must be ignored for CTR-GCN comparison. It is
  a special tuned delivery-only profile and was already marked unsafe in the
  report branch.
- Seed `33724876` reaches `17/24` on OP2 (`TP=7`, `TN=10`, `FP=2`, `FN=5`),
  which is one correct video above the strengthened TCN `16/24` runtime line at
  the same false-positive count.
- The five-seed CTR-GCN replay line is not stable: four of five seeds do not
  reach the strengthened TCN `16/24` replay reference, and two seeds miss every
  fall clip under OP2.
- OP1 and gated variants confirm that replay behavior is strongly policy-shaped:
  lower thresholds recover falls but quickly increase ADL false alarms.
- Reusing the strengthened TCN Candidate A/D thresholds directly on CTR-GCN does
  not work. The thresholds are model-probability-scale specific: Candidate A
  parameters suppress every CTR-GCN fall alert, and Candidate D parameters peak
  at only `14/24`.
- The careful claim is therefore: CTR-GCN has a promising single-seed replay
  result on the 24-clip surface, but it is not yet a robust deployment upgrade
  over the strengthened TCN line.
