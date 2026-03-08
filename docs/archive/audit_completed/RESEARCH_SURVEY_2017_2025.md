# RESEARCH_SURVEY_2017_2025

Date: 2026-03-02
Coverage: 2017-2025 (with 2024-2025 priority), focused on fall detection + skeleton action methods relevant to this repo.

## 1) Fall-Detection-Focused Papers (2024-2025 priority)

| Year | Work | Core idea | Implementation detail you would code | On-device impact | Evidence strength | Mapping to this repo |
|---|---|---|---|---|---|---|
| 2025 | *A comparative study for pre-impact and post-impact fall detection in wearable systems* (Pervasive and Mobile Computing) | Jointly evaluate pre/post impact detection tradeoffs in deploy settings | Extend event-policy objective to separate pre-impact recall vs post-impact FA constraints | low runtime overhead (policy-level) | medium-high (journal comparative study) | `src/fall_detection/evaluation/fit_ops.py`, `metrics_eval.py`, ops schema |
| 2025 | *Skeleton-based human fall detection with person tracking and temporal deep learning* (Automation in Construction) | Multi-person tracking + temporal model around skeleton streams | Add tracked-person selection and temporal identity-consistent windows before model inference | moderate CPU increase (tracking) | medium (domain-specific journal) | `src/fall_detection/pose/*`, `data/windowing/*`, `deploy/run_modes.py` |
| 2025 | *A lightweight Transformer- and TCN-based architecture for robust, real-time fall detection* (Scientific Reports) | Lightweight temporal stack with real-time constraints, edge deployment framing | Borrow lightweight temporal attention block and benchmark latency p95 on CPU profile | moderate but controllable | medium (peer-reviewed, edge emphasis) | `core/models.py` (TCN block), `scripts/profile_infer.py` |
| 2024 | *Real-Time Fall Detection from Infrared Video Based on ST-GCN* (Sensors) | ST-GCN adaptation to fall detection under robust sensing modality | Re-validate ST-GCN variants + threshold policy under domain shift | model-side moderate | medium (single-paper, modality-specific) | `core/models.py` (GCN variants), `evaluation/fit_ops.py` |

Notes:
- Public reproducible repos for many 2024-2025 fall-specific papers are limited; prioritize methods with clear ablations and reproducible baselines before adopting novel blocks.

## 2) Skeleton Action Recognition Methods Adaptable to Fall Detection

| Year | Method | Core idea | What to implement exactly | Compute/latency | Evidence strength | Mapping to this repo |
|---|---|---|---|---|---|---|
| 2018 | ST-GCN | spatial-temporal graph conv baseline with fixed graph priors | Keep as baseline abstraction for graph-temporal decomposition | moderate | very high (foundational) | `src/fall_detection/core/models.py` |
| 2019 | 2s-AGCN | adaptive graph topology (A+B+C) + two-stream inputs | Add learnable residual adjacency and joint/motion branch controls | moderate+ | very high | `core/models.py`, `training/train_gcn.py` flags |
| 2019 | MS-TCN | multi-stage temporal refinement / dilated temporal context | Add optional multi-scale temporal stack for long-context fall transitions | moderate | high | `core/models.py` TCN path |
| 2019 | TSM | temporal shift at near-zero FLOPs | Insert shift op inside TCN blocks (online-safe variant) | low | high | `core/models.py`, `training/train_tcn.py` |
| 2021 | CTR-GCN | channel-wise topology refinement | Add channel-wise topology refine block in selected GCN layers | moderate | high | `core/models.py` |
| ongoing | MMAction2 skeleton ecosystem | standardized skeleton pipelines and configs | use as external validation bed for architecture parity and ablations | n/a (tooling) | high (widely used) | experiment config design + sanity comparisons |

## 3) Training/Policy Techniques Required by This Audit

| Technique | Research basis | Why relevant here | Minimal adoption plan |
|---|---|---|---|
| Temperature scaling | Guo et al., ICML 2017 | already present in ops calibration; ensure objective-level use is correct | keep scalar T, add risk-coverage reporting in ops outputs |
| Cost-sensitive thresholding | decision-theoretic detection literature + deployment utility | current F1-only or conservative tie-break can produce unusable OPs | add explicit `cost_fn`, `cost_fp`, `coverage` objective option in fit-ops |
| Coverage/abstention semantics | SelectiveNet (ICML 2019) and selective prediction literature | needed when uncertainty gating is introduced to avoid “abstain-all” failures | encode coverage KPI + reject-rate in reports |
| Hard-negative mining loop | widely adopted in detection systems | direct path to reducing FA/24h in ADL streams | standardize mined-window list and bounded replay ratio |
| Numeric fingerprint gate | engineering best-practice for cross-domain skeleton training | already identified in repo artifacts | enforce in CI via `make audit-numeric` |
| Temporal stride gate | physics-consistent window semantics | already identified in repo artifacts | enforce in CI via `make audit-temporal` |

## 4) On-Device Suitability Assessment

- Best low-latency candidates first:
  1. TSM-in-TCN (near-zero FLOPs)
  2. policy-level cost-sensitive calibration (no model FLOPs)
  3. selective hard-negative mining (training-only cost)
- Higher-cost candidates:
  1. full adaptive adjacency GCN
  2. channel-wise topology refinement across deep GCN stacks
  3. multi-person tracking before skeleton inference

## 5) Evidence-to-Repo Mapping (exact modules)

- Model architecture toggles:
  - `src/fall_detection/core/models.py`
- Train-time flag surfaces:
  - `src/fall_detection/training/train_tcn.py`
  - `src/fall_detection/training/train_gcn.py`
- Policy and calibration:
  - `src/fall_detection/evaluation/fit_ops.py`
  - `src/fall_detection/evaluation/metrics_eval.py`
- Online/offline policy parity:
  - `src/fall_detection/deploy/run_modes.py`
  - `src/fall_detection/deploy/run_alert_policy.py`
- Data-contract gates:
  - `scripts/audit_numeric.py`, `scripts/audit_temporal.py`, `Makefile` audit targets

## 6) Gaps / Verification Notes

- 2024-2025 fall-specific papers have uneven public-code availability; where no official repo exists, treat claims as “needs local reproduction.”
- For each imported technique, require one controlled ablation against current LE2i baseline before multi-dataset rollout.

## 7) References (primary links)

- ST-GCN (AAAI 2018): https://ojs.aaai.org/index.php/AAAI/article/view/12328
- 2s-AGCN (CVPR 2019): https://openaccess.thecvf.com/content_CVPR_2019/html/Shi_Two-Stream_Adaptive_Graph_Convolutional_Networks_for_Skeleton-Based_Action_Recognition_CVPR_2019_paper.html
- CTR-GCN (ICCV 2021): https://openaccess.thecvf.com/content/ICCV2021/html/Chen_Channel-Wise_Topology_Refinement_Graph_Convolution_for_Skeleton-Based_Action_Recognition_ICCV_2021_paper.html
- TSM (ICCV 2019): https://openaccess.thecvf.com/content_ICCV_2019/html/Lin_TSM_Temporal_Shift_Module_for_Efficient_Video_Understanding_ICCV_2019_paper.html
- MS-TCN (CVPR 2019): https://openaccess.thecvf.com/content_CVPR_2019/html/Abu_Farha_MS-TCN_Multi-Stage_Temporal_Convolutional_Network_for_Action_Segmentation_CVPR_2019_paper.html
- Temperature Scaling (ICML 2017): https://proceedings.mlr.press/v70/guo17a.html
- Focal Loss (ICCV 2017): https://openaccess.thecvf.com/content_iccv_2017/html/Lin_Focal_Loss_for_ICCV_2017_paper.html
- SelectiveNet (ICML 2019): https://proceedings.mlr.press/v97/geifman19a.html
- MMAction2 docs/repo: https://mmaction2.readthedocs.io/en/latest/model_zoo/skeleton.html and https://github.com/open-mmlab/mmaction2
- Sensors 2024 IR + ST-GCN fall detection: https://www.mdpi.com/1424-8220/24/13/4272
- Automation in Construction 2025 skeleton fall detection: https://www.sciencedirect.com/science/article/pii/S0926580525002054
- Pervasive and Mobile Computing 2025 comparative fall study: https://www.sciencedirect.com/science/article/pii/S1574119225000799
- Scientific Reports 2025 lightweight Transformer+TCN fall detection: https://www.nature.com/articles/s41598-025-99718-z
