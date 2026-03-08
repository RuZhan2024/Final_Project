# TECHNIQUE_TIMELINE

Date: 2026-03-02
Theme: skeleton-action foundations -> deployment-oriented fall detection.

## 2017
- **Focal Loss (ICCV 2017)** introduced robust class-imbalance optimization for dense detection.
- **Calibration framing (ICML 2017)** established temperature scaling as a reliable post-hoc confidence correction.
- Why it mattered: made probability quality and imbalance handling first-class concerns.

## 2018
- **ST-GCN (AAAI 2018)** established the graph-temporal backbone for skeleton action recognition.
- Why it mattered: provided a reusable operator family for skeleton sequences.

## 2019
- **2s-AGCN (CVPR 2019)**: adaptive adjacency + two-stream (joint/motion) representation.
- **TSM (ICCV 2019)**: near-zero-FLOPs temporal context injection.
- **MS-TCN (CVPR 2019)**: multi-scale temporal refinement.
- **SelectiveNet (ICML 2019)**: explicit risk-coverage prediction for abstention-aware systems.
- Why it mattered: moved from fixed-graph baselines to adaptive topology and practical temporal efficiency.

## 2020
- Consolidation period: action-recognition stacks increasingly emphasized lightweight temporal modules and better deployment ergonomics.
- Why it mattered: prepared transition from benchmark-only models to deployable variants.

## 2021
- **CTR-GCN (ICCV 2021)**: channel-wise topology refinement improved expressiveness without abandoning graph priors.
- Why it mattered: stronger accuracy/efficiency trade-off for skeleton models.

## 2022
- Broad ecosystem stabilization (e.g., MMAction2 skeleton support) improved reproducibility and engineering workflows.
- Why it mattered: easier cross-method ablations and robust baselines.

## 2023
- Increased focus on practical deployment metrics (false alarms, robustness under occlusion/domain shift).
- Why it mattered: highlighted that classification AP alone is insufficient for fall systems.

## 2024
- Fall-specific works continued adapting skeleton GCNs to robust sensing modalities (e.g., IR + ST-GCN).
- Why it mattered: modality/domain robustness became central for real deployments.

## 2025
- Stronger deployment-oriented fall papers (edge constraints, comparative protocols, temporal-lightweight stacks).
- Why it mattered: research emphasis shifted from raw detection score to real-time reliability and operational policy.

## 2026 (current planning horizon)
- Best practice is now:
  1. lock data/time/normalization contracts,
  2. enforce policy sanity gates,
  3. apply low-latency architecture upgrades (TSM/adaptive GCN) behind feature flags,
  4. optimize FA/24h via hard-negative loops and cost-sensitive calibrated thresholds.

## References
- ST-GCN: https://ojs.aaai.org/index.php/AAAI/article/view/12328
- 2s-AGCN: https://openaccess.thecvf.com/content_CVPR_2019/html/Shi_Two-Stream_Adaptive_Graph_Convolutional_Networks_for_Skeleton-Based_Action_Recognition_CVPR_2019_paper.html
- TSM: https://openaccess.thecvf.com/content_ICCV_2019/html/Lin_TSM_Temporal_Shift_Module_for_Efficient_Video_Understanding_ICCV_2019_paper.html
- MS-TCN: https://openaccess.thecvf.com/content_CVPR_2019/html/Abu_Farha_MS-TCN_Multi-Stage_Temporal_Convolutional_Network_for_Action_Segmentation_CVPR_2019_paper.html
- CTR-GCN: https://openaccess.thecvf.com/content/ICCV2021/html/Chen_Channel-Wise_Topology_Refinement_Graph_Convolution_for_Skeleton-Based_Action_Recognition_ICCV_2021_paper.html
- Focal Loss: https://openaccess.thecvf.com/content_iccv_2017/html/Lin_Focal_Loss_for_ICCV_2017_paper.html
- Temperature scaling: https://proceedings.mlr.press/v70/guo17a.html
- SelectiveNet: https://proceedings.mlr.press/v97/geifman19a.html
