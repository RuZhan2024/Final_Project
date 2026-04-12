# Paper Claims and Limitations Draft

Date: 2026-03-22

Purpose:
Provide paper-safe draft wording for contributions and limitations based on the current repository evidence.

Usage:
- Use this as source text for the abstract, introduction, discussion, and conclusion.
- Keep the wording bounded to current artifacts.

Protocol Reminder:
- `Paper Protocol Freeze v1`
- Primary dataset: `CAUCAFall`
- Comparative/generalization dataset: `LE2i`
- Current architecture-comparison wording:
  - `TCN trends stronger than GCN under the frozen protocol`
  - not `TCN significantly outperforms GCN`

## Draft Contribution Claims

### Contribution 1: End-to-end deployment-oriented fall-detection system

This work presents an end-to-end pose-based fall-detection system that spans preprocessing, temporal window generation, model training, operating-point fitting, backend inference, and replay-oriented runtime validation. The contribution is not only a benchmark model, but an integrated system whose deployable profile is locked and traceable through configuration, runbooks, and evidence artifacts.

Evidence anchors:
- [README.md](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/README.md)
- [DEPLOYMENT_LOCK.md](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/docs/project_targets/DEPLOYMENT_LOCK.md)
- [THESIS_EVIDENCE_MAP.md](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/docs/project_targets/THESIS_EVIDENCE_MAP.md)

Safe claim boundary:
- claim a working deployment-oriented software artifact
- do not claim mature real-world production deployment

### Contribution 2: Controlled TCN-vs-GCN comparison under a fixed protocol

This work provides a controlled comparison between temporal and graph-based skeleton models under matched preprocessing, feature construction, split policy, and evaluation rules. Under the current locked protocol, the final TCN candidate trends stronger than the final GCN candidate on the primary CAUCAFall evidence. The paper deliberately keeps this as a directional result rather than a definitive superiority claim, because the primary non-parametric significance test remains exploratory at the current `n=5`.

Evidence anchors:
- [FINAL_CANDIDATES.md](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/docs/project_targets/FINAL_CANDIDATES.md)
- [STABILITY_REPORT.md](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/docs/project_targets/STABILITY_REPORT.md)
- [SIGNIFICANCE_REPORT.md](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/docs/project_targets/SIGNIFICANCE_REPORT.md)

Safe claim boundary:
- claim a controlled comparison and directional advantage under the locked protocol
- do not claim definitive universal superiority of TCN over GCN
- do not use `significantly outperforms` language in the main paper draft

### Contribution 3: Operating-point calibration as a first-class evaluation layer

This work treats fall detection as an operational alerting problem rather than a score-ranking problem only. The project separates window-level discrimination quality from deployment-facing alert behavior by fitting validation-only operating points and reporting event-level outcomes such as recall, precision, and false alerts per 24 hours. This makes the evaluation more relevant to practical monitoring use cases than AP alone.

Evidence anchors:
- [RESEARCH_QUESTIONS_MAPPING.md](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/docs/project_targets/RESEARCH_QUESTIONS_MAPPING.md)
- [PAPER_PUBLICATION_READINESS_PLAN.md](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/docs/project_targets/PAPER_PUBLICATION_READINESS_PLAN.md)
- [CLAIM_TABLE.md](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/docs/project_targets/CLAIM_TABLE.md)

Safe claim boundary:
- claim that operating-point calibration is a distinctive part of the evaluation design
- do not claim that the current OP report is already complete unless the pending summary is refreshed

### Contribution 4: Explicit cross-dataset limitation analysis

This work reports bidirectional cross-dataset transfer results and uses them to argue against over-claiming universal robustness. The current transfer evidence is asymmetric, with at least one direction showing severe degradation relative to in-domain performance. This negative result is important because it bounds the scope of deployment claims and clarifies where dataset dependence remains strong.

Evidence anchors:
- [CROSS_DATASET_REPORT.md](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/docs/project_targets/CROSS_DATASET_REPORT.md)
- [cross_dataset_summary.csv](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/artifacts/reports/cross_dataset_summary.csv)

Safe claim boundary:
- claim honest generalization analysis and documented transfer limits
- do not claim strong cross-domain robustness

### Contribution 5: Bounded replay and custom replay validation evidence

This work includes bounded deployment-style evidence through locked replay validation and a four-folder custom replay check evaluated under the same canonical online profile as the main monitor. These artifacts support the claim that the system can be demonstrated end to end under a controlled review path. They should be used as deployment-oriented supporting evidence, not as a substitute for broad real-world validation.

Evidence anchors:
- [FOUR_VIDEO_DELIVERY_PROFILE.md](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/docs/reports/runbooks/FOUR_VIDEO_DELIVERY_PROFILE.md)
- [DEPLOYMENT_LOCK.md](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/docs/project_targets/DEPLOYMENT_LOCK.md)
- [THESIS_EVIDENCE_MAP.md](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/docs/project_targets/THESIS_EVIDENCE_MAP.md)

Safe claim boundary:
- claim bounded controlled replay evidence and explicit runtime-boundary analysis
- do not claim complete real-world deployment validation until field-validation artifacts are closed

## Draft Limitations Section

The present study has several limitations that should be kept explicit in the paper. First, the final architecture comparison is based on a relatively small multi-seed sample, and the primary Wilcoxon test remains exploratory at the current `n=5` setting. The observed direction of the effect is informative, but the paper should avoid presenting the comparison as definitive proof of architectural superiority.

Second, the cross-dataset evaluation shows that transfer is asymmetric and not robust in both directions. This means the reported system should be understood as effective under the locked primary-dataset protocol, while broader generalization remains limited by domain shift, camera conditions, and skeleton-quality differences.

Third, the strongest deployment-oriented evidence currently comes from replay-style validation and a bounded custom replay check rather than from a large real-world field study. The current unified-profile custom replay result is not a success claim; it is useful because it exposes runtime sensitivity and domain mismatch under the canonical online profile. Until the field-validation artifacts are fully closed, the paper should describe real-world usefulness as promising but still partially validated.

Fourth, the reproducibility story is strong at the artifact and documentation level, but practical reruns still depend on environment stability, dependency setup, and runtime assumptions. The paper should therefore distinguish between traceable reproducibility of results and friction-free reproduction on a fresh machine.

## Short Abstract-Safe Version

This study presents a pose-based fall-detection system with controlled TCN-vs-GCN comparison, deployment-oriented operating-point calibration, and explicit cross-dataset evaluation. Under the locked protocol, the final TCN candidate trends stronger than the final GCN candidate on the primary dataset, while cross-dataset transfer remains asymmetric and real-world validation is still bounded. The contribution is therefore best understood as a strong deployment-oriented system study with cautious comparative conclusions rather than a claim of universal robustness.

## Abstract-Ready Structured Version

Background:
Fall detection is often reported as a classification problem, but practical deployment also depends on alert-policy behavior, false alarms, and reproducibility.

Aim:
This study evaluates a pose-based fall-detection pipeline as a deployment-oriented system and compares temporal and graph-based skeleton models under a frozen protocol.

Methods:
We use a fixed preprocessing and evaluation pipeline, compare final TCN and GCN candidates on `CAUCAFall` and `LE2i`, report multi-seed results under a frozen seed set, and include replay-oriented deployment evidence together with bounded field-validation artifacts.

Results:
Under the frozen primary-dataset protocol, the final TCN candidate trends stronger than the matched GCN candidate on `CAUCAFall`, while the primary non-parametric significance result remains exploratory at `n=5`. Cross-dataset transfer is asymmetric, and deployment-oriented evidence is strongest for locked replay and delivery-package validation rather than broad field closure.

Conclusion:
The project is best interpreted as a strong deployment-oriented system study with cautious comparative conclusions and explicit generalization and validation limits.

## Short Introduction-Safe Contributions List

Use a compact 3-point version if the paper needs shorter contribution bullets:

1. We present an end-to-end pose-based fall-detection system that integrates preprocessing, training, operating-point fitting, and replay-oriented deployment validation.
2. We provide a controlled comparison between temporal and graph-based skeleton models under a shared protocol, with cautious statistical interpretation.
3. We report cross-dataset and deployment-oriented evidence in a way that emphasizes operational trade-offs and explicit limitations rather than over-claiming universal generalization.

## Discussion-Safe Main Findings Paragraph

The main finding of this study is not that one architecture has been definitively proven superior in general, but that under a frozen deployment-oriented protocol the TCN candidate trends stronger than the matched GCN candidate on the primary `CAUCAFall` evidence while retaining a cautious statistical interpretation. A second important finding is that cross-dataset transfer is clearly asymmetric, which limits any broader robustness claim. Finally, the project’s strongest practical contribution lies in its end-to-end system framing, operating-point-aware evaluation, and locked replay validation path rather than in a claim of fully closed real-world deployment.

## Limitations-Safe Short Paragraph

This study is limited by the small multi-seed sample used for the final architecture comparison, the asymmetric cross-dataset transfer behavior, and the fact that field-validation evidence is currently bounded rather than large scale. In addition, although artifact traceability is strong, friction-free reproduction on a fresh machine still depends on environment stability and dependency setup.
