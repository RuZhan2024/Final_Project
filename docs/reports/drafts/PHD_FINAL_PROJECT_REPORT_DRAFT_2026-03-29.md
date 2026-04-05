# Pose-Based Fall Detection with Temporal and Graph Neural Models: Deployment-Aware Evaluation, Alert Policy Calibration, and Runtime Analysis

Draft date: 2026-03-29  
Status: complete first full draft for supervisor-review iteration  
Evidence control: this draft is constrained by [PHD_FINAL_REPORT_TASKS_2026-03-29.md](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/docs/reports/notes/PHD_FINAL_REPORT_TASKS_2026-03-29.md) and [PHD_FINAL_REPORT_EVIDENCE_INVENTORY_2026-03-29.md](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/docs/reports/notes/PHD_FINAL_REPORT_EVIDENCE_INVENTORY_2026-03-29.md).

## Abstract

Fall detection is often presented as a classification problem, but practical deployment depends on more than classifier accuracy alone. A usable monitoring system must integrate pose extraction, temporal inference, alert-policy design, runtime latency constraints, and reproducible deployment behaviour. This project studies a pose-based fall-detection pipeline that combines preprocessing, temporal window generation, model training, operating-point fitting, backend inference, and monitoring-oriented validation.

The work compares a Temporal Convolutional Network (TCN) and a custom spatio-temporal Graph Convolutional Network (GCN) under a locked offline protocol, with `CAUCAFall` treated as the primary benchmark and deployment-target dataset and `LE2i` treated as comparative generalisation evidence. A secondary `MUVIM` track is retained as supporting exploratory work rather than as a co-equal primary benchmark. In addition to model comparison, the project treats alerting as an operational decision problem and therefore evaluates validation-side operating-point calibration, temporal smoothing, and alert-policy logic rather than relying on raw window scores alone. The implementation also includes a full-stack monitoring system with a React frontend, FastAPI backend, and persistence/notification path, allowing replay and limited live validation beyond offline metrics.

Under the current frozen protocol, the final TCN candidate trends stronger than the matched GCN candidate on the primary `CAUCAFall` evidence, while the comparative statistical interpretation remains cautious at the current seed budget. Cross-dataset transfer is asymmetric, which limits any claim of broad universal robustness. Deployment-oriented evidence is strongest for locked replay and bounded delivery-style validation rather than for large-scale field closure. The most defensible interpretation of the project is therefore that it delivers a strong deployment-oriented pose-based fall-detection system study with explicit calibration, runtime, and generalisation boundaries, rather than a claim of fully solved real-world fall detection.

## 1. Introduction

### 1.1 Background and Motivation

Falls are a clinically and socially significant event, especially in older-adult and assisted-living settings, because delayed detection can increase both injury severity and time to intervention. For that reason, automated fall detection has been studied across wearable, ambient, and vision-based paradigms. Yet the practical value of a fall-detection system does not depend on recognition accuracy alone. In a monitoring setting, a usable system must also control false alarms, maintain tractable runtime behaviour, and expose decisions that remain meaningful under deployment constraints.

This project treats fall detection as a deployment-oriented sequence-modelling problem rather than as a narrow benchmark classification task. Instead of predicting falls from isolated frames, the system models short pose sequences and converts window-level outputs into operational alerts through an explicit policy layer. The project therefore combines model development with systems concerns: pose extraction, preprocessing, training, operating-point fitting, backend inference, replay monitoring, and bounded deployment validation are treated as connected parts of one end-to-end pipeline.

### 1.2 Why Pose-Based Detection

The project focuses on pose-based vision rather than raw RGB classification or wearable-only sensing for three reasons. First, skeletal pose representations provide a compact and relatively interpretable motion signal, which is well suited to temporal modelling and downstream alert logic. Second, pose-based processing reduces dependence on scene appearance and background texture compared with direct RGB modelling, while still preserving body-configuration and motion cues relevant to falls. Third, a pose-based frontend fits the project’s local/on-device monitoring framing, because browser-side processing can extract landmarks locally and send structured temporal windows to a backend inference path.

This design does not remove deployment difficulty. Pose quality is still sensitive to occlusion, camera placement, motion blur, and frontend runtime behaviour. These limitations are central rather than incidental, because downstream inference can only be as reliable as the skeleton sequence presented to the backend. For that reason, pose quality and runtime stability are treated later in the report as explicit parts of the system analysis rather than hidden behind aggregate benchmark scores.

### 1.3 Project Framing

The project is best understood as a fall-detection system study with three tightly connected layers:

1. A controlled model-comparison layer, focused on temporal and graph-based sequence models.
2. An alert-policy layer, focused on converting window-level model outputs into practical fall alerts through operating-point calibration and temporal decision rules.
3. A deployment/runtime layer, focused on replay validation, latency behaviour, frontend pose quality, and bounded realtime feasibility.

Keeping these layers separate is essential. Offline model evidence cannot be replaced by demo behaviour, and replay/deployment results cannot be treated as equivalent to unseen-test generalisation. A central aim of the report is therefore methodological discipline: each claim is tied to the specific evidence layer that actually supports it.

### 1.4 Problem Statement

The core problem addressed in this work is not simply whether a classifier can separate fall windows from non-fall windows. The stronger practical question is whether a pose-based temporal system can support credible alerting under controlled evaluation and bounded deployment conditions. This leads to a broader technical challenge: temporal model comparison, calibration, alert-policy design, and deployment analysis must be integrated into one coherent study without allowing one evidence layer to illegitimately stand in for another.

### 1.5 Contributions of This Project

The most defensible contributions of the project are:

1. An end-to-end pose-based fall-detection system that spans preprocessing, model training, operating-point fitting, backend inference, replay-oriented monitoring, and reproducible deployment artifacts.
2. A controlled comparison between a TCN and a custom spatio-temporal GCN under a locked primary-dataset protocol, with cautious comparative interpretation.
3. A deployment-aware alerting formulation in which validation-side operating-point fitting, smoothing, and policy rules are treated as first-class components rather than as ad hoc thresholding.
4. An explicit analysis of runtime and deployment boundaries, including replay-path behaviour, latency effects, and the limits of current realtime and field evidence.

### 1.6 Report Structure

The remainder of the report is organised as follows. Section 2 places the work in the context of prior fall-detection and pose-sequence literature. Section 3 defines the research questions and scope boundaries used throughout the report. Section 4 describes the system architecture and responsibility split between the frontend, backend, and deployment paths. Section 5 explains the data roles and experimental protocol. Section 6 describes the model families, while Section 7 explains calibration and alert-policy design. Section 8 summarises the implementation architecture and refactoring rationale. Section 9 presents the core results, including offline comparison, cross-dataset evidence, and deployment/runtime findings. Section 10 discusses the results in direct relation to the research questions. Sections 11 and 12 cover limitations and future work, Section 13 concludes the report, and Section 14 indicates the most relevant appendices and supporting artifact families.

## 2. Background and Related Work

### 2.1 Fall Detection as a Multi-Layer Problem

Fall detection is often presented as a pattern-recognition problem: given sensor observations, determine whether a fall has occurred. In practice, however, the problem is layered. A model may rank fall-like windows effectively while still producing an operationally weak monitoring system if false alarms remain high, decisions are unstable across time, or runtime delays prevent timely alerting. This distinction is central to the present project because the final system is evaluated not only as a classifier, but also as an alerting pipeline with temporal policy and deployment constraints.

### 2.2 Vision-Based and Pose-Based Approaches

Vision-based fall detection can operate on raw RGB frames, optical flow, depth, or derived pose representations. Direct RGB models may capture rich scene context, but they also inherit strong dependence on appearance, lighting, and environment-specific texture. Pose-based methods instead compress the scene into an articulated body representation. This makes them attractive for action-like reasoning, temporal modelling, and partial privacy preservation relative to full-frame storage. At the same time, it makes them especially sensitive to pose extraction quality, missing landmarks, and frontend runtime effects. These trade-offs are central to this project because the deployed monitor pipeline depends on browser-side pose extraction before backend inference can occur.

### 2.3 Temporal and Graph Sequence Models

Skeleton sequences can be modelled in multiple ways. Temporal convolutional models treat the pose sequence as a structured temporal signal and learn discriminative motion patterns through convolution over time. Graph-based models preserve joint topology more explicitly and are therefore attractive when reasoning over articulated motion. Rather than assuming one family is inherently superior, this project treats the comparison as an empirical question under a controlled protocol. The TCN and the custom spatio-temporal GCN are therefore evaluated under matched preprocessing, feature construction, and operating-point rules.

### 2.4 Calibration, Thresholding, and Operational Alerting

A major limitation of many fall-detection discussions is that they stop at classifier outputs. In deployment, an alert is rarely the direct consequence of a single probability threshold. Temporal smoothing, multi-window confirmation, cooldown logic, and false-alarm control are often needed to convert noisy scores into meaningful alerts. This project treats that conversion process explicitly. Validation-side temperature scaling and operating-point fitting are used to shape the alert-policy layer, and the report therefore distinguishes model discrimination quality from deployment-facing alert behaviour.

### 2.5 Deployment-Oriented Evaluation

Benchmark performance alone is not sufficient to justify a monitoring claim. Runtime latency, frontend pose stability, replay-vs-live pipeline differences, and the availability of bounded field evidence all affect what can honestly be claimed about practical usefulness. For this reason, the project includes replay-oriented validation and deployment analysis in addition to offline evaluation. At the same time, these deployment results are not treated as substitutes for formal unseen-test evidence. Maintaining that separation is one of the report’s central methodological commitments.

## 3. Research Questions and Scope

### 3.1 Locked Research Questions

The report is organised around three research questions selected to align with the strongest evidence currently available in the repository while preserving strict separation between model comparison, alert-policy design, and deployment/runtime analysis.

**RQ1. Comparative Offline Performance.** Under the locked offline evaluation protocol, how do the TCN and the custom spatio-temporal GCN compare on the primary fall-detection task?

**RQ2. Calibration and Operational Alerting.** How does validation-side operating-point calibration influence the conversion of window-level model outputs into practical alert decisions?

**RQ3. Deployment Feasibility and Runtime Limits.** What do replay deployment evidence and limited realtime validation show about the practical feasibility and current runtime limits of the system?

### 3.2 Scope Boundaries

The project scope is intentionally bounded in the following ways.

First, the primary result-bearing dataset is `CAUCAFall`, while `LE2i` is treated as comparative and generalisation evidence. The main technical story is therefore anchored on the locked primary-dataset protocol rather than on a pooled multi-dataset claim.

Second, replay validation and limited live validation are treated as system-level evidence, not as replacements for formal offline model evaluation. Replay-oriented tuning or deployment-specific adjustments, if discussed, must therefore be framed as deployment or demonstration calibration only.

Third, the secondary `MUVIM` track is acknowledged as real project work, but it is not promoted into the primary comparative claim. Its role is to support methodological discussion about experimentation breadth, operating-point fitting, and metric-contract interpretation.

Fourth, calibration is used in the operating-point fitting pipeline and must be described precisely in that role. The report should not imply that all deployment-time probabilities are explicitly temperature-calibrated at runtime unless the runtime path demonstrably does so.

Fifth, realtime evidence remains bounded. The report may discuss feasibility and runtime behaviour, but it should not claim broad real-world deployment closure or clinical validation.

### 3.3 Expected Answers

The expected shape of the answers is bounded in advance.

For RQ1, the report seeks a controlled and cautious comparative conclusion rather than a universal architectural ranking. For RQ2, the report seeks to show that alert-policy calibration is materially important to practical monitoring. For RQ3, the report seeks to identify what the current system can already demonstrate and what remains incomplete or environment-sensitive.

## 4. System Architecture

### 4.1 Architectural Overview

The implemented system is a full-stack fall-detection architecture composed of four main layers:

1. A pose and preprocessing layer that converts video or image streams into skeleton sequences and fixed-length temporal windows.
2. A model and calibration layer that trains TCN and GCN candidates, fits operating points, and defines deployable alert-policy profiles.
3. A backend inference and persistence layer implemented in FastAPI, responsible for online prediction, session-aware monitor logic, and event handling.
4. A frontend monitoring layer implemented in React, responsible for live and replay interaction, browser-side pose extraction, and operator-facing runtime control.

The codebase is organised around this architecture. Core model, training, evaluation, and deploy-time feature preparation live under `src/fall_detection`. The backend API lives under `server`, while the monitoring frontend lives under `apps`. Configuration and evidence artifacts are preserved under `configs`, `artifacts`, and `docs`. This separation is relevant to the report because it mirrors the way claims are distributed across modelling, policy, runtime, and deployment layers.

**Figure 1. System architecture and decision path**

Asset:
- [system_architecture_diagram.svg](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/artifacts/figures/report/system_architecture_diagram.svg)

Figure 1 should be interpreted as a responsibility diagram rather than as a low-level implementation trace. The important point is that pose extraction occurs on the client side, model inference and alert policy are applied on the backend side, and the final monitoring behaviour depends on the full decision path rather than on model output alone. The figure also clarifies why local and cloud deployment paths can share the same alert policy while still exhibiting different runtime characteristics.

### 4.2 Local On-Device Path Versus Cloud Deployment Path

The project’s primary framing is local/on-device monitoring. In this path, pose extraction occurs on the local client, and the monitoring interface can be run either with a lightweight local backend or with a fuller persistent system mode. The repository explicitly supports both a low-friction local demonstration path and a more complete Docker-backed persistent path.

Cloud deployment is treated as an extension rather than the primary scientific claim. In the deployment shape documented by the project, the frontend may be hosted separately from the backend, while the backend serves the inference and persistence path. This distinction matters because frontend pose extraction still depends on the user’s local browser and device even when the backend is remote. Several deployment findings later in the report arise directly from this split between local pose generation and remote inference.

### 4.3 Decision Flow

The system’s decision flow is more structured than a direct model call. Video or live frames are first converted into skeleton landmarks. These landmarks are then transformed into feature windows with a fixed temporal contract. The backend performs model inference using the active runtime profile, but the final monitor state is not simply the raw output probability. Instead, the runtime path combines the fitted operating-point profile with temporal policy components such as smoothing, `k/n` logic, cooldown, and optional confirmation behaviour. This architecture reflects the project’s central position that fall detection in deployment is an alerting problem rather than only a score-ranking problem.

### 4.4 Replay and Realtime as Distinct Runtime Paths

Replay and realtime must be treated as distinct runtime paths. Replay uses fixed stored clips and is useful for reproducible validation, profile comparison, and bounded deployment demonstration. Realtime depends more directly on browser-side pose extraction quality, camera behaviour, and runtime latency. Several engineering findings from the project show that these paths should not be conflated. In particular, replay inconsistencies were at times driven by frontend window-production rate or backend inference latency rather than by changes in the underlying model thresholds alone. This distinction becomes important later when interpreting deployment and runtime evidence.

### 4.5 Software Architecture and Refactoring Rationale

The final codebase reflects substantial refactoring intended to improve clarity, reduce coupling, and preserve traceability between report claims and implementation behaviour. On the frontend, feature-level API logic, monitoring transport, and prediction shaping were separated from page-level rendering code. On the backend, route handlers were thinned by moving data access and core logic into dedicated service and repository layers. These refactors are relevant to the report not as a software-engineering digression, but because they improved the interpretability and maintainability of the runtime path that underpins the deployment-oriented claims.

## 5. Data and Experimental Protocol

### 5.1 Dataset Roles

The project uses two benchmark datasets with deliberately different reporting roles. `CAUCAFall` is the primary benchmark and deployment-target dataset. It anchors the main comparative model claim and the deployment-facing operating-point profile. `LE2i` is retained as mandatory comparative evidence, but it does not determine the primary deployment claim. This asymmetry is intentional and reflects the current evidence base rather than an arbitrary reporting preference.

The repository also contains a genuine `MUVIM` experiment track. However, `MUVIM` is not used as a primary result-bearing dataset in this report. Instead, it is treated as a secondary exploratory track that exercised the training, operating-point, and metric-validation pipeline beyond the main `CAUCAFall` / `LE2i` protocol. This distinction matters because the `MUVIM` artifacts are useful supporting evidence for experimentation breadth and metric-contract discipline, but they should not blur the locked comparative story that underpins the main conclusions.

The project also contains replay clips, delivery-style replay packs, and a small field-validation pack. These artifacts are useful, but they occupy a different methodological layer from the benchmark datasets. Replay clips are used for deployment validation and runtime debugging. The field-validation pack is used only for bounded feasibility evidence. Neither source is allowed to replace formal offline benchmark evaluation.

### 5.2 Frozen Candidate Protocol

The report follows the frozen candidate policy defined in [FINAL_CANDIDATES.md](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/docs/project_targets/FINAL_CANDIDATES.md). The primary architecture comparison is therefore constrained to the following candidates:

**Table 1. Frozen candidate protocol summary**

| Dataset role | Dataset | Architecture | Frozen root | Report role |
| --- | --- | --- | --- | --- |
| Primary benchmark and deployment target | CAUCAFall | TCN | `outputs/caucafall_tcn_W48S12_r2_train_hneg` | Main comparative and deployment-facing candidate |
| Primary benchmark and deployment target | CAUCAFall | GCN | `outputs/caucafall_gcn_W48S12_r2_recallpush_b` | Matched comparison baseline |
| Comparative and generalisation evidence | LE2i | TCN | `outputs/le2i_tcn_W48S12_opt33_r2` | In-domain comparative candidate |
| Comparative and generalisation evidence | LE2i | GCN | `outputs/le2i_gcn_W48S12_opt33_r2` | In-domain comparative candidate |

- `CAUCAFall` TCN: `outputs/caucafall_tcn_W48S12_r2_train_hneg`
- `CAUCAFall` GCN: `outputs/caucafall_gcn_W48S12_r2_recallpush_b`
- `LE2i` TCN: `outputs/le2i_tcn_W48S12_opt33_r2`
- `LE2i` GCN: `outputs/le2i_gcn_W48S12_opt33_r2`

The frozen seed set is:

- `1337`
- `17`
- `2025`
- `33724876`
- `42`

This freeze matters because it prevents the report from drifting toward post hoc checkpoint selection or ad hoc single-run winners. Any comparative statement in the results chapter must remain traceable to this protocol and to the corresponding stability and significance artifacts.

### 5.3 Pose Preprocessing and Temporal Window Contract

The model families operate on skeleton sequences rather than raw RGB frames. The preprocessing path converts pose landmarks into fixed-size temporal windows with a locked window size and stride. The cross-dataset protocol explicitly states the invariant `W=48` and `S=12`, and the same contract is retained in the primary benchmark path. This matters because the project’s results are not only functions of model architecture, but also functions of a stable windowing and feature-generation interface. If the window contract changes, both offline comparison and deployment behaviour become difficult to interpret.

The report therefore treats the pose-window contract as part of the formal method rather than as an implementation detail. In deployment, the same temporal contract also links browser-side pose extraction to backend inference. This is one reason why frontend skeleton quality and runtime window production later appear in the deployment discussion.

### 5.4 Evaluation Policy

Four policy rules constrain the formal evaluation:

1. Operating-point fitting is performed on validation data only.
2. Test data are not used to tune thresholds or alert-policy parameters.
3. Cross-dataset evaluation fits operating points on the source validation split, not on the target test split.
4. Replay or deployment-specific tuning is not permitted to retroactively modify the formal offline claim.

These rules are stricter than a pure “best metrics wins” workflow, but they are necessary if the report is to preserve a defensible separation between model evidence and system evidence.

## 6. Model Design

### 6.1 Temporal Convolutional Network

The TCN acts as the primary temporal sequence model and the main deployment-facing candidate. Conceptually, it treats each pose window as a structured time series and learns discriminative temporal patterns through convolutional processing over fixed windows. This family is attractive in the present project because it offers a relatively direct path from windowed skeleton features to stable backend inference, which proved useful when the system later had to support practical alerting and latency-constrained runtime paths.

The final frozen `CAUCAFall` TCN candidate is the most important model instance in the report because it also underlies the deployable runtime profile. The TCN is therefore not only an offline benchmark candidate, but also the operationally relevant architecture for the monitoring system.

### 6.2 Custom Spatio-Temporal GCN

The GCN baseline must be described carefully. It is not a strict, official reproduction of a single canonical `ST-GCN` release. Rather, it is a custom spatio-temporal graph-based skeleton model used as a matched comparison architecture within the project. The methodological value of this model is therefore comparative rather than archival: it allows the report to ask whether a graph-structured skeleton model offers advantages over the matched temporal-convolutional alternative under the same frozen protocol.

This wording matters for methodological honesty. The report should not imply a benchmark against a fully standardised `ST-GCN` implementation if the actual code is a project-specific graph baseline.

### 6.3 Feature Parity and Fair Comparison

The architecture comparison is meaningful only if the models are trained and evaluated under aligned preprocessing and policy conditions. The project therefore treats the TCN-GCN comparison as a controlled protocol question rather than a broad architectural popularity contest. The evidence later shows directional advantage for the TCN under the frozen conditions, but that conclusion remains bounded by the specific feature contract, seed budget, and operating-point workflow used here.

## 7. Calibration and Alert Policy

### 7.1 Why Calibration Matters Here

In this project, the model does not directly emit the final monitoring decision. Instead, it emits window-level outputs that must be converted into alert states suitable for a monitoring interface. This conversion is where calibration and operating-point fitting become central. A raw score can rank windows reasonably well while still producing unstable or operationally poor alert behaviour if it is used without smoothing, temporal aggregation, or false-alarm control.

### 7.2 Validation-Side Temperature Scaling

The repository contains an explicit temperature-scaling implementation in `src/fall_detection/core/calibration.py`, and the operating-point fitting path in `src/fall_detection/evaluation/fit_ops.py` uses that logic during validation-side fitting. This is an important methodological point. Calibration is part of the operating-point fitting pipeline, not merely a rhetorical label for threshold selection.

At the same time, the report must remain precise about what calibration means here. The safest statement is that temperature scaling is used during validation-side operating-point fitting, and that the resulting calibration metadata are stored alongside the fitted operating-point configuration. The report should not overstate this as a guarantee that deployment-time probabilities are always explicitly temperature-corrected at runtime unless the runtime code path demonstrably applies the same transformation online.

### 7.3 Operating Points as Deployable Policy Profiles

The project’s deployable alert path depends on fitted operating-point configurations rather than on a single universal threshold. The relevant configuration files, such as [tcn_caucafall.yaml](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/configs/ops/tcn_caucafall.yaml) and [gcn_caucafall.yaml](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/configs/ops/gcn_caucafall.yaml), store policy-level parameters including threshold values and temporal decision settings. In other words, the operational behaviour of the deployed monitor is defined by a fitted profile, not by an informal threshold chosen at the UI layer.

### 7.4 Alert Policy Beyond Single-Window Scores

The project’s alert layer includes more than thresholding. Smoothing, `k/n` logic, cooldown, and optional confirmation behaviour all contribute to the final monitoring state. This is why the report treats alerting as an operational decision problem rather than as a direct alias for classification accuracy. It also explains why deployment behaviour can change when runtime latency, window production rate, or replay-specific path differences change, even if the underlying model checkpoint remains the same.

## 8. Implementation and Refactoring

### 8.1 Frontend Responsibilities

The frontend is responsible for user interaction, browser-side pose extraction, replay/live source management, and presentation of monitor state. During the project, the frontend accumulated substantial complexity because pose extraction, transport logic, session handling, and UI rendering were initially entangled. The final architecture refactors this responsibility split so that feature-level API logic and monitor-specific transport behaviour are separated from page components and presentation logic. This matters because the monitor path is both user-facing and methodologically important: it is the place where replay and live evidence are actually produced.

### 8.2 Backend Responsibilities

The backend is responsible for inference, policy application, session-aware monitor behaviour, persistence, and system-level integrations. Earlier route handlers contained a large amount of mixed responsibility, including request parsing, database access, policy logic, and response construction. The refactored backend introduces clearer separation between routes, services, and repositories so that business logic and data-access concerns are easier to trace. This is especially relevant in the monitor path, where alert policy, runtime defaults, session state, and persistence all need to remain interpretable if deployment findings are to be reported credibly.

### 8.3 Why Refactoring Matters to the Report

This refactoring work is relevant because the report is not only about static model artifacts. It is also about a monitoring system whose deployment claims depend on the runtime path being understandable and reproducible. The architectural cleanup therefore supports the report indirectly by reducing hidden coupling, making transport and policy boundaries more explicit, and improving confidence that the runtime behaviour analysed in the deployment section corresponds to a coherent code path rather than to a patchwork of ad hoc logic.

### 8.4 Reproducibility and Artifact Discipline

The implementation also reflects an explicit artifact discipline. Candidate checkpoints, operating-point files, summary reports, figures, and deployment notes are tracked in a way that allows report claims to be mapped back to concrete files. This does not make the project fully audit-proof in the strictest reproducibility sense, but it substantially improves traceability. In practice, this means that major report claims can be linked to frozen candidate roots, tracked metrics summaries, configuration files, and deployment validation notes rather than to undocumented development memory.

## 9. Results

### 9.1 Offline Comparative Results on the Frozen Protocol

The primary offline comparison is summarised by the frozen five-seed stability results in [stability_summary.csv](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/artifacts/reports/stability_summary.csv). On the primary `CAUCAFall` dataset, the TCN shows higher mean performance than the matched GCN across the main tracked metrics:

**Figure 3. Offline stability comparison across the frozen protocol**

Asset:
- [offline_stability_comparison.png](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/artifacts/figures/report/offline_stability_comparison.png)

**Table 2. Main offline comparative results under the frozen five-seed protocol**

| Dataset | Model | AP mean | F1 mean | Recall mean | Precision mean | FA24h mean |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| CAUCAFall | TCN | 0.9819 | 0.8611 | 0.7600 | 1.0000 | 0.0000 |
| CAUCAFall | GCN | 0.9706 | 0.5873 | 0.4400 | 1.0000 | 0.0000 |
| LE2i | TCN | 0.8389 | 0.8235 | 0.7778 | 0.8750 | 581.5843 |
| LE2i | GCN | 0.7471 | 0.7500 | 0.6667 | 0.8571 | 581.5843 |

This is the strongest directional model-comparison evidence currently available in the repository. It is also more informative than a single best-run comparison because it incorporates run-to-run variability under the frozen seed set. Figure 3 complements Table 2 by showing the relative separation of the frozen candidates across the report's main offline metrics. On `CAUCAFall`, the TCN not only attains higher mean `F1`, `Recall`, and `AP`, but also exhibits substantially narrower dispersion than the GCN for the same primary event metrics. This improves confidence that the observed directional advantage is not merely an artifact of one unusually favourable run.

However, the statistical interpretation must remain cautious. The paired five-seed significance analysis in [SIGNIFICANCE_REPORT.md](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/docs/project_targets/SIGNIFICANCE_REPORT.md) shows that the Wilcoxon signed-rank test does not cross the conventional `0.05` threshold for the primary metrics under the current `n=5` seed budget. On `CAUCAFall`, the TCN-GCN mean differences favour the TCN for `F1`, `Recall`, and `AP`, but Wilcoxon yields `p=0.125` for `F1` and `Recall` and `p=0.0625` for `AP`.

**Table 3. Statistical caution summary for the primary architecture comparison**

| Dataset | Metric | Mean difference (TCN - GCN) | Wilcoxon p | Interpretation |
| --- | --- | ---: | ---: | --- |
| CAUCAFall | F1 | 0.2738 | 0.1250 | Direction favours TCN; not formally significant at `alpha=0.05` |
| CAUCAFall | Recall | 0.3200 | 0.1250 | Direction favours TCN; not formally significant at `alpha=0.05` |
| CAUCAFall | AP | 0.0113 | 0.0625 | Borderline directional evidence; still above the primary cutoff |
| LE2i | F1 | 0.0735 | 0.0625 | Direction favours TCN; exploratory only |
| LE2i | Recall | 0.1111 | 0.0625 | Direction favours TCN; exploratory only |
| LE2i | AP | 0.0918 | 0.0625 | Direction favours TCN; exploratory only |

The most defensible interpretation is therefore directional rather than definitive: under the frozen protocol, the TCN trends stronger than the matched custom GCN, but the current statistical evidence should not be framed as a conclusive proof of superiority.

### 9.2 Cross-Dataset Results and Generalisation Boundary

Cross-dataset evaluation is included to bound generalisation rather than to prove universal robustness. The summary file [cross_dataset_summary.csv](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/artifacts/reports/cross_dataset_summary.csv) shows a strongly asymmetric transfer pattern.

**Figure 4. Cross-dataset transfer summary**

Asset:
- [cross_dataset_transfer_summary.png](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/artifacts/figures/report/cross_dataset_transfer_summary.png)

**Table 4. Cross-dataset transfer summary**

| Transfer direction | Model | In-domain AP | Cross-domain AP | Delta AP | In-domain F1 | Cross-domain F1 | Delta F1 | In-domain Recall | Cross-domain Recall | Delta Recall |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| LE2i -> CAUCAFall | TCN | 0.8541 | 0.8873 | +0.0333 | 0.8235 | 0.8889 | +0.0654 | 0.7778 | 0.8000 | +0.0222 |
| CAUCAFall -> LE2i | TCN | 0.9819 | 0.5797 | -0.4022 | 0.8889 | 0.0000 | -0.8889 | 0.8000 | 0.0000 | -0.8000 |
| LE2i -> CAUCAFall | GCN | 0.7523 | 0.8265 | +0.0741 | 0.7500 | 0.8889 | +0.1389 | 0.6667 | 0.8000 | +0.1333 |
| CAUCAFall -> LE2i | GCN | 0.9812 | 0.4687 | -0.5125 | 0.5714 | 0.0000 | -0.5714 | 0.4000 | 0.0000 | -0.4000 |

When moving from `CAUCAFall` to `LE2i`, both architectures collapse sharply. The TCN drops from in-domain `AP=0.9819` to cross-domain `AP=0.5797`, and from `F1=0.8889` to `F1=0.0`; the GCN similarly drops from in-domain `AP=0.9812` to `0.4687` and from `F1=0.5714` to `0.0`. The false-alert profile also worsens in that direction. In contrast, the `LE2i -> CAUCAFall` direction appears numerically less damaging and in some metrics even superficially improves. Figure 4 makes the asymmetry visible at a glance by plotting transfer deltas rather than raw in-domain scores alone. This asymmetry should not be over-read as broad cross-domain robustness. Instead, it indicates that transfer behaviour is domain-sensitive and that cross-dataset performance is highly contingent on dataset characteristics, motion statistics, and event definitions.

The correct conclusion is therefore not that the models generalise well across datasets, but that cross-dataset transfer exposes a strong limitation boundary. This is exactly the kind of bounded evidence that belongs in the discussion and limitations sections rather than in an aggressive model-success claim.

### 9.3 Secondary MUVIM Exploratory Track

Although `MUVIM` is not part of the primary locked result narrative, the project did carry out substantial work on that dataset. The corrected summaries in [muvim_r2_summary.md](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/artifacts/reports/tuning/muvim_r2_summary.md) and [muvim_r3_summary.md](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/artifacts/reports/tuning/muvim_r3_summary.md) show that `MUVIM` was used as a secondary track for operating-point refitting and training-side exploration after a metric-contract issue had been identified and corrected.

The most useful report-level interpretation of the `MUVIM` work is methodological rather than headline comparative. First, it shows that the project extended beyond the final two-dataset comparative core. Second, it provides a concrete example of why metric-contract discipline matters: on `MUVIM`, correcting the event-metric contract materially changed how event-level results should be interpreted, and later operating-point refits improved alert-oriented event metrics without changing threshold-independent score quality. Third, it shows that not every additional experiment was promoted into the final claim set, which is consistent with the report's evidence-locking discipline. In that sense, `MUVIM` strengthens the credibility of the process more than it changes the main comparative answer.

For that reason, `MUVIM` is retained here as supporting experimental evidence rather than as a third co-equal benchmark. It strengthens the account of experimental breadth and pipeline maturity, but it does not alter the report's main comparative conclusion, which remains anchored on the locked `CAUCAFall` / `LE2i` protocol.

### 9.4 Calibration and Alert-Policy Results

The project’s second research question is not answered by benchmark discrimination metrics alone. It is addressed by the fact that the deployable monitor path is governed by fitted operating-point configurations rather than by a manually chosen threshold. The operating-point files for the primary candidates capture threshold and policy structure that are then consumed by the deployable runtime path. This means that alert behaviour is the result of a fitted policy profile rather than a simple post hoc threshold chosen after inspecting test outputs.

The most important result here is conceptual but still evidence-grounded: the project successfully operationalises a calibration-aware alert layer. In other words, the system does not stop at “which model scores higher,” but proceeds to “how should a monitoring system convert model scores into actionable alert states?” That shift is important because practical fall-detection systems are judged by the behaviour of alerts rather than by score curves alone.

**Figure 2. Calibration-aware alert decision flow**

Asset:
- [alert_policy_flow.svg](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/artifacts/figures/report/alert_policy_flow.svg)

Figure 2 emphasises that the deployed alert is not the direct output of a single thresholded probability. It is the output of a fitted and temporally structured decision layer. This figure is therefore methodological rather than decorative: it explains why the project treats operating-point fitting and temporal policy as part of the main contribution.

### 9.5 Deployment and Runtime Results

Deployment evidence is strongest in the locked replay and delivery-style path. The deployment-lock validation report in [deployment_lock_validation.md](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/artifacts/reports/deployment_lock_validation.md) captures an earlier locked replay workflow, while the current online review preset has been standardised to `LE2I + TCN + OP-2` with a shared `k=2, n=3` temporal policy. The replay acceptance and deployment notes therefore remain useful as workflow evidence, but they should not be read as exact reproductions of the current review preset.

**Table 5. Deployment and runtime evidence summary**

| Evidence slice | Source artifact | Main result | Reporting role |
| --- | --- | --- | --- |
| Deployment lock validation | [deployment_lock_validation.md](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/artifacts/reports/deployment_lock_validation.md) | Locked replay workflow validated; replay spot checks passed | Confirms that the deployed runtime path is reviewable end to end |
| Bounded custom replay matrix with uncertainty-gate check | [online_mc_replay_matrix_20260402.csv](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/artifacts/reports/online_mc_replay_matrix_20260402.csv) | On the current fixed raw online replay path, MC-on produced no change across any of the `12` combinations | Supporting runtime-boundary evidence only |
| Field validation sample | [deployment_field_validation_summary.md](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/artifacts/reports/deployment_field_validation_summary.md) | `event_recall_proxy=1.0`, `event_precision_proxy=0.5`, `fa24h_estimate=24.0`, based on `n_clips=3` | Bounded feasibility evidence only |
| Runtime diagnostics | deployment/runtime logs and latency notes | Runtime behaviour depends on both frontend window production and backend inference latency | System discussion, not model-comparison evidence |

The bounded custom replay set remains useful, but it now functions primarily as a runtime-interpretation matrix rather than as a delivery-success package. The current fixed raw online replay matrix in [online_mc_replay_matrix_20260402.csv](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/artifacts/reports/online_mc_replay_matrix_20260402.csv) shows that enabling the uncertainty-aware live gate did not change any of the `12` model-dataset-operating-point combinations on the 24 labeled clips. This is still a useful negative result: it shows that the uncertainty path was genuinely exercised on boundary windows without yielding video-level gains on this bounded replay set. The replay matrix therefore supports cautious runtime interpretation, not a claim that the uncertainty mechanism improves deployment accuracy.

The bounded field-validation summary is weaker and should be reported as such. The current field summary in [deployment_field_validation_summary.md](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/artifacts/reports/deployment_field_validation_summary.md) is based on only three clips, with one true positive clip, one false alert, and one unknown case. The resulting `event_recall_proxy=1.0`, `event_precision_proxy=0.5`, and `fa24h_estimate=24.0` are therefore too data-limited to support a broad field-readiness claim. They are still useful as evidence that the system has entered a genuine end-to-end field-check stage rather than remaining purely synthetic, but they must remain bounded.

### 9.6 Runtime Interpretation

The runtime evidence collected during deployment work shows that monitoring behaviour depends not only on model weights and thresholds but also on the production rate of valid windows and the latency of the inference path. This is why replay and realtime are separated analytically throughout the report. In earlier deployment iterations, runtime discrepancies were traced to a combination of frontend pose-window production and backend inference latency rather than to a single threshold mismatch. This observation strengthens the report’s central methodological claim that deployment analysis belongs to the system layer, not to the pure model-comparison layer.

This point also clarifies why runtime improvement work, API-path stabilisation, and frontend monitor refactoring are relevant to the scientific interpretation of the project. They do not change the frozen offline evidence, but they do affect whether the deployment path is capable of expressing that evidence under practical runtime conditions.

## 10. Discussion

### 10.1 Answer to RQ1

RQ1 asked how the TCN and the custom GCN compare under the locked offline protocol. The evidence supports a cautious directional answer: the TCN trends stronger than the matched GCN on the primary `CAUCAFall` protocol and also remains stronger on the in-domain `LE2i` comparison. However, the current five-seed inferential budget does not justify a strong claim of definitive statistical superiority under the primary Wilcoxon analysis. The correct answer is therefore “directional advantage with bounded statistical certainty,” not “final proof that TCN is universally better.”

### 10.2 Answer to RQ2

RQ2 asked whether validation-side calibration and operating-point fitting materially affect practical alerting. The answer is yes. The repository’s deployable behaviour depends on fitted operating-point profiles, not on raw single-window probabilities alone. The use of temperature scaling during operating-point fitting, together with policy parameters such as smoothing, `k/n`, and cooldown, shows that the project’s alerting path is explicitly calibrated as an operational decision layer. This is an important systems contribution because it moves the project beyond a benchmark-only classifier framing and makes the final monitoring behaviour interpretable in deployment terms.

### 10.3 Answer to RQ3

RQ3 asked what replay deployment evidence and limited realtime validation reveal about practical feasibility and runtime limits. The answer is mixed but useful. Replay-oriented deployment evidence is strong enough to support a practical system claim in bounded controlled conditions, with the strongest bounded replay row observed for `caucafall_tcn OP-2`. However, field evidence remains small and runtime behaviour is still sensitive to path-specific latency and frontend pose quality. The system therefore demonstrates practical feasibility in controlled deployment settings, but not broad field closure.

### 10.4 Model Quality Versus System Quality

A key lesson of the project is that model quality and system quality are not interchangeable. A strong checkpoint can still yield poor alerting if policy selection is weak or runtime latency distorts the sequence of windows seen by the backend. Conversely, a carefully tuned deployment path may appear highly effective on replay evidence even though it does not justify a broader unseen-test claim. This distinction is one of the main reasons why the report insists on separating offline, replay, and limited realtime evidence.

### 10.5 Interpretation of the Project’s Strongest Contribution

The project’s strongest contribution is not a claim that one neural architecture definitively solves fall detection. The stronger contribution is that it integrates controlled model comparison, calibration-aware alert policy, and deployment/runtime analysis within one coherent pipeline. This is what makes the final system study more useful than a narrow benchmark note. It provides not only model evidence, but also a structured account of how benchmark outputs become operational decisions and where that process remains fragile. That combination of model evidence and runtime interpretation is the report's most defensible claim to originality and maturity.

### 10.6 Role of the MUVIM Track

The `MUVIM` work helps clarify the scope of the project without changing its primary conclusions. It shows that the project did not stop at a single benchmark pair, and it provides useful supporting evidence that the evaluation and operating-point pipeline was exercised on a broader experimental track. At the same time, the report deliberately does not elevate `MUVIM` into a main result axis, because doing so would weaken the evidence hierarchy and risk conflating exploratory work with the locked comparative protocol. This is the correct trade-off: acknowledge the work, preserve it in the narrative, but keep the main claims tied to the strongest and cleanest evidence.

## 11. Limitations

The first major limitation is statistical. The primary architecture comparison currently relies on a frozen five-seed comparison. This is enough to support a directional interpretation, but not enough to justify a strong non-parametric significance claim for the main event metrics.

The second major limitation is generalisation. Cross-dataset evidence is clearly asymmetric and does not support a claim of universal robustness across domains, cameras, or motion statistics.

The project also includes a secondary `MUVIM` experiment track, but that work is not integrated into the main comparative claim because its strongest value is exploratory and methodological rather than central to the final locked evidence hierarchy.

The third limitation is measurement quality. Because the deployed system depends on browser-side or frontend pose extraction, degradation in skeleton quality can propagate directly into inference and alerting behaviour. This is a fundamental systems issue rather than a small implementation nuisance.

The fourth limitation is deployment closure. Replay-oriented deployment evidence is much stronger than field or live evidence. The current field-validation pack is too small to support broad real-world conclusions, and realtime results remain bounded by environment and runtime constraints.

The fifth limitation is interpretive. Some replay or delivery-style validation artifacts are the result of deployment-aware tuning and path stabilisation. They are valuable as engineering evidence, but they cannot be promoted into the same evidential status as locked unseen-test benchmark results.

Taken together, these limitations imply that the correct scientific stance is bounded confidence rather than maximalist claim-making. The project demonstrates a serious and coherent system study, but it does not yet justify claims of universal real-world robustness, clinical readiness, or definitive architecture superiority.

## 12. Future Work

Future work should proceed along four directions.

1. Expand the live and field-validation protocol so that deployment claims can be supported by a larger and more varied runtime evidence base.
2. Improve domain robustness across datasets, camera setups, and pose-quality conditions rather than treating in-domain performance as sufficient.
3. Rework the uncertainty path so that it is either validated under a stronger runtime protocol or simplified further, since the current bounded replay matrix does not show deployment gains from the uncertainty-aware live gate.
4. Continue simplifying the runtime architecture, especially in the live monitor path, so that the deployment system remains maintainable as evaluation complexity increases.

## 13. Conclusion

This project delivers a deployment-oriented study of pose-based fall detection rather than a narrow benchmark exercise. Its main contribution is not only the construction of TCN and custom GCN candidates, but also the integration of preprocessing, operating-point fitting, alert-policy logic, runtime inference, and monitoring-oriented deployment analysis into one coherent system.

Under the locked primary-dataset protocol, the TCN trends stronger than the matched custom GCN, although the current five-seed inferential evidence supports only a cautious comparative claim. Validation-side operating-point calibration is a substantive part of the system because alert behaviour depends on fitted policy profiles rather than on raw classifier scores alone. Replay-oriented deployment evidence shows that the system can operate effectively in bounded runtime conditions, while also showing that optional uncertainty-aware live inference did not improve the current fixed raw online replay matrix. Cross-dataset and field evidence make clear that generalisation and real-world closure remain incomplete.

The most defensible final interpretation is therefore that the project successfully demonstrates a serious, end-to-end, deployment-aware fall-detection system with explicit model, policy, and runtime analysis, while also making its current boundaries clear. It also shows broader experimentation, including the secondary `MUVIM` track, without collapsing exploratory work into the main claim set. That is a stronger and more credible contribution than an overstated claim of solved real-world fall detection.

## 14. Appendices and Supporting Artifacts

The final submitted report should include appendices that make the evidence trail easier to audit without overloading the main text. The most useful appendix items are likely to be:

1. Frozen candidate roots and seed list, derived from [FINAL_CANDIDATES.md](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/docs/project_targets/FINAL_CANDIDATES.md).
2. Additional operating-point details drawn from the active `configs/ops/*.yaml` files for the reported candidates.
3. Supplementary significance and stability summaries that are too detailed for the main narrative.
4. Deployment and replay validation notes, especially where they clarify the boundary between bounded system evidence and formal model evidence.
5. Supporting notes for the secondary `MUVIM` track, retained explicitly as exploratory material rather than as core comparative evidence.

These appendices should support the main report, not replace it. Their role is to improve auditability, artifact traceability, and methodological transparency.
