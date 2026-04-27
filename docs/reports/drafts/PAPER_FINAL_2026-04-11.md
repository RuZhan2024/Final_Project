Draft date: 2026-04-09  
Type: compact paper/manuscript draft  
Status: active supervisor-review paper draft after evidence refreeze  
Evidence control: this draft is constrained by [PAPER_WRITING_MASTER_PLAN_2026-04-09.md](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/docs/reports/notes/PAPER_WRITING_MASTER_PLAN_2026-04-09.md) and [WEEK1_RESULTS_AND_FRAMING_LOCK_2026-04-27.md](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/docs/reports/notes/WEEK1_RESULTS_AND_FRAMING_LOCK_2026-04-27.md).

# Abstract

Fall detection is often reported as a classification task, but practical monitoring depends on more than classifier accuracy alone. This paper studies an end-to-end pose-based fall-detection stack in which offline comparison, operating-point fitting, runtime interpretation, and caregiver-facing delivery are treated as one connected artifact. The system combines browser-side pose extraction, temporal window inference, validation-fitted alert profiles, replay-oriented monitoring, persistent event history, and Telegram-first alert delivery. Under a locked offline protocol, the Temporal Convolutional Network (TCN) trends stronger than the matched custom spatio-temporal Graph Convolutional Network (GCN) on the primary `CAUCAFall` dataset, while cross-dataset transfer remains asymmetric and bounded. Later bounded retraining improves the defended custom replay surfaces modestly, but not enough to support a solved deployment claim. The most defensible contribution is therefore a deployment-oriented systems study showing how pose-based monitoring can be made operationally coherent under explicit evidence boundaries.

# Introduction

## Background and Motivation

Falls are clinically and socially significant because delayed detection can increase both injury severity and time to intervention. Automated fall detection has therefore been studied across wearable, ambient, and vision-based paradigms. Yet practical usefulness depends on more than recognition accuracy alone. A monitoring system must also control false alarms, remain tractable at runtime, and expose decisions that stay meaningful under deployment constraints.

This paper therefore treats fall detection as a deployment-oriented sequence-modelling problem rather than as a narrow benchmark exercise. Instead of predicting falls from isolated frames, the system models short pose sequences and converts window-level outputs into operational alerts through an explicit policy layer. The central question is not only which model scores highest offline, but whether a pose-based monitoring stack remains coherent once inference, alert policy, persistence, and runtime behaviour are treated as one integrated path.

## Why Pose-Based Detection

The project focuses on pose-based vision rather than raw RGB classification or wearable-only sensing for three reasons. First, skeletal pose provides a compact and relatively interpretable motion signal suited to temporal modelling and downstream alert logic. Second, pose-based processing reduces dependence on scene appearance compared with direct RGB modelling while preserving body-configuration and motion cues relevant to falls. Third, a pose-based frontend fits the project’s local/on-device monitoring framing, because browser-side processing can extract landmarks locally and send structured temporal windows to a backend inference path.

This design does not remove deployment difficulty. Pose quality remains sensitive to occlusion, camera placement, motion blur, and frontend runtime behaviour. These limitations are central rather than incidental because downstream inference can only be as reliable as the skeleton sequence presented to the backend. For that reason, pose quality and runtime stability are treated later as explicit parts of the system analysis rather than hidden behind aggregate benchmark scores.

## Project Framing

The paper is best understood as a fall-detection systems study with three linked layers:

1. A controlled model-comparison layer, focused on temporal and graph-based sequence models.
2. An alert-policy layer, focused on converting window-level model outputs into practical fall alerts through operating-point calibration and temporal decision rules.
3. A deployment/runtime layer, focused on replay validation, latency behaviour, frontend pose quality, and bounded realtime feasibility.

Keeping these layers separate is essential. Offline model evidence cannot be replaced by demo behaviour, and replay/deployment results cannot be treated as equivalent to unseen-test generalisation. A central aim of the manuscript is therefore methodological discipline: each claim is tied to the evidence layer that actually supports it.

## Problem Statement

The core problem addressed here is not simply whether a classifier can separate fall windows from non-fall windows. The stronger practical question is whether a pose-based temporal system can support credible alerting under controlled evaluation and bounded deployment conditions. This requires offline comparison, calibration-aware alerting, and deployment analysis to be integrated into one coherent study without allowing one evidence layer to stand in for another.

## Contributions of This Project

The manuscript makes three bounded contributions:

1. It presents a deployment-oriented pose-based monitoring system that links preprocessing, temporal inference, operating-point fitting, replay/live monitoring, persistent event handling, and caregiver-facing delivery within one audited pipeline.
2. It reports a controlled TCN-versus-custom-GCN comparison under a locked primary-dataset protocol, interpreted cautiously rather than as a universal architectural ranking.
3. It shows that alerting must be treated as a fitted policy layer, and that bounded replay/runtime analysis is necessary to distinguish operational coherence from broader deployment success.

## Paper Structure

The remainder of the paper places the work in context, defines the locked protocol and system architecture, presents the offline, policy, and deployment-facing results, and then closes with bounded interpretation, limitations, and future work.

# Background and Related Work

## Fall Detection as a Multi-Layer Problem

Fall detection is often presented as a pattern-recognition problem: given sensor observations, determine whether a fall has occurred. In practice, however, the problem is layered. A model may rank fall-like windows effectively while still producing an operationally weak monitoring system if false alarms remain high, decisions are unstable across time, or runtime delays prevent timely alerting. This distinction is central to the present project because the final system is evaluated not only as a classifier, but also as an alerting pipeline with temporal policy, persistence, and deployment constraints.

## Vision-Based and Pose-Based Approaches

Vision-based fall detection can operate on raw RGB frames, optical flow, depth, or derived pose representations. Direct RGB models may capture rich scene context, but they also inherit strong dependence on appearance, lighting, and environment-specific texture. Pose-based methods instead compress the scene into an articulated body representation. This makes them attractive for action-like reasoning, temporal modelling, and partial privacy preservation relative to full-frame storage. At the same time, it makes them especially sensitive to pose extraction quality, missing landmarks, and frontend runtime effects. These trade-offs are central to this project because the deployed monitor pipeline depends on browser-side pose extraction before backend inference can occur, and several of the project’s runtime findings arise exactly at that interface.

## Temporal and Graph Sequence Models

Skeleton sequences can be modelled in multiple ways. Temporal convolutional models treat the pose sequence as a structured temporal signal and learn discriminative motion patterns through convolution over time. Graph-based models preserve joint topology more explicitly and are therefore attractive when reasoning over articulated motion. Rather than assuming one family is inherently superior, this project treats the comparison as an empirical question under a controlled protocol. The TCN and the custom spatio-temporal GCN are therefore evaluated under matched preprocessing, feature construction, and operating-point rules, so that the final comparison reflects architecture behaviour under one locked contract rather than under loosely comparable training setups.

## Calibration, Thresholding, and Operational Alerting

A major limitation of many fall-detection discussions is that they stop at classifier outputs. In deployment, an alert is rarely the direct consequence of a single probability threshold. Temporal smoothing, multi-window confirmation, cooldown logic, and false-alarm control are often needed to convert noisy scores into meaningful alerts. This paper treats that conversion process explicitly. Validation-side temperature scaling and operating-point fitting are used to shape the alert-policy layer, and the manuscript therefore distinguishes model discrimination quality from deployment-facing alert behaviour. This distinction is especially important here because the same backend path must support both bounded replay validation and limited live monitoring without allowing replay-oriented interpretation to overwrite the formal offline result layer.

## Deployment-Oriented Evaluation

Benchmark performance alone is not sufficient to justify a monitoring claim. Runtime latency, frontend pose stability, replay-vs-live pipeline differences, and the availability of bounded field evidence all affect what can honestly be claimed about practical usefulness. For this reason, the project includes replay-oriented validation, persistent event review, and deployment analysis in addition to offline evaluation. At the same time, these deployment results are not treated as substitutes for formal unseen-test evidence. Maintaining that separation is one of the paper’s central methodological commitments.

# Research Questions and Scope

## Locked Research Questions

The paper is organised around three research questions selected to align with the strongest evidence currently available in the repository while preserving strict separation between model comparison, alert-policy design, and deployment/runtime analysis. These questions are intentionally narrower than a generic “does the system solve fall detection?” framing, because the current evidence base is strongest when interpreted through bounded comparative, policy, and runtime questions.

**RQ1. Comparative Offline Performance.** Under the locked offline evaluation protocol, how do the TCN and the custom spatio-temporal GCN compare on the primary fall-detection task?

**RQ2. Calibration and Operational Alerting.** How does validation-side operating-point calibration influence the conversion of window-level model outputs into practical alert decisions?

**RQ3. Deployment Feasibility and Runtime Limits.** What do replay deployment evidence and limited realtime validation show about the practical feasibility and current runtime limits of the system?

## Scope Boundaries

The project scope is intentionally bounded in the following ways.

First, the primary result-bearing dataset is `CAUCAFall`, while `LE2i` is treated as comparative and generalisation evidence. The main technical story is therefore anchored on the locked primary-dataset protocol rather than on a pooled multi-dataset claim.

Second, replay validation and limited live validation are treated as system-level evidence, not as replacements for formal offline model evaluation. Replay-oriented tuning or deployment-specific adjustments, if discussed, must therefore be framed as deployment or demonstration calibration only.

Third, the secondary `MUVIM` track is acknowledged as real project work, but it is not promoted into the primary comparative claim. Its role is to support methodological discussion about experimentation breadth, operating-point fitting, and metric-contract interpretation.

Fourth, calibration is used in the operating-point fitting pipeline and must be described precisely in that role. The paper should not imply that all deployment-time probabilities are explicitly temperature-calibrated at runtime unless the runtime path demonstrably does so. The safe statement is that validation-side calibration informs the fitted operating-point profiles consumed by deployment.

Fifth, realtime evidence remains bounded. The paper may discuss feasibility and runtime behaviour, but it should not claim broad real-world deployment closure or clinical validation. The same caution applies to the Telegram-first notification path: it is valid system evidence that end-to-end alert delivery works, but it does not strengthen model-comparison claims by itself.

## Expected Answers

The expected shape of the answers is bounded in advance.

For RQ1, the paper seeks a controlled and cautious comparative conclusion rather than a universal architectural ranking. For RQ2, the paper seeks to show that alert-policy calibration is materially important to practical monitoring. For RQ3, the paper seeks to identify what the current system can already demonstrate and what remains incomplete or environment-sensitive. In practice, that means the final answers should read as bounded evidence statements, not as maximal deployment claims.

# System Architecture

## Architectural Overview

The implemented system is a full-stack fall-detection architecture composed of four main layers:

1. A pose and preprocessing layer that converts video or image streams into skeleton sequences and fixed-length temporal windows.
2. A model and calibration layer that trains TCN and GCN candidates, fits operating points, and defines deployable alert-policy profiles.
3. A backend inference and persistence layer implemented in FastAPI, responsible for online prediction, session-aware monitor logic, and event handling.
4. A frontend monitoring layer implemented in React, responsible for live and replay interaction, browser-side pose extraction, and operator-facing runtime control.

The codebase is organised around this architecture. Core model, training, evaluation, and deploy-time feature preparation live under `ml/src/fall_detection`. The backend API lives under `applications/backend`, while the monitoring frontend lives under `applications/frontend`. Configuration and operational scripts are preserved under `ops/configs` and `ops/scripts`, while evidence and supporting documentation are preserved under `artifacts` and `docs`. This separation is relevant to the paper because it mirrors the way claims are distributed across modelling, policy, runtime, and deployment layers.

In the final implementation, the backend does more than expose a single prediction endpoint. It owns the active runtime profile, resolves the fitted operating-point configuration, persists fall events, and dispatches the current Telegram-first notification path with generated summary text. The frontend, by contrast, is responsible for obtaining pose landmarks, assembling temporal windows, and presenting the resulting monitor state to the operator. This split matters because several runtime findings later in the paper arise from the contract between browser-side window production and backend-side policy evaluation rather than from model weights alone.

**Figure 1. System architecture and decision path**

Asset:
- [system_architecture_diagram.svg](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/artifacts/figures/report/system_architecture_diagram.svg)

![Figure 1. System architecture and decision path](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/artifacts/figures/report/system_architecture_diagram.svg){ width=85% }

Figure 1 should be interpreted as a responsibility diagram rather than as a low-level implementation trace. The important point is that pose extraction occurs on the client side, model inference and alert policy are applied on the backend side, and the final monitoring behaviour depends on the full decision path rather than on model output alone. The figure also clarifies why local and cloud deployment paths can share the same alert policy while still exhibiting different runtime characteristics.

## Local On-Device Path Versus Cloud Deployment Path

The project’s primary framing is local/on-device monitoring. In this path, pose extraction occurs on the local client, and the monitoring interface can be run either with a lightweight local backend or with a fuller persistent system mode. The repository explicitly supports both a low-friction local demonstration path and a more complete Docker-backed persistent path.

Cloud deployment is treated as an extension rather than the primary scientific claim. In the deployment shape documented by the project, the frontend may be hosted separately from the backend, while the backend serves inference, persistence, and notification delivery. This distinction matters because frontend pose extraction still depends on the user’s local browser and device even when the backend is remote. Several deployment findings later in the paper arise directly from this split between local pose generation and remote inference, especially when pose-window production rate and backend processing latency move out of step.

## Decision Flow

The system’s decision flow is more structured than a direct model call. Video or live frames are first converted into skeleton landmarks. These landmarks are then transformed into feature windows with a fixed temporal contract. The backend performs model inference using the active runtime profile, but the final monitor state is not simply the raw output probability. Instead, the runtime path combines the fitted operating-point profile with temporal policy components such as smoothing, `k/n` logic, cooldown, and optional confirmation behaviour. When the resulting state is persisted as a fall event, that event then becomes the source for dashboard summaries, event history, and the current Telegram-first caregiver notification path. This architecture reflects the project’s central position that fall detection in deployment is an alerting problem rather than only a score-ranking problem.

## Replay and Realtime as Distinct Runtime Paths

Replay and realtime must be treated as distinct runtime paths. Replay uses fixed stored clips and is useful for reproducible validation, profile comparison, and bounded deployment demonstration. Realtime depends more directly on browser-side pose extraction quality, camera behaviour, and runtime latency. Several engineering findings from the project show that these paths should not be conflated. In particular, replay inconsistencies were at times driven by frontend window-production rate or backend inference latency rather than by changes in the underlying model thresholds alone. The current interface also treats replay persistence explicitly rather than implicitly: replay can be used either as a visual/runtime inspection path or as an event-producing path when event persistence is deliberately enabled. This distinction becomes important later when interpreting deployment and runtime evidence.

## Software Architecture and Refactoring Rationale

The final codebase reflects substantial refactoring intended to improve clarity, reduce coupling, and preserve traceability between manuscript claims and implementation behaviour. On the frontend, feature-level API logic, monitoring transport, and prediction shaping were separated from page-level rendering code. On the backend, route handlers were thinned by moving data access and core logic into dedicated service and repository layers. These refactors are relevant to the paper not as a software-engineering digression, but because they improved the interpretability and maintainability of the runtime path that underpins the deployment-oriented claims.

# Data and Experimental Protocol

## Dataset Roles

The project uses two benchmark datasets with deliberately different reporting roles. `CAUCAFall` is the primary benchmark and deployment-target dataset. It anchors the main comparative model claim and the deployment-facing operating-point profile. `LE2i` is retained as mandatory comparative evidence, but it does not determine the primary deployment claim. This asymmetry is intentional and reflects the current evidence base rather than an arbitrary reporting preference.

The repository also contains a genuine `MUVIM` experiment track. However, `MUVIM` is not used as a primary result-bearing dataset in this paper. Instead, it is treated as a secondary exploratory track that exercised the training, operating-point, and metric-validation pipeline beyond the main `CAUCAFall` / `LE2i` protocol. This distinction matters because the `MUVIM` artifacts are useful supporting evidence for experimentation breadth and metric-contract discipline, but they should not blur the locked comparative story that underpins the main conclusions.

The project also contains replay clips, delivery-style replay packs, and a small field-validation pack. These artifacts are useful, but they occupy a different methodological layer from the benchmark datasets. Replay clips are used for deployment validation and runtime debugging. The field-validation pack is used only for bounded feasibility evidence. Neither source is allowed to replace formal offline benchmark evaluation.

The three datasets also operate under different nominal frame-rate contracts in the repository: `CAUCAFall` is treated as `23 FPS`, `LE2i` as `25 FPS`, and `MUVIM` as `30 FPS`. This matters because raw label spans, pose windows, and deploy-time timing metadata all depend on these contracts. Treating dataset identity and frame rate consistently is therefore part of the formal protocol rather than a minor preprocessing detail.

## Frozen Candidate Protocol

The paper follows the frozen candidate policy recorded in the current Week 1 framing-and-results lock and in the locked output roots preserved on this branch. The primary architecture comparison is therefore constrained to the following candidates:

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

This freeze matters because it prevents the paper from drifting toward post hoc checkpoint selection or ad hoc single-run winners. Any comparative statement in the results chapter must remain traceable to this protocol and to the corresponding stability artifacts.

## Pose Preprocessing and Temporal Window Contract

The model families operate on skeleton sequences rather than raw RGB frames. The preprocessing path converts pose landmarks into fixed-size temporal windows with a locked window size and stride. The cross-dataset protocol explicitly states the invariant `W=48` and `S=12`, and the same contract is retained in the primary benchmark path. Within each exported window, start and end positions are treated as frame-index metadata on the target FPS grid rather than as free-form timestamps. This matters because the project’s results are not only functions of model architecture, but also functions of a stable windowing and feature-generation interface. If the window contract changes, both offline comparison and deployment behaviour become difficult to interpret.

The paper therefore treats the pose-window contract as part of the formal method rather than as an implementation detail. In deployment, the same temporal contract also links browser-side pose extraction to backend inference. This is one reason why frontend skeleton quality and runtime window production later appear in the deployment discussion. It is also why the code-review phase treated window metadata semantics, recursive window discovery, and dataset-specific FPS assumptions as result-relevant correctness issues rather than as cosmetic implementation cleanups.

## Evaluation Policy

Four policy rules constrain the formal evaluation:

1. Operating-point fitting is performed on validation data only.
2. Test data are not used to tune thresholds or alert-policy parameters.
3. Cross-dataset evaluation fits operating points on the source validation split, not on the target test split.
4. Replay or deployment-specific tuning is not permitted to retroactively modify the formal offline claim.

These rules are stricter than a pure “best metrics wins” workflow, but they are necessary if the manuscript is to preserve a defensible separation between model evidence and system evidence. They also imply that some artifacts in the repository must remain explicitly secondary: replay matrices, delivery-style custom checks, and field-validation notes are informative for deployment interpretation, but they do not outrank frozen unseen-test summaries when the paper answers RQ1.

# Model Design

## Temporal Convolutional Network

The TCN acts as the primary temporal sequence model and the main deployment-facing candidate. Conceptually, it treats each pose window as a structured time series and learns discriminative temporal patterns through convolutional processing over fixed windows. This family is attractive in the present project because it offers a relatively direct path from windowed skeleton features to stable backend inference, which proved useful when the system later had to support practical alerting and latency-constrained runtime paths.

The final frozen `CAUCAFall` TCN candidate is the most important model instance in the paper because it also underlies the deployable runtime profile. In the current system state, the preferred live demo preset remains `CAUCAFall + TCN + OP-2`. That choice should be interpreted as a defended demonstration preset rather than as proof of a uniformly strongest replay profile. The TCN is therefore not only an offline benchmark candidate, but also the operationally relevant architecture for the monitoring system.

## Custom Spatio-Temporal GCN

The GCN baseline must be described carefully. It is not a strict, official reproduction of a single canonical `ST-GCN` release. Rather, it is a custom spatio-temporal graph-based skeleton model used as a matched comparison architecture within the project. The methodological value of this model is therefore comparative rather than archival: it allows the paper to ask whether a graph-structured skeleton model offers advantages over the matched temporal-convolutional alternative under the same frozen protocol.

This wording matters for methodological honesty. The paper should not imply a benchmark against a fully standardised `ST-GCN` implementation if the actual code is a project-specific graph baseline.

## Feature Parity and Fair Comparison

The architecture comparison is meaningful only if the models are trained and evaluated under aligned preprocessing and policy conditions. The project therefore treats the TCN-GCN comparison as a controlled protocol question rather than a broad architectural popularity contest. Both families consume the same pose-window contract and the same feature channel family, including motion-aware channels and confidence-aware preprocessing in the locked training/evaluation path. The evidence later shows directional advantage for the TCN under the frozen conditions, but that conclusion remains bounded by the specific feature contract, seed budget, and operating-point workflow used here.

# Calibration and Alert Policy

## Why Calibration Matters Here

In this project, the model does not directly emit the final monitoring decision. Instead, it emits window-level outputs that must be converted into alert states suitable for a monitoring interface. This conversion is where calibration and operating-point fitting become central. A raw score can rank windows reasonably well while still producing unstable or operationally poor alert behaviour if it is used without smoothing, temporal aggregation, or false-alarm control.

## Validation-Side Temperature Scaling

The repository contains an explicit temperature-scaling implementation in `ml/src/fall_detection/core/calibration.py`, and the operating-point fitting path exposed through [ops/scripts/fit_ops.py](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/ops/scripts/fit_ops.py) delegates to `ml/src/fall_detection/evaluation/fit_ops.py` during validation-side fitting. This is an important methodological point. Calibration is part of the operating-point fitting pipeline, not merely a rhetorical label for threshold selection.

At the same time, the paper must remain precise about what calibration means here. The safest statement is that temperature scaling is used during validation-side operating-point fitting, and that the resulting calibration metadata are stored alongside the fitted operating-point configuration. The paper should not overstate this as a guarantee that deployment-time probabilities are always explicitly temperature-corrected at runtime unless the runtime code path demonstrably applies the same transformation online.

## Operating Points as Deployable Policy Profiles

The project’s deployable alert path depends on fitted operating-point configurations rather than on a single universal threshold. The relevant configuration files, such as [tcn_caucafall.yaml](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/ops/configs/ops/tcn_caucafall.yaml) and [gcn_caucafall.yaml](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/ops/configs/ops/gcn_caucafall.yaml), store policy-level parameters including threshold values and temporal decision settings. The three operating points should therefore be understood as distinct policy intents: `OP-1` prioritises earlier or looser alerting, `OP-2` acts as the balanced deployable profile, and `OP-3` applies a stricter interpretation under the same fitted family. In other words, the operational behaviour of the deployed monitor is defined by a fitted profile, not by an informal threshold chosen at the UI layer.

## Alert Policy Beyond Single-Window Scores

The project’s alert layer includes more than thresholding. Smoothing, `k/n` logic, cooldown, and optional confirmation behaviour all contribute to the final monitoring state. The current code path also distinguishes between immediate monitor-state transitions, persisted fall events, and downstream notification delivery. This is why the paper treats alerting as an operational decision problem rather than as a direct alias for classification accuracy. It also explains why deployment behaviour can change when runtime latency, window production rate, or replay-specific path differences change, even if the underlying model checkpoint remains the same.

# Implementation and Refactoring

## Frontend Responsibilities

The frontend is responsible for user interaction, browser-side pose extraction, replay/live source management, and presentation of monitor state. During the project, the frontend accumulated substantial complexity because pose extraction, transport logic, session handling, and UI rendering were initially entangled. The final architecture refactors this responsibility split so that feature-level API logic and monitor-specific transport behaviour are separated from page components and presentation logic. This matters because the monitor path is both user-facing and methodologically important: it is the place where replay and live evidence are actually produced. It is also where several user-facing contracts are enforced, such as the explicit replay-event persistence toggle, the live-video preview option used for demonstration recording, and the fallback handling around monitor/session state.

## Backend Responsibilities

The backend is responsible for inference, policy application, session-aware monitor behaviour, persistence, and system-level integrations. Earlier route handlers contained a large amount of mixed responsibility, including request parsing, database access, policy logic, and response construction. The refactored backend introduces clearer separation between routes, services, and repositories so that business logic and data-access concerns are easier to trace. This is especially relevant in the monitor path, where alert policy, runtime defaults, session state, and persistence all need to remain interpretable if deployment findings are to be reported credibly. The same backend layer also now owns the Telegram-first notification path and its audit store, which helps keep notification evidence aligned with actual event generation rather than with a legacy UI queue model.

## Why Refactoring Matters to the Paper

This refactoring work is relevant because the paper is not only about static model artifacts. It is also about a monitoring system whose deployment claims depend on the runtime path being understandable and reproducible. The architectural cleanup therefore supports the paper indirectly by reducing hidden coupling, making transport and policy boundaries more explicit, and improving confidence that the runtime behaviour analysed in the deployment section corresponds to a coherent code path rather than to a patchwork of ad hoc logic. In practice, this includes making replay persistence explicit, aligning active operating-point defaults across frontend and backend, and ensuring that event history, dashboard summaries, and caregiver notifications all observe the same persisted event semantics.

## Reproducibility and Artifact Discipline

The implementation also reflects an explicit artifact discipline. Candidate checkpoints, operating-point files, summary reports, figures, and deployment notes are tracked in a way that allows paper claims to be mapped back to concrete files. This does not make the project fully audit-proof in the strictest reproducibility sense, but it substantially improves traceability. In practice, this means that major paper claims can be linked to frozen candidate roots, tracked metrics summaries, configuration files, and deployment validation notes rather than to undocumented development memory.

# Results

## Offline Comparative Results on the Frozen Protocol

The primary offline comparison is currently defended through the frozen five-seed metric files under `outputs/metrics/tcn_caucafall_stb_s*.json`, `outputs/metrics/gcn_caucafall_stb_s*.json`, `outputs/metrics/tcn_le2i_stb_s*.json`, and `outputs/metrics/gcn_le2i_stb_s*.json`. On the primary `CAUCAFall` dataset, the TCN shows higher mean performance than the matched GCN across the main tracked metrics:

**Figure 2. Offline stability comparison across the frozen protocol**

Asset:
- [offline_stability_comparison.png](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/artifacts/figures/report/offline_stability_comparison.png)

![Figure 2. Offline stability comparison across the frozen protocol](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/artifacts/figures/report/offline_stability_comparison.png){ width=90% }

**Table 2. Main offline comparative results under the frozen five-seed protocol**

| Dataset | Model | AP mean | F1 mean | Recall mean | Precision mean | FA24h mean |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| CAUCAFall | TCN | 0.9819 | 0.8611 | 0.7600 | 1.0000 | 0.0000 |
| CAUCAFall | GCN | 0.9706 | 0.5873 | 0.4400 | 1.0000 | 0.0000 |
| LE2i | TCN | 0.8389 | 0.8235 | 0.7778 | 0.8750 | 581.5843 |
| LE2i | GCN | 0.7471 | 0.7500 | 0.6667 | 0.8571 | 581.5843 |

This is the clearest directional model-comparison evidence currently available in the repository. It is also more informative than a single best-run comparison because it incorporates run-to-run variability under the frozen seed set. Figure 2 complements Table 2 by showing the relative separation of the frozen candidates across the paper's main offline metrics. On `CAUCAFall`, the TCN not only attains higher mean `F1`, `Recall`, and `AP`, but also exhibits substantially narrower dispersion than the GCN for the same primary event metrics. This improves confidence that the observed directional advantage is not merely an artifact of one unusually favourable run.

However, the inferential interpretation must remain cautious. The current branch preserves the five-seed frozen metric files but not the earlier standalone significance summary bundle, so the defensible statement here is directional rather than definitive: under the frozen protocol, the TCN trends stronger than the matched custom GCN, but the manuscript should not present this as final proof of architectural superiority.

## Cross-Dataset Results and Generalisation Boundary

Cross-dataset evaluation is included to bound generalisation rather than to prove universal robustness. The current branch preserves the frozen cross-domain result files directly under `outputs/metrics/cross_*_frozen_20260409.json`, and these show a strongly asymmetric transfer pattern.

**Figure 3. Cross-dataset transfer summary**

Asset:
- [cross_dataset_transfer_summary.png](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/artifacts/figures/report/cross_dataset_transfer_summary.png)

![Figure 3. Cross-dataset transfer summary](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/artifacts/figures/report/cross_dataset_transfer_summary.png){ width=90% }

**Table 4. Cross-dataset transfer summary**

| Transfer direction | Model | In-domain AP | Cross-domain AP | Delta AP | In-domain F1 | Cross-domain F1 | Delta F1 | In-domain Recall | Cross-domain Recall | Delta Recall |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| LE2i -> CAUCAFall | TCN | 0.8301 | 0.9121 | +0.0820 | 0.8235 | 1.0000 | +0.1765 | 0.7778 | 1.0000 | +0.2222 |
| CAUCAFall -> LE2i | TCN | 0.9680 | 0.5257 | -0.4424 | 0.8889 | 0.0000 | -0.8889 | 0.8000 | 0.0000 | -0.8000 |
| LE2i -> CAUCAFall | GCN | 0.8250 | 0.7640 | -0.0610 | 0.6667 | 1.0000 | +0.3333 | 0.5556 | 1.0000 | +0.4444 |
| CAUCAFall -> LE2i | GCN | 0.9683 | 0.4154 | -0.5528 | 0.3333 | 0.7778 | +0.4444 | 0.2000 | 0.7778 | +0.5778 |

The frozen rerun shows a more nuanced limitation boundary than the older summary that had both `CAUCAFall -> LE2i` rows collapsing to `F1=0.0`. The TCN still fails sharply in that direction: `AP` drops from `0.9680` to `0.5257`, while event-level `F1` and `Recall` both fall to `0.0`. The GCN, however, behaves differently under the frozen rerun. Its cross-domain `AP` still degrades severely (`0.9683 -> 0.4154`), but event-level `Recall` rises from `0.2000` in-domain to `0.7778` cross-domain and `F1` rises from `0.3333` to `0.7778`. That apparent improvement is not a robustness win: it occurs together with a very poor false-alert profile (`FA24h = 1163.17` in the transfer direction), meaning the model becomes far less selective even while it recovers event hits.

The new per-video failure analysis shows why this matters. On the frozen `CAUCAFall -> LE2i` surface, the TCN is dominated by missed-fall collapse (`TP=2`, `FP=0`, `FN=7`, `TN=4`), whereas the matched GCN recovers more hits only by becoming far less selective (`TP=7`, `FP=2`, `FN=2`, `TN=2`) and by generating extremely large false-alert rates on the negative clips. This gives the paper a cleaner limitation statement: transfer failure is model-family specific, and the relevant difference is not simply “which F1 is higher,” but whether recovered recall is achieved at a practical false-alert cost. The targeted note and derived figures are preserved in [TARGETED_FAILURE_ANALYSIS_2026-04-27.md](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/docs/reports/notes/TARGETED_FAILURE_ANALYSIS_2026-04-27.md), [cross_tcn_caucafall_to_le2i_failure_box.png](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/artifacts/figures/report/cross_tcn_caucafall_to_le2i_failure_box.png), and [cross_gcn_caucafall_to_le2i_failure_box.png](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/artifacts/figures/report/cross_gcn_caucafall_to_le2i_failure_box.png).

In contrast, the `LE2i -> CAUCAFall` direction is materially less damaging and can even improve event-level metrics under the frozen policy. Figure 3 makes the asymmetry visible by plotting transfer deltas rather than raw in-domain scores alone. The correct interpretation is therefore not "cross-dataset transfer works well" and not "both directions collapse equally." It is that transfer behaviour is strongly directional and metric-sensitive: score ranking quality, event recall, and false-alert cost can move in different directions across domains. This is limitation evidence, not a universal-strength claim.

The correct conclusion is therefore not that the models generalise well across datasets, but that cross-dataset transfer exposes a strong limitation boundary. This is exactly the kind of bounded evidence that belongs in the discussion and limitations sections rather than in an aggressive model-success claim.

## Secondary MUVIM Exploratory Track

Although `MUVIM` is not part of the primary locked result narrative, the project did carry out substantial work on that dataset. On the current branch, the most defensible way to present `MUVIM` is as a secondary exploratory track supported by archived operating-point families and exploratory outputs rather than by a live headline summary pack.

The most useful paper-level interpretation of the `MUVIM` work is methodological rather than headline comparative. First, it shows that the project extended beyond the final two-dataset comparative core. Second, it provides a concrete example of why metric-contract discipline matters: on `MUVIM`, correcting the event-metric contract materially changed how event-level results should be interpreted, and later operating-point refits improved alert-oriented event metrics without changing threshold-independent score quality. Third, it shows that not every additional experiment was promoted into the final claim set, which is consistent with the paper's evidence-locking discipline. In that sense, `MUVIM` strengthens the credibility of the process more than it changes the main comparative answer.

For that reason, `MUVIM` is retained here as supporting experimental evidence rather than as a third co-equal benchmark. It strengthens the account of experimental breadth and pipeline maturity, but it does not alter the paper's main comparative conclusion, which remains anchored on the locked `CAUCAFall` / `LE2i` protocol.

## Calibration and Alert-Policy Results

The project’s second research question is not answered by benchmark discrimination metrics alone. It is addressed by the fact that the deployable monitor path is governed by fitted operating-point configurations rather than by a manually chosen threshold. The operating-point files for the primary candidates capture threshold and policy structure that are then consumed by the deployable runtime path. This means that alert behaviour is the result of a fitted policy profile rather than a simple post hoc threshold chosen after inspecting test outputs.

The central result here is conceptual but still evidence-grounded: the system operationalises a calibration-aware alert layer. In other words, it does not stop at “which model scores higher,” but proceeds to “how should a monitoring system convert model scores into actionable alert states?” That shift matters because practical fall-detection systems are judged by alert behaviour rather than by score curves alone.

**Figure 4. Calibration-aware alert decision flow**

Asset:
- [alert_policy_flow.svg](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/artifacts/figures/report/alert_policy_flow.svg)

![Figure 4. Calibration-aware alert decision flow](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/artifacts/figures/report/alert_policy_flow.svg){ width=85% }

Figure 4 emphasises that the deployed alert is not the direct output of a single thresholded probability. It is the output of a fitted and temporally structured decision layer. This figure is therefore methodological rather than decorative: it explains why the project treats operating-point fitting and temporal policy as part of the main contribution.

**Table 5. Current fitted operating-point trade-offs for the active CAUCAFall TCN profile**

| Operating point | Intent | Recall | F1 | FA24h | Mean delay (s) | Practical interpretation |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| OP-1 | Faster, lower-persistence alerting | 0.60 | 0.75 | 0.00 | 0.63 | Earlier alerts, but some falls are missed under the current fitted profile |
| OP-2 | Balanced deployable profile | 1.00 | 1.00 | 0.00 | 3.34 | Highest recall under the current fit, but with a substantially longer mean alert delay |
| OP-3 | Stricter thresholding under the same fitted family | 0.60 | 0.75 | 0.00 | 0.63 | Similar bounded outcome to OP-1 in the current fit; shows that fitted OPs can collapse or partially overlap rather than always producing three sharply separated behaviours |

This table uses the current active [tcn_caucafall.yaml](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/ops/configs/ops/tcn_caucafall.yaml) profile, which records fitted thresholds together with event-level recall, `F1`, `FA24h`, and mean delay. The key point is not that one operating point is universally best, but that the deployable alert layer changes practical behaviour in measurable ways. Under the current fit, `OP-2` preserves the strongest recall, while `OP-1` and `OP-3` trade that recall for faster alert timing. This is sufficient to justify treating operating-point selection as a substantive part of the system design rather than as a cosmetic threshold choice.

## Deployment and Runtime Results

Deployment evidence is strongest in bounded replay and live demonstration evidence, not in broad field closure. On the current branch, the preferred live demo preset remains `CAUCAFall + TCN + OP-2`, but the replay artifacts show that runtime behaviour is highly profile-sensitive and must not be narrated as uniformly strong.

**Table 6. Deployment and runtime evidence summary**

| Evidence slice | Source artifact | Main result | Reporting role |
| --- | --- | --- | --- |
| Canonical replay summary | `artifacts/ops_delivery_verify_20260315/online_replay_summary.json` | `caucafall_tcn OP-1` reaches `accuracy=1.0`, `recall=1.0`, `specificity=1.0` on the bounded 10-video replay check | Bounded replay evidence only |
| Same canonical replay summary | `artifacts/ops_delivery_verify_20260315/online_replay_summary.json` | `caucafall_tcn OP-2` drops to `accuracy=0.5`, `recall=0.0`, `specificity=1.0` on the same replay surface | Shows that profile choice materially changes runtime behaviour |
| Bounded 24-clip custom replay matrix | `artifacts/fall_test_eval_20260315/summary_tcn_caucafall_locked_op2.csv` | the defended baseline `tcn_caucafall_locked_op2` path produces `15/24` correct video-level outcomes with `TP=5`, `TN=10`, `FP=2`, `FN=7` | Runtime-boundary evidence only |
| Same custom replay surface | `artifacts/fall_test_eval_20260315/summary_tcn_caucafall_locked_op1.csv` | the looser `tcn_caucafall_locked_op1` path produces `11/24` correct video-level outcomes with `TP=9`, `TN=2`, `FP=10`, `FN=3` | Shows the recall-versus-selectivity trade-off inside one model family |
| Same custom replay surface | `artifacts/fall_test_eval_20260315/summary_gcn_caucafall_locked_op2.csv` | the matched `gcn_caucafall_locked_op2` path produces `12/24` correct video-level outcomes with `TP=12`, `TN=0`, `FP=12`, `FN=0` | Shows why full fall capture can still correspond to a weak runtime preset |
| Custom replay runbook | `docs/reports/runbooks/FOUR_VIDEO_DELIVERY_PROFILE.md` | the current four-folder check is explicitly bounded and no longer supports any earlier perfect-profile narrative | Workflow-boundary evidence only |
| Retraining strengthening track | `docs/reports/notes/CANDIDATE_A_RUNTIME_EVAL_2026-04-27.md`, `docs/reports/notes/CANDIDATE_D_RUNTIME_EVAL_2026-04-27.md` | recall-oriented continuations improve the bounded TCN custom surfaces from `13/24` or `15/24` to `16/24` without increasing ADL false positives | Modest improvement only; both candidates require confirm-disabled `fit_ops` fallback |
| Realtime evidence pack | `artifacts/evidence/realtime/*.mp4` | the monitor, persisted-event, and caregiver-facing alert path can be demonstrated under a live bounded run | Live demonstration evidence only |
| Runtime diagnostics | deployment/runtime logs and latency notes | runtime behaviour depends on frontend window production and backend inference latency as well as model/policy choice | System discussion, not model-comparison evidence |

**Figure 5. Online replay accuracy by dataset, model, and operating point**

Asset:
- [online_replay_accuracy_heatmap.png](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/artifacts/figures/report/online_replay_accuracy_heatmap.png)

![Figure 5. Online replay accuracy by dataset, model, and operating point](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/artifacts/figures/report/online_replay_accuracy_heatmap.png){ width=90% }

**Figure 6. MC-dropout effect on the bounded online replay matrix**

Asset:
- [online_mc_dropout_delta.png](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/artifacts/figures/report/online_mc_dropout_delta.png)

![Figure 6. MC-dropout effect on the bounded online replay matrix](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/artifacts/figures/report/online_mc_dropout_delta.png){ width=82% }

**Figure 7. Realtime evidence chain under the defended demo preset**

Assets:
- [hifi_live_monitor.png](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/artifacts/figures/report/appendix/hifi_live_monitor.png)
- [hifi_event_history.png](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/artifacts/figures/report/appendix/hifi_event_history.png)

![Figure 7a. Representative live monitor capture for the defended runtime path](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/artifacts/figures/report/appendix/hifi_live_monitor.png){ width=92% }

![Figure 7b. Representative Event History capture for the defended review path](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/artifacts/figures/report/appendix/hifi_event_history.png){ width=92% }

Supplementary videos:
- `Supplementary Video S1`: [realtime_fall_submission.mp4](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/artifacts/evidence/realtime/realtime_fall_submission.mp4)
- `Supplementary Video S2`: [realtime_adl_submission.mp4](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/artifacts/evidence/realtime/realtime_adl_submission.mp4)
- mirrored realtime evidence folder: `https://drive.google.com/drive/folders/19xGW69UNtjJOA8BRw4kLd6fe0rVwQnS1?usp=sharing`

The bounded custom replay set remains useful, but it now functions primarily as a runtime-interpretation matrix rather than as a delivery-success package. Figure 5 shows the fixed raw online replay results across the `12` model-dataset-operating-point combinations. On the current branch, the safe interpretation is not that `OP-2` is uniformly strongest, but that replay behaviour varies materially by dataset, model, and operating point, and that the current `CAUCAFall + TCN + OP-2` profile is better treated as the preferred demo configuration than as a universally strongest replay row.

The newer custom replay breakdown makes that statement more concrete. `TCN + OP-2` is comparatively selective on the ADL folders but misses a substantial subset of fall clips. `TCN + OP-1` recovers more fall clips on the same surface, but only by sharply increasing ADL false alarms. The matched `GCN + OP-2` profile is more extreme again: it captures all positive folders while collapsing both ADL folders into false alarms. The result is a clearer deployment lesson: the defended demo preset is not the “best overall row,” but the most interpretable bounded trade-off currently available in the active runtime family. The supporting runtime note is preserved in [CUSTOM_REPLAY_RUNTIME_ANALYSIS_2026-04-27.md](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/docs/reports/notes/CUSTOM_REPLAY_RUNTIME_ANALYSIS_2026-04-27.md).

The later strengthening runs show that this bounded runtime line is improvable, but not yet transformed. A recall-oriented continuation (`Candidate A`) improved the canonical four-folder bounded replay check from `13/24` to `16/24` and the historical locked 24-clip surface from `15/24` to `16/24` without increasing ADL false positives. A second continuation (`Candidate D`) reached the same `16/24` outcome on both bounded surfaces. This is useful deployment-oriented evidence because it shows that retraining can reduce false negatives on the defended custom surfaces. It is not a decisive runtime breakthrough, however, because both candidates still depended on a confirm-disabled fallback during `fit_ops`, and neither run improved the `kitchen` fall subset. The strengthening evidence is preserved in [CANDIDATE_A_RUNTIME_EVAL_2026-04-27.md](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/docs/reports/notes/CANDIDATE_A_RUNTIME_EVAL_2026-04-27.md), [CANDIDATE_A_LOCKED_SURFACE_EVAL_2026-04-27.md](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/docs/reports/notes/CANDIDATE_A_LOCKED_SURFACE_EVAL_2026-04-27.md), [CANDIDATE_D_RUNTIME_EVAL_2026-04-27.md](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/docs/reports/notes/CANDIDATE_D_RUNTIME_EVAL_2026-04-27.md), and [candidate_ad_runtime_surface_compare_2026-04-27.csv](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/docs/reports/notes/candidate_ad_runtime_surface_compare_2026-04-27.csv).

Figure 6 captures the uncertainty-gate result more directly than a textual summary alone. On the fixed raw online replay path, enabling the uncertainty-aware MC path did not change any of the `12` combinations at the video level. This is still a useful negative result: the uncertainty path was genuinely exercised on boundary windows, but it did not improve the replay matrix. The correct interpretation is not that MC dropout improved deployment accuracy, but that the uncertainty-aware path is methodologically relevant without yet showing replay gains.

Figure 7 complements the replay matrix by showing the bounded realtime chain for the defended deployment profile. The first panel shows the monitor surface used for runtime interpretation. The second panel shows the review surface used for persisted event history. This closes the semantic gap between transient runtime output and persisted review state without pretending that one screenshot pair adds new statistical authority beyond the replay and offline layers.

The project also includes a smaller four-folder custom replay check aligned to the same canonical `CAUCAFall + TCN + OP-2` runtime profile. That check is useful as a supporting custom-validation slice, but it is weaker than the 24-clip bounded matrix and should therefore remain secondary in the written interpretation. Its role is to support the statement that the monitor path was exercised on a small custom set under the same active policy family, not to replace the broader replay matrix.

The current branch does not preserve a stable field-validation summary pack strong enough to support a field-readiness paragraph. The safer statement is that live evidence exists and end-to-end runtime behaviour has been demonstrated, but broad field closure remains outside the current defended evidence base.

## Runtime Interpretation

The runtime evidence collected during deployment work shows that monitoring behaviour depends not only on model weights and thresholds but also on the production rate of valid windows and the latency of the inference path. This is why replay and realtime are separated analytically throughout the paper. In earlier deployment iterations, runtime discrepancies were traced to a combination of frontend pose-window production and backend inference latency rather than to a single threshold mismatch. This observation strengthens the paper’s central methodological claim that deployment analysis belongs to the system layer, not to the pure model-comparison layer.

This point also clarifies why runtime improvement work, API-path stabilisation, and frontend monitor refactoring are relevant to the scientific interpretation of the project. They do not change the frozen offline evidence, but they do affect whether the deployment path is capable of expressing that evidence under practical runtime conditions.

# Discussion

## Answer to RQ1

RQ1 asked how the TCN and the custom GCN compare under the locked offline protocol. The evidence supports a cautious directional answer: the TCN trends stronger than the matched GCN on the primary `CAUCAFall` protocol and also remains stronger on the in-domain `LE2i` comparison. However, the current five-seed inferential budget does not justify a strong claim of definitive statistical superiority under the primary Wilcoxon analysis. The correct answer is therefore “directional advantage with bounded statistical certainty,” not “final proof that TCN is universally better.”

## Answer to RQ2

RQ2 asked whether validation-side calibration and operating-point fitting materially affect practical alerting. The answer is yes. The repository’s deployable behaviour depends on fitted operating-point profiles, not on raw single-window probabilities alone. The use of temperature scaling during operating-point fitting, together with policy parameters such as smoothing, `k/n`, and cooldown, shows that the project’s alerting path is explicitly calibrated as an operational decision layer. This is an important systems contribution because it moves the project beyond a benchmark-only classifier framing and makes the final monitoring behaviour interpretable in deployment terms.

## Answer to RQ3

RQ3 asked what replay deployment evidence and limited realtime validation reveal about practical feasibility and runtime limits. The answer is mixed but useful. Replay-oriented deployment evidence is strong enough to support a practical system claim in controlled conditions, and the bounded realtime evidence chain in Figure 7 and Supplementary Video S1 further shows that the defended profile can support monitor-visible runtime behaviour and persisted Event History state in a live run. At the same time, the replay artifacts make clear that deployment behaviour is not uniform across datasets and profiles. On the custom replay surface, the preferred `TCN + OP-2` preset is more selective but miss-prone, while `TCN + OP-1` recovers more falls only by becoming much noisier on ADL clips; the matched `GCN + OP-2` profile is less selective still. The strengthening runs improve this picture modestly: the best retrained continuations raise the defended bounded replay surfaces to `16/24` without worsening ADL false positives, but they do not remove the need for careful policy fitting and they do not solve all subset-level misses. The uncertainty-aware MC path also does not improve the fixed replay matrix. Runtime behaviour therefore remains sensitive to policy choice, path-specific latency, and frontend pose quality. The system demonstrates practical feasibility in controlled deployment settings, but not broad field closure or uniform runtime robustness.

## Model Quality Versus System Quality

A key lesson of the work is that model quality and system quality are not interchangeable. A strong checkpoint can still yield poor alerting if policy selection is weak or runtime latency distorts the sequence of windows seen by the backend. Conversely, a carefully stabilised deployment path may appear highly effective on replay evidence even though it does not justify a broader unseen-test claim. The replay matrix illustrates both sides of this distinction: `CAUCAFall + TCN + OP-2` is strong enough to support a live-demo preset, while the same runtime surface still exposes weaker `LE2i` behaviour and a neutral uncertainty-path result. This is one of the main reasons why the paper insists on separating offline, replay, and limited realtime evidence.

## Interpretation of the Project’s Strongest Contribution

The project’s strongest contribution is not a claim that one neural architecture definitively solves fall detection. The stronger contribution is that it integrates controlled model comparison, calibration-aware alert policy, and deployment/runtime analysis within one coherent pipeline. This makes the final system study more useful than a narrow benchmark note. It provides not only model evidence, but also a structured account of how benchmark outputs become operational decisions and where that process remains fragile under bounded runtime validation. That combination of model evidence, policy design, and runtime interpretation is the paper's most defensible claim to originality and maturity.

## Role of the MUVIM Track

The `MUVIM` work helps clarify the scope of the project without changing its primary conclusions. It shows that the project did not stop at a single benchmark pair, and it provides useful supporting evidence that the evaluation and operating-point pipeline was exercised on a broader experimental track. At the same time, the paper deliberately does not elevate `MUVIM` into a main result axis, because doing so would weaken the evidence hierarchy and risk conflating exploratory work with the locked comparative protocol. This is the correct trade-off: acknowledge the work, preserve it in the narrative, but keep the main claims tied to the strongest and cleanest evidence.

# Limitations

The first major limitation is statistical. The primary architecture comparison currently relies on a frozen five-seed comparison. This is enough to support a directional interpretation, but not enough to justify a strong non-parametric significance claim for the main event metrics.

The second major limitation is generalisation. Cross-dataset evidence is clearly asymmetric and does not support a claim of universal robustness across domains, cameras, or motion statistics. More specifically, the frozen `CAUCAFall -> LE2i` transfer boundary exposes different failure modes for the two model families: the TCN collapses toward missed falls, while the matched GCN recovers recall only by becoming far less selective.

The project also includes a secondary `MUVIM` experiment track, but that work is not integrated into the main comparative claim because its strongest value is exploratory and methodological rather than central to the final locked evidence hierarchy.

The third limitation is measurement quality. Because the deployed system depends on browser-side or frontend pose extraction, degradation in skeleton quality can propagate directly into inference and alerting behaviour. This is a fundamental systems issue rather than a small implementation nuisance.

The fourth limitation is deployment closure. Replay-oriented deployment evidence is much stronger than field or live evidence. The current field-validation pack is too small to support broad real-world conclusions, and realtime results remain bounded by environment and runtime constraints. Even within replay, the bounded runtime matrix supports a defended demo preset rather than a uniform deployment-success claim across all dataset-model-operating-point combinations. The custom replay surface is especially clear on this point: more fall capture can be bought by materially increasing ADL false alarms, so runtime quality depends on the trade-off one is willing to defend. The strengthening runs improve the defended TCN custom surfaces, but only modestly, and they still depend on confirm-disabled `fit_ops` fallback. That means the project can now claim bounded deployment improvement under retraining, but not a solved deployment profile.

The fifth limitation is interpretive. Some replay or delivery-style validation artifacts are the result of deployment-aware tuning and path stabilisation. They are valuable as engineering evidence, but they cannot be promoted into the same evidential status as locked unseen-test benchmark results. The current replay matrix is especially important in this respect: it is useful because it is bounded, repeatable, and now tied to the active runtime profile, but it still remains system evidence rather than formal generalisation evidence.

The sixth limitation is runtime uncertainty interpretation. The uncertainty-aware path is methodologically relevant and fully part of the implemented system, but the 24-clip replay matrix does not show a deployment gain from enabling MC-based uncertainty handling. That makes the evidence neutral rather than supportive for a stronger uncertainty-benefit claim.

Taken together, these limitations imply that the correct scientific stance is bounded confidence rather than maximalist claim-making. The project demonstrates a serious and coherent system study, but it does not yet justify claims of universal real-world robustness, clinical readiness, or definitive architecture superiority.

# Future Work

Future work should proceed along five directions.

1. Expand the live and field-validation protocol so that deployment claims can be supported by a larger and more varied runtime evidence base.
2. Improve domain robustness across datasets, camera setups, and pose-quality conditions rather than treating in-domain performance as sufficient.
3. Rework the uncertainty path so that it is either validated under a stronger runtime protocol or simplified further, since the current bounded replay matrix does not show deployment gains from the uncertainty-aware live gate.
4. Continue simplifying the runtime architecture, especially in the live monitor path, so that the deployment system remains maintainable as evaluation complexity increases.
5. Extend the current Telegram-first notification path into a broader multi-channel escalation design only after the runtime and field-validation evidence base is stronger.

# Conclusion

This project delivers a deployment-oriented study of pose-based fall detection rather than a narrow benchmark exercise. Its main contribution is not only the construction of TCN and custom GCN candidates, but also the integration of preprocessing, operating-point fitting, alert-policy logic, runtime inference, and monitoring-oriented deployment analysis into one coherent system.

Under the locked primary-dataset protocol, the TCN trends stronger than the matched custom GCN, although the current five-seed inferential evidence supports only a cautious comparative claim. Validation-side operating-point calibration is a substantive part of the system because alert behaviour depends on fitted policy profiles rather than on raw classifier scores alone. Replay-oriented deployment evidence shows that the system can operate effectively in runtime conditions bounded by the current protocol, while also showing that replay behaviour is profile-sensitive and that optional uncertainty-aware live inference did not improve the fixed raw online replay matrix. Cross-dataset and field evidence make clear that generalisation and broad deployment closure remain incomplete.

The most defensible final interpretation is therefore that the project successfully demonstrates a serious, end-to-end, deployment-aware fall-detection system with explicit model, policy, runtime, and notification-path analysis, while also making its current boundaries clear. It also shows broader experimentation, including the secondary `MUVIM` track, without collapsing exploratory work into the main claim set. That is a stronger and more credible contribution than an overstated claim of solved real-world fall detection.

# Appendices and Supporting Artifacts

The final manuscript should include supporting appendices or supplementary materials that make the evidence trail easier to audit without overloading the main text. The most useful supporting items are likely to be:

1. Frozen candidate roots and seed list, derived from the locked candidate set recorded in this manuscript and the current Week 1 framing/results lock.
2. Additional operating-point details drawn from the active `ops/configs/ops/*.yaml` files for the reported candidates.
3. Supplementary significance and stability summaries that are too detailed for the main narrative.
4. Deployment and replay validation notes, especially where they clarify the boundary between bounded system evidence and formal model evidence.
5. Supporting notes for the secondary `MUVIM` track, retained explicitly as exploratory material rather than as core comparative evidence.

These materials should support the main paper, not replace it. Their role is to improve auditability, artifact traceability, and methodological transparency.

# References

The bibliography below is a provisional final-draft reference set aligned with the current paper wording. It should still be normalized to the final supervisor- or venue-required citation style before submission.

1. Yan, S., Xiong, Y., and Lin, D. (2018). Spatial Temporal Graph Convolutional Networks for Skeleton-Based Action Recognition. *Proceedings of the AAAI Conference on Artificial Intelligence*.
2. Shi, L., Zhang, Y., Cheng, J., and Lu, H. (2019). Two-Stream Adaptive Graph Convolutional Networks for Skeleton-Based Action Recognition. *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*.
3. Chen, Y., Zhang, Z., Yuan, C., Li, B., Deng, Y., and Hu, W. (2021). Channel-Wise Topology Refinement Graph Convolution for Skeleton-Based Action Recognition. *Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)*.
4. Abu Farha, Y. and Gall, J. (2019). MS-TCN: Multi-Stage Temporal Convolutional Network for Action Segmentation. *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*.
5. Lin, J., Gan, C., and Han, S. (2019). TSM: Temporal Shift Module for Efficient Video Understanding. *Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)*.
6. Guo, C., Pleiss, G., Sun, Y., and Weinberger, K. Q. (2017). On Calibration of Modern Neural Networks. *Proceedings of the 34th International Conference on Machine Learning (ICML)*.
7. Lin, T.-Y., Goyal, P., Girshick, R., He, K., and Dollar, P. (2017). Focal Loss for Dense Object Detection. *Proceedings of the IEEE International Conference on Computer Vision (ICCV)*.
8. Geifman, Y. and El-Yaniv, R. (2019). SelectiveNet: A Deep Neural Network with an Integrated Reject Option. *Proceedings of the 36th International Conference on Machine Learning (ICML)*.
9. OpenMMLab. MMAction2 Skeleton-Action Documentation and Repository. Available at: https://mmaction2.readthedocs.io/en/latest/model_zoo/skeleton.html and https://github.com/open-mmlab/mmaction2
10. Real-Time Fall Detection from Infrared Video Based on ST-GCN. *Sensors* (2024).
11. Skeleton-Based Human Fall Detection with Person Tracking and Temporal Deep Learning. *Automation in Construction* (2025).
12. A Comparative Study for Pre-Impact and Post-Impact Fall Detection in Wearable Systems. *Pervasive and Mobile Computing* (2025).
13. A Lightweight Transformer- and TCN-Based Architecture for Robust, Real-Time Fall Detection. *Scientific Reports* (2025).
