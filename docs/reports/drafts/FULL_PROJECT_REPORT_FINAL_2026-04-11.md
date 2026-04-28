Date: 2026-04-09  
Type: full final-year project report  
Status: final submission version

# Abstract

This report presents a research-led, full-stack study of pose-based fall detection under a bounded and explicitly controlled evidence framework. This project is an end-to-end pose-based fall-detection and monitoring system study, supported by controlled model comparison and deployment-oriented evaluation. It investigates fall detection not only as a window-level classification task, but as a monitoring problem that also requires alert-policy design, runtime interpretation, event persistence, and caregiver-facing delivery.

The study compares a Temporal Convolutional Network (TCN) with a matched custom spatio-temporal Graph Convolutional Network (GCN) under a shared pose-window protocol, frozen data splits, and validation-side operating-point fitting. `CAUCAFall` is used as the primary benchmark and deployment-target dataset, `LE2i` provides comparative and transfer-boundary evidence, and `MUVIM` is retained as a secondary exploratory track. Beyond offline comparison, the report evaluates how model outputs are transformed into reviewable alert behaviour through temperature-informed fitting, temporal policy, and bounded replay/runtime validation.

The strongest scientific finding is a cautious directional advantage for the TCN under the frozen primary-dataset protocol. The strongest systems finding is that a pose-based monitoring pipeline can be made coherent and reviewable when model inference, alert-policy interpretation, event persistence, and caregiver-facing notification are treated as one integrated technical artifact rather than as isolated layers.

The report does not claim solved fall detection, broad clinical readiness, or unrestricted deployment robustness. Its contribution is instead a serious and defensible end-to-end project study that keeps model evidence, policy evidence, runtime evidence, and deployment interpretation analytically separate while showing how they can still be made operationally consistent.

# Introduction

## Background and Motivation

Automatic fall detection remains an important but difficult problem because falls are both clinically significant and operationally hard to monitor continuously. In supervised settings, a caregiver may recognise a fall immediately. In home-style or semi-independent settings, however, harmful delay can arise between the incident itself and the moment at which another person becomes aware of it. The practical aim of fall detection is therefore not only to classify motion correctly, but to shorten the time between an incident and a reviewable response.

That practical aim creates a broader technical challenge than window-level recognition alone. A useful fall-detection system must observe a person reliably, represent motion in a machine-readable form, distinguish falls from daily activities, transform score streams into interpretable alert states, and communicate those states through a reviewable interface. Many studies focus primarily on the classifier. This project instead treats fall detection as a multi-layer monitoring problem in which model quality, policy design, runtime behaviour, and downstream review all matter.

A pose-based approach is attractive in this setting because it offers a compact and interpretable motion representation that can be shared across offline evaluation and runtime deployment. At the same time, pose quality becomes a first-order system constraint. For that reason, a pose-based monitor cannot be evaluated only through offline model metrics. Its behaviour also depends on how browser-side pose extraction, backend inference, temporal policy, and event semantics interact at runtime. This tension motivates the report’s central design choice: to study pose-based fall detection as an end-to-end system while keeping the limits of the current evidence explicit.

## Problem Framing

The project is organised around one central question: can a pose-based temporal monitoring stack produce credible fall alerts under a controlled and reviewable protocol without overstating what the available evidence can support? This project is an end-to-end pose-based fall-detection and monitoring system study, supported by controlled model comparison and deployment-oriented evaluation. This question is narrower than asking whether fall detection in general is solved, but broader than asking which architecture attains the best offline score.

In practice, the work sits at the intersection of three linked problems. The first is a model-comparison problem: compare a strong temporal baseline with a matched graph-based alternative under one frozen data and fitting protocol. The second is an alert-policy problem: determine how validation-fitted operating points and temporal rules convert score streams into operational states. The third is a system-integration problem: make monitoring, persistence, review, and caregiver-facing delivery function coherently enough to support bounded deployment claims.

This framing matters because none of these layers is sufficient on its own. Offline benchmark results alone do not show whether a system can support reviewable alert behaviour. Conversely, runtime demonstrations alone are weak unless they are grounded in a disciplined offline protocol. The report therefore moves from data and training contracts to policy fitting, then from policy fitting to runtime interpretation, persistence, and deployment boundaries.

## Objectives

The first objective was to produce a fair comparison between a Temporal Convolutional Network (TCN) and a custom spatio-temporal Graph Convolutional Network (GCN) under shared preprocessing, windowing, and evaluation rules. The second objective was to fit deployable operating-point profiles on validation data only, so that runtime alert behaviour could be discussed under a disciplined evidence policy rather than through ad hoc threshold selection. The third objective was to implement a working monitoring system with frontend, backend, persistence, and caregiver-notification components rather than leaving the work at notebook level. The fourth objective was to preserve evidence discipline through audit, code review, and freeze-state cleanup so that final claims remained traceable to defended artifacts.

These objectives also define what the report does not attempt to do. It does not claim universal domain robustness across all fall datasets. It does not treat bounded replay or live demonstrations as substitutes for formal benchmark evaluation. It does not equate notification delivery with detection validity. Instead, the report aims to show that a final-year research project can produce a technically serious, end-to-end, and carefully bounded study of pose-based fall monitoring.

## Contributions

This report makes four main contributions.

1. It presents a locked-protocol comparison between a TCN and a matched custom spatio-temporal GCN under a shared pose-window representation, fixed evidence policy, and frozen multi-seed evaluation design.
2. It treats validation-side operating-point fitting and temporal alert policy as first-class methodological objects, rather than reducing deployment behaviour to a single manually chosen threshold.
3. It implements and analyses an integrated monitoring pipeline that connects browser-side pose extraction, backend inference, runtime interpretation, event persistence, dashboard review, and caregiver-facing Telegram delivery.
4. It demonstrates a bounded evidence framework in which offline benchmark results, cross-dataset transfer evidence, replay/runtime validation, and delivery-path evidence are kept analytically separate while still being linked through one coherent system artifact.

# Project Context and Requirements

## Project Goals

The project was not defined as a pure benchmark exercise. Its goals combine scientific comparison, system implementation, and bounded deployment validation. At the highest level, the work set out to answer whether a pose-based temporal system could support credible fall alerting while remaining auditable and maintainable enough for a final-year systems project. This produced four concrete goals.

The first goal was comparative: implement and evaluate a strong temporal baseline and a matched graph-based baseline under one locked protocol. This goal matters because the report’s model-level claim depends on fair comparison rather than on isolated best-run numbers.

The second goal was operational: move from classifier scores to deployable alert behaviour. That required fitting operating points on validation data, encoding temporal policy in the runtime path, and measuring alert-oriented outcomes rather than reporting threshold-free metrics alone.

The third goal was systems-oriented: build a working monitoring stack that joins browser-side pose extraction, backend inference, event persistence, dashboard/event review, and caregiver notification delivery. This distinguishes the project from a notebook-only model study.

The fourth goal was methodological: maintain evidence discipline. Offline benchmark results, replay validation, and limited field checks had to remain analytically separate so that deployment work could strengthen the system story without contaminating the formal evaluation story.

## Primary Use Cases

The system was developed around a small number of concrete use cases rather than an abstract claim of universal monitoring support.

The first use case is supervised bounded monitoring. In this setting, an operator or reviewer runs the monitor locally or in a Docker-backed environment, observes pose extraction and monitor state in real time, and uses the interface to understand how the active profile behaves. This use case is central because it provides the clearest path from code to runtime evidence.

The second use case is replay-based validation. Here, stored clips are passed through the same monitoring surface to test how the deployed path behaves under known and repeatable inputs. This use case is valuable because it bridges the gap between frozen benchmark evaluation and live demonstration without pretending to be equivalent to either.

The third use case is reviewable event handling. In this setting, a fall-like runtime condition becomes a persisted event, is visible through dashboard and event views, and can trigger downstream caregiver notification. This use case matters because it turns the project from a classifier demo into a monitoring system with review semantics.

The final use case is caregiver-facing alert delivery. Even though the active delivery path is intentionally narrow, the project still aims to demonstrate that an event can progress through the full operational chain to a real notification endpoint. This use case is modest in scope, but it is important because it closes the loop between inference and external action.

## Functional Requirements

From the beginning, the system needed to support more than model training. The implemented platform was expected to provide a usable end-to-end path from video input to monitor-visible alert state. In practice, this produced the following functional requirements.

1. The system must ingest live camera input and stored replay clips through one monitoring interface.
2. The frontend must convert visual input into pose sequences and package them into fixed-length windows suitable for backend inference.
3. The backend must apply the active model and fitted operating-point profile, then emit monitor states that can be interpreted as operational alert states rather than raw classifier outputs alone.
4. When alert conditions are met, the system must be able to persist fall events into a history that can later drive dashboard summaries and event review.
5. The system must support caregiver-notification delivery on the active deployment path. In the final implementation, this requirement is satisfied by the Telegram-first notification route.
6. The system must support replay-based evaluation and demonstration without forcing replay evidence to act as a substitute for formal test-set evidence.

These requirements are intentionally system-level. They explain why frontend, backend, persistence, and notification behaviour all matter to the final report, even though the project also includes a conventional model-comparison component.

The requirements are also linked rather than independent. Live and replay ingestion are only useful if their outputs are converted into windows under one stable contract. Persisted events are only valuable if the dashboard and events page can later render them under the same semantics. Notification delivery only matters if it is triggered by reviewable event logic rather than by an isolated transient state. For that reason, the full report should treat requirement satisfaction as an end-to-end property rather than a checklist of isolated features.

**Table 1. Functional requirement traceability matrix**

| ID | Requirement | Primary implementation locus | Verification evidence | Main report anchor |
| --- | --- | --- | --- | --- |
| FR-1 | ingest live camera input and replay clips through one monitor surface | `applications/frontend/src/pages/Monitor.tsx`, `applications/frontend/src/pages/monitor/hooks/usePoseMonitor.ts` | live/replay monitor path, runtime validation chapter | Sections `System Architecture`, `System Validation, Testing, and Audit` |
| FR-2 | convert visual input into pose-derived temporal windows for backend inference | browser-side pose extraction plus frontend window packaging, backend monitor contract | replay/live bounded runtime checks, pipeline contract review | Sections `System Architecture`, `Data and Experimental Protocol` |
| FR-3 | apply active model and fitted operating-point profile to produce operational monitor states | `applications/backend/services/monitor_runtime_service.py`, active ops configs under `ops/configs/ops/*` | canonical tests, replay/runtime validation, code review | Sections `Calibration and Alert Policy`, `System Validation, Testing, and Audit` |
| FR-4 | persist reviewable fall events for dashboard and event history | backend persistence layer, monitor repository, event routes | event-history verification, dashboard/event contract review | Sections `System Architecture`, `Implementation and Refactoring` |
| FR-5 | deliver caregiver-facing notification on the active runtime path | Telegram notification manager and audit store | live/test notification checks, notification audit path | Sections `Notification Architecture`, `Deployment and Runtime Results` |
| FR-6 | support replay validation without conflating replay with benchmark evidence | explicit replay controls and realtime-only event persistence semantics | replay validation chapter, audit/code-review findings | Sections `Research Questions and Scope`, `System Validation, Testing, and Audit` |

## Success Criteria

The project uses layered success criteria rather than one monolithic pass/fail definition. At the model-comparison layer, success means producing a fair and reproducible comparison between the TCN and the custom GCN under a frozen protocol. At the alert-policy layer, success means producing fitted operating-point profiles that yield interpretable and reviewable behaviour rather than arbitrary threshold choices. At the systems layer, success means the monitor, persistence, dashboard, and notification path can operate together coherently under bounded conditions.

This layered approach matters because different artifacts answer different questions. A benchmark table cannot by itself show that caregiver notification delivery works. A live demo cannot by itself establish architectural preference. A high-quality full report therefore needs success criteria that are aligned with the evidence hierarchy.

The most important practical success criteria in the current project are:

1. the repository must support a frozen offline comparison that can be regenerated from tracked artifacts
2. the alert-policy layer must be validation-fitted and reviewable
3. the preferred deployment profile must behave credibly on the bounded replay matrix
4. the runtime path must support persisted event history and downstream notification delivery
5. the final repository and report must survive audit and code review without major mismatch between claims and implementation

## Non-Functional Requirements

Several non-functional requirements were equally important because they determine whether the system is scientifically credible and operationally understandable.

First, the project needed reproducibility. Reported results had to map back to specific artifacts, candidate roots, configuration files, and regeneration commands. This led to the freeze/evidence discipline recorded elsewhere in the repository.

Second, the project needed maintainability. The monitor path, notification path, and evaluation pipeline all became too complex to leave in an ad hoc state. Refactoring was therefore not an optional cleanup exercise; it became necessary for runtime interpretation and final report traceability.

Third, the system needed runtime interpretability. Operators and reviewers had to be able to distinguish replay from realtime, understand which operating-point profile was active, and inspect persisted events and notification outputs without guessing which path had produced them.

Fourth, the project needed bounded deployment realism. It was not enough to report benchmark metrics in isolation. The runtime path had to demonstrate that alert-policy logic could survive contact with browser-side pose extraction, backend latency, and reviewable event history. At the same time, the evidence had to stay honest about the current lack of broad field closure.

## Risk Framing

The project also carries explicit technical and interpretive risks. The most obvious technical risk is upstream pose fragility: if browser-side skeleton quality degrades, backend inference and alert policy can degrade even when model checkpoints remain unchanged. The most important interpretive risk is evidence conflation: replay, live demonstration, and frozen offline evaluation can be mistakenly read as though they support the same claims when they do not.

There are also repository-level risks. As the project accumulated code, artifacts, and writing outputs, there was a growing risk that final claims would drift away from the code and evidence that actually supported them. This is one reason later audit and freeze work belongs naturally inside the full report rather than being treated as administrative cleanup.

# Background and Related Work

Fall detection sits at the intersection of activity recognition, time-series decision making, and safety-adjacent monitoring. As a result, the literature is wider than a single benchmark tradition. Some studies are primarily concerned with representation and classification accuracy under tightly controlled datasets. Others are more concerned with operational realism, including alert stability, sensor limitations, and deployment workflow. This report draws on both strands. It treats fall detection as a problem that begins with recognition but does not end there, because a deployable monitor must also transform model outputs into reviewable incident states.

For that reason, the literature reviewed here is organised by problem layer rather than by dataset name alone or by model family alone. The discussion moves from the general structure of fall detection as a layered problem, to representation choice, to temporal and graph-based modelling, and finally to calibration, alert policy, and deployment-oriented evaluation. This structure is important because contributions at one layer do not automatically transfer to another. A model may improve score ranking without improving event-level behaviour, and a system may produce a convincing live demonstration without strengthening the formal benchmark claim.

## Fall Detection as a Multi-Layer Problem

A common weakness in fall-detection writing is to treat the task as though it were exhausted by clip-level or window-level recognition. In practice, however, fall detection is a layered decision problem. A system must first obtain a usable representation of human motion, then classify whether a fall-like transition is present, and then decide whether the resulting evidence is strong and stable enough to justify an alert. Each stage introduces its own error modes. Representation quality can degrade because of viewpoint, occlusion, or noisy pose extraction. Classification can fail because of domain mismatch or limited temporal context. Alerting can fail because thresholds, smoothing, and persistence logic produce unstable operational behaviour.

This layered interpretation matters because it changes what counts as success. In some parts of the literature, success is defined primarily through benchmark discrimination under fixed dataset conditions. In more deployment-oriented work, success also depends on whether the system can suppress false alarms, preserve useful recall, and maintain clear semantics between transient model output and incident-level response. The present project adopts the second view. It keeps a formal offline comparison, but it also treats alert generation, persistence, review, and caregiver-facing delivery as part of the technical artifact.

This positioning also helps explain the structure of the present report. The project is not framed as a pure leaderboard exercise, nor as a purely software-driven proof of concept. Instead, it asks how far a pose-based monitoring stack can be defended when benchmark evidence, policy evidence, and runtime evidence are all kept visible. In that sense, the report belongs to a strand of work that treats monitoring systems as multi-layer artifacts whose value depends on the consistency of the whole path from representation to reviewable action.

## Vision-Based and Pose-Based Detection

Fall detection has been approached through a range of input representations, including raw RGB video, silhouettes, handcrafted motion features, trajectories, and skeleton-based pose sequences. Earlier vision-based work such as Charfi et al. (2013) demonstrates the value of spatio-temporal visual descriptors, while more recent fall-detection studies continue to explore real-time video and skeleton-oriented variants. RGB-heavy methods can exploit rich scene context and appearance cues, but they often inherit scene-specific bias and can be sensitive to lighting variation, clothing, camera placement, and background structure. These properties can make them powerful in-domain while also making transfer and deployment interpretation more difficult.

Pose-based methods pursue a different compromise. By reducing the input to body structure and motion over time, they shift attention away from appearance and toward movement dynamics. This can improve interpretability and can make the representation more compatible with privacy-oriented deployment designs. In the present project, that advantage is especially important because the same pose abstraction is used in both offline evaluation and the runtime monitor. This continuity is methodologically valuable: it avoids a hidden gap between the representation used to report performance and the representation used to produce operational behaviour.

At the same time, skeleton-based methods are not inherently robust. They depend on upstream pose quality, confidence handling, temporal smoothing choices, and frame-rate consistency. In many systems, those assumptions remain implicit. In this project, they are treated as part of the method, because they directly shape both training-time windows and runtime behaviour. This is one of the reasons pose-based monitoring is attractive here: not because it removes deployment difficulty, but because it makes the interaction between representation quality and monitoring behaviour easier to analyse explicitly.

The project therefore aligns most closely with literature that views pose not merely as an offline research convenience, but as a deployable representation in its own right. That is a stricter and more demanding design choice than using skeletons only inside a benchmark pipeline. It requires tighter control of preprocessing and timing semantics, but it also produces a cleaner systems argument: the representation used for evaluation is materially the same as the representation used in monitoring.

Unlike work that evaluates pose features only as an offline input representation, this project keeps the pose-window contract active across both benchmark evaluation and the runtime monitor.

## Temporal and Graph Sequence Models

Because falls are structured events that unfold over time, temporal modelling is central to modern fall-detection pipelines. A purely frame-wise view is usually too weak, since the distinction between a fall and an ordinary activity often lies in the temporal pattern of motion rather than in any single pose. This is why temporal architectures remain central in the literature, even when the input representation is already compact.

Temporal Convolutional Networks offer one strong and practical approach, with work such as MS-TCN showing how temporal convolution can model structured time-series behaviour efficiently. They can model local and medium-range temporal dynamics, train stably, and fit naturally into fixed-window inference pipelines. Graph-based sequence models pursue a different hypothesis: if the body is represented as an articulated structure, then explicitly modelling joint topology may improve recognition by preserving relational information that a plain temporal encoder may flatten or ignore. This line connects to ST-GCN and later adaptive graph-convolution work in skeleton action recognition.

The comparison between these families is therefore meaningful, but it also needs to be interpreted carefully. Much of the broader literature compares architectures under changing datasets, preprocessing choices, evaluation rules, and decision policies. Such comparisons may still be informative, but they make it difficult to isolate what the model family itself contributes. The present project narrows that problem deliberately. It compares a strong TCN baseline with a matched custom spatio-temporal GCN under a shared pose-window contract, shared fitting rules, and a shared alert interpretation framework.

This narrower comparison does not claim to resolve the value of graph structure in general. Instead, it asks a more defensible question for a research-led project dissertation: under one controlled operational contract, does a topology-aware alternative improve practical fall-detection behaviour relative to a strong temporal baseline? This framing is academically stronger than a more generic “TCN versus GCN” comparison because it makes clear what has been controlled and what has not. It also aligns with the later system chapters, where model outputs are not interpreted in isolation but under a fixed policy and runtime path.

Unlike broader architecture surveys, the contribution here is not to rank model families in the abstract. It is to compare two concrete families after fixing representation, windowing, fitting, and runtime interpretation.

## Calibration and Alerting in Monitoring Systems

A second weakness in parts of the fall-detection literature is that deployment behaviour is often reduced to a threshold selected late in the pipeline or described only briefly after the main classification results. That simplification is convenient for compact benchmark papers, but it is inadequate for a monitoring system. A deployed system does not act on threshold-free metrics. It acts on interpreted score streams.

This is where calibration and operating-point selection become methodologically important. Calibration work such as Guo et al. (2017) motivates treating probability interpretation as a separate concern from ranking accuracy, while selective-prediction work such as Geifman and El-Yaniv (2019) reinforces the value of handling uncertain cases explicitly. In a monitoring context, raw model outputs must be transformed into alert states through a policy layer that may include threshold fitting, smoothing, temporal aggregation, cooldown logic, and persistence rules. These choices affect false-alert burden, latency, event fragmentation, and the distinction between transient suspicion and reviewable incident state. As a result, alert policy should not be treated as a cosmetic post-processing step. It is part of the deployment problem itself.

The present project takes this view explicitly. Validation-side temperature-informed fitting and tracked operating-point profiles are used so that deployment behaviour can be discussed as a defended and reviewable object rather than as a set of undocumented manual choices. This matters especially in safety-adjacent monitoring, where false positives and false negatives both carry operational cost. A model with attractive ranking metrics may still produce poor monitoring behaviour if its policy surface is unstable. Conversely, relatively small changes in smoothing or persistence rules can materially alter event-level performance without changing the underlying classifier.

By making calibration, operating-point fitting, and runtime semantics explicit, the project aligns with a more systems-aware reading of the literature. It also differentiates itself from narrower studies that report classification performance without closely examining how score sequences become operational decisions. In this report, the alert layer is not peripheral. It is the bridge between model evidence and system behaviour.

This differs from a benchmark-only threshold discussion because the fitted policy is carried into the monitor, event persistence, and notification path.

## Deployment-Oriented Evaluation

A final body of related work concerns evaluation beyond frozen offline benchmarking. Between formal test-set evaluation and uncontrolled real-world deployment lies an intermediate space: replay analysis, runtime matrices, operator-facing demonstrations, and delivery-path checks. These forms of evidence are useful because they test the integrated system rather than only the trained model. They can expose mismatches between frontend representation, backend interpretation, persistence semantics, and notification behaviour that do not appear in offline metrics alone.

However, deployment-oriented evidence also needs careful interpretation. Replay and live demonstrations can easily be over-read as though they established broad generalisation or field readiness. In reality, they answer a different question: whether the monitored path behaves coherently under bounded conditions. This distinction is central to the present project. Offline frozen evaluation is used to answer the architecture-comparison question most directly. Replay and live evidence are used to answer the system-feasibility question. Delivery-path checks are used to show that persisted events can reach caregiver-facing notification, not to strengthen the underlying detection claim.

This layered evidence policy places the project between two familiar traditions. On one side are benchmark-driven studies that prioritise model comparison and say relatively little about runtime integration. On the other side are systems-oriented studies whose contribution lies in workflow, integration, or operator-facing behaviour even when they do not redefine the best benchmark model. The present report draws from both, but its contribution is strongest when read as a research-and-systems study with explicit evidence boundaries.

That mixed position is one of the report’s main strengths. It allows the project to retain benchmark discipline while also asking a harder systems question: whether the selected monitoring pipeline still makes sense once inference, alert policy, persistence, review, and delivery are treated as parts of one technical artifact. The literature reviewed above motivates exactly that question and explains why the rest of the report is organised around it.

# Research Questions and Scope

## Locked Research Questions

The full report uses the same locked research questions as the compact paper line, but here they are expanded in a form more suitable for a supervisor-facing project report.

RQ1 asks how the TCN and the custom spatio-temporal GCN compare under the frozen offline protocol. This question covers the model-comparison layer and is answered primarily with stability and significance evidence rather than with replay or field artifacts.

RQ2 asks how validation-side operating-point fitting changes practical alerting. This question exists because the project is not only about score ranking; it is about how model outputs become decisions that a monitoring system can act on.

RQ3 asks what replay deployment evidence and limited realtime validation reveal about practical feasibility and runtime limits. This question explicitly accepts that deployment evidence is a different evidence layer from unseen-test benchmark evidence. It is designed to capture what the system can honestly demonstrate today without overstating field readiness.

## Scope Boundaries

The project scope is deliberately narrower than a claim of general-purpose fall-detection readiness.

The primary benchmark and deployment-target dataset is `CAUCAFall`. `LE2i` is retained as mandatory comparative and generalisation evidence, but the final deployment-facing narrative is not built around `LE2i`. This asymmetry is not arbitrary; it reflects the strength and quality of the current evidence base.

`MUVIM` is included as a real experimental track, but not as a third co-equal benchmark in the main claim hierarchy. Its role in the full report is to show experimentation breadth, metric-contract discipline, and the wider use of the pipeline rather than to redefine the main results chapter.

Replay clips, delivery-style custom checks, and the small field-validation pack are treated as system-evidence artifacts. They are valid and useful, but they do not replace frozen offline benchmark evidence. This distinction must remain explicit throughout the full report, especially because the larger report format makes it easy to accidentally give secondary evidence too much narrative weight.

The report also remains bounded in what it says about calibration and notification delivery. Validation-side calibration is part of operating-point fitting, and Telegram delivery is part of the implemented system path. Neither point should be stretched into a claim that every runtime score is fully calibrated online or that caregiver notification delivery by itself proves model quality.

# System Architecture

## Overall Architecture

The system is organised as a full-stack pose-based fall-detection platform rather than as a disconnected set of scripts. At a high level, it consists of five interacting subsystems:

1. pose extraction and preprocessing
2. temporal window generation and feature construction
3. model inference and operating-point policy evaluation
4. event persistence and review
5. caregiver-notification delivery

The implementation boundary between these subsystems is important to the report because many of the project’s later findings are not caused by one model checkpoint in isolation. Instead, they arise from interactions between frontend pose quality, backend inference timing, and the policy layer that turns window-level outputs into alert states.

The final repository structure mirrors this system view. Core data, model, and deploy-time logic live under `ml/src/fall_detection`. The API and persistence path live under `applications/backend`. The user-facing monitoring interface lives under `applications/frontend`. Runtime configuration artifacts are preserved under `ops/configs`, promoted deployment assets under `ops/deploy_assets`, and experimental outputs under `outputs`. This separation is one reason the project was able to support a substantial later audit and code-review phase without collapsing into an unrecoverable state.

Architecturally, the most important design decision is that the system is split by responsibility rather than by development stage. The frontend is not treated as a thin visual shell, and the backend is not treated as a generic prediction box. Instead, the frontend owns input capture and pose-window construction, while the backend owns operating-point selection, temporal policy evaluation, persistence semantics, and downstream delivery. This boundary is analytically useful because it explains where runtime deviations can arise. If a replay result changes, the cause may be frontend pose quality, backend policy interpretation, or the interaction between the two, not simply a different checkpoint.

The architecture also reflects a deliberate separation between benchmark and deployment concerns. Offline data preparation, training, operating-point fitting, and frozen summaries all exist independently of the monitoring application. That makes it possible to defend the benchmark layer even when runtime behaviour is still being refined. At the same time, the deployment path can be studied as a genuine software system because it consumes tracked operating-point artifacts and real model checkpoints rather than placeholder rules.

This structure has practical consequences for traceability. The report can point from a narrative claim to a configuration file, from a configuration file to a fitted policy, and from that policy to runtime behaviour observed in the monitor and event history.

**Figure 1. System architecture and decision path**

Asset:
- [system_architecture_diagram.svg](artifacts/figures/report/system_architecture_diagram.svg)

![Figure 1. System architecture and decision path](artifacts/figures/report/system_architecture_diagram.svg){ width=85% }

Figure 1 presents the system as a responsibility map. The frontend produces pose-derived windows, the backend owns model execution and alert-policy interpretation, persistence records fall events for later review, and the notification layer consumes persisted outcomes rather than bypassing the event path.

## Frontend Architecture

The frontend is implemented in React and acts as the operator-facing control surface for both live and replay monitoring. Its responsibilities go beyond page rendering. It must acquire camera or replay input, run browser-side pose extraction, assemble temporal windows at the required contract, submit those windows to the backend, and present the resulting alert state in a way that remains intelligible to the operator.

The monitoring page is therefore the most important frontend subsystem. It exposes the active dataset/model/operating-point selection, live-versus-replay mode selection, and several runtime controls such as preview visibility, replay playback, and realtime event persistence semantics. These are not cosmetic controls. They directly shape whether the monitoring path is being used as a demonstration path, a bounded replay-validation path, or an event-producing system path.

The frontend also contains dashboard, settings, and events pages. The dashboard summarises persisted outcomes, the settings page controls active runtime preferences and caregiver information, and the events page exposes reviewable event history. Together these interfaces mean that the system is more than a live demo viewer; it is a small but coherent monitoring application.

Architecturally, later refactoring separated feature-level API logic, monitor hooks, and page components. This was necessary because the initial monitor path mixed state management, media handling, API transport, and UI rendering in ways that made runtime behaviour difficult to reason about.

This separation makes the monitor page easier to explain as a composition of concerns: media capture, pose generation, stateful runtime control, and render-layer presentation. It also makes later fixes, such as replay-versus-realtime persistence semantics and live-video preview toggling, easier to locate in the codebase.

The frontend also embodies several trade-offs that are worth stating in the report. Keeping pose extraction in the browser reduces backend responsibility and supports a more privacy-oriented flow, but it also means that frontend performance and pose quality directly affect system behaviour. Exposing replay and live modes through the same monitor improves consistency, but it increases the burden on state handling because the same page must support demonstration, debugging, validation, and event-producing operation without silently changing semantics. The later code review showed that these trade-offs were real and that the final implementation is stronger precisely because they were made explicit.

The same monitor surface is used for live demonstration, replay validation, and event-producing operation, but these modes are not semantically identical. The final frontend design is stronger because these differences are represented as explicit state and controls rather than hidden UI side effects.

## Backend Architecture

The backend is implemented in FastAPI and acts as the coordination layer for deployable inference. It owns the active runtime profile, resolves operating-point configurations, executes model inference, applies temporal policy, persists events, and exposes reviewable data through route handlers.

The most important backend path is the monitor runtime path. It receives frontend-produced windows, applies the active model and fitted policy, and decides whether the system is in a neutral, uncertain, or fall-like state. This path does not simply return a thresholded score. It is responsible for policy-aware behaviour, including smoothing, `k/n` logic, cooldown handling, and the distinction between transient monitor state and persisted fall event.

The backend also supports a repository/service split. Route handlers are now thinner than they were in earlier versions of the project, while data-access and runtime logic are delegated to dedicated layers. This matters because event history, dashboard summaries, and notification delivery all need to observe the same persisted semantics if the deployment chapter is to remain credible.

The backend’s design problem is therefore not only one of inference latency. It must maintain semantic consistency. The active dataset, model family, operating-point code, runtime defaults, persistence policy, and notification settings all need to align across monitor requests, dashboard summaries, event records, and caregiver-delivery logic. Several of the later review findings came from exactly this class of issue: not that one endpoint crashed, but that different parts of the system were interpreting the same runtime state slightly differently.

This is why the service and repository split matters. A route-centric implementation would have made it much harder to reason about shared contracts such as event status values, active operating-point normalization, notification truth sources, or dashboard counting semantics. In the current architecture, those behaviours can be discussed as explicit backend contracts rather than as accidental properties of route handlers.

The backend is also where the project’s distinction between transient evidence and persistent evidence becomes operational. A monitor may briefly enter a fall-like state without creating a persisted event, depending on temporal policy and session mode. That distinction is one of the most important design choices in the system, because it prevents momentary runtime fluctuations from being confused with reviewable incident history.

The final architecture is stronger because persistence, dashboard summaries, and delivery logic are all downstream of the same interpreted event path rather than parallel approximations of it. This is one reason the code review placed so much emphasis on shared semantics such as event type, status, operating-point normalization, and notification truth sources.

## Notification Architecture

The final implemented notification path is Telegram-based. When a fall event is persisted and the active notification path is enabled, the backend dispatches a caregiver-facing Telegram message. The notification includes a generated summary rather than only a bare alert label, so the delivery path is informative as well as immediate.

This notification architecture is intentionally narrower than the earlier multi-channel concept discussed during development. Email, SMS, and phone escalation remain future-work directions rather than active delivery claims. This is the correct trade-off for the current report: it lets the system demonstrate true end-to-end alert delivery without forcing unsupported claims about broader notification infrastructure.

The notification path also has its own audit store, which is important for report credibility. Delivery attempts should be interpreted through the actual notification audit path rather than through any older UI-queue abstractions that may still exist in historical development material.

Narrowing the active channel to Telegram made it possible to complete and verify one real end-to-end delivery path instead of leaving several partially wired channels in an unfinished state. The audit store attached to the notification manager keeps the deployment chapter aligned with real delivery state rather than optimistic UI summaries.

## Deployment Modes

The repository supports multiple deployment modes, but they should not be conflated.

The first mode is lightweight local use, where the frontend and backend are run in a development-style environment and the goal is rapid iteration or local demonstration. The second mode is a fuller Docker-backed deployment path, which better reflects persistent operation and integrated service behaviour. The third is a more remote deployment shape in which the frontend can be hosted separately from the backend.

The key architectural point is that frontend pose extraction remains local to the client/browser even when the backend is remote. This means that cloud hosting of the backend does not remove frontend pose-quality dependence. Many runtime behaviours remain jointly determined by client-side pose generation and backend-side policy evaluation.

This distinction matters because deployment discussion can easily become misleading if the report talks about "the deployed system" as though there were only one runtime shape. A local demo on one machine, a Docker-backed integrated stack, and a split frontend-backend deployment all exercise different operational assumptions. The report therefore uses deployment-mode language carefully and specifies which mode underlies each validation claim.

In practical terms, the current deployment story is strongest in local and Docker-backed bounded operation, where the full event path can be observed and audited. Remote hosting remains technically possible, but it does not remove the frontend’s role in pose generation and therefore does not eliminate the main source of runtime input variability. This is another reason the report keeps deployment claims bounded rather than generic.

This deployment discussion also reinforces a larger theme of the report: system quality is shaped by where interpretation happens. In this project, some interpretation happens in the browser through pose and window formation, some in the backend through policy application, and some in the review/delivery layers through persistence and notification. A high-quality report therefore cannot talk about deployment as if it were only a server-hosting question. It is an end-to-end question about which parts of the stack own which pieces of meaning.

# Data and Experimental Protocol

This chapter defines the experimental contract used throughout the report. Its purpose is not only to describe data preparation, but also to specify the methodological boundaries that make later comparison, policy fitting, and runtime interpretation defensible. In this project, dataset role assignment, split construction, preprocessing, temporal windowing, and evaluation policy are all part of the evidence design rather than incidental implementation detail.

## Dataset Roles

The report uses three datasets, but they do not play identical argumentative roles.

`CAUCAFall` is the primary benchmark and the primary deployment-target dataset. It anchors the main offline comparison, the preferred fitted operating-point profile, and the strongest bounded replay/runtime evidence. It therefore carries the greatest weight in the report’s final defended system narrative.

`LE2i` is retained as a mandatory comparative dataset, but its role is different. It functions primarily as comparative and transfer-boundary evidence rather than as the centre of the deployment-facing story. This distinction is deliberate. `LE2i` remains essential because it prevents the project from collapsing into a single-dataset success claim, but it is not the dataset on which offline comparison, fitted policy, and runtime evidence align most cleanly.

`MUVIM` is included as a secondary exploratory track. Its role is not to compete directly with the main `CAUCAFall`-`LE2i` comparison, but to demonstrate that the pipeline was exercised beyond the final two-dataset core and that metric-contract and fitting issues mattered outside the main benchmark path.

The report also distinguishes benchmark datasets from deployment-support evidence. Replay clips, bounded live checks, and delivery-style validation slices are used as system-evidence artifacts, not as substitutes for frozen benchmark test splits. This separation is central to the report’s evidence discipline.

**Figure 2. Dataset roles and evidence hierarchy**

Asset:
- [dataset_roles_evidence_hierarchy.svg](artifacts/figures/report/dataset_roles_evidence_hierarchy.svg)

![Figure 2. Dataset roles and evidence hierarchy](artifacts/figures/report/dataset_roles_evidence_hierarchy.svg){ width=85% }

Figure 2 makes the evidence hierarchy explicit. It shows that `CAUCAFall`, `LE2i`, and `MUVIM` do not play identical roles in the report, and that replay, field, and live artifacts are linked to system validation rather than to the primary benchmark claim.

## Label and Split Construction

Labels, spans, and split files are treated as tracked artifacts rather than invisible preprocessing side effects. This is methodologically important because the validity of the later comparison depends on stable train/validation/test boundaries and on reproducible mappings from raw source sequences to benchmark units.

For `CAUCAFall`, label generation and span reconstruction are tied to the dataset’s project-level FPS contract and to the sequence metadata used in deployment-facing window generation. For `LE2i`, labeling and split logic follow its own dataset conventions. In both cases, split construction defines the boundary between training evidence, validation evidence, and held-out evaluation evidence. It is therefore part of the formal protocol, not just a convenience step in the pipeline.

This distinction matters for two reasons. First, operating-point fitting is performed on validation outputs only, so unstable split logic would directly weaken later policy claims. Second, cross-dataset transfer results are only interpretable if source and target boundaries remain fixed and reproducible. For this reason, labels, spans, split files, and window metadata are treated as evidence-bearing artifacts in the repository.

## Dataset-Specific Protocol Notes

The datasets differ not only in content, but also in the way they constrain interpretation.

For `CAUCAFall`, the main methodological issue is alignment between raw labels, reconstructed spans, FPS assumptions, and the deployment-facing window contract. Because this dataset also supports the strongest bounded runtime evidence, consistency between offline protocol and runtime timing semantics is especially important.

For `LE2i`, the main issue is its dual role. It functions both as an in-domain benchmark dataset and as a transfer-boundary target. This makes it valuable for comparative evaluation while also exposing where fitted behaviour degrades under domain shift.

For `MUVIM`, the main issue is not headline comparison but methodological breadth. It demonstrates that the same data, fitting, and evaluation machinery was used beyond the final two-dataset core, and it helps show that metric-contract discipline affected interpretation in more than one experimental setting.

## Pose Preprocessing

The preprocessing path converts raw pose outputs into cleaned temporal skeleton sequences suitable for model training and runtime inference. This includes confidence-aware handling, temporal smoothing, gap handling where appropriate, and the enforcement of dataset-specific timing assumptions. These stages are not treated as cosmetic cleanup. They define the quality of the representation consumed by both the offline pipeline and the deployed monitor.

A core design choice in the project is that the runtime system and the offline experiments share the same pose-first representation family. This improves interpretability and strengthens the connection between benchmark and deployment behaviour, but it also means that preprocessing becomes a first-order methodological concern. If pose quality degrades, then both offline comparison and runtime alert behaviour may degrade for reasons that are upstream of the classifier itself.

Dataset-specific frame-rate contracts are therefore treated as part of preprocessing truth rather than as incidental defaults. In the current protocol, `CAUCAFall` is handled at `23 FPS`, `LE2i` at `25 FPS`, and `MUVIM` at `30 FPS`. These contracts affect label-span alignment, window timing, event interpretation, and replay/runtime consistency. As a result, preprocessing choices are not neutral with respect to later deployment claims.

## Temporal Window Contract

The main evaluation path is built around a locked temporal contract of `W = 48` and `S = 12`. This contract defines the temporal view presented to both architectures, constrains feature construction, and provides a shared interface between offline inference and backend runtime interpretation.

The importance of this contract is methodological as well as technical. It ensures that the TCN and the matched custom GCN are compared under the same temporal geometry, and it prevents later deployment analysis from quietly shifting to a different runtime representation. In other words, the reported results are functions not only of architecture choice, but also of a stable pose-window interface.

Window metadata are also part of the contract. In the reviewed codebase, window start and end positions are treated as frame-index metadata on the target FPS grid rather than as loosely defined timestamps. This matters because span overlap, event grouping, replay inspection, and later audit work all assume that window boundaries have stable semantics.

A second important distinction follows from this design. Window labels are assigned offline through span overlap under the locked temporal contract, whereas persisted events are created later through policy interpretation, smoothing, and persistence logic. The two layers are related, but they are not identical. Keeping that distinction explicit prevents the report from collapsing offline labeling and runtime incident semantics into a single step.

**Figure 3. Temporal window contract and span-overlap semantics**

Asset:
- [temporal_window_contract.svg](artifacts/figures/report/temporal_window_contract.svg)

![Figure 3. Temporal window contract and span-overlap semantics](artifacts/figures/report/temporal_window_contract.svg){ width=85% }

Figure 3 clarifies one of the most important protocol boundaries in the project. Window labels are assigned through span overlap under the locked `W=48`, `S=12` contract, but later persisted events are formed through additional runtime interpretation. This figure therefore helps the report distinguish offline labeling from deployment-time event semantics without making those two layers look unrelated.

## Evaluation Policy

The project uses a strict evidence policy to prevent contamination between methodological layers.

Operating-point fitting is performed on validation data only. Test data are not used to choose thresholds, tune temporal alert-policy parameters, or refine deployment profiles. In cross-dataset transfer evaluation, fitting is performed on the source-side validation split rather than on the target test split. Replay and bounded deployment tuning are not allowed to redefine the formal offline claim.

These rules create a clear separation of evidential roles. Frozen offline evaluation answers the architecture-comparison question. Validation-side fitting answers the alert-policy question. Replay and live runtime evidence answer the bounded system-feasibility question. Delivery-path checks answer whether persisted events can reach caregiver-facing notification. None of these layers is allowed to substitute for another.

This policy is one of the report’s most important methodological safeguards. Because the repository contains many forms of evidence, strong-looking artifacts from one layer could easily overshadow weaker but more appropriate artifacts from another. The evaluation policy acts as a firewall against that drift. It allows the report to be broad without becoming methodologically loose.

For that reason, this chapter forms part of the project’s defended evidence design rather than background setup. Dataset roles, split construction, preprocessing, temporal windowing, and evaluation policy together define the contract under which the later results can be interpreted.

# Model Design and Training

This chapter describes the model families, shared representation design, and training protocol used in the project. Its purpose is not only to introduce the architectures, but also to show how architectural comparison was kept methodologically controlled. In the present study, the central question is not which model can be made to look best under loosely varying conditions, but whether a strong temporal baseline and a matched graph-based alternative behave differently under one locked pose-window contract and one defended evaluation policy.

## Temporal Convolutional Network

The Temporal Convolutional Network (TCN) serves as the strongest practical baseline in the project and also as the model family underlying the preferred bounded deployment preset. Conceptually, it treats each pose window as a structured temporal signal and learns fall-relevant motion patterns through convolution over time. This makes it well suited to a fixed-window monitoring pipeline, where stable temporal inference matters as much as raw benchmark discrimination.

The TCN is methodologically important for two reasons. First, it provides a strong baseline against which a topology-aware alternative can be compared under the same protocol. Second, it is operationally compatible with the runtime system because it maps naturally from fixed-length pose windows to policy-shaped monitoring behaviour. In the present report, the TCN is therefore more than a benchmark model. It is the architecture family that most cleanly connects offline comparison, operating-point fitting, and bounded replay/runtime evidence.

This point is important for interpretation. The TCN is not treated as an abstract architecture evaluated in isolation, but as one component in a larger monitoring stack. Its reported behaviour is therefore understood as the product of model design, shared preprocessing, locked temporal windowing, and downstream policy interpretation.

## Custom Spatio-Temporal GCN

The graph-based comparison model is a custom spatio-temporal Graph Convolutional Network rather than a strict reproduction of one canonical external implementation. This distinction should remain explicit. The purpose of the GCN in this project is comparative rather than exhaustive: it provides a topology-aware alternative to the TCN under the same pose-window contract, shared feature family, and frozen fitting protocol.

This makes the GCN methodologically useful even where it underperforms the TCN. Its role is to test whether preserving joint-graph structure yields a practical advantage once the rest of the monitoring contract is held fixed. That question matters because graph-based models are often assumed to be naturally stronger for skeleton data. The present project does not adopt that assumption uncritically. Instead, it asks whether explicit body-topology modelling improves practical fall-monitoring behaviour under controlled conditions.

The results are best read through this narrower lens. The comparison does not establish a universal verdict on graph-based modelling in fall detection. It establishes something more defensible: under one locked operational contract, the matched TCN remains easier to defend as the stronger architecture family.

## Feature Construction

A central design decision in the project is that both model families operate over the same basic pose-window representation. This is essential to the fairness of the comparison. The feature path includes motion-aware temporal channels, confidence-aware preprocessing, and the project’s standard skeleton representation under one locked windowing rule. As a result, the TCN-versus-GCN comparison is not a comparison of unrelated data pipelines, but a comparison of two architectures exposed to the same underlying evidence contract.

This shared feature design matters scientifically because it reduces a common source of ambiguity in architectural comparison. If preprocessing, representation, and temporal geometry all vary between models, it becomes difficult to identify which part of the stack is actually responsible for an observed advantage. By contrast, the present design narrows that ambiguity. Feature parity does not remove every possible source of bias, but it materially strengthens the interpretability of the comparison.

It also matters systemically. The backend runtime path reconstructs inference windows under the same semantics assumed by the offline pipeline. This continuity is one of the reasons the report can connect model comparison to bounded deployment evidence without quietly switching representation between the two settings.

## Training Protocol

The training protocol is governed by a frozen-candidate discipline. Final candidate models are selected under tracked commands, candidate roots, and reproducible artifact paths. After this freeze step, the main comparative results are reported through fixed-seed summaries rather than through isolated best-run anecdotes. This design protects the report against post hoc selection and makes the model-comparison layer more auditable.

The seed policy is particularly important. The report’s main architectural claim is not built on a single convenient run, but on a fixed multi-seed comparison. This makes the final claim appropriately cautious. The evidence supports a directional advantage under the frozen protocol, not a universal claim of architectural superiority. In a research-led project dissertation, that restraint is methodologically preferable to a stronger but less defensible result statement.

Checkpoint provenance is treated with the same discipline. Training outputs record only metadata that can be justified from actual run configuration and tracked artifacts, rather than writing assumed development history back into checkpoints after the fact. This matters because later offline summaries, fitted operating points, and deployment presets must remain traceable to real model roots. In that sense, the training protocol is not only an optimisation recipe; it is part of the defended evidence chain of the report.

## ML Pipeline Strategy and Selection Logic

The repository preserves far more ML-facing work than the main report can narrate usefully. That breadth is an asset for traceability, but it would weaken the report if every sweep, candidate family, and temporary metric dump were promoted into the main argument. The correct report-level question is therefore not “what experiments exist?” but “which ML work materially strengthens the defended project story?”

The answer is selective. The highest-value ML work in the project falls into five categories. First, the locked offline comparison provides the formal model-evaluation backbone and should remain the primary source for `RQ1`. Second, validation-side operating-point fitting and temperature-informed policy design matter because the project is not only comparing architectures; it is turning score streams into alert-worthy runtime behaviour. Third, the online recovery sequence matters because gate fixes, motion-support fixes, and online refitting explain why deployment-facing behaviour changed and why some runtime failures were system-path failures rather than simple checkpoint failures. Fourth, cross-dataset transfer work matters because it defines the limitation boundary that prevents the project from overclaiming general robustness. Fifth, the later targeted retraining work matters because it responded to a concrete missed-fall weakness and produced a modest but real bounded runtime improvement.

This filtering principle also clarifies what does *not* deserve equal narrative weight. Full threshold sweeps, every historical experiment family, temporary metric rechecks, and exploratory side tracks are useful for reproducibility and audit, but they do not all belong in the main text. In the present report, they are retained as supporting or archival evidence unless they directly strengthen a defended conclusion. This is why `MUVIM` is treated as a secondary methodological track rather than as a third co-equal benchmark, why representative TCN and GCN optimisation families are discussed only briefly, and why detailed sweep logs are kept below the main argument.

This strategy matters methodologically. It shows that the project did not equate “more runs” with “stronger evidence.” Instead, it promoted only the ML work that clarified one of three things: the locked architecture comparison, the policy layer that turns classifier scores into deployable behaviour, or the bounded strengthening work that improved the primary `CAUCAFall + TCN` line without pretending to solve deployment. That discipline is one of the reasons the final report can remain broad without collapsing into an experiment diary.

# Calibration and Alert Policy

This chapter addresses the second major layer of the project: how model outputs become operational monitoring behaviour. In a benchmark-only study, score ranking may be close to the end of the argument. In a monitoring system, it is only the beginning. A deployed runtime path must transform raw score streams into interpretable, reviewable, and bounded alert states. For that reason, calibration, operating-point fitting, temporal policy, and runtime semantics are treated here as first-class methodological components rather than as minor post-processing details.

## Validation-Side Temperature Scaling

Temperature scaling is used in the repository as part of the validation-side calibration path. In this project, calibration is not presented as a decorative add-on or as a claim that all runtime outputs are universally well calibrated. Its practical function is narrower and more important: it improves the usefulness of validation outputs for downstream operating-point fitting.

This distinction remains explicit in the report. The defensible claim is that calibration informs operating-point design on validation data. The report does not imply that every probability emitted by the runtime system is fully recalibrated online unless that transformation is demonstrably present in the active inference path. This level of precision matters because calibration language can easily be overstated in deployment-facing writing.

Within the present project, calibration is therefore best understood as part of policy design rather than as an independent headline result. It strengthens the interpretability of the alert layer, but its value is realised through the fitted policy profiles that are later applied in runtime behaviour.

## Operating-Point Fitting

Operating-point fitting is the mechanism that turns model outputs into deployable alert profiles. Rather than choosing a threshold by inspection, the project fits a family of operating points on validation outputs and stores the resulting policy parameters in tracked YAML artifacts. These fitted profiles are then used by the runtime path. This makes deployment behaviour auditable and prevents the alert layer from becoming an undocumented collection of manual threshold choices.

The three operating points used in the active profile family are not arbitrary labels. They represent distinct policy intents under the same fitted model family. `OP-1` favours looser or faster alerting, `OP-2` functions as the balanced deployable profile, and `OP-3` applies a stricter interpretation. Their relative behaviour is itself informative. If operating points partially overlap or collapse under one fitted family, that is evidence about the score surface and policy regime, not merely a naming inconvenience.

This is one of the project’s most important methodological contributions. It shifts the question from “which architecture scores higher?” to “how should a monitored system convert score streams into reviewable alert behaviour?” That shift is central to the whole dissertation, because the project’s contribution lies not only in model comparison, but also in making the alert layer explicit, fitted, and reviewable.

**Figure 4. Alert-policy decision path**

Asset:
- [alert_policy_flow.svg](artifacts/figures/report/alert_policy_flow.svg)

![Figure 4. Alert-policy decision path](artifacts/figures/report/alert_policy_flow.svg){ width=85% }

Figure 4 is the clearest compact representation of the policy layer. It shows that the runtime path is not a single threshold crossing but a sequence of interpretation stages that include smoothing, policy application, persistence decisions, and downstream review/delivery handling. In the full report this figure plays an important bridging role between model design and runtime results.

## Temporal Policy Layer

The deployable alert path is not defined by one threshold alone. It includes temporal smoothing, `k/n` logic, cooldown handling, and optional confirmation behaviour. Together, these components determine whether the monitor remains neutral, becomes uncertain, or crosses into a fall-like state that can later produce a persisted event.

This temporal layer explains why offline classifier quality and deployable alert quality are not interchangeable. A checkpoint may rank windows well but still produce unstable incident behaviour once temporal policy is applied. Conversely, a policy can suppress false alerts while also masking weak recall. The practical question is therefore not only whether the model separates classes, but whether the fitted policy produces an operationally credible region of behaviour.

In the present report, this layer is not treated as peripheral engineering. It is the operational bridge between statistical output and system action. That is why the alert-policy chapter sits between model design and implementation, rather than appearing as a minor subsection at the end of the results chapter.

## Runtime Interpretation

Runtime interpretation is where model behaviour, fitted policy, and monitoring semantics finally meet. In the implemented system, the backend distinguishes between transient monitor states, persisted fall events, and downstream caregiver-facing delivery. These distinctions are methodologically important. A short-lived rise in fall-like score is not the same thing as a reviewable incident, and a persisted incident is not the same thing as a successful notification event.

Several of the project’s most important runtime lessons arise from this distinction. Late-stage debugging and code review showed that mismatches were often caused not by threshold choice alone, but by the interaction between frontend window production, backend policy application, and persistence semantics. For that reason, runtime interpretation should be understood as part of the scientific method of the project rather than as a purely engineering afterthought.

This perspective also explains why the alert layer belongs at the centre of the report’s argument. The system does not merely emit scores. It interprets them, groups them, persists them, and exposes them to human review and delivery logic. In a monitoring study, those steps are not downstream decoration. They are part of what determines whether the system is technically credible at all.

# Implementation and Refactoring

## Frontend Implementation

The frontend implementation centres on a React monitoring application that supports live input, replay input, dashboard review, event browsing, and settings management. The monitoring page is the critical path because it produces the pose-derived windows that the backend later interprets.

Over time, this frontend became complex enough that modularisation was required. Media handling, monitoring state, API transport, replay/live distinctions, and page-level rendering all needed clearer separation. The final frontend structure separates page components, feature-level API calls, and monitor hooks under `applications/frontend`.

From the report’s perspective, the frontend matters in two roles. It is the human-visible part of the system, and it is also one of the places where deployment evidence is actually produced. This is especially true for replay validation and live demonstration work.

The most important implementation outcome is that monitor behaviour is now represented through explicit state contracts rather than implicit UI side effects. Dataset and operating-point selection, replay controls, monitoring enablement, caregiver settings, realtime persistence behaviour, and preview state are all represented as controlled state. This reduces the chance that the page appears correct while the backend receives a different operational context.

Three frontend subsystems are especially important. The monitor hook layer coordinates capture mode, active profile state, replay controls, and backend interaction. The feature-level API layer isolates transport logic and makes API assumptions easier to test. The page/component layer keeps presentation concerns separate from runtime-state semantics. This division allowed later fixes such as replay-versus-realtime persistence clarification, live-preview toggling, and fallback-profile alignment to be implemented without deepening hidden coupling.

The settings and dashboard paths also matter more than they might in a smaller project. Settings are where caregiver configuration, active runtime preferences, and fallback semantics become user-visible. Dashboard and events views are where persisted meaning is inspected after the fact. Together they form the interpretive half of the application: without them, the monitor would remain a transient display rather than part of a reviewable system.

The frontend also acts as a temporal coordinator. Browser-side pose extraction and window packaging must remain compatible with the backend's expectation of dataset-specific frame rates and fixed window geometry. Operator input passes through selected state, derived runtime context, feature-level request shaping, and backend submission. Backend responses are then rendered as monitor state, event markers, and page-visible summaries. This flow is why preview toggles, replay controls, and fallback-profile alignment are treated as evidence-relevant implementation details rather than as cosmetic UI choices.

## Backend Implementation

The backend implementation centres on FastAPI routes supported by service and repository layers. It owns model loading, operating-point resolution, runtime-state transitions, event persistence, and caregiver-notification delivery. This architecture is particularly important in the monitor path, because the backend is where the system decides whether a score sequence has become a persistent event rather than a temporary visual state.

Earlier versions of the backend contained more mixed responsibility inside route handlers. Later refactoring moved more logic into services and repositories so that state transitions, persistence semantics, and schema compatibility could be reasoned about more clearly. This matters to the full report because the deployment story depends on the backend being understandable enough to audit.

The backend also now owns the Telegram notification path and its audit store. This is part of the system narrative, because the final report can show true end-to-end event-to-delivery behaviour without pretending that broader multi-channel delivery has already been completed.

One of the most important backend implementation lessons was that semantic consistency matters as much as raw functionality. Dashboard summaries, event-history rows, and notification delivery must be derived from the same event interpretation, not from loosely related intermediate states. Several late-stage fixes corrected exactly this kind of issue.

The backend can therefore be read as three cooperating layers. The inference-and-policy layer resolves the active profile, loads the relevant model family, and applies runtime interpretation. The persistence-and-summary layer stores events and exposes them through dashboard and event views. The delivery-and-audit layer turns persisted fall events into caregiver-facing notifications and records what actually happened in that path.

Runtime requests follow a checkpointed path: active dataset/model/operating-point context, frozen profile and checkpoint resolution, policy interpretation, event-persistence decision, review exposure, and then optional delivery. This flow makes later dashboard and notification evidence interpretable because monitor output, event state, review history, and delivery state are all tied to the same backend semantics.

Repository and schema compatibility work supported this same goal. Event-history views, dashboard summaries, and delivery logs are only useful if they reflect compatible event types, statuses, timestamps, and truth sources. The later repository and route work turned several hidden compatibility assumptions into explicit behaviour that could be reviewed and tested.

## Report-Relevant Frontend and Backend Work

Not every frontend or backend change deserves equal space in the report. The work that materially strengthens the project story is the work that turned a model demo into a reviewable monitoring system.

The first high-value frontend/backend contribution is the monitor-path responsibility split. The frontend is not a decorative shell around backend inference. It owns live/replay media intake, browser-side pose extraction, window construction, and operator-facing runtime controls. The backend is not a thin prediction endpoint. It owns active profile resolution, policy interpretation, event creation, persistence semantics, and downstream delivery. This split is worth promoting because it explains why the project should be evaluated as an end-to-end monitoring system rather than as a disconnected classifier plus interface.

The second high-value contribution is the separation of replay and realtime semantics. Earlier in the project, the same monitor surface could too easily encourage replay output, transient runtime state, and persisted evidence to be read as though they meant the same thing. The final frontend/backend contracts make that distinction explicit. Replay is now a bounded validation and demonstration path. Realtime monitoring is the event-producing path whose outputs can be persisted, reviewed, and delivered onward. This is more than a usability refinement. It is part of the report’s evidence discipline, because it prevents controlled replay behaviour from being mistaken for stored incident evidence.

The third high-value contribution is the persisted event and review path. The system does not stop at a monitor state or a probability trace. Backend policy can create a reviewable fall event, the event can appear in dashboard and event-history views, and that persisted record can then act as the truth source for caregiver-facing delivery. This workflow is one of the clearest reasons the project is stronger than a model-only prototype. It shows that the monitored path, review path, and delivery path are technically connected rather than being discussed as independent aspirations.

The fourth high-value contribution is runtime-path debugging and deployment recovery. Some of the most important late-stage improvements were not achieved by changing architecture family, but by repairing the interaction between frontend window production and backend policy evaluation. Gate fixes, motion-support fixes, online operating-point refits, and monitor-path stabilisation changed what the deployed system actually did under replay/runtime conditions. This matters to the report because it shows that deployment behaviour was improved through system-path understanding rather than through arbitrary threshold changes alone.

The fifth high-value contribution is frontend/backend contract alignment. Active profile defaults, fallback behaviour, monitor-state semantics, event-history meaning, and notification truth sources all had to be made consistent across both sides of the stack. That work deserves report space because it directly affects interpretability. A runtime result is only defendable if the frontend, backend, persistence layer, and delivery layer all agree on what state has actually occurred.

The sixth high-value contribution is the notification path itself, but only in a bounded sense. The important point is not simply that Telegram messages exist. The important point is that the system reaches a real end-to-end state in which a persisted event can produce caregiver-facing output through an auditable backend path. This supports the report’s systems contribution without requiring exaggerated claims about multi-channel deployment maturity.

Taken together, these frontend and backend contributions justify treating implementation work as evidence-bearing work rather than as secondary engineering detail. They are valuable in the report because they explain how the project moved from offline model outputs to a coherent monitored path with explicit runtime semantics, reviewable persistence, and bounded end-to-end delivery.

## Refactoring Timeline and Rationale

Refactoring became necessary because the project outgrew a small-prototype architecture. As more responsibilities accumulated in the monitor path, the code began to mix configuration defaults, runtime state, persistence semantics, transport logic, and UI behaviour in ways that made later debugging unnecessarily difficult.

The refactoring should not be described as an abstract software-quality exercise. It had a direct methodological purpose: reduce hidden coupling, stabilise runtime contracts, and make the relationship between code and report claims easier to trace. In practice, this included clarifying active operating-point defaults, aligning replay and realtime semantics, cleaning up notification-path truth sources, and separating UI concerns from transport and session concerns.

The timing of this work is important. Much of it happened after the project already had attractive results and a working monitor. The project first proved that the system could work, then invested in making that system understandable enough to defend. The later phases therefore targeted points where semantic drift threatened the final argument: default profile resolution, replay-versus-realtime meaning, event persistence, notification truth sources, and evidence/report boundaries.

The resulting implementation story is therefore cumulative rather than cosmetic. Early work established functionality. Mid-stage work added broader experimentation and system integration. Late-stage refactoring and review then reduced the gap between “the system appears to work” and “the system can be described precisely enough to defend.” That final reduction in ambiguity is one of the reasons the full report can sustain stronger conclusions than a loosely integrated prototype ever could.

## Audit and Code Review Work

The project later underwent a full-stack audit and a full code-review pass. These reviews were important because by that stage the repository contained enough code, artifacts, and report material that mismatch risk had become a real threat. The audit work focused on evidence drift, freeze-state discipline, configuration and artifact boundaries, and report-to-code consistency. The code review then moved deeper into the implementation itself, including ML contracts, backend runtime semantics, frontend state handling, and reproducibility tooling.

This phase produced substantive changes rather than only documentation. Examples include clarifying replay-versus-realtime persistence semantics, aligning operating-point defaults across frontend and backend, fixing notification truth sources, and tightening data/evaluation contracts in the ML pipeline. The importance of this phase to the full report is that it improves confidence that the final system description is not merely aspirational, but anchored to reviewed code paths.

The full report includes this audit and review work because it shows that the project underwent real late-stage verification rather than stopping once headline metrics existed. The implementation story is therefore cumulative: first functional, then structurally clearer, and finally easier to defend under scrutiny.

# Experimental Results

This chapter reports the main empirical findings of the project under the defended evidence hierarchy introduced earlier in the report. The results are organised by evidential role rather than by a single flat leaderboard. Frozen offline comparison provides the strongest basis for answering the architecture question. Cross-dataset transfer clarifies limitation boundaries. Operating-point and alert-policy results address how model outputs become deployable behaviour. Replay and runtime evidence then test whether the defended monitoring path remains coherent once it is exercised as software rather than only as an offline benchmark pipeline.

**Headline results.** The strongest defended model line is the `CAUCAFall` TCN under the frozen protocol. Cross-dataset transfer remains asymmetric and does not support broad robustness claims. The preferred runtime preset is `CAUCAFall + TCN + OP-2`, but the current bounded replay artifacts show profile-sensitive runtime behaviour rather than one uniformly strongest replay row. The uncertainty-aware runtime path is implemented, but it does not improve the fixed replay matrix. Telegram delivery is achieved on the active notification path; SMS and phone-call escalation remain future work.

## Offline Comparative Results

The strongest model-comparison evidence in the project comes from the frozen multi-seed offline protocol. This is the correct starting point for the results chapter because it is the least contaminated by deployment-side interpretation and the most directly aligned with the architecture-comparison question posed in `RQ1`.

Under the primary `CAUCAFall` protocol, the TCN shows higher mean `F1`, `Recall`, and `AP` than the matched custom GCN, while both models retain `FA24h = 0.0` under the frozen reporting contract. On the current branch, the five preserved `OP2` summaries yield mean `F1=0.8611`, mean recall `0.7600`, and mean `AP=0.9819` for the TCN, versus mean `F1=0.5873`, mean recall `0.4400`, and mean `AP=0.9706` for the matched GCN. This matters because it shows that the TCN advantage is not bought by accepting extra false-alert burden on the primary deployment-target dataset.

**Table 2. Main defended results synthesis**

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

**Figure 5. Offline stability comparison across frozen candidates**

Asset:
- [offline_stability_comparison.png](artifacts/figures/report/offline_stability_comparison.png)

![Figure 5. Offline stability comparison across frozen candidates](artifacts/figures/report/offline_stability_comparison.png){ width=90% }

Figure 5 is useful because it makes three points visible at once. First, the TCN advantage is not driven by a single exceptional seed. Second, the clearest gap appears on the primary dataset rather than being uniformly distributed across all settings. Third, false-alert burden remains visible alongside `F1` and `Recall`, which prevents the comparison from collapsing into score-only interpretation.

The `LE2i` in-domain comparison remains important, but it functions as comparative evidence rather than as the anchor of the main defended system claim. Under the frozen protocol, the TCN still trends stronger than the GCN on `LE2i`, although both models show substantially higher alert-rate burden than on `CAUCAFall`. This supports the report’s broader argument that `LE2i` is essential as boundary evidence even though it is not the main deployment-facing benchmark.

The inferential interpretation should also constrain the language used in this chapter. The current branch preserves the five-seed frozen summaries but not the older standalone significance bundle, so the strongest defensible conclusion is directional rather than definitive: the TCN trends stronger than the matched GCN under the frozen protocol, especially on `CAUCAFall`, but this should not be phrased as a conclusive universal superiority result.

Taken together, the offline results support three main conclusions. First, the cleanest comparative story is on `CAUCAFall`. Second, the TCN advantage survives frozen multi-seed comparison rather than depending on one convenient run. Third, the appropriate interpretation remains directional and bounded. This gives the chapter a disciplined empirical base before the argument moves into transfer, policy, and runtime behaviour.

## Cross-Dataset Results

Cross-dataset evaluation provides some of the strongest limitation evidence in the project. Rather than demonstrating broad robustness, these results show where transfer breaks and how metric behaviour changes when training and evaluation domains no longer match.

The most important finding is that transfer is asymmetric. The `CAUCAFall -> LE2i` direction remains a strong limitation boundary. For the TCN, event-level `F1` and `Recall` collapse to `0.0` even while `AP` remains non-zero. This is methodologically important because it shows that score ranking quality and final event behaviour can diverge sharply under domain shift. The GCN produces a different but not stronger pattern: event-level recall and `F1` partially recover in the transfer direction, but this recovery is accompanied by a poor false-alert profile, so the result does not translate into a clean operational success story.

The new targeted failure analysis makes this boundary more specific rather than merely more emphatic. On the frozen `CAUCAFall -> LE2i` transfer surface, the TCN failure mode is dominated by missed-fall collapse (`TP=2`, `FP=0`, `FN=7`, `TN=4`), whereas the matched GCN recovers more event hits only by becoming far less selective (`TP=7`, `FP=2`, `FN=2`, `TN=2`) and by producing extremely large false-alert rates on the negative clips. This means the cross-dataset limitation is not a generic “performance drop.” It is a trade-off shift in which one family collapses into false negatives while the other recovers recall only by inflating alert noise. That is a stronger limitation claim than a single aggregate `F1` delta because it identifies the actual failure structure. The derived note and figures are preserved in [TARGETED_FAILURE_ANALYSIS_2026-04-27.md](docs/reports/notes/TARGETED_FAILURE_ANALYSIS_2026-04-27.md), [cross_tcn_caucafall_to_le2i_failure_box.png](artifacts/figures/report/cross_tcn_caucafall_to_le2i_failure_box.png), and [cross_gcn_caucafall_to_le2i_failure_box.png](artifacts/figures/report/cross_gcn_caucafall_to_le2i_failure_box.png).

**Figure 6. Cross-dataset transfer summary**

Asset:
- [cross_dataset_transfer_summary.png](artifacts/figures/report/cross_dataset_transfer_summary.png)

Figure 6 should be interpreted as a directionality figure rather than as a leaderboard. Its purpose is to show how far each transfer condition moves away from its in-domain anchor and how that failure manifests across metrics. In this context, the most important question is not which bar is highest in isolation, but which transfer direction fails most severely and why.

The opposite direction, `LE2i -> CAUCAFall`, is materially less damaging and can even preserve stronger event-level behaviour under the frozen policy. This asymmetry is one of the most informative findings in the chapter. It shows that cross-dataset transfer in this project cannot be reduced to a single scalar notion of “generalisation quality.” It depends on direction, score surface, event policy, and false-alert burden at the same time.

These results support two important conclusions. First, the project genuinely tests transfer boundaries rather than remaining inside an entirely in-domain comparison. Second, the evidence does not support claims of broad cross-domain deployment robustness. This is precisely why the final deployment-facing narrative remains centred on `CAUCAFall`: not because weaker transfer results were ignored, but because they were examined and shown to be materially less stable.

## Secondary MUVIM Track

The `MUVIM` track is retained as supporting evidence for experimentation breadth and pipeline maturity. It is not presented as a co-equal benchmark axis with `CAUCAFall` and `LE2i`, but it remains valuable for interpretation.

Its primary role is methodological. `MUVIM` shows that the project’s data handling, fitting, and evaluation machinery was exercised beyond the final two-dataset comparative core. It also exposed how sensitive alert-oriented conclusions can be to metric-contract correctness and operating-point interpretation. In particular, refitting after event-metric corrections materially changed how the resulting operating behaviour should be read, even where threshold-independent score quality did not move in the same way.

This is useful in the present report for two reasons. First, it shows that the pipeline was not written only to produce one desired two-dataset conclusion. Second, it strengthens the credibility of the final evidence hierarchy by showing that the project encountered and corrected interpretation issues in a broader experimental setting. `MUVIM` therefore supports the methodological seriousness of the project without changing the main defended conclusion set.

## Calibration and Alert-Policy Results

The second research question is answered primarily through operating-point behaviour rather than through threshold-free model discrimination alone. This is why alert-policy results deserve their own place in the results chapter rather than being folded invisibly into the model-comparison discussion.

The active operating-point artifacts show that the deployed path is defined by a fitted policy profile rather than by a manually chosen threshold. In practice, this means that final monitoring behaviour depends on validation-side fitting, smoothing, `k/n` logic, cooldown, and related policy parameters rather than on a single score cutoff. The fitted operating-point table for the active `CAUCAFall` TCN profile provides the clearest concrete example of this layer in use.

Under the active profile family, `OP-2` preserves the strongest recall and therefore functions as the balanced deployable profile, while `OP-1` and `OP-3` trade recall for looser/faster or stricter behaviour. The fact that these operating points can partially overlap or collapse under one fitted family is itself informative. It shows that operating-point selection is a real policy-design problem, not a cosmetic renaming exercise applied after classification is complete.

This matters because the project’s strongest system contribution lies partly in this interpretive layer. The key question is no longer only which model scores higher, but how a monitored system should convert score streams into reviewable alert behaviour. This also explains later runtime evidence: when replay results differ across `OP-1`, `OP-2`, and `OP-3`, the difference is evidence about fitted policy shape under runtime conditions, not merely about arbitrary threshold choice.

More broadly, these findings show that alert-policy behaviour is not just a downstream reflection of model quality. Two models with similar threshold-free behaviour may still behave very differently once smoothing, cooldown, and persistence semantics are applied. Making this layer explicit is one of the report’s main methodological strengths.

## Deployment and Runtime Results

Deployment evidence in this project is strongest in bounded replay and delivery-style validation. This layer is valuable, but it remains distinct from the formal offline benchmark layer. Its purpose is to test whether the defended monitoring path remains coherent when exercised as software.

The most important runtime result is not a single triumphant replay number, but the fact that the bounded replay surface exposes meaningful profile-sensitive behaviour. The canonical replay summary preserved under `artifacts/ops_delivery_verify_20260315/online_replay_summary.json` shows that `caucafall_tcn OP-1` reaches `accuracy=1.0`, `recall=1.0`, and `specificity=1.0` on a bounded 10-video replay check, while `caucafall_tcn OP-2` drops to `accuracy=0.5`, `recall=0.0`, and `specificity=1.0` on that same surface. The aligned 24-clip custom replay trace is also bounded rather than uniformly strong: the defended baseline `tcn_caucafall_locked_op2` path yields `15/24` correct video-level outcomes (`TP=5`, `TN=10`, `FP=2`, `FN=7`), while `gcn_caucafall_locked_op2` yields `12/24` (`TP=12`, `TN=0`, `FP=12`, `FN=0`). The safe report-level interpretation is therefore not “one row wins everywhere,” but “runtime behaviour is real, bounded, and policy-shaped.”

The newer custom replay breakdown clarifies the trade-off structure inside those totals. `TCN + OP-2` is comparatively selective on the ADL folders, but it misses a substantial subset of fall clips, especially in the `kitchen` and part of the `corridor` set. `TCN + OP-1` recovers more fall hits on the same custom surface, but does so by sharply increasing ADL false alarms. The matched `GCN + OP-2` profile is more extreme still: it captures all positive folders while also collapsing both ADL folders into false alarms. In other words, the preferred `TCN + OP-2` preset is best understood as a bounded selectivity-versus-miss trade-off, not as a universally strongest replay line. This runtime-side error analysis is preserved in [CUSTOM_REPLAY_RUNTIME_ANALYSIS_2026-04-27.md](docs/reports/notes/CUSTOM_REPLAY_RUNTIME_ANALYSIS_2026-04-27.md) and [custom_replay_profile_breakdown_2026-04-27.csv](docs/reports/notes/custom_replay_profile_breakdown_2026-04-27.csv).

The clip-level error pattern makes the strengthening result more interpretable. Under the current `16/24` bounded result, the two persistent false positives are `kitchen_lie_1.mp4` and `kitchen_lie_2.mp4`, both of which involve controlled sofa-lying actions that remain close enough to fall-like descent-and-low-posture behaviour to trigger the policy path. The persistent false negatives are concentrated in `corridor_back_1.mp4`, `corridor_side_1.mp4`, `kitchen_back_1.mp4`, `kitchen_front_1.mp4`, `kitchen_front_2.mp4`, and `kitchen_side_2.mp4`. The strengthening runs recovered one corridor fall clip on the historical locked surface and several additional corridor-style cases on the canonical four-folder check, but they did not meaningfully improve the kitchen-heavy residual set. This is useful because it shows that the move from `13/24` or `15/24` to `16/24` was not a vague aggregate gain. It was a specific reduction in missed-fall behaviour on part of the surface while a smaller cluster of geometry-sensitive and persistence-sensitive failures remained.

A second point worth making explicit is that the deployment optimisation sequence was cumulative rather than single-stage. The initial bounded replay path did not fail only because one threshold was wrong. Early online re-verification showed that the runtime surface could suppress genuine falls almost completely. The later direct-window gate fix recovered part of that loss by allowing true positive windows to pass through the monitor path more reliably. The motion-support fix then improved how the runtime path interpreted movement and low-motion evidence, reducing another source of false suppression. Online operating-point refitting aligned the deployable alert profiles to the repaired runtime path rather than to a stale offline interpretation. Only after those system-path repairs did the later recall-oriented retraining make sense as a final strengthening step. The bounded runtime gains from `Candidate A` and `Candidate D` should therefore be read as the last stage of a broader deployment-alignment sequence: first repair the runtime decision chain, then refit the active policy, and only then push the primary `CAUCAFall + TCN` model line upward in a targeted way.

A plausible system-level explanation for the weaker `kitchen` subset is camera geometry rather than a purely abstract model deficiency. In the corridor setup, the camera is mounted higher (`~2.2m`) and the subject is typically farther from the lens (`~3m`), which yields a more stable whole-body view. In the kitchen setup, the camera is lower (`~2.0m`) and the subject is often much closer (`~1–1.5m`). At that shorter distance, the body is more likely to be only partially captured, perspective distortion is stronger, and the resulting skeleton track is less stable. This matters because the backend is trained on a fixed pose-window contract: when the runtime pose geometry departs more strongly from the training scenes, the model is asked to interpret windows whose observability and scale differ from the learned distribution. The subset pattern is consistent with that explanation: the strengthening runs improved the corridor fall clips, but the kitchen fall clips remained flat. One kitchen case also suggests a second mechanism beyond raw observability alone. In `kitchen_front_2`, the subject rises from the sofa and then falls forward very quickly in a flat collapse. That clip produces a very high instantaneous fall score, but still fails under `OP-2`, which is consistent with a short-lived fall spike that does not persist long enough to satisfy the stricter event-formation logic.

The project-strengthening retraining track slightly improves this runtime picture without changing its bounded interpretation. A recall-oriented continuation of the `CAUCAFall + TCN` line (`Candidate A`) improves the canonical four-folder replay check from `13/24` to `16/24` and the historical locked 24-clip surface from `15/24` to `16/24`, both without increasing ADL false positives. A second continuation run (`Candidate D`) reaches the same bounded runtime outcome. These are useful gains because they show that retraining can reduce false negatives on the defended custom surfaces, but they remain modest rather than decisive, and both candidates still required a confirm-disabled fallback during `fit_ops`. The strengthening artifacts are preserved in [CANDIDATE_A_RUNTIME_EVAL_2026-04-27.md](docs/reports/notes/CANDIDATE_A_RUNTIME_EVAL_2026-04-27.md), [CANDIDATE_A_LOCKED_SURFACE_EVAL_2026-04-27.md](docs/reports/notes/CANDIDATE_A_LOCKED_SURFACE_EVAL_2026-04-27.md), [CANDIDATE_D_RUNTIME_EVAL_2026-04-27.md](docs/reports/notes/CANDIDATE_D_RUNTIME_EVAL_2026-04-27.md), and [candidate_ad_runtime_surface_compare_2026-04-27.csv](docs/reports/notes/candidate_ad_runtime_surface_compare_2026-04-27.csv).

**Table 3. Deployment and runtime evidence synthesis**

| Scenario | Evidence type | Defended finding | Main limitation | Artifact anchor |
| --- | --- | --- | --- | --- |
| Canonical replay summary on `ops_delivery_verify_20260315` | replay/runtime summary | `caucafall_tcn OP-1` achieved `accuracy=1.0`, `recall=1.0`, `specificity=1.0` on the bounded `10`-video check | bounded replay evidence, not formal unseen-test evidence | `artifacts/ops_delivery_verify_20260315/online_replay_summary.json` |
| Same canonical replay summary | replay/runtime summary | `caucafall_tcn OP-2` fell to `accuracy=0.5`, `recall=0.0`, `specificity=1.0` on that same replay surface | shows replay/runtime behaviour is highly profile-sensitive | `artifacts/ops_delivery_verify_20260315/online_replay_summary.json` |
| Same canonical replay summary | replay/runtime summary | `caucafall_gcn` remained weak across `OP-1` to `OP-3` with `accuracy=0.5`, `recall=0.0`, `specificity=1.0` | runtime replay evidence does not rescue the weaker model line | `artifacts/ops_delivery_verify_20260315/online_replay_summary.json` |
| Bounded 24-clip custom replay matrix | per-video runtime trace | `tcn_caucafall_locked_op2` produced `15/24` correct video-level outcomes with `TP=5`, `TN=10`, `FP=2`, `FN=7` | more selective than the looser `OP-1` preset, but miss-prone on several fall clips | `artifacts/fall_test_eval_20260315/summary_tcn_caucafall_locked_op2.csv` |
| Same custom replay surface | per-video runtime trace | `tcn_caucafall_locked_op1` produced `11/24` correct video-level outcomes with `TP=9`, `TN=2`, `FP=10`, `FN=3` | recovers falls by substantially increasing ADL false alarms | `artifacts/fall_test_eval_20260315/summary_tcn_caucafall_locked_op1.csv` |
| Bounded 24-clip custom replay matrix | per-video runtime trace | `gcn_caucafall_locked_op2` produced `12/24` correct video-level outcomes with `TP=12`, `TN=0`, `FP=12`, `FN=0` | fall capture can coexist with unusable non-fall control on the same replay surface | `artifacts/fall_test_eval_20260315/summary_gcn_caucafall_locked_op2.csv` |
| Four-folder custom replay runbook | narrative boundary control | the current canonical four-folder profile is bounded and no longer supports any earlier perfect-profile reading | workflow evidence only | `docs/reports/runbooks/FOUR_VIDEO_DELIVERY_PROFILE.md` |
| Retraining strengthening track | bounded replay improvement evidence | `Candidate A` and `Candidate D` both improve the canonical four-folder replay check to `16/24`; `Candidate A` also improves the defended `15/24` locked-surface line to `16/24` | improvement is modest and both candidates require confirm-disabled `fit_ops` fallback | `docs/reports/notes/CANDIDATE_A_RUNTIME_EVAL_2026-04-27.md`, `docs/reports/notes/CANDIDATE_D_RUNTIME_EVAL_2026-04-27.md` |
| Realtime evidence pack | live demonstration evidence | the repository contains live-run evidence showing monitor execution and event-history style behaviour under a bounded live path | demonstration evidence only; not broad field validation | `artifacts/evidence/realtime/realtime_fall_submission.mp4`, `artifacts/evidence/realtime/realtime_adl_submission.mp4` |

This deployment synthesis table is useful because it forces the runtime story into a defensible shape. Instead of flattening all system evidence into one “works/does not work” claim, it keeps replay summaries, replay traces, workflow runbooks, and live demos in their proper interpretive roles.

**Figure 7. Online replay accuracy across dataset, model, and operating point**

Asset:
- [online_replay_accuracy_heatmap.png](artifacts/figures/report/online_replay_accuracy_heatmap.png)

![Figure 7. Online replay accuracy across dataset, model, and operating point](artifacts/figures/report/online_replay_accuracy_heatmap.png){ width=90% }

Figure 7 makes the runtime story concrete. It shows not only that runtime behaviour varies systematically across datasets and operating points, but also that the runtime layer should be read as policy-shaped behaviour rather than as a portable substitute for frozen offline ranking metrics.

At the same time, the replay matrix makes the limits of the deployment story visible. `LE2i` remains materially weaker than `CAUCAFall` on the same runtime surface, even after replay/runtime corrections. The uncertainty-aware MC path also fails to improve any of the `12` evaluated combinations at video level, which turns uncertainty handling into a meaningful but currently neutral result rather than a demonstrated deployment gain.

**Figure 8. MC-dropout delta on the fixed online replay matrix**

Asset:
- [online_mc_dropout_delta.png](artifacts/figures/report/online_mc_dropout_delta.png)

![Figure 8. MC-dropout delta on the fixed online replay matrix](artifacts/figures/report/online_mc_dropout_delta.png){ width=82% }

Figure 8 is worth retaining because negative deployment results are easy to omit in a shorter manuscript. Here, however, the absence of improvement is itself informative. It shows that the project tested whether a more sophisticated runtime option produced material bounded deployment gain and reported the answer honestly when it did not.

The field-validation layer is weaker still. On the current branch, there is no stable field-summary artifact strong enough to anchor a field-readiness paragraph. The safe interpretation is only that the system moved beyond purely synthetic inputs into bounded live demonstration evidence.

Finally, the deployment chapter is strongest when read as a coherence chapter rather than as a pure performance chapter. The central question is not only how many clips were handled correctly, but whether the chosen profile survived the full path from monitored input to interpreted state, persisted event, and downstream caregiver-facing delivery. That is the level at which the present project makes its bounded systems contribution.

# System Validation, Testing, and Audit

This chapter evaluates the system as software and as a defended technical artifact. Its purpose is not to replace the benchmark and runtime chapters, but to show how implementation correctness, runtime coherence, and evidence discipline were verified after the system had reached functional maturity. In a project of this scope, model evaluation alone is not enough. The report must also show that runtime semantics, persistence behaviour, notification flow, and active evidence paths remained aligned under review.

## Canonical Tests

The repository includes a stratified canonical testing path rather than one undifferentiated test surface. This structure is important because the project depends on components with different runtime assumptions, and a single flat “all tests” narrative would obscure which parts of the system are actually being verified.

The current test workflow separates torch-free regression checks, frontend regression checks, and environment-sensitive contract and monitor checks. Torch-free tests provide the main fast regression layer for repository contracts, fallback behaviour, settings semantics, notification logic, and selected ML/data assumptions without requiring full torch-backed execution. Frontend tests verify selected API and UI expectations on the active monitor path. Environment-sensitive contract and monitor tests then sit on top of these lower layers and provide bounded verification of runtime behaviour when the execution environment is stable enough to support them.

This layered structure strengthens the report in two ways. First, it keeps validation informative even when a particular machine cannot run every torch-backed path reliably. Second, it makes explicit that different parts of the system fail differently and therefore should not be validated as though they carry identical risk.

**Table 4. Canonical test matrix**

| Test mode | Purpose | Environment requirement | Current role in validation | Key covered paths |
| --- | --- | --- | --- | --- |
| `torch-free` | verify repository contracts, settings/runtime defaults, notification semantics, selected ML/data regressions without torch-backed monitor execution | standard Python environment | primary fast regression layer | data contracts, settings fallbacks, notification manager, repository semantics |
| `frontend` | verify selected API/UI assumptions on the active monitor path | Node/npm test environment | focused frontend regression layer | monitor API fallback behaviour, frontend contract assumptions |
| `contract` | verify environment-sensitive backend/model contracts that require torch-backed import and execution | stable torch environment | deferred/conditional validation layer | monitor contract checks, selected deploy/runtime ML paths |
| `monitor` | verify runtime monitor-oriented tests that exercise live/replay path assumptions more directly | stable torch environment | bounded runtime verification layer | monitor runtime service, policy application, event-persistence path |

**Table 5. Examiner-facing test summary**

| Test area | What was checked | Result in the defended snapshot | Remaining risk |
| --- | --- | --- | --- |
| backend health and API contract | health route, settings defaults, monitor payload shape, and fallback behaviour | passed under the maintained contract checks | full torch-backed paths still depend on local environment stability |
| frontend monitor behaviour | replay controls, monitor API handling, UI state assumptions, and runtime fallback behaviour | passed on the targeted frontend regression layer | browser performance can still affect realtime demonstration quality |
| event persistence | realtime fall events, event-history visibility, dashboard/event count semantics, and review status handling | implemented and regression-checked after audit fixes | replay mode is intentionally visual-only and does not create event history records |
| notification delivery | Telegram-first delivery path, notification audit semantics, and caregiver-facing message generation | implemented and verified on the active delivery path | SMS and phone-call escalation are not part of the final implemented claim |
| replay/runtime validation | bounded replay matrix, preferred runtime preset behaviour, and retraining improvement checks | baseline bounded replay remained mixed, while the strengthening track improved the defended TCN custom surfaces modestly without worsening ADL false positives | replay evidence remains controlled system evidence, not broad field validation |
| release/readiness checks | frontend production build, canonical test entrypoints, and release-boundary scripts | used as final software-artifact readiness checks | raw-dataset reproduction remains outside the default review path |

## Replay and Runtime Validation

Replay and runtime validation occupy a middle evidential layer between formal offline benchmarking and uncontrolled real-world deployment. Their role is not to strengthen the architecture claim directly, but to show whether the monitored path behaves coherently once the system is run as software rather than only as an offline pipeline.

The replay path is especially valuable because it exposes interactions between frontend window production, backend policy interpretation, persistence semantics, and caregiver-facing delivery. Several of the project’s most important runtime lessons emerged at this layer, including the need to keep replay semantics distinct from realtime semantics and to avoid treating replay output as stored event evidence. These findings are systemically important even though replay evidence remains narrower than benchmark evidence.

Live demonstration evidence should be interpreted in the same bounded way. A successful live run is meaningful as operational confirmation that the integrated path can function under the defended demo preset. It is not equivalent to population-level deployment validation. The report is strongest because it keeps these two meanings separate.

**Figure 9. Runtime evidence panel**

The active runtime-evidence panel is illustrated here with representative interface captures drawn from the defended monitor path. The monitor screenshot shows the live runtime surface used to inspect prediction state, timeline behaviour, and active model/profile metadata. The Event History screenshot shows the review surface used to inspect persisted event records separately from the live monitor.

Active evidence assets:
- [hifi_live_monitor.png](artifacts/figures/report/appendix/hifi_live_monitor.png)
- [hifi_event_history.png](artifacts/figures/report/appendix/hifi_event_history.png)

![Figure 9a. Representative live monitor capture for the defended runtime path](artifacts/figures/report/appendix/hifi_live_monitor.png){ width=92% }

![Figure 9b. Representative Event History capture for the defended review path](artifacts/figures/report/appendix/hifi_event_history.png){ width=92% }

Supplementary videos:
- `Supplementary Video S1`: [realtime_fall_submission.mp4](artifacts/evidence/realtime/realtime_fall_submission.mp4)
- `Supplementary Video S2`: [realtime_adl_submission.mp4](artifacts/evidence/realtime/realtime_adl_submission.mp4)

Figure 9 provides bounded interface-level systems evidence. It shows that the defended artifact includes a monitor surface for runtime interpretation and a review surface for persisted events, but it does not add new statistical authority beyond the replay and offline chapters.

**Table 6. Validation-interpretation matrix**

| Validation surface | What it can support | What it cannot support on its own |
| --- | --- | --- |
| frozen offline test metrics | comparative model performance under the declared protocol | direct claims about integrated runtime behaviour |
| cross-dataset transfer runs | limitation boundaries and directionality of domain shift | claims of robust cross-domain deployment |
| replay matrix | bounded runtime behaviour under controlled clips and fixed policy profiles | field-readiness or population-level deployment confidence |
| live demonstration | operational existence of the end-to-end path | benchmark-quality statistical validation |
| notification delivery checks | proof that persisted events can reach caregiver-facing delivery | proof that delivery quality improves detection quality |

This matrix is useful because the project includes several different kinds of validation that could otherwise be over-interpreted when placed side by side. A strong report should show not only what was tested, but also what kind of claim each validation layer can and cannot support.

## Code Audit Findings

The full-stack audit and later code-review pass form a substantive part of the defended evidence chain. Their purpose was not only to improve code quality in the abstract, but to reduce mismatch risk between implementation, artifacts, and report claims.

The most important findings were cross-layer rather than cosmetic. They included evidence-chain drift, replay/runtime truth-source mismatch, frontend/backend fallback inconsistency, and ambiguity around persisted event semantics. These were significant because the project’s most serious risks often lived between modules and evidence layers rather than inside one isolated function. The remediation work therefore materially strengthened the report’s credibility by aligning code paths more closely with the claims made about them.

**Table 7. Issue-to-fix summary**

| Symptom or risk | Root cause | Fix or control | Evidence of improvement |
| --- | --- | --- | --- |
| report claims could drift from the active implementation | old paths and archived artifacts remained close to active material | active code paths, promoted assets, and release boundaries were clarified | final documentation now points to `applications/`, `ml/`, `ops/`, and promoted runtime assets |
| replay behaviour could be mistaken for stored event evidence | replay and realtime modes shared a monitor surface but had different evidence meanings | replay was kept visual-only, while event history was tied to realtime monitoring events | Event History semantics now distinguish controlled replay input from stored realtime incidents |
| dashboard/event counts could be misread | backend totals and currently visible rows represented different scopes | UI copy and count semantics were tightened around the displayed event set | event-history display now avoids presenting a global backend total as visible records |
| notification evidence could be over-claimed | earlier multi-channel concepts were broader than the verified delivery route | final claim was narrowed to Telegram-first delivery | SMS and phone-call escalation are explicitly future work, not completed functionality |
| runtime behaviour was hard to explain from code alone | monitor logic mixed media, pose, API, state, and rendering responsibilities | monitor logic was refactored into typed hooks, components, and service contracts | frontend/backend structure is easier to map to the report’s runtime workflow |
| release readiness was vulnerable to environment assumptions | frontend and torch-backed checks have different prerequisites | canonical checks were separated by environment requirement | fast checks remain available while torch-backed validation is recorded as conditional where necessary |

## Freeze and Handoff State

By the end of the review and cleanup cycle, the repository had reached a stable freeze-core state and later a clean handoff state. This matters because a final report should describe a defended system snapshot rather than a moving target.

Freeze and handoff work clarified which artifacts belong to the active evidence chain and which belong to supporting or archival history. That clarification has direct methodological value. It sharpens the meaning of earlier validation layers by making it easier to identify which figures, fitted profiles, summaries, test entrypoints, and screenshots are authoritative for the final report. In that sense, freeze state is not a packaging afterthought. It is the final wrapper that makes the rest of the evidence chain easier to defend.

## Final Evaluation Summary

The report’s main objectives and requirements can be summarised directly as follows.

**Table 8. Objectives and requirements revisited**

| Objective or requirement | Status | Main evidence | Boundary or limitation |
| --- | --- | --- | --- |
| compare TCN and custom GCN under a shared frozen protocol | Achieved | frozen offline comparison, stability figures, locked candidate artifacts | conclusion remains directional rather than universal |
| fit deployable operating-point profiles using validation-side evidence | Achieved | operating-point YAMLs, policy fitting results, replay/runtime preset selection | profile quality depends on the frozen validation contract |
| implement a working monitoring system with frontend and backend inference | Achieved | React monitor, FastAPI backend, replay mode, realtime mode, runtime model panel | realtime performance depends on browser and recording load |
| persist reviewable fall events for later inspection | Achieved | realtime Event History workflow, dashboard/event records, event replay evidence | replay-mode detections are visual-only and are not stored |
| deliver caregiver-facing notification on the active path | Achieved for Telegram | Telegram delivery evidence and notification audit path | SMS and phone-call escalation are future work |
| support reproducible review of the software artifact | Achieved for the default demo path | README, local bootstrap path, frontend build check, canonical test entrypoints | raw datasets and full training reproduction require separate dataset access |
| demonstrate broad field readiness | Not claimed | bounded realtime evidence and replay matrix | current system is a review-support prototype, not a clinically validated safety product |

# Ethics, Privacy, and Operational Constraints

## Privacy by Design

One of the clearest architectural decisions in the project is the use of skeleton-based monitoring rather than raw cloud-hosted video classification. This does not remove every privacy concern, but it does reduce the amount of directly identifying visual information that must pass through the inference path. In the implemented runtime design, browser-side pose extraction produces pose-derived windows, and the backend consumes those windows rather than full RGB streams.

This privacy-oriented separation is important for two reasons. First, it aligns with the project's goal of demonstrating a practical monitoring pipeline that is sensitive to the realities of domestic and care-adjacent settings. Second, it supports a more defensible software architecture because it narrows the responsibility of the backend to inference, policy, persistence, and delivery, instead of turning it into a general video-processing store.

The report remains careful here. A skeleton stream is not identical to anonymous data, and fall events remain sensitive records. The correct claim is therefore privacy-oriented design rather than absolute privacy.

This distinction is important because it prevents the privacy discussion from becoming decorative. The choice to work with pose-derived windows rather than raw backend video is not only a public-facing ethical preference; it also shapes the technical system. It narrows what the backend must store, changes what kinds of artifacts can later be audited, and affects how runtime evidence should be captured for the report. The project therefore treats privacy-oriented design as a constraint on architecture and evidence handling, not merely as a statement of intent.

There is also a trade-off embedded in this decision. By preferring skeletonized representations, the system gains a cleaner privacy posture and a more compact runtime contract, but it also becomes more dependent on the quality and stability of browser-side pose extraction. In other words, privacy-oriented design and representation fragility are linked. A mature report should say this explicitly rather than presenting privacy-conscious design as if it were a free gain without technical cost.

## Data Handling and Retention

The repository contains raw data, processed windows, fitted operating-point files, runtime artifacts, and review documents. That breadth makes data-handling discipline especially important. The later freeze work addressed not only code cleanliness but also the distinction between active evidence, supporting material, and archive-only artifacts.

From a report perspective, data minimisation appears in two places. At runtime, the system prefers pose-derived windows and persisted fall-event summaries over long-lived raw video handling inside the backend path. In the repository, the project distinguishes between evidence that is necessary for defended results and historical material that should be archived rather than left in the active surface.

This is not a legal compliance analysis in the formal sense, but it is a technically relevant design consideration. A full project report should show that storage and traceability decisions were not accidental side effects of experimentation.

It is also useful to connect data handling directly to report defensibility. A long technical report accumulates draft figures, exploratory metrics, runtime logs, and historical notes very quickly. If those materials are not separated into active, supporting, and archived layers, then later interpretation becomes much harder. The repository's freeze and archive work therefore has an ethical dimension as well as a methodological one. It reduces the chance that sensitive or misleading artifacts remain visible in places where they can be mistaken for current evidence.

## Operational Risk and Human Review

Fall detection is a safety-adjacent application. That means the project has to acknowledge operational risks even when the technical artifact is bounded to a final-year setting. Two risks dominate. The first is the missed-fall risk associated with weak recall or domain shift. The second is alert fatigue caused by excessive false alarms or unstable replay/runtime interpretation.

The project addresses these risks through operating-point fitting, temporal policy, event review, and bounded claims rather than through any claim of perfect automated detection. In practice, the system is better understood as a review-support tool than as an autonomous safety device. Persisted events, dashboard summaries, and caregiver notifications are all designed around the assumption that a human remains part of the response loop.

This framing is important for the report because it affects how the system should be judged. A bounded monitoring aid with explicit review semantics is a credible final-year contribution. A claim of autonomous safety-critical readiness would not be.

The ethical and operational discussion also clarifies why some technical choices were intentionally conservative. The project retains explicit review state, persists incidents for later inspection, and keeps caregiver-facing delivery downstream of event creation rather than treating delivery as a direct translation of transient monitor state. These are not only engineering preferences. They are safeguards against turning an uncertain runtime signal into an overconfident operational action.

This is one of the clearest places where ethics and systems design meet. False negatives and false positives are not abstract metric inconveniences in this setting; they map onto missed intervention risk and alert-fatigue risk. The report is stronger because it acknowledges that mapping directly and then shows how alert policy, review semantics, and bounded claims were chosen in response. In a high-quality full report, ethics is most persuasive when it can be traced into system behaviour rather than left as a generic closing reflection.

# Project Management, Iteration, and Risk Control

## Iterative Delivery Structure

Although the strongest evidence in the report comes from frozen artifacts, the project itself was delivered iteratively. This matters because many of the final design decisions only make sense when seen as responses to concrete development pressure rather than as ideas chosen once at the beginning and never revisited.

The earlier planning material shows a staged progression in which application work, data pipeline work, model fitting, and evaluation work were advanced in parallel rather than as one linear waterfall. That structure was useful because the project had genuine cross-dependencies. A monitor page could not remain stable if backend event semantics were still changing, and backend event semantics could not be defended if operating-point fitting and data contracts were still unclear.

**Figure 10. Iteration timeline from the design-proposal phase**

Asset:
- [iteration_timeline.png](artifacts/figures/report/appendix/iteration_timeline.png)

Figure 10 is retained not as active evidence for the final result, but as project-process evidence. It illustrates that the system, pipeline, and evaluation workstreams were planned as interacting strands. In the full report this is useful because it shows that later audit and refactor work did not come out of nowhere; they emerged from the increasing complexity of a multi-strand project.

This iterative structure exposed system interactions early: window contracts mattered to the monitor, operating-point fitting mattered to dashboard meaning, and repository organization mattered to the report itself.

## Design Evolution

The project also evolved visually and structurally. Early wireframes emphasised dashboard, monitor, event-history, and settings views as the core user-visible surfaces. Later higher-fidelity mockups sharpened this into a more coherent monitor-centered workflow. These materials are worth preserving in the report, not because they prove technical performance, but because they show that interface structure and review semantics were designed deliberately rather than improvised at the end.

The key lesson from this design evolution is that frontend design in this project was not only cosmetic. Page layout controlled observability. Dashboard summaries, event lists, settings, monitor controls, and caregiver-notification configuration all contributed to whether the system could be interpreted as a monitoring platform instead of as a raw model demo.

The report is explicit about the status of these visuals. Proposal-stage wireframes and higher-fidelity previews are useful because they expose intended workflow shape, information density, and review semantics. They are not evidence that the final implementation reached the same quality automatically. Their value lies in traceability: they show how early design intent later interacted with refactoring, deployment constraints, and audit findings.

## Risk Control Through Staged Verification

Project management also became a form of technical risk control. As the codebase and artifact set grew, the main danger was no longer simply “the model might underperform.” It became increasingly likely that one part of the system would drift from another: a report could cite an outdated figure, a runtime path could interpret an operating-point code differently from the settings page, or replay behaviour could silently diverge from persisted-event semantics.

This is why the later phases of the project included not just coding and experimentation, but freeze control, code review, audit, artifact allowlists, and canonical test entrypoints. Some risks could only be controlled once several artifacts existed at once: evidence drift becomes visible when figures, results text, and active artifacts coexist, while runtime semantic drift becomes visible when monitor UI, persistence, and delivery are all active.

**Table 9. Project risk-control matrix**

| Risk class | Typical manifestation | Control mechanism | Final status |
| --- | --- | --- | --- |
| evidence drift | report text, figures, and active artifacts fall out of alignment | evidence maps, figure-family cleanup, freeze inventory, late-stage audit | materially reduced |
| runtime semantic drift | transient monitor state diverges from persisted event or delivery state | replay/realtime semantic split, repository/service cleanup, notification truth-source alignment | materially reduced |
| configuration drift | frontend, backend, and fallback presets resolve different profiles | unified operating-point normalization and preset alignment across layers | materially reduced |
| environment-sensitive verification | torch-backed checks fail on some local machines | stratified canonical tests, conditional contract layer, documented environment caveat | reduced but not eliminated |
| design-to-implementation drift | early UI concepts no longer match final workflow | design-evolution review and appendix traceability | reduced |

## Legacy and Final Architecture

The earlier proposal-stage architecture is still useful as a contrast artifact because it reveals how the system was originally conceptualised in tiered form before later refactoring introduced more precise contract boundaries.

**Figure 11. Architecture evolution from proposal-stage tiers to final contract-oriented layers**

Asset:
- [architecture_evolution_comparison.svg](artifacts/figures/report/architecture_evolution_comparison.svg)

![Figure 11. Architecture evolution from proposal-stage tiers to final contract-oriented layers](artifacts/figures/report/architecture_evolution_comparison.svg){ width=85% }

Appendix reference assets:
- [legacy_system_tier_diagram.jpg](artifacts/figures/report/appendix/legacy_system_tier_diagram.jpg)
- [system_architecture_diagram.svg](artifacts/figures/report/system_architecture_diagram.svg)

The contrast is meaningful because the final architecture is not merely a prettier redrawing of the original tier diagram. The later version encodes clearer boundaries around browser-side pose extraction, backend policy interpretation, event persistence, and delivery audit. The proposal-stage architecture captured intent, but not yet those later semantic refinements.

Figure 11 makes this change visible in a way that prose alone does not. The proposal-stage view grouped concerns into broad tiers that were useful for early planning, but it left important operational meanings implicit. The final architecture is still recognisably related to that earlier concept, yet it exposes a much stronger distinction between browser-side capture, backend interpretation, persisted review state, and downstream delivery audit. This difference matters because late-stage code review showed that the project’s most serious mismatch risks lived exactly at those boundaries.

## Iteration Outcome

The cumulative management outcome is best described as controlled narrowing rather than unchecked expansion. Early in the project it was still plausible to chase multiple datasets, multiple architectures, richer uncertainty handling, several delivery channels, and both compact-paper and long-report outputs simultaneously. Later stages had to reduce that breadth and convert it into a defended core. The final active profile, the Telegram-first delivery path, the bounded replay interpretation, and the freeze-core artifact set are all products of that narrowing process.

This matters because a high-quality full report should not pretend that every explored branch matured equally. The project’s management outcome was not maximal scope retention, but disciplined convergence toward a defendable system and evidence base.

# Discussion

This chapter answers the three research questions in light of the full evidence hierarchy developed throughout the report. The project’s contribution is strongest when the answers are kept at the right level of authority: stronger where frozen comparison and bounded runtime evidence are clear, and more cautious where inferential or deployment limits remain.

## Answer to RQ1

`RQ1` asked how the TCN and the custom spatio-temporal GCN compare under the locked offline protocol. The answer is cautious but directionally clear. Under the frozen primary-dataset protocol, the TCN trends stronger than the matched GCN, and the same directional preference also appears in the in-domain `LE2i` comparison. However, the current inferential budget does not justify maximalist language about universal architectural superiority. The correct interpretation is therefore directional advantage with bounded statistical certainty.

This answer remains important even after the system side of the project is taken seriously. Replay, audit, and runtime chapters do not overturn the offline comparison. Instead, they show that architecture is only one layer of the final system story. That balance is one of the report’s strengths. The TCN is not emphasised simply because it produced the most attractive number, but because it remained easier to defend once representation, policy, runtime interpretation, and deployment boundaries were all taken into account.

This also clarifies what the report is not claiming. It is not concluding that graph-based temporal reasoning lacks value in fall detection as a field. It is concluding that, within this repository’s chosen representation, temporal contract, and policy layer, the custom GCN did not displace the TCN as the strongest defended baseline. That narrower statement is methodologically stronger because it is directly tied to the conditions that were actually frozen and reviewed.

## Answer to RQ2

`RQ2` asked whether validation-side calibration and operating-point fitting materially influence practical alerting. The answer is yes. In the present system, deployable behaviour is governed by fitted operating-point profiles rather than by raw single-window probabilities alone. Alert behaviour is therefore the product of calibration-aware fitting, temporal smoothing, `k/n` logic, cooldown behaviour, and persistence semantics rather than of classifier discrimination alone.

This is a substantive contribution because it shows that alerting is a methodological and engineering layer in its own right. The monitor does not merely expose model outputs. It interprets them through a reviewable policy. That is why the report can say something meaningful about deployment-shaped behaviour while still keeping broader field claims bounded.

The deeper lesson is that policy fitting changes the unit of success. Once the project is treated as a monitoring system rather than a ranking exercise, the key question is whether score streams can be converted into reviewable and operationally stable states. The report’s answer is that this conversion can indeed be made explicit, validation-fitted, and auditable. That is stronger than simply observing that thresholds matter in the abstract. It shows that the alert layer itself can be brought under artifact control and linked coherently to persistence, dashboard review, and caregiver-facing delivery.

## Answer to RQ3

`RQ3` asked what replay deployment evidence and limited realtime validation reveal about practical feasibility and runtime limits. The answer is mixed but meaningful. The system is strong enough to support a bounded practical deployment claim under replay-oriented and live-demo conditions, but the current branch evidence also shows that runtime behaviour is highly profile-sensitive. The canonical replay summary, the 24-clip custom replay trace, and the live evidence pack collectively show that the monitor path is real and reviewable, while also showing that runtime success is not uniform across operating points or datasets and that the current uncertainty-aware path does not improve the bounded replay matrix.

The correct interpretation is therefore practical feasibility in controlled settings rather than broad deployment closure. This is a valuable answer because it captures the actual maturity level of the artifact. The project is neither a pure benchmark script nor a field-validated product. It occupies the middle ground of a research-led monitoring system whose strongest claims concern coherence, traceability, and bounded operational credibility. The subset-level kitchen weakness also reinforces that deployment readiness depends on camera placement and pose observability, not only on checkpoint quality and threshold selection.

## Research Contribution by Boundary Discipline

The phrase *boundary discipline* best captures the deeper contribution of the report. Throughout the project, the most important improvements came from tightening boundaries: between offline and runtime evidence, between transient state and persisted state, between persisted state and downstream delivery, and between active defended artifacts and archived exploratory history.

These boundaries are not rhetorical devices. They are the reason the system can now be described in layers without constant backtracking over hidden ambiguity. They also explain why the long-report format is justified. A compact paper can present the strongest numerical findings, but the full report can additionally show how those findings survived integration pressure, review, runtime interpretation, and freeze-state cleanup. In that sense, the report’s contribution is not only that it obtained a set of results, but that it turned a broad mixed project into a defendable technical artifact.

# Limitations

The report’s conclusions should be read with bounded confidence. The project demonstrates a serious end-to-end system study, but it does not justify claims of universal robustness, clinical readiness, or final field closure. The limitations below are therefore not marginal caveats. They define the correct scope of the defended contribution.

The first limitation is inferential. The frozen five-seed comparison is sufficient to support a careful directional architecture claim, but it is not sufficient to justify stronger significance language on the main event metrics.

The second limitation is generalisation. Cross-dataset transfer remains asymmetric and does not support claims of broad robustness across camera conditions, pose quality variation, scene statistics, or deployment domains.

The third limitation is systems-dependent measurement quality. Because the monitoring path depends on browser-side pose extraction, degradation in skeleton quality can propagate directly into backend behaviour. Some runtime failure modes are therefore caused by system interactions rather than by model weights alone.

The fourth limitation is deployment closure. Replay evidence is materially stronger than field evidence, and the current branch no longer carries a stable field-summary pack strong enough to support broad real-world deployment claims. This is why the report repeatedly frames deployment support as bounded.

The deployment limitation is also geometric. The bounded custom replay results suggest that camera placement, capture distance, and subject observability materially affect pose stability. The corridor setup uses a higher camera and a longer subject-to-camera distance, while the kitchen setup uses a lower camera and a much shorter distance. In the kitchen clips, subjects are more likely to be partially framed or strongly distorted by perspective, which makes skeleton generation less stable and pushes runtime windows farther from the scene geometry seen during training. In at least one kitchen clip (`kitchen_front_2`), the remaining miss also appears to be temporal: the fall is a very rapid forward collapse after standing from the sofa, which is consistent with a strong but short-lived fall spike failing to persist under the stricter `OP-2` event logic. This is a plausible explanation for why the kitchen subset remained harder even after retraining improvement on the broader custom surfaces.

The fifth limitation is uncertainty interpretation. The uncertainty-aware runtime path is implemented and methodologically meaningful, but the current bounded replay matrix does not show a deployment gain from enabling it.

These limitations also differ in kind, and the report is stronger when that difference is made explicit. Some are inferential limitations, such as the current seed budget and the restricted field sample. Some are architectural limitations, such as dependence on browser-side pose quality and environment-sensitive verification. Others are methodological limitations, such as the fact that replay evidence, however useful, remains narrower than genuine deployment evidence.

This distinction matters because different limitations call for different responses. Larger experiments alone do not solve architecture-bound runtime fragility, and cleaner runtime engineering alone does not resolve statistical caution. Some limitations primarily constrain how strongly the report can argue today; others primarily constrain how easily the system could mature later. The field-evidence budget and seed budget mainly restrict inferential authority. By contrast, pose-quality dependence, environment-sensitive verification, and transfer weakness mainly restrict expansion into a broader operational artifact.

Making these categories explicit is one of the report’s strengths. Large mixed projects often appear impressive while remaining poorly bounded. The present report takes the opposite route. It names which limitations affect interpretation, which affect engineering scale, and which affect both. That is one reason the conclusions can remain strong without pretending that the project is already complete.

# Future Work

Future work should extend the project in ways that strengthen the current defended claim set rather than simply enlarging the codebase.

The first priority is stronger deployment-oriented evidence. The live and field-validation protocol should be expanded so that bounded deployment claims can rest on a wider and more varied evidence base. The second priority is stronger robustness analysis across datasets, cameras, and pose-quality conditions. The third is a clearer decision on uncertainty handling: the uncertainty-aware runtime path should either be validated under a stronger runtime protocol or simplified further if it continues to show neutral deployment value.

Near-term work should therefore reinforce the current contribution: richer bounded runtime evidence, stronger field-style validation, more stable torch-backed verification, and continued runtime simplification. Longer-term work can broaden scope through multi-channel delivery, larger external deployment studies, and additional architecture families. Broader delivery infrastructure would enlarge the system artifact, but it would not automatically strengthen model evidence unless paired with stronger operational validation.

# Data Availability Statement

The code submission does not redistribute the raw training datasets used in this project. Public datasets used in the project, including `CAUCAFall`, `LE2i`, and `URFD`, should be obtained from their original public sources rather than from this repository.

`MUVIM` was obtained from the Intelligent Assistive Technology and Systems Lab (IATSL), University of Toronto, under a signed research data agreement. The dataset is used for research purposes only and is not redistributed in this repository. Access is restricted and must be arranged directly with the data owner, subject to the provider's terms.

Research outputs using `MUVIM` should cite S. Denkovski, S. S. Khan, B. Malamis, S. Y. Moon, B. Ye and A. Mihailidis, "Multi Visual Modality Fall Detection Dataset," *IEEE Access*, 2022, doi: `10.1109/ACCESS.2022.3211939`, and acknowledge the support of the Intelligent Assistive Technology and Systems Lab (IATSL) at the University of Toronto through the sharing of data related to this research.

# Final Evaluation

The project succeeded most clearly as a working research-and-systems artifact. The strongest completed outcome is the integrated monitoring workflow: replay or realtime input can be processed through the frontend and backend, interpreted under an active model/profile, displayed in the monitor, and, for realtime incidents, persisted for later review. This directly satisfies the software-artifact requirement because the system demonstrates a meaningful end-to-end feature rather than only describing one.

The project also succeeded in keeping the model evidence and system evidence separate. The TCN result is defended through the frozen offline protocol and bounded replay evidence, while the deployed monitor is defended as a working prototype with explicit limits. This separation prevents the report from overstating what the demo proves.

The project did not complete every early ambition. SMS and phone-call notification were not implemented as final delivery claims. Broad field validation was also not achieved. The current realtime evidence is useful for demonstrating feasibility, but it is not large enough to support clinical or deployment-readiness claims.

The main lesson is that a fall-detection system cannot be judged only by a classifier score. The final artifact became stronger when the project controlled the whole path from input capture to inference, policy interpretation, event persistence, review, and notification, while also stating which parts remain bounded.

# Conclusion

This report has presented a research-led study of pose-based fall detection as an integrated monitoring problem rather than as a classifier-only benchmark task. The project combined frozen offline comparison, validation-side operating-point fitting, temporal policy design, runtime monitoring, event persistence, and caregiver-facing delivery within one coherent technical pipeline.

At the model-comparison layer, the strongest defended result is a cautious directional advantage for the TCN over the matched custom GCN under the frozen primary-dataset protocol. At the policy layer, the report shows that deployable behaviour depends not only on score ranking quality but also on how validation-fitted operating points, smoothing, cooldown, and persistence semantics shape alert interpretation. At the systems layer, the project demonstrates that a pose-based monitoring pipeline can support bounded end-to-end behaviour in which monitored input can become interpreted runtime state, persisted incident history, and caregiver-facing notification under a reviewed path.

Just as importantly, the report shows where stronger claims cannot yet be made. Cross-dataset transfer remains asymmetric, replay evidence remains narrower than broad field validation, and the current system should be interpreted as a review-support artifact rather than an autonomous safety-critical device.

On that basis, the report’s main contribution is not a claim of solved fall detection. It is the demonstration that a final-year research project can produce a serious, bounded, and defensible end-to-end study in which model, policy, runtime, persistence, delivery, and evidence control remain answerable to one another under review.

# Acknowledgments

The author acknowledges the support of the Intelligent Assistive Technology and Systems Lab (IATSL) at the University of Toronto through the sharing of data related to this research.

# References

The bibliography below is a working final reference set covering the main methodological and comparative foundations used in the report. It replaces the earlier empty placeholder, but it should still be checked against the final supervisor-preferred citation style before submission.

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
14. Eraso Guerrero, J. C., Muñoz España, E., Muñoz Añasco, M., and Pinto Lopera, J. E. (2022). Dataset for human fall recognition in an uncontrolled environment. *Data in Brief*, 45, 108610. https://doi.org/10.1016/j.dib.2022.108610
15. Charfi, I., Mitéran, J., Dubois, J., Atri, M., and Tourki, R. (2013). Optimized spatio-temporal descriptors for real-time fall detection: comparison of support vector machine and Adaboost-based classification. *Journal of Electronic Imaging*, 22(4), 041106. https://doi.org/10.1117/1.JEI.22.4.041106
16. Denkovski, S., Khan, S. S., Malamis, B., Moon, S. Y., Ye, B., and Mihailidis, A. (2022). Multi Visual Modality Fall Detection Dataset. *IEEE Access*, 10, 108288-108300. https://doi.org/10.1109/ACCESS.2022.3211939

# Appendices

## Appendix A: Frozen Candidates and Config Snapshots

This appendix documents the final candidate and operating-point artifacts that anchor the report. The purpose is not to reproduce raw YAML files line by line, but to make the defended artifact set auditable.

Appendix A exists because the main chapters rely repeatedly on a relatively small defended core inside a much larger repository. A high-quality full report should not assume that readers will infer that core automatically from prose references to checkpoints, operating-point names, or figure files. By listing the defended candidate families explicitly, the appendix turns “frozen core” from a phrase into something that can be inspected.

**Table A1. Primary frozen candidates and active profiles**

| Role | Dataset | Architecture / profile | Artifact family | Report use |
| --- | --- | --- | --- | --- |
| primary offline candidate | `CAUCAFall` | `TCN` | `outputs/caucafall_tcn_W48S12/*` | main comparative result line |
| matched offline baseline | `CAUCAFall` | `GCN` | `outputs/caucafall_gcn_W48S12/*` | matched comparison line |
| preferred replay/runtime preset | `CAUCAFall` | `TCN + OP-2` | `ops/configs/ops/tcn_caucafall*.yaml` | preferred live and replay profile |
| comparative deployable profile | `LE2i` | `GCN` | `ops/configs/ops/gcn_le2i_opt33_r2.yaml` and linked metrics | comparative runtime/profile evidence |
| exploratory/supporting family | `MUVIM` | mixed | supporting and archived config families | methodological breadth only |

This appendix should be used together with the repository evidence map. Its function is to show which artifact families still define the active report and which have been demoted to supporting or archive status.

**Table A2. Active figure families**

| Figure role | Active artifact | Status in main report |
| --- | --- | --- |
| system architecture | `artifacts/figures/report/system_architecture_diagram.svg` | main figure |
| offline stability | `artifacts/figures/report/offline_stability_comparison.png` | main figure |
| cross-dataset transfer | `artifacts/figures/report/cross_dataset_transfer_summary.png` | main figure |
| alert-policy flow | `artifacts/figures/report/alert_policy_flow.svg` | main figure |
| bounded replay runtime | `artifacts/figures/report/online_replay_accuracy_heatmap.png` | main figure |
| uncertainty delta | `artifacts/figures/report/online_mc_dropout_delta.png` | main figure |
| supporting cross-dataset absolute F1 | `artifacts/figures/report/cross_dataset_f1_comparison.png` | appendix/supporting |
| supporting seed stability | `artifacts/figures/report/stability_f1_errorbars.png` | appendix/supporting |

**Table A3. Active artifact-state boundaries**

| Artifact class | Active role | Excluded from defended core |
| --- | --- | --- |
| report drafts and plans | current paper and full-report drafts, active master plans | deleted or superseded merged planning drafts |
| figures | main figures under `artifacts/figures/report/` | diagnostic or appendix-only figures unless explicitly labeled |
| operating-point profiles | active frozen YAML families for defended presets | exploratory or archived tuning families |
| metrics and summaries | frozen comparative and runtime summaries referenced in the report | legacy pre-refreeze summaries and diagnostic-only outputs |
| audit materials | active full-stack/code-review summaries that explain defended verification | historical notes that no longer control the live report |

This table makes the freeze logic explicit inside the report. It tells the reader that the repository was not treated as an undifferentiated pile of outputs. Instead, the submission distinguishes defended artifacts from supporting or archived material, which is essential in a project with many experiment branches and late-stage review passes.

**Table A4. Evidence-lineage summary**

| Claim layer | Primary source type | Typical traced artifact | Why the lineage matters |
| --- | --- | --- | --- |
| model comparison | frozen metrics summaries and seed-comparison artifacts | `outputs/metrics/*stb_*.json`, `outputs/metrics/*locked.json` | supports comparative claims without runtime conflation |
| policy interpretation | fitted operating-point files and policy sweeps | `ops/configs/ops/*.yaml`, saved sweep JSON files | links threshold/policy prose to tracked deployable profiles |
| runtime interpretation | replay matrices, runtime figures, persisted event path | `artifacts/ops_delivery_verify_20260315/online_replay_summary.json`, `artifacts/fall_test_eval_20260315/summary_*.csv`, monitor/event route behaviour | supports bounded deployment/system claims |
| delivery evidence | Telegram delivery audit and event-linked notification path | delivery audit store, runtime screenshots, notification tests | shows that persisted incidents can reach caregiver-facing output |
| repository defensibility | audit, freeze, and review-control documents | freeze manifests, code-review summaries, audit reports | explains why the final submission state is trustworthy |

This lineage table is useful because a long report can easily lose track of which artifact family is supposed to support which kind of statement. By summarising those mappings explicitly, the appendix makes it easier for a reader, examiner, or supervisor to trace major claims back to the right technical evidence.

**Table A5. Active profile summary**

| Profile family | Dataset | Intended role | Interpretation note |
| --- | --- | --- | --- |
| `tcn_caucafall_*` | `CAUCAFall` | primary offline and preferred runtime line | strongest defended alignment across offline, policy, and replay evidence |
| `gcn_caucafall_*` | `CAUCAFall` | matched baseline and policy contrast | important for fair architecture comparison, not the preferred deployable preset |
| `gcn_le2i_opt33_r2` | `LE2i` | comparative deployable profile | useful for boundary analysis, not a replacement for the primary runtime line |
| `tcn_le2i_*` | `LE2i` | comparative model evidence | useful for in-domain comparison and limitations, but not the main demo path |
| archived `muvim` families | `MUVIM` | supporting breadth and exploratory work | methodologically useful, but outside the defended core contribution |

This profile-summary table is useful because operating-point files can easily look interchangeable when read only as YAML names. In reality they serve different evidential roles. The appendix makes that role separation explicit so the reader can see which profiles define the defended narrative and which remain supporting context.

**Table A6. Profile-intent summary**

| Profile group | Main intent | Typical downstream use |
| --- | --- | --- |
| `OP-1` family | looser or earlier alert posture within one fitted family | contrastive runtime interpretation, not preferred defended demo state |
| `OP-2` family | balanced deployable profile | preferred replay/live demonstration and bounded runtime discussion |
| `OP-3` family | stricter interpretation with higher alert conservatism | contrastive runtime interpretation and false-alert sensitivity discussion |
| source-fitted transfer profiles | preserve protocol discipline under cross-dataset evaluation | limitation analysis rather than deployment recommendation |
| archived exploratory profiles | document search breadth and historical fitting paths | appendix/supporting context only |

This table is useful because it translates profile names into methodological intent. A reader who only sees `OP-1`, `OP-2`, and `OP-3` in prose may not immediately understand why one profile became the preferred live/replay path while another remained only a contrastive reference.

**Table A7. Config-snapshot highlights**

| Config family | Snapshot property worth preserving in the report | Why it matters |
| --- | --- | --- |
| primary `CAUCAFall` TCN ops | balanced `OP-2` path under one fitted family | explains why the preferred demo profile is policy-based rather than ad hoc |
| matched `CAUCAFall` GCN ops | same downstream policy framing on a weaker model line | preserves fairness of architecture comparison at the deployable-profile level |
| `LE2i` deployable profile | comparative runtime profile with weaker operational story | keeps boundary analysis tied to a tracked config rather than to prose alone |
| transfer profiles | source-fitted policy discipline | shows that cross-dataset runs respected protocol boundaries |
| archived exploratory families | historical breadth without defended-core authority | explains how breadth was retained without polluting the final claim surface |

This snapshot table is intentionally selective rather than exhaustive. Its job is not to reproduce the YAMLs. Its job is to tell the reader what aspects of those YAML families actually matter for interpretation.

**Table A8. Artifact-to-chapter map**

| Artifact family | Main chapter that depends on it | Report function |
| --- | --- | --- |
| frozen metrics outputs | `Experimental Results` | supports comparative and transfer claims |
| operating-point YAMLs | `Calibration and Alert Policy`, `Deployment and Runtime Results` | supports deployable policy interpretation |
| runtime figures and replay summaries | `Deployment and Runtime Results` | supports bounded system-evidence claims |
| audit and code-review documents | `System Validation, Testing, and Audit` | supports defended-snapshot and mismatch-reduction claims |
| appendix-only figures | `Appendix B` | supports narrower interpretation or design-evolution context |

This mapping is useful because it shows that artifacts are not only stored; they are narratively assigned. A defended report depends not only on having the right files, but also on using each artifact family in the right argumentative location.

## Appendix B: Additional Figures and Tables

This appendix contains supporting visual material that is useful but not central enough to remain in the main narrative.

Its role is not decorative. Appendix B protects the main argument from visual overload while still preserving figures that are meaningful for narrower questions about transfer shape, seed stability, and interface evolution. In a long report, that separation is valuable because it lets the main chapters remain selective without hiding the broader visual record of the project.

### B.1 Supporting Quantitative Figures

Recommended appendix figures and tables include:

1. [cross_dataset_f1_comparison.png](artifacts/figures/report/cross_dataset_f1_comparison.png)  
   Supporting only. Useful for absolute `F1` comparison, but weaker than the main transfer-delta figure for argumentation.

2. [stability_f1_errorbars.png](artifacts/figures/report/stability_f1_errorbars.png)  
   Supporting only. Useful as a compact seed-stability supplement.

3. [final_replay_confusion_matrix_caucafall_tcn_op2.png](artifacts/figures/report/appendix/final_replay_confusion_matrix_caucafall_tcn_op2.png)
   Supporting only. Useful as a bounded replay diagnostic for the preferred `CAUCAFall + TCN + OP-2` runtime preset. It should not be read as a replacement for the frozen offline benchmark figures.

4. Diagnostic visuals under `artifacts/figures/report/diagnostic/`
   Diagnostic only. These should remain clearly labeled as pre-fix or diagnostic materials if included.

These supporting quantitative figures are useful when the reader wants a more granular view than the main narrative can comfortably carry. They are not weak because they are in the appendix; they are in the appendix because they answer narrower questions than the main report needs to answer first.

**Table B1. Supporting-figure interpretation guide**

| Supporting figure family | Best use | Should not be used for |
| --- | --- | --- |
| absolute cross-dataset `F1` bars | quick visual support for transfer asymmetry | replacing the main delta-based transfer interpretation |
| seed-stability error bars | compact reinforcement of multi-seed behaviour | standalone proof of operational usefulness |
| bounded replay confusion matrix | quick diagnostic view of the preferred runtime preset | replacement of frozen offline benchmark evidence |
| diagnostic pre-fix visuals | explaining historical mismatch or debugging context | defended final performance claims |

This table is helpful because appendix figures often create interpretation drift when they are read without context. The report is stronger when it tells the reader what each supporting figure is for instead of assuming that appendix placement alone makes its role self-evident.

### B.2 Design-Evolution Figures

1. Proposal-stage low-fidelity wireframes:
   [wireframe_dashboard_lowfi.jpg](artifacts/figures/report/appendix/wireframe_dashboard_lowfi.jpg),
   [wireframe_monitor_lowfi.jpg](artifacts/figures/report/appendix/wireframe_monitor_lowfi.jpg),
   [wireframe_events_lowfi.jpg](artifacts/figures/report/appendix/wireframe_events_lowfi.jpg), and
   [wireframe_settings_lowfi.jpg](artifacts/figures/report/appendix/wireframe_settings_lowfi.jpg)
   Design-evolution evidence. Useful for showing that dashboard, monitor, event-history, and settings were already treated as core workflow surfaces during the proposal stage. These are not final-system screenshots.

2. [wireframe_dashboard.png](artifacts/figures/report/appendix/wireframe_dashboard.png) and [wireframe_settings.png](artifacts/figures/report/appendix/wireframe_settings.png)
   Supporting wireframe exports. Useful for showing early interaction structure and requirement shaping.

3. [hifi_live_monitor.png](artifacts/figures/report/appendix/hifi_live_monitor.png) and [hifi_event_history.png](artifacts/figures/report/appendix/hifi_event_history.png)
   Higher-fidelity design evidence. Useful for discussing interface evolution, but should not be confused with final runtime screenshots.

4. [legacy_op_tradeoff_concept.png](artifacts/figures/report/appendix/legacy_op_tradeoff_concept.png)
   Early conceptual visualization of safety-versus-alarm trade-off. Useful as historical design context, not as final evidence.

5. [legacy_system_tier_diagram.jpg](artifacts/figures/report/appendix/legacy_system_tier_diagram.jpg)
   Proposal-stage architecture reference. Useful when read together with `Figure 11` in the main text to show what later refactoring clarified.

The appendix is also the correct place to preserve a strict distinction between main figures and supporting figures. Diagnostic-only visuals should appear here with explicit explanatory captions rather than being allowed to compete with the main evidence figures in the core results sections.

The design-evolution figures are valuable because they document how the interface and operational framing matured. They are presented as process evidence, not as proof that the final implemented pages match every element of the early mockups. Their function is to show deliberate evolution from early wireframe reasoning to later interface structure.

**Table B2. Design-evolution figure roles**

| Figure family | Stage represented | Contribution to the report |
| --- | --- | --- |
| low-fidelity wireframes | early workflow planning | shows that dashboard, monitor, event-history, and settings roles were designed intentionally |
| supporting wireframe exports | early workflow planning | preserves additional proposal-stage interface evidence |
| high-fidelity previews | later interaction refinement | shows how observability and review semantics were made more explicit |
| legacy architecture visuals | proposal-stage system thinking | provides contrast for later contract-oriented refactoring |
| early operating-point concept sketch | conceptual policy framing | shows that safety-versus-alert-burden trade-off was recognized before final fitting artifacts existed |

This appendix section therefore does more than preserve old images. It documents the evolution of system thinking: from workflow sketches, to refined interface intent, to final code and artifact boundaries. That progression is valuable in a full report because it shows that the final artifact emerged from structured iteration rather than from isolated coding bursts.

**Table B3. Appendix-figure usage policy**

| Appendix figure class | May support | Must not be used to support |
| --- | --- | --- |
| supporting quantitative figure | narrower secondary interpretation of already reported quantitative results | replacement of the main defended result figure |
| design-evolution figure | process maturity, design traceability, workflow intent | final implementation quality claims by itself |
| diagnostic figure | explanation of historical mismatch or debugging context | final performance or deployment claims |

This usage-policy table is valuable because it prevents the appendix from becoming a loophole through which weaker visuals silently gain stronger authority than they deserve.

## Appendix C: Deployment and Replay Notes

This appendix summarises the practical runtime presets used during replay and live testing, including the preferred `CAUCAFall + TCN + OP-2` demo line, the rationale for keeping replay visual-only, and the relationship between monitor state, event persistence, and downstream Telegram delivery.

It also preserves the most important bounded-interpretation notes: replay evidence is system evidence, not unseen-test evidence; field-validation clips remain too sparse for broad claims; and the uncertainty-aware path currently shows neutral deployment value on the fixed replay matrix.

Appendix C records the practical runtime semantics that support the defended deployment interpretation, including the preferred preset, replay-versus-realtime persistence meaning, and runtime-evidence capture discipline.

### C.1 Preferred Runtime Preset

The preferred deployment-facing preset in the current project is `CAUCAFall + TCN + OP-2`. This line is preferred not simply because it is the strongest single replay row, but because it is also the line where offline primary-dataset evidence, active operating-point interpretation, and bounded runtime evidence align most cleanly. That alignment makes it the most defensible demo and replay preset in the full report.

### C.2 Replay and Realtime Persistence Semantics

One of the most important late-stage clarifications in the repository was that replay and realtime should not share semantics implicitly. Replay mode is used as a visual demonstration and bounded runtime check. It does not create Event History records in the final software artifact. Realtime monitoring is the event-producing path: when the backend policy confirms a realtime fall event, that event can be persisted, reviewed, and passed to the notification path. This split prevents controlled replay output from being mistaken for stored incident evidence.

### C.3 Monitor State, Event State, and Delivery State

The deployment path follows a three-stage progression. First, the monitor can enter a fall-like or uncertain state. Second, depending on policy and session mode, that state may become a persisted event. Third, if notification delivery is enabled and the event qualifies, the event can trigger downstream Telegram delivery. This distinction prevents operational language from becoming imprecise.

**Table C1. Runtime-state interpretation guide**

| Stage | Meaning | Evidence type |
| --- | --- | --- |
| monitor state | transient runtime interpretation visible on the monitor page | operational runtime evidence |
| persisted event | reviewable incident written to history/dashboard | system-state evidence |
| Telegram delivery | caregiver-facing downstream notification | delivery-path evidence |

### C.4 Boundaries of Deployment Interpretation

This appendix is read together with the main runtime-results chapter. The key boundary remains unchanged: replay and live checks support bounded system claims, not additional benchmark claims. They strengthen the argument that the integrated stack works, but they do not replace frozen offline test evidence.

### C.5 Runtime-Evidence Capture Discipline

The runtime evidence panel was captured from the defended demo preset. The same capture discipline applies if the screenshots are ever replaced: the monitor view, persisted event view, and delivery view should all come from one coherent same-incident runtime sequence rather than from visually cleaner but semantically unrelated sessions.

## Appendix D: Audit and Code Review Summaries

This appendix provides condensed summaries of the full-stack audit and full code-review pass. The main body already explains why these reviews mattered; the appendix preserves the operational details, such as the major mismatch classes that were corrected, the freeze and allowlist logic, and the final status of batch-based code review across ML, backend, frontend, and scripts/tests.

The appendix role here is archival and defensive. It lets the report show serious verification work without turning the main narrative into a review log.

That archival role is important in a long report because verification evidence can otherwise become oddly under-specified. A reader may accept that audit and code review occurred, yet still have no concrete sense of what kinds of mismatch were actually being controlled. Appendix D solves that problem by giving the report a compact but specific record of review scope, major issue classes, and remaining conditional risk. In that sense it functions as the bridge between the narrative claim “the repository was reviewed” and the more defensible claim “these are the categories of inconsistency that were actually checked and reduced.”

### D.1 Review Logic

The review work proceeded in two layers. The first layer was repository and evidence audit, concerned with active versus archive material, figure and artifact authority, freeze state, and report-evidence consistency. The second layer was code review, concerned with line-level logic, cross-module contract consistency, and workflow mismatch risk across ML, backend, frontend, and scripts/tests.

This distinction is useful because it clarifies what kind of problem each review pass was meant to catch. The repository/evidence audit focused on whether the project could still defend its claims coherently. The code review focused on whether the implementation actually behaved according to those defended contracts.

**Table D1. High-level review closure summary**

| Review area | Representative issue class | Closure outcome |
| --- | --- | --- |
| ML pipeline | dataset-contract and evaluation-contract mismatch | corrected and regression-checked |
| backend runtime | persistence semantics, notification truth source, active profile normalization | corrected and test-covered |
| frontend | replay/live state semantics, fallback contract drift, monitor control meaning | corrected and spot-tested |
| scripts/tests | canonical test coverage gaps, build invocation friction, freeze verification | corrected with updated script entrypoints |
| repository/evidence layer | active vs archive confusion, report/evidence drift | corrected through freeze and inventory work |

The purpose of this appendix table is not to replace the audit documents in the repository. It is to make the final report self-contained enough that a reader can understand what kinds of review work were performed and why that work improved the credibility of the final artifact.

### D.2 Representative Corrected Mismatch Classes

The most important corrected mismatch classes include:

1. configuration and fallback drift, where different parts of the stack were previously capable of interpreting the active profile differently
2. replay-versus-realtime semantic ambiguity, especially around preventing replay output from being mistaken for persisted event evidence
3. notification truth-source drift, where older abstractions could have been mistaken for actual delivery evidence
4. data and evaluation contract drift, especially around FPS assumptions, window metadata semantics, and recursive artifact discovery

This list matters because it demonstrates that the reviews were materially connected to the report’s central methodological concerns. They were not generic cleanup passes detached from the scientific content.

### D.3 Review Boundary and Remaining Conditional Risk

The review and remediation work substantially reduced mismatch risk, but it did not erase every conditional boundary. The main remaining conditional item is environment-sensitive torch-backed verification on machines where `import torch` is unstable. This is important to preserve in the appendix because it distinguishes a real environment caveat from an unresolved logic defect in the reviewed code. The full report is stronger when it records that difference explicitly.

The practical implication is that the repository now has a stronger defended core than it had before review, but that defended core is still conditioned by environment quality for a subset of monitor-facing verification paths. The report therefore treats the torch-sensitive layer as conditional verification rather than as an ignored gap. That framing is more rigorous than pretending all validation layers were equally available on every local machine.

**Table D2. Review-phase summary**

| Review phase | Main question | Representative outcome |
| --- | --- | --- |
| evidence audit | do claims, figures, and active artifacts still line up? | report/evidence drift reduced through freeze and inventory work |
| ML code review | do data and evaluation contracts still support the claimed protocol? | FPS, window-metadata, and recursive-discovery mismatches corrected |
| backend code review | do runtime state, persistence, and delivery share one interpretation? | event and notification truth sources aligned |
| frontend code review | do UI controls and displayed state map cleanly to backend contracts? | replay/live semantics and fallback meanings clarified |
| scripts/tests review | do build and test entrypoints still reflect the defended repository state? | canonical test entrypoints and report build path cleaned up |

This table is useful because it converts a long review history into a readable verification narrative. It shows that review was staged and purposeful rather than a single undifferentiated cleanup burst.

Appendix D therefore supports one of the report's broader methodological claims: systems credibility is not only a matter of adding more test counts. It is also a matter of making cross-layer verification legible. The appendix keeps that legibility available to the reader without forcing the main chapters to become dominated by review bookkeeping.

## Appendix E: Reproducibility Commands

This appendix lists the canonical commands needed to rebuild the main report artifacts and key regression outputs.

Appendix E is important because a report of this scale cannot rely on implicit reproducibility. The main text can explain which artifacts matter, but it is the appendix that shows how those artifacts are actually regenerated and under what prerequisites. This is especially useful in the present project, where some rebuild paths are straightforward while others depend on environment-sensitive torch availability. By preserving that distinction explicitly, the appendix turns reproducibility from an aspiration into a structured part of the defended submission.

**Table E1. Reproducibility prerequisites**

| Layer | Minimum requirement | Reason |
| --- | --- | --- |
| Python/runtime | active project virtual environment with repository `PYTHONPATH` | required for scripts, tests, and pipeline commands |
| Node/frontend | working Node/npm environment inside `applications/frontend/` | required for frontend regression checks |
| document build | `pandoc` and `xelatex` available to `build_report.sh` | required for PDF export |
| torch-backed checks | stable local environment where `import torch` succeeds | required for `contract` and `monitor` validation modes |
| repository state | active figures and active frozen artifacts present | required so the report rebuild matches the defended snapshot |

This prerequisite table is useful because reproducibility in this project is layered rather than monolithic. Some commands are fully available in a standard environment; others depend on a stable torch setup. Making that distinction explicit is more informative than presenting one flat command list without environment context.

**Table E2. Verification-outcome summary**

| Verification layer | Current outcome in the defended snapshot | Interpretation |
| --- | --- | --- |
| report build | passes | source markdown can be turned into reviewable PDF output |
| torch-free canonical tests | passes | core repository and contract logic are regression-covered |
| frontend regression layer | passes on the active targeted test set | selected UI/API assumptions are checked |
| torch-backed contract layer | conditional on machine stability | environment caveat remains explicit rather than hidden |
| CAUCAFall data/eval regression | rerun and checked | primary data/evaluation path was revalidated after code review |
| LE2i data/eval regression | rerun and checked | comparative data/evaluation path was revalidated after code review |

This table is helpful because it turns the command appendix into a defended-state appendix. It tells the reader not only what can be run, but what the current reviewed repository state says about the outcome of those runs.

**Table E3. Runtime-evidence status**

| Item | Current state | Notes |
| --- | --- | --- |
| live monitor + Telegram screenshot | inserted from defended realtime evidence | same-incident capture retained |
| persisted event-history screenshot | inserted from the same defended incident chain | same-incident capture retained |
| supplementary fall video | compressed and linked as `Supplementary Video S1` | submission copy present |
| supplementary ADL video | compressed and linked as `Supplementary Video S2` | submission copy present |

This table records the final runtime-evidence assets referenced by the report.

**Table E4. Command-to-output map**

| Command family | Main output or state change | Report relevance |
| --- | --- | --- |
| report build | current PDF under `artifacts/report_build/` | proves that the report is buildable as a defended artifact |
| canonical tests | regression pass/fail state across torch-free, frontend, and conditional torch-backed layers | supports software-validation claims |
| freeze verification | defended-core path existence and cleanliness check | supports repository defensibility claims |
| data/eval regression commands | regenerated labels, windows, and locked metric outputs for primary datasets | supports post-review protocol integrity |
| figure regeneration commands | refreshed report figures under the unified figure directory | supports figure traceability and rebuildability |

This map is useful because reproducibility is easier to understand when commands are linked to the artifacts they are supposed to regenerate, not only listed as shell text.

The appendix therefore does more than preserve shell snippets. It shows that the defended report, the defended tests, and the defended data/evaluation paths all have concrete regeneration routes. In a high-quality long report, that kind of explicit rebuildability is part of what separates a polished artifact from a merely descriptive one.

**Report build**

```bash
./scripts/build_report.sh docs/reports/drafts/FULL_PROJECT_REPORT_FINAL_2026-04-11.md "" "" --pdf-only
./scripts/build_report.sh docs/reports/drafts/PAPER_FINAL_2026-04-11.md "" "" --pdf-only
```

**Canonical tests**

```bash
./ops/scripts/run_canonical_tests.sh torch-free
./ops/scripts/run_canonical_tests.sh frontend
./ops/scripts/run_canonical_tests.sh contract
./ops/scripts/run_canonical_tests.sh monitor
```

**Freeze verification**

```bash
./ops/scripts/release_manifest.sh
```

**Primary `CAUCAFall` data/evaluation regression**

```bash
make -B labels-caucafall
make -B splits-caucafall
make -B WIN_CLEAN=1 windows-caucafall
make -B WIN_EVAL_CLEAN=1 windows-eval-caucafall
make -B repro-best-tcn-caucafall
make -B repro-best-gcn-caucafall
```

**Primary `LE2i` data/evaluation regression**

```bash
make -B labels-le2i
make -B splits-le2i
make -B WIN_CLEAN=1 windows-le2i
make -B WIN_EVAL_CLEAN=1 windows-eval-le2i
make -B repro-best-gcn-le2i-paper
```

**Figure regeneration**

```bash
# regenerate the branch-local cross-dataset summary CSV from the frozen eval JSON files
python3 ops/scripts/build_cross_dataset_summary.py --manifest docs/reports/notes/cross_dataset_manifest_2026-04-27.json --out_csv docs/reports/notes/cross_dataset_summary_2026-04-27.csv
python3 ops/scripts/plot_cross_dataset_transfer.py --summary_csv docs/reports/notes/cross_dataset_summary_2026-04-27.csv --out_fig artifacts/figures/report/cross_dataset_transfer_summary.png
```

The key value of this appendix is not convenience alone. It demonstrates that the final report remains tied to executable paths in the repository rather than to one-off undocumented manual steps.
