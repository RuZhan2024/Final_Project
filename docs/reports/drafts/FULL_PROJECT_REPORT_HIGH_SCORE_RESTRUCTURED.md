# Design and Evaluation of a Pose-Based Fall Detection and Monitoring System
## From Offline Model Comparison to Bounded Deployment Evidence

# Abstract

Falls are clinically important because delayed detection can increase injury severity and time to intervention. In practice, however, fall monitoring is not only a classification problem. A useful system must convert incoming visual input into operationally meaningful alerts, keep false alarms under control, and preserve event semantics that support later review and caregiver notification. This report studies that broader problem through a pose-based monitoring system rather than through a model-only benchmark exercise.

The implemented system uses browser-side pose extraction to convert live or replay video into fixed temporal windows for backend inference. Two model families are compared under a shared pose-window contract: a Temporal Convolutional Network (TCN) and a matched custom spatio-temporal Graph Convolutional Network (GCN). Validation-side operating-point fitting, temporal alert policy, event persistence, and Telegram-based caregiver notification are treated as explicit parts of the method rather than as informal post-processing.

Under the locked primary `CAUCAFall` protocol, the TCN shows a cautious directional advantage over the matched custom GCN. The strongest offline comparative evidence comes from the frozen multi-seed summaries, while cross-dataset results show asymmetric transfer rather than broad robustness. At the system level, the project delivers an end-to-end monitoring artifact that supports pose-window inference, fitted runtime policy, persisted event history, dashboard review, and caregiver-facing delivery. Runtime replay analysis shows profile-sensitive behaviour rather than a single uniformly strongest deployment profile, and targeted retraining modestly improves selected bounded replay surfaces to `16/24`.

The report therefore makes a bounded claim. It shows that pose-window inference, alert-policy fitting, persistence, review, and Telegram delivery can be integrated into one monitoring system under controlled conditions. It does not show broad deployment robustness, clinical readiness, or solved real-home performance. Replay and live evidence are reported as system-validation evidence, not as substitutes for frozen benchmark evaluation.

# 1. Introduction

Falls are a serious safety concern because delayed intervention can worsen injury outcomes, especially in domestic or semi-independent settings. Automated fall detection is therefore attractive, but practical monitoring is difficult for two reasons. First, falls must be distinguished from ordinary activities such as sitting, lying, or recovering posture under variable camera conditions. Second, even when model discrimination is strong, operational usefulness still depends on how scores are turned into alerts, how incidents are stored, and how caregivers are notified.

This report studies that problem through a pose-based monitoring system. The project does not rely on raw RGB classification alone. Instead, it uses browser-side pose extraction to generate temporal skeleton windows, compares temporal and graph-based sequence models under a shared protocol, and then studies how validation-fitted operating points and temporal alert policy shape runtime behaviour. This framing is important because a monitoring system cannot be evaluated only as a window-level classifier. It must also be evaluated as a path from observed motion to interpreted alert state, persisted incident record, and downstream delivery.

Pose-based monitoring is a reasonable direction for this problem because it provides an interpretable temporal representation, supports a more privacy-oriented deployment shape than backend video storage, and fits the project’s design goal of a local or browser-led monitoring front end. At the same time, pose-based systems inherit sensitivity to occlusion, camera geometry, pose quality, and frontend runtime behaviour. These constraints are therefore treated as part of the technical problem rather than as peripheral caveats.

The aim of this project is to design, implement, and evaluate a pose-based fall monitoring system that connects controlled model comparison with bounded runtime evidence.

The project has four main objectives. First, it aims to compare a TCN and a custom spatio-temporal GCN under shared preprocessing, windowing, and evaluation rules. Second, it aims to fit deployment-facing operating-point profiles on validation data only, so that runtime behaviour is governed by tracked policy rather than by ad hoc threshold choice. Third, it aims to implement a working monitoring system with frontend, backend, persistence, and caregiver-notification components rather than stopping at notebook-level evaluation. Fourth, it aims to preserve evidence discipline through audit, code review, and freeze-state cleanup so that final claims remain traceable to defended artifacts.

This report makes four main contributions.

1. It presents a locked-protocol comparative result in which the TCN trends stronger than the matched custom spatio-temporal GCN under a shared pose-window representation and frozen multi-seed evaluation design.
2. It establishes validation-side operating-point fitting and temporal alert policy as explicit methodological layers rather than leaving deployment behaviour to manual threshold selection.
3. It implements an integrated monitoring artifact that joins browser-side pose extraction, backend inference, runtime interpretation, event persistence, dashboard review, and caregiver-facing Telegram delivery.
4. It provides a layered evidence framework that separates frozen offline results, cross-dataset transfer, replay/runtime validation, and notification-path evidence while still supporting one coherent system story.

The report proceeds as follows. Chapter 2 places the work in the context of pose-based fall detection, temporal and graph sequence models, and deployment-oriented evaluation. Chapter 3 defines the system requirements, research questions, and evidence boundaries. Chapters 4 to 7 describe the final architecture, experimental protocol, modelling and alert-policy design, and implementation. Chapters 8 and 9 present the comparative results and the system-validation layer. Chapters 10 to 12 discuss the findings, define the principal limitations, and conclude.

# 2. Background and Related Work

## 2.1 Vision-Based Fall Detection

Vision-based fall detection has been studied through RGB video, depth, infrared, and derived motion representations. Earlier work such as Charfi et al. (2013) focused on engineered spatio-temporal descriptors and conventional classifiers, while more recent systems increasingly rely on deep temporal models and dataset-specific evaluation pipelines. The appeal of vision-based sensing is clear: it does not require the monitored person to wear or charge a device, and it can support room-level monitoring in domestic spaces. Its main limitations are equally clear: appearance dependence, camera-position sensitivity, privacy concerns, and the difficulty of translating raw model scores into actionable alerts.

These limitations matter because the project is not trying to solve an abstract action-recognition problem in isolation. It is trying to build a monitoring system whose outputs remain interpretable once persistence, review, and caregiver notification are introduced. That requirement creates a gap between pure video recognition work and deployment-oriented monitoring work.

## 2.2 Skeleton and Pose-Based Fall Detection

Pose-based fall detection compresses the scene into articulated landmark trajectories rather than full-frame appearance. This direction is attractive because it reduces dependence on texture and lighting, provides a structured temporal signal, and fits a more privacy-oriented deployment shape than backend video storage. The approach is not cost-free. A pose-based system becomes directly dependent on landmark quality, missing joints, viewpoint distortion, and frontend runtime stability.

This project uses `CAUCAFall`, `LE2i`, and `MUVIM` to test that trade-off under different evidence roles. `CAUCAFall` provides the primary benchmark and deployment-target dataset (Eraso Guerrero et al., 2022), `LE2i` supports in-domain comparison and transfer-boundary analysis (Charfi et al., 2013), and `MUVIM` supports broader exploratory experimentation (Denkovski et al., 2022). Together these datasets show that pose-based monitoring can be technically viable while still remaining sensitive to camera geometry, occlusion, and domain mismatch.

That dataset split is important for the literature positioning of the report. `CAUCAFall` is the dataset where benchmark comparison, fitted operating points, and bounded deployment-facing replay can be narrated together most cleanly. `LE2i` plays a different role: it prevents the project from collapsing into a single-dataset success claim and exposes how strongly a pose-based system can depend on domain direction, scene geometry, and policy interpretation. `MUVIM` is not used to inflate the main benchmark story; it is retained as evidence that the pipeline and representation choices were exercised beyond the final two-dataset core.

## 2.3 Temporal and Graph Models for Skeleton Sequences

Skeleton sequences can be modelled with temporal convolutions, graph-based joint reasoning, or hybrid forms. Temporal Convolutional Networks treat the sequence as a structured time series and learn discriminative motion patterns over fixed windows. Graph models preserve joint topology more explicitly and are attractive when body structure is expected to help action discrimination. Representative literature includes ST-GCN (Yan et al., 2018), 2s-AGCN (Shi et al., 2019), CTR-GCN (Chen et al., 2021), MS-TCN (Abu Farha and Gall, 2019), and lightweight temporal video modules such as TSM (Lin et al., 2019).

The current project does not attempt to claim a universal ranking among all skeleton architectures. Instead, it asks a narrower question: under one locked representation, one window contract, and one fitting protocol, does the matched custom GCN displace the TCN as the strongest defended line? That question is methodologically stronger for a final-year project because it controls more variables than a broad benchmark survey.

## 2.4 Calibration, Thresholding, and Alert Policy

Many fall-detection discussions stop at classifier outputs, but monitored alerting is rarely determined by a single raw threshold. Calibration, temporal smoothing, confirmation rules, cooldown handling, and persistence semantics all shape what a deployed system actually does. Temperature scaling (Guo et al., 2017) and selective or uncertainty-aware decision ideas such as SelectiveNet (Geifman and El-Yaniv, 2019) are relevant here, but in this project their role is bounded. Validation-side calibration informs operating-point fitting; it is not used as a claim that all runtime probabilities are universally well calibrated online.

The project therefore treats alert policy as an explicit methodological layer. That layer is part of the answer to how model outputs become reviewable incident states, and it is one of the main reasons the report is framed as a system study rather than a classifier comparison alone.

## 2.5 Deployment-Oriented Evaluation

Deployment-oriented evaluation asks a different question from pure benchmark evaluation. It asks whether model outputs survive replay, persistence, user-facing interpretation, and delivery logic. In this project, replay validation, live demonstration, event history, and Telegram delivery all contribute to system evidence. They do not replace unseen-test benchmark evidence. Runtime behaviour can be highly policy-sensitive even when the underlying checkpoint remains the same.

A recurring limitation in deployment-facing fall-detection work is that runtime demonstrations are often presented without a clear boundary between benchmark evidence and system evidence. The project’s replay and live evidence are therefore reported as bounded system-validation evidence. This keeps the report honest about what runtime checks can and cannot support.

That distinction becomes especially important in a monitoring project. A benchmark result may show that a model can separate positive and negative windows under one frozen protocol, but a monitoring artifact still has to survive score smoothing, profile fitting, event persistence, user-facing review, and caregiver delivery. These later layers are not cosmetic. They determine whether a high-scoring window becomes a transient state, a reviewable event, or no delivered alert at all. For that reason, the present report treats deployment-oriented evaluation as a structured continuation of the model story rather than as an informal demonstration attached at the end.

## 2.6 Research Gap

The practical gap addressed in this report is not only the lack of another fall classifier. It is the lack of a controlled study that links:

- a shared pose-window representation,
- a locked temporal-versus-graph comparison,
- a validation-fitted alert-policy layer,
- and a working monitoring path with persistence and caregiver notification.

Existing work often addresses some of these layers but not all of them together under one disciplined evidence structure. This project addresses that gap by treating model comparison, alert interpretation, and runtime integration as one connected problem while keeping the corresponding evidence layers analytically separate.

# 3. Requirements and Research Questions

## 3.1 System Requirements

The project requirements are system-level rather than model-only. The monitor must support both live and replay input, convert visual input into pose-derived temporal windows, apply the active model and fitted profile on the backend path, persist reviewable fall events, and deliver caregiver-facing notification on the active runtime path. It must also support replay validation without allowing replay evidence to overwrite benchmark evidence.

These requirements explain why the project includes frontend, backend, persistence, and notification behaviour in addition to model evaluation. They also explain why requirement satisfaction must be judged end to end: monitor input is only useful if it becomes windows under one stable contract, and notification is only useful if it is triggered by reviewable event semantics rather than by a transient raw score.

**Table 1. Functional requirements and verification summary**

| ID | Requirement | Implementation route | Verification |
| --- | --- | --- | --- |
| FR-1 | Support live and replay input through one monitor surface. | Monitor UI and pose-monitor hook. | Live and replay runtime validation. |
| FR-2 | Convert visual input into pose-derived temporal windows. | Browser-side pose extraction and window packaging. | Runtime checks and contract review. |
| FR-3 | Apply the active model and fitted profile to produce monitor states. | Backend runtime service and active ops configs. | Canonical tests, replay validation, and code review. |
| FR-4 | Persist reviewable fall events for later inspection. | Event repository and event routes. | Event-history and dashboard checks. |
| FR-5 | Deliver caregiver-facing notification on the active path. | Telegram manager and notification audit store. | Notification checks and delivery-path verification. |
| FR-6 | Keep replay validation separate from benchmark evidence. | Replay controls and realtime-only persistence semantics. | Replay validation and audit findings. |

## 3.2 Non-Functional Requirements

The most important non-functional requirements are reproducibility, interpretability, auditability, and bounded runtime coherence. Reproducibility requires frozen candidates, tracked validation-fit operating points, and explicit dataset timing contracts. Interpretability requires that operators can distinguish replay from realtime, see which model/profile is active, and inspect persisted events separately from transient monitor state. Auditability requires that the active system state, supporting artifacts, and report claims remain aligned after cleanup and review. Runtime coherence requires that the monitor, persistence, dashboard, and notification path behave as one integrated system rather than as loosely related components.

These non-functional requirements are especially important because the strongest contribution of the project is systemic. The project depends on both controlled model evidence and an interpretable software artifact.

## 3.3 Research Questions

The report is organised around three locked research questions.

**RQ1. Comparative Offline Performance.** Under the locked offline evaluation protocol, how do the TCN and the custom spatio-temporal GCN compare on the primary fall-detection task?

**RQ2. Operating-Point Fitting and Operational Alerting.** How does validation-side operating-point fitting influence the conversion of window-level model outputs into practical alert decisions?

**RQ3. Runtime Feasibility and Deployment Limits.** What do replay deployment evidence and limited realtime validation show about the practical feasibility and current runtime limits of the system?

These questions are narrower than a generic “does the system solve fall detection?” framing, but they better match the strongest current evidence in the repository.

## 3.4 Evidence Boundaries

The report uses a layered evidence policy.

- Frozen offline summaries support comparative model claims.
- Cross-dataset transfer runs support limitation and directionality claims.
- Replay matrices support bounded runtime and policy claims.
- Live demonstrations support feasibility claims about the integrated monitoring path.
- Notification checks support the existence of an end-to-end delivery path, not improved model quality.

This separation is central to the final report. Each claim is kept at the level of evidence that actually supports it.

# 4. System Architecture

The implemented system is a full-stack pose-based fall-detection and monitoring artifact rather than a disconnected set of scripts. The architecture links browser-side capture and pose extraction, temporal window packaging, backend inference and policy interpretation, event persistence, dashboard review, and Telegram-based delivery.

At the highest level, the system has five functional layers:

1. frontend monitoring and capture
2. backend runtime inference
3. alert-policy interpretation
4. persistence and review
5. caregiver notification

The most important architectural boundary is the split between frontend and backend responsibility. The frontend is not only a display surface. It owns input capture, browser-side pose extraction, and temporal window packaging. The backend is not merely a prediction endpoint. It owns active profile resolution, model inference, temporal policy application, event semantics, persistence, and downstream delivery. This split is analytically useful because several runtime findings in the report arise from the interaction between frontend pose quality and backend policy rather than from checkpoint quality alone.

The architecture also preserves a distinction between offline pipeline and monitoring runtime. Offline preprocessing, training, fitting, and evaluation can be studied independently of the monitor. Runtime behaviour is then analysed as a software path that consumes tracked models and tracked operating-point profiles. This is what allows the report to keep benchmark evidence and deployment evidence separate without treating them as unrelated projects.

**Figure 1. System architecture and decision path**

![](artifacts/figures/report/system_architecture_diagram.svg){ width=85% }

Figure 1 shows the runtime decision path from browser-side pose-window construction to backend policy interpretation, persisted event creation, and caregiver-facing notification. The critical architectural constraint is that notification is downstream of persisted event state rather than a direct reaction to raw model scores.

**Table 2. Architecture responsibility summary**

| Layer | Responsibility | Input | Output |
| --- | --- | --- | --- |
| Frontend monitor | Capture live or replay input, extract pose, package temporal windows. | Camera stream or replay clip. | Pose-derived window payloads. |
| Backend runtime | Load active model/profile, run inference, apply runtime policy. | Pose-window payloads. | Monitor state and event decisions. |
| Alert policy | Smooth and interpret score streams under fitted operating points. | Window-level model outputs. | Fall-like state, persisted-event eligibility. |
| Persistence | Store and expose reviewable incidents for dashboard and event history. | Confirmed runtime event. | Event-history record and dashboard-visible state. |
| Notification | Deliver caregiver-facing Telegram alert from persisted incident state. | Persisted fall event. | Telegram message and delivery audit record. |

This preserves the same event truth source across monitor state, review, and notification.

Replay and realtime share the same monitor surface, but they do not carry the same evidence meaning. Replay is used for bounded validation and demonstration. Realtime is the event-producing path whose outputs can be persisted, reviewed, and delivered onward.

The repository supports local and Docker-backed runtime paths. The main architectural lesson is that deployment in this project is not just a hosting question. Even when the backend is remote, browser-side pose extraction remains local to the client. Runtime behaviour therefore depends jointly on frontend capture quality and backend policy interpretation.

# 5. Data and Experimental Protocol

The project’s conclusions depend on a controlled protocol rather than on one-off runs. Dataset role assignment, split construction, FPS handling, temporal windowing, and validation-only operating-point fitting are all part of that protocol. The report therefore treats the data and evaluation contract as part of the method rather than as invisible preprocessing.

## 5.1 Dataset Roles

`CAUCAFall` is the primary benchmark and the primary deployment-target dataset (Eraso Guerrero et al., 2022). It anchors the strongest offline comparison, the preferred operating-point family, and the bounded runtime story. `LE2i` is retained as comparative and transfer-boundary evidence (Charfi et al., 2013). It is essential for showing that the project does not claim broad robustness from a single in-domain result. `MUVIM` is retained as a secondary exploratory track rather than a co-equal benchmark axis (Denkovski et al., 2022).

The report also distinguishes benchmark datasets from runtime-support artifacts. Replay clips, delivery-style replay packs, and live demonstration assets are system-validation artifacts, not substitutes for frozen offline test splits.

**Figure 2. Dataset roles and evidence hierarchy**

![](artifacts/figures/report/dataset_roles_evidence_hierarchy.svg){ width=85% }

Figure 2 distinguishes the benchmark role of `CAUCAFall`, the comparative and transfer-boundary role of `LE2i`, and the secondary exploratory role of `MUVIM`, while also separating benchmark, replay, live, and delivery evidence.

## 5.2 Pose Extraction and Preprocessing

The system operates on skeleton sequences rather than raw RGB inputs. Browser-side or preprocessing-time pose extraction produces structured pose observations, which are then converted into temporal feature windows. In the defended implementation, browser-side and preprocessing-time extraction both depend on MediaPipe Pose as an implementation dependency rather than as a report contribution. This design narrows backend responsibility and aligns the runtime path with the same representation family used in offline comparison. It also means that pose quality becomes a first-order methodological concern. Degraded landmarks, missing joints, or unstable frontend timing can affect both benchmark interpretation and runtime behaviour.

In the defended preprocessing path, confidence-aware short-gap filling, weighted temporal smoothing, and torso-centred normalization are applied before final window packaging. These steps preserve sequence length while making the pose stream more stable for both offline comparison and monitor replay.

**Table 3. Pose preprocessing summary**

| Step | Purpose |
| --- | --- |
| landmark extraction | convert visual input into pose sequences under one shared representation |
| confidence handling | standardise missing-value semantics and reduce the influence of unreliable pose observations |
| short-gap filling | interpolate brief missing landmark gaps without changing sequence length |
| temporal smoothing | reduce frame-level landmark jitter before window construction |
| torso normalization | apply body-centric normalization so scale and camera-position variation are reduced |
| FPS alignment | keep timing metadata consistent across datasets and runtime surfaces |
| temporal packaging | convert framewise pose into fixed windows for inference |

## 5.3 Label and Split Construction

Labels, spans, and split files are treated as tracked artifacts rather than invisible pipeline side effects. Operating-point fitting is performed on validation outputs only, and cross-dataset evaluation remains interpretable only if the training, validation, and test boundaries stay fixed and reproducible. For `CAUCAFall`, split logic is tied to the project’s FPS and windowing semantics. For `LE2i`, split logic follows its own dataset conventions while still remaining inside the shared evaluation policy.

For `CAUCAFall`, the defended split is subject-independent, so windows from the same subject are not allowed to leak across train, validation, and test boundaries. This avoids the much weaker alternative of random window-level splitting, which would risk placing highly similar windows from the same subject or sequence into different evaluation partitions.

## 5.4 Temporal Windowing

The primary evaluation path uses a locked temporal contract of `W = 48` and `S = 12`. The same contract is used in the main offline comparison and in the deployable monitoring path. This prevents the report from comparing architectures on one temporal geometry and then interpreting runtime behaviour on another.

The project also uses inclusive `w_end` semantics for temporal windows. This detail matters because alert timing, label alignment, and event-level interpretation all depend on consistent frame-index meaning.

**Figure 3. Temporal window contract**

![](artifacts/figures/report/temporal_window_contract.svg){ width=85% }

Figure 3 fixes the shared timing semantics used across training, evaluation, and monitor replay so that alert timing and label alignment remain comparable.

## 5.5 Evaluation Metrics

The report uses a layered metric logic.

- `AP` is the threshold-free ranking metric for the underlying score surface.
- `F1`, `Recall`, and `FA/24h` are interpreted under the fitted operating-point and event-policy surface used for the corresponding frozen summary.
- `FA/24h` expresses false-alert burden in deployment-shaped terms.
- Cross-dataset transfer results show where event-level behaviour collapses or remains asymmetric.
- Replay matrices express bounded runtime outcomes at clip level.
- Accuracy is reported only where it helps describe bounded replay surfaces; it is not the main offline comparison metric.

This metric mix is necessary because no single scalar can capture both offline model behaviour and runtime system behaviour.

`FA/24h` is computed as `false alert count / (evaluated duration in seconds / 86,400)`, so it expresses alert burden in an operational time-normalised form.

## 5.6 Evidence Policy

Validation-side operating points are fitted on validation data only. Test data are not used to tune thresholds or alert-policy parameters. Cross-dataset evaluation fits operating points on the source validation split, not on the target test split. Replay evidence is reported as system-validation evidence, not benchmark evidence.

**Table 4. Protocol summary**

| Item | Protocol value |
| --- | --- |
| Primary benchmark / deployment target | `CAUCAFall` |
| Comparative / transfer-boundary dataset | `LE2i` |
| Secondary exploratory dataset | `MUVIM` |
| `CAUCAFall` FPS | `23` |
| `LE2i` FPS | `25` |
| `MUVIM` FPS | `30` |
| Window size | `W = 48` |
| Stride | `S = 12` |
| Window semantics | inclusive `w_end` timing |
| Operating-point fitting | validation only |
| Replay evidence | bounded system validation, not benchmark evidence |

# 6. Model and Alert-Policy Design

The project method is not only model selection. It is a chain:

**pose window -> model score -> fitted operating point -> temporal alert policy -> runtime state or persisted event**

This chapter therefore combines the architecture comparison and the alert-policy layer in one methodological chapter.

## 6.1 TCN Baseline

The TCN is treated as the primary temporal baseline and later becomes the strongest defended line in the reported results. It treats each pose window as a temporal signal and learns motion patterns through convolution over fixed-length windows. In this repository, the TCN proved easier to defend not only because of its offline results, but also because it aligned more cleanly with the fitted alert-policy and bounded runtime interpretation on `CAUCAFall`.

The TCN produces a window-level fall score rather than a final incident decision. That score is later interpreted by the fitted alert-policy layer, so the model is evaluated both as a classifier and as part of a policy-shaped monitoring path.

## 6.2 Custom Spatio-Temporal GCN

The graph baseline is a custom spatio-temporal GCN rather than a direct official reproduction of one canonical public model family. Its value is comparative. It preserves joint-topology reasoning under the same pose-window contract used by the TCN so that the report can ask whether graph-structured motion modelling yields a practical advantage in this project’s setting.

The custom GCN uses skeleton connectivity as an inductive bias so that joint relations remain explicit before temporal aggregation. Its role is not to represent the strongest possible graph model, but to provide a matched topology-aware comparison under the same input and policy contract.

## 6.3 Shared Pose-Window Feature Contract

Both model families operate on the same pose-window representation. At model entry, each sample is treated under a `[T, V, F]` contract, where `T` is temporal length, `V` is the number of pose landmarks, and `F` is the per-landmark feature dimension. This is a major methodological safeguard because it reduces the chance that later differences can be attributed to changing preprocessing, timing, or feature semantics rather than to model family. The same contract also supports the monitoring path, which means the runtime system consumes the same class of windowed pose representation used in the offline experiments.

## 6.4 Training and Candidate Selection

The repository contains a broader training history than the report can promote directly. The final report therefore uses a frozen candidate policy covering four defended lines: `CAUCAFall` TCN, `CAUCAFall` GCN, `LE2i` TCN, and `LE2i` GCN. The exact artifact roots are recorded in Appendix A. The frozen seed set is `1337`, `17`, `2025`, `33724876`, and `42`, so the comparative story is based on multi-seed stability summaries rather than on one convenient checkpoint.

Candidate selection is validation-side rather than test-tuned. The frozen candidates are promoted from tracked training histories into the locked comparative protocol only after validation-side fitting and review.

## 6.5 Validation-Side Operating-Point Fitting

Operating-point fitting turns model outputs into tracked alert profiles. Instead of selecting a threshold manually, the project fits a family of operating points on validation outputs and stores the resulting parameters in YAML artifacts. These profiles then define the runtime monitor path. In practice this means the deployed system is governed by a fitted policy family rather than by an undocumented score cutoff.

The active profile family uses `OP-1`, `OP-2`, and `OP-3` as distinct policy intents. `OP-2` is the preferred balanced deployable profile on the primary `CAUCAFall` TCN line, while `OP-1` and `OP-3` are contrastive looser or stricter variants.

## 6.6 Runtime Alert Policy

The runtime alert layer includes smoothing, temporal confirmation, cooldown behaviour, and persistence semantics. It is therefore possible for a model to rank windows well while still producing weak event-level behaviour, or to appear conservative in replay while remaining operationally useful in bounded deployment settings.

**Figure 4. Alert-policy decision path**

![](artifacts/figures/report/alert_policy_flow.svg){ width=85% }

Figure 4 shows how fitted operating points, smoothing, persistence, and cooldown convert raw window scores into runtime state and persisted-event eligibility.

**Table 5. Model and alert-policy summary**

| Component | Input | Mechanism | Output | Role |
| --- | --- | --- | --- | --- |
| TCN | `[T, V, F]` pose-window tensor | temporal convolution over fixed windows | window-level fall score | primary comparative and deployment-facing line |
| Custom GCN | same `[T, V, F]` pose-window tensor | spatio-temporal graph reasoning over skeleton structure | window-level fall score | matched comparison baseline |
| Alert-policy layer | model score stream plus fitted profile | operating-point fitting, smoothing, temporal decision rules | monitor state, persisted-event eligibility | deployment-facing interpretation layer |

**Table 6. Alert-policy components**

| Component | Purpose | Runtime effect |
| --- | --- | --- |
| `tau_high` | define the high-confidence alert boundary | controls entry into fall-like state |
| `tau_low` | define the lower hysteresis / sustain boundary | helps maintain or release fall-like state without immediate oscillation |
| EMA smoothing | reduce score volatility | suppresses transient spikes |
| `k/n` confirmation | require persistence across windows | reduces one-window alerting |
| cooldown | prevent repeated rapid alert bursts | limits alert fatigue and duplicate incidents |
| persistence rule | separate transient state from stored incident | determines whether a fall-like state becomes a reviewable event |

Window-level model output is therefore not the same thing as a persisted fall event. The project’s system contribution depends on making that distinction explicit.

# 7. Implementation

The final implemented system is stronger than a notebook-level prototype because it integrates frontend monitoring, backend runtime interpretation, persistence, review, and caregiver-facing notification in one artifact. This chapter describes the final system state rather than the full development diary.

At runtime, the main request path is:

**video frame or replay clip -> browser-side pose extraction -> `W = 48` window packaging -> backend monitor request -> active profile/model inference -> alert-policy interpretation -> optional event persistence -> dashboard/event history -> optional Telegram delivery**

## 7.1 Frontend Monitor

The frontend is implemented in React. The monitor page is the critical path because it coordinates live or replay input, browser-side pose extraction, temporal window packaging, and backend communication. It also exposes the active dataset/model/profile selection, replay controls, preview state, and session-mode semantics needed to interpret the runtime path correctly.

Later refactoring separated feature-level API logic, monitor hooks, and page components. This reduced hidden coupling and made replay-versus-realtime state handling easier to reason about.

This frontend behaviour is important to the final claim because the browser is not just a passive UI shell. It is where camera or replay input becomes the pose-window representation consumed by the backend. As a result, frontend timing, pose quality, and mode semantics directly affect the evidence surface presented later in the report. Replay and realtime intentionally share the same surface so that the runtime path remains consistent, but they do not share the same persistence meaning: replay exists for bounded validation and demonstration, while realtime is the path that can generate persisted incident state.

## 7.2 Backend Runtime Service

The backend is implemented in FastAPI and owns active profile resolution, model loading, policy interpretation, and event creation. It does more than return predictions. It distinguishes between transient monitor state, persisted event state, and downstream notification delivery. That distinction is essential because the runtime story depends on semantics, not only on raw probabilities.

In practical terms, a backend monitor response is therefore richer than a classifier output. The service resolves which model family and fitted operating-point profile are active, applies the corresponding policy semantics, and decides whether the current score stream should remain a transient monitor condition or move toward persisted-event eligibility. This makes the backend the main location where the project’s methodological claim is enforced: the same offline score can lead to meaningfully different runtime behaviour once alert policy is applied.

## 7.3 Event Persistence and Dashboard Review

Persisted incidents are written into an event-history path that can later drive dashboard summaries and review views. This means the system includes a genuine review layer rather than only a live monitor. Replay and realtime do not share persistence semantics implicitly. Realtime is the event-producing path; replay is a bounded validation and demonstration path.

This persistence layer is also where the project becomes more than a monitoring demo. A fall-like state that never leaves the live view would be difficult to audit, difficult to review, and difficult to connect to downstream caregiver communication. By persisting the event layer separately, the system keeps monitor behaviour, stored incidents, and notification delivery aligned to one interpretation of runtime state.

## 7.4 Notification Delivery

The active implemented delivery path is Telegram-based. When notification is enabled and the runtime path produces a qualifying persisted incident, the backend sends a caregiver-facing Telegram message and records delivery through the notification audit path. The end-to-end delivery path is therefore real, but it remains bounded to the Telegram-first implementation. SMS and phone-call escalation are future work rather than completed functionality.

This design choice is intentionally conservative. Telegram delivery is presented as the active implemented notification path because it can be demonstrated, audited, and connected to persisted incidents without pretending that a broader escalation stack has already been completed. That makes the delivery claim narrower, but it also makes it much stronger.

## 7.5 Deployment Configuration

The repository supports local and Docker-backed deployment shapes. In both cases, frontend pose extraction remains browser-side while backend inference and policy interpretation remain backend-side. This is why deployment quality depends on both frontend pose quality and backend interpretation.

The implementation was later refactored into frontend hooks, backend services, repositories, and audit paths to reduce hidden coupling and align runtime semantics with report claims. Detailed audit closure is provided in Appendix C, and rebuild/runtime evidence status is summarised in Appendix D.

**Table 7. Implementation summary**

| Component | Technology / location | Responsibility | Validation evidence |
| --- | --- | --- | --- |
| Frontend monitor | `applications/frontend` / React | live/replay controls, pose extraction, window packaging | replay/live validation |
| Backend runtime | `applications/backend` / FastAPI | active profile resolution, inference, policy interpretation | API and runtime validation |
| ML package | `ml/src/fall_detection` | model, window, fitting, and evaluation logic | offline and runtime validation |
| Persistence | backend repositories and routes | event history and dashboard-visible state | event-history checks |
| Notification | Telegram delivery manager and audit path | caregiver-facing delivery and audit recording | notification verification |
| Deployment | local and Docker-backed runtime shapes | integrated end-to-end monitor path | build and runtime validation |

# 8. Experimental Results

This chapter presents the strongest comparative, transfer, policy, and bounded runtime findings. The main report separates these layers rather than flattening them into one large synthesis table.

## 8.1 Offline Model Comparison

The strongest model-comparison evidence comes from the frozen multi-seed offline protocol. On the primary `CAUCAFall` dataset, the TCN shows higher mean `F1`, `Recall`, and `AP` than the matched custom GCN while both lines retain `FA24h = 0.0` under the frozen reporting contract. On `LE2i`, the TCN also remains directionally stronger in-domain, although both models show far higher alert-rate burden than on `CAUCAFall`.

These summaries should be read strictly as locked comparative evidence under the shared offline protocol. They are not deployment metrics.

**Table 8. Frozen in-domain five-seed comparison**

| Dataset | Model/profile | F1 | Recall | AP | FA/24h | Interpretation |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `CAUCAFall` | `TCN OP-2` | `0.8611` | `0.7600` | `0.9819` | `0.0` | strongest current primary-dataset line |
| `CAUCAFall` | `GCN OP-2` | `0.5873` | `0.4400` | `0.9706` | `0.0` | weaker event-level behaviour under the same protocol |
| `LE2i` | `TCN OP-2` | `0.8235` | `0.7778` | `0.8389` | `581.5843` | stronger than matched GCN, but high alert burden |
| `LE2i` | `GCN OP-2` | `0.7500` | `0.6667` | `0.7471` | `581.5843` | secondary in-domain comparison |

**Figure 5. Offline stability comparison across frozen candidates**

![](artifacts/figures/report/offline_stability_comparison.png){ width=90% }

The GCN’s relatively high `AP` but much lower `OP-2` `F1` and `Recall` indicate that ranking quality alone did not translate into equally strong fitted-policy event behaviour.

This is one of the clearest examples of why the report separates ranking evidence from policy-shaped behaviour. A model can still place many positive windows above many negative ones and therefore retain a strong `AP`, yet behave poorly once a particular operating point and temporal event policy are applied. In this report, the stronger TCN line is not preferred only because its ranking metric is higher, but because its fitted `OP-2` behaviour is materially more usable on the primary dataset.

The frozen offline results therefore support a bounded directional TCN advantage on `CAUCAFall`, not a universal architecture ranking.

## 8.2 Cross-Dataset Transfer

Cross-dataset evaluation provides some of the strongest limitation evidence in the project. The results show asymmetric transfer rather than broad robustness. In the `CAUCAFall -> LE2i` direction, the TCN collapses into missed-fall behaviour, while the GCN recovers event hits only by carrying a poor false-alert burden. In the reverse `LE2i -> CAUCAFall` direction, the damage is materially smaller.

**Table 9. Cross-dataset transfer summary**

| Transfer | Model | F1 | Recall | FA/24h | Interpretation |
| --- | --- | ---: | ---: | ---: | --- |
| `CAUCAFall -> LE2i` | `TCN` | `0.0` | `0.0` | `1163.1686` | missed-fall collapse |
| `CAUCAFall -> LE2i` | `GCN` | `0.7778` | `0.7778` | `1163.1686` | recall recovery with severe alert burden |
| `LE2i -> CAUCAFall` | `TCN` | `1.0` | `1.0` | `0.0` | less damaging transfer direction |
| `LE2i -> CAUCAFall` | `GCN` | `1.0` | `1.0` | `0.0` | same directional asymmetry |

**Figure 6. Cross-dataset transfer summary**

![](artifacts/figures/report/cross_dataset_transfer_summary.png){ width=90% }

The transfer results support an asymmetric boundary claim rather than a generalisation success claim. The `1.0` reverse-transfer scores should be read only within this evaluation surface, not as proof of broad cross-domain robustness.

The practical implication is that transfer direction matters as much as transfer existence. A result that reaches `1.0` on one bounded surface does not erase the collapse seen in the other, and it does not remove the need for broader camera, subject, and environmental variation. The transfer chapter therefore supports a limitation argument about domain sensitivity rather than a success claim about cross-domain deployment.

The same caution applies to surface size. These transfer results are reported on bounded target evaluation surfaces and should be interpreted with the dataset-role restrictions defined in Chapter 5 rather than as large-scale evidence of cross-domain deployment behaviour.

## 8.3 Operating-Point and Alert-Policy Behaviour

The active profile family shows that runtime behaviour is shaped by fitted operating points rather than by one model score threshold. Under the active `CAUCAFall` TCN family, `OP-2` is the preferred balanced deployable profile, while `OP-1` and `OP-3` represent looser or stricter policy variants. The system’s practical behaviour is therefore partly a policy-design result, not only a model-ranking result.

**Table 10. Operating-point interpretation summary**

| Profile | Intended behaviour | Main runtime reading |
| --- | --- | --- |
| `OP-1` | looser, recall-favouring profile | can recover more falls but may increase ADL burden |
| `OP-2` | balanced defended profile | preferred current runtime line, but replay-sensitive |
| `OP-3` | stricter profile | suppresses alerts more aggressively, but can miss falls |

These results answer the second research question by showing that validation-side operating-point fitting materially changes the meaning of model outputs by converting score streams into policy-shaped alert behaviour. The replay differences between `OP-1` and `OP-2` show that the same model family can produce sharply different runtime outcomes under different fitted profiles.

## 8.4 Runtime Replay Results

The runtime chapter uses three bounded surfaces:

- a canonical `10`-video replay summary,
- a historical locked `24`-clip custom replay surface,
- and a live/demo feasibility layer.

These surfaces answer different questions and should not be read as one flat leaderboard.

The canonical `10`-video replay summary shows strong profile sensitivity. `caucafall_tcn OP-1` reaches `accuracy=1.0`, `recall=1.0`, and `specificity=1.0` on that bounded surface, while `caucafall_tcn OP-2` falls to `accuracy=0.5`, `recall=0.0`, and `specificity=1.0`. The historical locked `24`-clip custom replay surface is also mixed rather than uniformly strong: `tcn_caucafall_locked_op2` gives `15/24`, while the looser `OP-1` recovers more falls at the cost of much worse ADL control. The matched GCN remains operationally weak on the same custom replay surface.

`OP-1`'s `1.0` result on the canonical `10`-video replay surface is not sufficient to make it the preferred deployment profile. The preferred profile is selected from the wider validation-fitted and evidence-boundary logic rather than from one small replay surface alone.

Targeted retraining modestly improves this picture. Here, `Candidate A` and `Candidate D` refer to targeted retraining candidates introduced after the locked runtime failure analysis; they are reported only as bounded strengthening evidence rather than as replacements for the main frozen comparison. Both improve the compact four-folder replay surface to `16/24`, and `Candidate A` also lifts the defended locked `15/24` line to `16/24`, both without increasing ADL false positives. These improvements remain bounded and do not justify a solved deployment claim.

The remaining misses also make the limitation more concrete. The runtime path is still sensitive to pose observability, camera geometry, and persistence requirements. In other words, the system is not failing only because one threshold is poorly chosen; it is also constrained by what the pose stream looks like once the subject is partially framed, unusually close to the camera, or represented by a short-lived fall spike that does not persist long enough to satisfy the preferred event logic.

**Table 11. Runtime evidence summary**

| Evidence surface | Main result | Interpretation | Limit |
| --- | --- | --- | --- |
| Canonical `10`-video replay | `TCN OP-1` reaches `1.0`; `TCN OP-2` shows recall collapse | runtime behaviour is highly profile-sensitive | too small to act as benchmark evidence |
| Historical locked `24`-clip replay | `TCN OP-2 = 15/24`; `GCN OP-2 = 12/24` with poor ADL control | bounded runtime behaviour differs materially by model and profile | still miss-prone on several fall clips |
| Strengthened custom replay surfaces | `Candidate A/D` improve to `16/24` | retraining can reduce false negatives on bounded runtime surfaces | gains remain modest and depend on the current confirmation-fallback policy |
| Live monitor and Telegram path | monitor, event history, and Telegram delivery work under bounded live conditions | supports end-to-end system feasibility | not field-readiness evidence |

**Figure 7. Online replay accuracy across dataset, model, and operating point**

![](artifacts/figures/report/online_replay_accuracy_heatmap.png){ width=90% }

The replay matrix therefore shows policy-sensitive runtime behaviour, not deployment readiness.

## 8.5 Uncertainty-Aware Runtime Check

The uncertainty-aware MC path is methodologically meaningful, but on the fixed online replay matrix it does not improve any of the `12` evaluated combinations at video level. This is a negative result, but it is worth reporting because it prevents the project from overstating the value of a more sophisticated runtime option.

**Figure 8. MC-dropout delta on the fixed online replay matrix**

![](artifacts/figures/report/online_mc_dropout_delta.png){ width=82% }

This suggests that uncertainty-aware logic alone cannot repair the main replay failure modes without stronger underlying score behaviour or a different runtime-policy design.

## 8.6 Result Summary

Taken together, the results support three concise conclusions.

1. The frozen offline comparison supports a cautious directional TCN advantage on the primary `CAUCAFall` protocol.
2. Cross-dataset transfer is asymmetric rather than broadly robust.
3. Runtime behaviour is real and policy-shaped, but still bounded and profile-sensitive.

# 9. System Validation and Deployment Evidence

This chapter asks whether the implemented artifact works as a system. It therefore focuses on software tests, replay and live runtime checks, persistence, delivery, and the evidence boundary between these layers.

## 9.1 Canonical Software Tests

The repository uses stratified verification rather than one undifferentiated test surface. Torch-free checks provide the main fast regression layer. Frontend checks validate the active monitor path and selected UI/API assumptions. Contract and monitor checks provide bounded torch-backed validation where the environment is stable enough to run them.

**Table 12. Examiner-facing test summary**

| Test area | What was checked | Result | Remaining risk |
| --- | --- | --- | --- |
| backend health and API contract | health route, defaults, monitor payload shape, fallback behaviour | recorded as passed in the defended verification snapshot | full torch-backed paths remain environment-sensitive |
| frontend monitor behaviour | replay controls, API handling, UI state assumptions | recorded as passed on the targeted defended regression layer | browser performance can still affect live demonstration quality |
| event persistence | realtime fall events, event-history visibility, dashboard/event semantics | implemented and checked in the defended regression snapshot | replay mode is intentionally visual-only |
| notification delivery | Telegram-first delivery path and audit semantics | implemented and verified in the defended snapshot | SMS and phone-call escalation are future work |
| replay/runtime validation | preferred preset and bounded replay checks | bounded replay remains mixed, strengthening gains are modest | replay remains system evidence, not field evidence |

## 9.2 Replay and Runtime Validation

Replay and runtime validation occupy a middle evidential layer between formal offline benchmarking and uncontrolled real-world deployment. They are valuable because they expose how frontend window production, backend policy interpretation, persistence semantics, and caregiver-facing delivery interact once the system is exercised as software.

Replay is useful because it is repeatable and bounded. Live demonstration is useful because it confirms that the integrated path can function under the defended demo preset. Neither should be mistaken for population-level deployment validation.

**Figure 9. Runtime evidence panel**

![](artifacts/figures/report/runtime_evidence_panel.png){ width=95% }

## 9.3 Event Persistence and Dashboard Verification

The system’s review layer is a substantial part of its contribution. A fall-like runtime state can become a persisted event, that event can later appear in Event History and dashboard summaries, and the same persisted state can support downstream notification. This means the monitor is not only a transient viewer. It is connected to a reviewable incident path.

## 9.4 Notification Verification

Notification verification shows that persisted incidents can produce caregiver-facing Telegram alerts. This strengthens the system story, but only at the delivery-path level. It does not raise model-comparison claims or prove that delivery quality improves detection quality.

## 9.5 Evidence Boundary and Audit Summary

The later audit and code review were important because they aligned runtime semantics, active artifacts, and report claims. The main issues corrected were cross-layer: replay-versus-realtime semantic ambiguity, notification truth-source drift, data/evaluation contract drift, and active-versus-archive evidence confusion.

**Table 13. Validation interpretation matrix**

| Validation surface | Can support | Cannot support on its own |
| --- | --- | --- |
| frozen offline test metrics | comparative model performance under the declared protocol | direct claims about integrated runtime behaviour |
| cross-dataset transfer runs | limitation boundaries and domain-shift directionality | claims of robust cross-domain deployment |
| replay matrix | bounded runtime behaviour under controlled clips and fixed policy profiles | field-readiness or population-level deployment confidence |
| live demonstration | operational existence of the end-to-end path | benchmark-quality statistical validation |
| notification checks | evidence that persisted events can reach caregiver-facing delivery | claims that delivery quality improves detection quality |

## 9.6 Ethics, Privacy, and Human Review

Pose-based representation reduces raw visual exposure compared with backend RGB video storage, but it does not remove privacy concerns. The system should therefore be read as a review-support artifact rather than as a privacy-complete safety product. Persisted incidents, dashboard review, and Telegram delivery all assume human oversight, and notifications are reported as caregiver-facing signals rather than as autonomous medical decisions.

# 10. Discussion

## 10.1 RQ1: TCN versus Custom GCN

The TCN is the strongest defended architecture under the locked `CAUCAFall` primary protocol. This is a directional rather than universal claim. The custom GCN remains methodologically useful because it shows that graph-structured skeleton reasoning did not automatically displace the temporal-convolutional baseline in the current representation and policy setting. The report therefore does not argue that graph models are unimportant in fall detection as a field. It argues that the matched custom GCN did not replace the TCN as the strongest defended line in this repository.

The comparison also shows that high ranking metrics alone are not sufficient. The preferred system line depends on how score surfaces behave after operating-point fitting and temporal policy, not only on whether one model can rank positive windows strongly in isolation.

## 10.2 RQ2: Operating-Point Fitting and Alert Policy

Validation-side operating-point fitting materially changes the meaning of model outputs. Once the project is treated as a monitoring system rather than only as a ranking exercise, success depends on how score streams are smoothed, thresholded, confirmed, and converted into reviewable incidents. The project shows that these steps can be made explicit, tracked, and audited. This is a methodological contribution because it brings the alert layer itself under experimental control.

The `OP-1` / `OP-2` replay contrast shows that policy choice can change runtime outcomes even when the underlying model family remains fixed.

## 10.3 RQ3: Runtime Feasibility and Deployment Limits

The runtime system is feasible and reviewable under bounded replay and live conditions, but it is not strong enough to support field-readiness claims. Replay results expose profile sensitivity and residual miss-prone behaviour. Live evidence shows that the end-to-end path exists and can operate coherently. The evidence therefore supports bounded runtime feasibility, not broad deployment closure.

## 10.4 Overall System Contribution and Reflection

The strongest contribution of the project is that it treats pose-based fall detection as a connected systems problem. The report does not stop at classifier performance. It links offline comparison, operating-point fitting, temporal policy, persistence, dashboard review, and Telegram delivery inside one monitored artifact. That is what allows the project to make a stronger systems claim than a model-only study while still staying within bounded evidence.

The most important project decision was therefore to stop treating the model as the whole system. Once the runtime path was examined carefully, the main risks shifted from model ranking alone to representation quality, operating-point choice, event persistence, and notification semantics. That shift also changed the evaluation strategy. The report could no longer ask only whether a window was classified correctly; it had to ask whether a score stream could become a stable, reviewable, and bounded incident path.

# 11. Limitations and Future Work

## 11.1 Dataset and Generalisation Limits

The project’s main benchmark story depends on `CAUCAFall`, while `LE2i` is used as comparative and transfer-boundary evidence. That is an appropriate project design, but it also limits generalisation claims. Cross-dataset transfer remains asymmetric, and the evidence does not support broad robustness across unseen homes, camera geometries, or deployment domains.

Future work should therefore expand evaluation across more varied datasets and camera layouts while preserving the same strict separation between benchmark and runtime evidence.

## 11.2 Pose and Runtime Dependence

The monitoring path depends directly on pose quality. Browser-side pose extraction, camera height, subject distance, partial framing, and perspective distortion all affect the windows consumed by the backend. The weaker `kitchen` subset is plausibly explained by this geometry-sensitive observability problem, and at least one miss (`kitchen_front_2`) also appears to involve a short-lived fall spike that fails under stricter event persistence.

Future work should strengthen close-range and geometry-sensitive robustness through targeted data collection, stronger skeleton modelling, and more specific hard-negative or failure-directed retraining.

## 11.3 Alert-Policy and False-Alarm Limits

The project shows that alert policy is essential, but the replay results remain profile-sensitive. Different operating points recover different balances between fall capture and ADL false alarms. Replay surfaces are informative, but they also show that runtime improvement is incremental rather than decisive.

Future work should continue targeted policy strengthening, especially around sofa-lying false alarms and persistent failure cases on the bounded replay surfaces.

## 11.4 Deployment and Human-Review Limits

The current system is better understood as a review-support prototype than as an autonomous safety device. Persisted events, dashboard review, and Telegram delivery all assume human oversight. Live evidence remains bounded and does not support clinical or field-readiness claims. Environment-sensitive torch-backed verification also remains a practical limitation on some local machines.

Future work should therefore include stronger field-style validation, more stable monitor-facing verification environments, and broader deployment evaluation that keeps human-review semantics explicit.

## 11.5 Future Work Priorities

The most direct future extensions are:

**Model robustness**
- stronger graph-model upgrades such as `CTR-GCN`
- targeted strengthening for close-range false negatives
- targeted negative mining for sofa-lying false alarms

**Evaluation robustness**
- broader dataset and camera-condition evaluation
- stronger bounded field-style validation

**System extension**
- multi-channel escalation beyond Telegram, with audit, as a later delivery extension

# 12. Conclusion

This project built a pose-based fall detection and monitoring system that links browser-side pose extraction, backend temporal inference, validation-fitted alert policy, persisted event history, dashboard review, and Telegram-based caregiver notification.

At the model-comparison layer, the strongest defended result is a cautious directional TCN advantage over the matched custom spatio-temporal GCN under the locked primary `CAUCAFall` protocol. The cross-dataset results do not support a broad robustness claim; instead they show asymmetric transfer and a sharp difference between in-domain behaviour and transfer behaviour.

At the system layer, the project demonstrates that pose-window inference, fitted alert policy, persistence, review, and caregiver delivery can be integrated into one monitoring artifact. Replay and live checks show that this path is operationally feasible under controlled conditions, while also making its limits visible.

The final claim is therefore bounded. The report supports an end-to-end, reviewable, deployment-oriented monitoring system with controlled offline comparison and bounded runtime evidence. It does not support clinical readiness, broad real-home robustness, or solved fall detection. The clearest future direction is to preserve the current system discipline while strengthening cross-domain robustness, failure-case handling, runtime validation, and stronger skeleton-model baselines such as CTR-GCN.

# References

Abu Farha, Y. and Gall, J. (2019). MS-TCN: Multi-Stage Temporal Convolutional Network for Action Segmentation. *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*.

Charfi, I., Mitéran, J., Dubois, J., Atri, M., and Tourki, R. (2013). Optimized spatio-temporal descriptors for real-time fall detection: comparison of support vector machine and Adaboost-based classification. *Journal of Electronic Imaging*, 22(4), 041106. https://doi.org/10.1117/1.JEI.22.4.041106

Chen, Y., Zhang, Z., Yuan, C., Li, B., Deng, Y., and Hu, W. (2021). Channel-Wise Topology Refinement Graph Convolution for Skeleton-Based Action Recognition. *Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)*.

Denkovski, S., Khan, S. S., Malamis, B., Moon, S. Y., Ye, B., and Mihailidis, A. (2022). Multi Visual Modality Fall Detection Dataset. *IEEE Access*, 10, 108288-108300. https://doi.org/10.1109/ACCESS.2022.3211939

Eraso Guerrero, J. C., Muñoz España, E., Muñoz Añasco, M., and Pinto Lopera, J. E. (2022). Dataset for human fall recognition in an uncontrolled environment. *Data in Brief*, 45, 108610. https://doi.org/10.1016/j.dib.2022.108610

Geifman, Y. and El-Yaniv, R. (2019). SelectiveNet: A Deep Neural Network with an Integrated Reject Option. *Proceedings of the 36th International Conference on Machine Learning (ICML)*.

Guo, C., Pleiss, G., Sun, Y., and Weinberger, K. Q. (2017). On Calibration of Modern Neural Networks. *Proceedings of the 34th International Conference on Machine Learning (ICML)*.

Lin, J., Gan, C., and Han, S. (2019). TSM: Temporal Shift Module for Efficient Video Understanding. *Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)*.

Shi, L., Zhang, Y., Cheng, J., and Lu, H. (2019). Two-Stream Adaptive Graph Convolutional Networks for Skeleton-Based Action Recognition. *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*.

Yan, S., Xiong, Y., and Lin, D. (2018). Spatial Temporal Graph Convolutional Networks for Skeleton-Based Action Recognition. *Proceedings of the AAAI Conference on Artificial Intelligence*.

# Appendices

## Appendix A. Frozen Profiles, Active Artifacts, and Evidence Lineage

This appendix records the defended artifact core used by the main report. It preserves enough traceability for review and reproduction without forcing the main text to act as an artifact register.

**Table A1. Primary frozen candidates and active profiles**

| Role | Dataset | Line / profile | Artifact family | Report use |
| --- | --- | --- | --- | --- |
| primary offline candidate | `CAUCAFall` | `TCN` | `outputs/caucafall_tcn_W48S12/*` | main comparative result line |
| matched offline baseline | `CAUCAFall` | `GCN` | `outputs/caucafall_gcn_W48S12/*` | matched comparison line |
| preferred replay/runtime preset | `CAUCAFall` | `TCN + OP-2` | `ops/configs/ops/tcn_caucafall*.yaml` | preferred live and replay profile |
| comparative deployable profile | `LE2i` | `GCN` | `ops/configs/ops/gcn_le2i_opt33_r2.yaml` and linked metrics | comparative runtime/profile evidence |
| exploratory/supporting family | `MUVIM` | mixed | supporting and archived config families | methodological breadth only |

**Table A2. Evidence-lineage summary**

| Claim layer | Primary source | Typical artifact family | Why it matters |
| --- | --- | --- | --- |
| model comparison | frozen metrics summaries and seed-comparison artifacts | `outputs/metrics/*stb_*.json`, `outputs/metrics/*locked.json` | supports comparative claims without runtime conflation |
| policy interpretation | fitted operating-point files and policy sweeps | `ops/configs/ops/*.yaml`, saved sweep JSON files | links threshold and policy prose to tracked deployable profiles |
| runtime interpretation | replay matrices, runtime figures, persisted event path | replay summaries, clip summaries, monitor/event route behaviour | supports bounded deployment and system claims |
| delivery evidence | Telegram delivery audit and event-linked notification path | delivery audit store, runtime screenshots, notification tests | shows that persisted incidents can reach caregiver-facing output |
| repository defensibility | audit, freeze, and review-control documents | freeze manifests, code-review summaries, audit reports | explains why the submission state is trustworthy |

**Table A3. Active profile summary**

| Profile family | Dataset | Intended role | Reading note |
| --- | --- | --- | --- |
| `tcn_caucafall_*` | `CAUCAFall` | primary offline and preferred runtime line | strongest defended alignment across offline, policy, and replay evidence |
| `gcn_caucafall_*` | `CAUCAFall` | matched baseline and policy contrast | useful for fair architecture comparison, not the preferred deployable preset |
| `gcn_le2i_opt33_r2` | `LE2i` | comparative deployable profile | useful for boundary analysis, not a replacement for the primary runtime line |
| `tcn_le2i_*` | `LE2i` | comparative model evidence | useful for in-domain comparison and limitations, but not the main demo path |
| archived `muvim` families | `MUVIM` | supporting breadth and exploratory work | methodologically useful, but outside the defended core contribution |

**Table A4. Strengthening-candidate context**

| Candidate | Role | Surface | Report interpretation | Policy caveat |
| --- | --- | --- | --- | --- |
| `Candidate A` | recall-oriented retraining candidate on the primary `CAUCAFall + TCN` line | compact four-folder replay and locked `24`-clip replay | bounded strengthening evidence, not a new default deployment line | fitted under a confirmation-disabled fallback after a degenerate confirmation sweep; reported as replay-strengthening evidence rather than a promoted replacement for the preferred runtime preset |
| `Candidate D` | extended hard-negative continuation candidate on the same primary line | compact four-folder replay and locked `24`-clip replay | bounded strengthening evidence, simpler to compare against `Candidate A` than to promote independently | fitted under the same confirmation-disabled fallback condition, with no claim that it should replace the preferred defended deployment profile |

## Appendix B. Supporting Figures and Design-Evolution Material

The main report retains only the figures needed for the central argument. Supporting quantitative figures, design-evolution figures, and historical visuals remain here so that the visual record is preserved without overloading the main text.

**Table B1. Supporting-figure interpretation guide**

| Supporting figure family | Best use | Not for |
| --- | --- | --- |
| absolute cross-dataset `F1` bars | quick visual support for transfer asymmetry | replacing the main delta-based transfer interpretation |
| seed-stability error bars | compact reinforcement of multi-seed behaviour | standalone proof of operational usefulness |
| bounded replay confusion matrix | quick diagnostic view of the preferred runtime preset | replacement of frozen offline benchmark evidence |
| diagnostic pre-fix visuals | explaining historical mismatch or debugging context | defended final performance claims |

**Figure B1. Iteration timeline from the design-proposal phase**

![](artifacts/figures/report/appendix/iteration_timeline.png){ width=90% }

**Figure B2. Architecture evolution from proposal-stage tiers to final contract-oriented layers**

![](artifacts/figures/report/architecture_evolution_comparison.svg){ width=85% }

**Table B2. Design-evolution figure roles**

| Figure family | Stage represented | Contribution |
| --- | --- | --- |
| low-fidelity wireframes | early workflow planning | shows that dashboard, monitor, event-history, and settings roles were designed intentionally |
| supporting wireframe exports | early workflow planning | preserves additional proposal-stage interface evidence |
| high-fidelity previews | later interaction refinement | shows how observability and review semantics were made more explicit |
| legacy architecture visuals | proposal-stage system thinking | provides contrast for later contract-oriented refactoring |
| early operating-point concept sketch | conceptual policy framing | shows that safety-versus-alert-burden trade-off was recognised before final fitting artifacts existed |

## Appendix C. Audit and Verification Summary

The report’s late-stage review work is preserved here rather than in the main narrative. The key purpose of this appendix is to show what kinds of mismatch were corrected and how the defended snapshot was stabilised.

**Table C1. High-level review closure summary**

| Review area | Representative issue | Closure outcome |
| --- | --- | --- |
| ML pipeline | dataset-contract and evaluation-contract mismatch | corrected and regression-checked |
| backend runtime | persistence semantics, notification truth source, active profile normalization | corrected and test-covered |
| frontend | replay/live state semantics, fallback contract drift, monitor control meaning | corrected and spot-tested |
| scripts/tests | canonical test coverage gaps, build invocation friction, freeze verification | corrected with updated script entrypoints |
| repository/evidence layer | active vs archive confusion, report/evidence drift | corrected through freeze and inventory work |

**Table C2. Review-phase summary**

| Review phase | Main question | Outcome |
| --- | --- | --- |
| evidence audit | do claims, figures, and active artifacts still line up? | report/evidence drift reduced through freeze and inventory work |
| ML code review | do data and evaluation contracts still support the claimed protocol? | FPS, window-metadata, and recursive-discovery mismatches corrected |
| backend code review | do runtime state, persistence, and delivery share one interpretation? | event and notification truth sources aligned |
| frontend code review | do UI controls and displayed state map cleanly to backend contracts? | replay/live semantics and fallback meanings clarified |
| scripts/tests review | do build and test entrypoints still reflect the defended repository state? | canonical test entrypoints and report build path cleaned up |

## Appendix D. Reproducibility and Runtime-Evidence Status

This appendix preserves the defended rebuild and verification context without forcing the main report to carry command-level detail.

**Table D1. Reproducibility prerequisites**

| Layer | Minimum requirement | Reason |
| --- | --- | --- |
| Python/runtime | active project virtual environment with repository `PYTHONPATH` | required for scripts, tests, and pipeline commands |
| Node/frontend | working Node/npm environment inside `applications/frontend/` | required for frontend regression checks |
| document build | `pandoc` and `xelatex` available to `build_report.sh` | required for PDF export |
| torch-backed checks | stable local environment where `import torch` succeeds | required for `contract` and `monitor` validation modes |
| repository state | active figures and active frozen artifacts present | required so the report rebuild matches the defended snapshot |

**Table D2. Verification-outcome summary**

| Verification layer | Recorded defended outcome | Interpretation |
| --- | --- | --- |
| report build | passes | source markdown can be turned into reviewable PDF output |
| torch-free canonical tests | passes | core repository and contract logic are regression-covered |
| frontend regression layer | passes on the active targeted test set | selected UI/API assumptions are checked |
| torch-backed contract layer | conditional on machine stability | environment caveat remains explicit rather than hidden |
| `CAUCAFall` data/eval regression | rerun and checked | primary data/evaluation path was revalidated after code review |
| `LE2i` data/eval regression | rerun and checked | comparative data/evaluation path was revalidated after code review |

**Table D3. Runtime-evidence status**

| Item | Current state | Notes |
| --- | --- | --- |
| live monitor + Telegram screenshot | inserted from defended realtime evidence | same-incident capture retained |
| persisted event-history screenshot | inserted from the same defended incident chain | same-incident capture retained |
| supplementary fall video | compressed and linked as `Supplementary Video S1` | submission copy present |
| supplementary ADL video | compressed and linked as `Supplementary Video S2` | submission copy present |

**Table D4. Command-family map**

| Command family | Main output or state change | Report relevance |
| --- | --- | --- |
| report build | current PDF or DOCX under `artifacts/report_build/` | proves that the report is buildable as a defended artifact |
| canonical tests | regression pass/fail state across torch-free, frontend, and conditional torch-backed layers | supports software-validation claims |
| freeze verification | defended-core path existence and cleanliness check | supports repository defensibility claims |
| data/eval regression commands | regenerated labels, windows, and locked metric outputs for primary datasets | supports post-review protocol integrity |
| figure regeneration commands | refreshed report figures under the unified figure directory | supports figure traceability and rebuildability |
