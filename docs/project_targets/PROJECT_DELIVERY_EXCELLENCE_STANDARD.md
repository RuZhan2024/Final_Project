# Project Delivery Excellence Standard

## Purpose
This document defines the final delivery bar above the minimum submission requirement.
It is designed for:
- Research-grade evidence quality (defensible, reproducible, auditable)
- Production-grade software quality (stable, observable, maintainable, demo-safe)

---

## 1) Deliverable Package (Higher-than-Baseline)

### 1.1 Source Code Package
Required:
- Public/private Git repository with a clearly marked release snapshot (`tag` or `release`).
- Clean repository state for the release commit (no unexplained local-only edits).
- Structured folders for:
  - training/evaluation code
  - backend API
  - frontend UI
  - docs/runbooks
  - artifacts/reports

Quality gates:
- `README.md` provides one-command workflow for:
  - ML pipeline reproduction
  - backend+frontend demo startup
- No hardcoded machine-specific absolute paths in runnable configs.
- All core scripts pass compile/import smoke checks.

### 1.2 Operator/User Guide
Required:
- Quickstart (5-minute path)
- Full setup (fresh environment path)
- Runtime modes:
  - replay mode (mandatory demo-safe mode)
  - live mode (performance-dependent mode)
- Exact expected outputs for health checks and inference checks.

Quality gates:
- A marker can run without private tribal knowledge.
- Failure cases are documented with recovery steps.

### 1.3 Demonstration Evidence
Required:
- A short demo video showing:
  - end-to-end input → inference → decision → event output
  - at least one positive and one negative case
  - model/mode selection and policy switching (OP-1/2/3 or equivalent)

Quality gates:
- Demo behavior is consistent with the documented runbook and reported metrics.

---

## 2) Research Evidence Standard

## 2.1 Claim Discipline
Every claim must be:
- measurable
- falsifiable
- linked to exact artifact(s)
- linked to a reproduce command

Minimum claim table fields:
- claim_id
- metric + threshold
- protocol/split
- artifact path
- reproduce command
- pass/fail rule

## 2.2 Reproducibility Contract
For every reported result:
- fixed experiment ID
- config hash or explicit config block
- seed
- commit hash
- generated artifacts (ckpt/ops/metrics/log)

No untracked numbers in report/dissertation text.

## 2.3 Evaluation Integrity
Must explicitly document:
- what is tuned on validation vs what is evaluated on test
- policy fitting data boundaries (no test leakage)
- metric definitions (AP/F1/Recall/Precision/FA24h/delay/latency)
- dataset caveats and limitations

## 2.4 Stability & Significance
Expected:
- multi-seed summary for final candidates
- confidence intervals or variance reporting
- pre-registered key comparisons (not unlimited post-hoc comparisons)

## 2.5 Negative Results and Limitations
Mandatory:
- what did not work
- why it likely failed
- what is deferred
- risk to external validity

This is scored positively when honest and evidence-based.

---

## 3) Engineering/Deployment Standard

## 3.1 End-to-End Runtime Reliability
System must demonstrate:
- startup health checks
- predictable error handling
- no silent critical failures
- graceful degradation under weak inputs (occlusion, low-FPS, missing keypoints)

## 3.2 API/Frontend Contract Consistency
Must have:
- explicit route contract (request/response schema)
- synchronized frontend usage
- no stale/phantom endpoint calls

## 3.3 Observability
Required runtime visibility:
- decision outputs (`p_fall`, policy state, final action)
- latency timing (at least p50/p95 in reports)
- key failure counters (dropped windows, invalid payloads, cooldown suppressions)

## 3.4 Demo Safety Profile
Default demo profile should prioritize:
- stable behavior over aggressive recall
- replay-mode determinism for examiner demonstrations
- one-event dedup/cooldown behavior (avoid alert spam)

## 3.5 Portability
Target:
- can run on a second machine with documented setup only
- no dependence on hidden local files
- environment variables documented in `.env.example`/runbook

---

## 4) Acceptance Rubric (Internal Release Gate)

Release is accepted only if all are true:
- `R1` Reproducibility: key metrics can be regenerated from documented commands.
- `R2` Integrity: no known leakage/protocol violations in reported numbers.
- `R3` Operability: backend/frontend and core inference flow run from runbook.
- `R4` Explainability: system decisions and policy behavior can be explained from logs/outputs.
- `R5` Limitation disclosure: known weaknesses are explicitly documented.

Recommended evidence files:
- `docs/project_targets/THESIS_EVIDENCE_MAP.md`
- `docs/project_targets/DEPLOYMENT_LOCK.md`
- `docs/project_targets/FINAL_SUBMISSION_CHECKLIST.md`
- `artifacts/reports/*` (metrics + validation outputs)

---

## 5) Final Submission Checklist (Excellence Bar)

- [ ] Release tag created for markable snapshot.
- [ ] README quickstart verified on clean environment.
- [ ] User guide/runbook verified by a second-person dry run.
- [ ] Demo video matches current release behavior.
- [ ] Claim table fully linked to artifacts + commands.
- [ ] Metrics used in report exist on disk and are reproducible.
- [ ] Known limitations and future work explicitly stated.
- [ ] No critical TODOs that contradict report claims.

---

## 6) Practical Target for This Project

For this project, “excellent delivery” means:
- A reproducible fall-detection pipeline with locked configs and traceable metrics.
- A working end-to-end monitor workflow (replay mandatory, live optional with constraints documented).
- Policy-driven deployment behavior that is explainable and testable.
- Documentation that allows an examiner to run and validate the system without developer intervention.

---

## 7) Post-Submission Architecture Optimization Backlog

This backlog is intentionally incremental (no broad refactor), focused on stability and maintainability.

### P0 — Configuration and Entry Convergence
- Unify training/eval/deploy parameter sources to reduce drift between Makefile, scripts, and YAML.
- Goal: one authoritative definition per experiment/runtime profile.

### P0 — Legacy Compatibility Cleanup
- Remove obsolete backward-compat branches and old path toggles that are no longer required.
- Goal: reduce hidden runtime branches and inconsistent behavior.

### P1 — Module Boundary Hardening
- Enforce clear one-way boundaries: `data -> features -> model -> policy -> api`.
- Goal: make frontend/backend/runtime issues isolate to one layer quickly.

### P1 — Runtime Contract Tests
- Add minimal end-to-end contract tests for payload shape, FPS/resampling, and OP policy application.
- Goal: prevent regressions where offline metrics and online behavior diverge.

### P2 — Documentation-to-Code Mapping
- Maintain a concise module contract page per critical component (inputs/outputs/constraints).
- Goal: improve maintainability, onboarding speed, and defense readiness.
