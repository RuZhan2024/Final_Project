# Project Deep Audit Tasks - 2026-05-06

## Goal

Run a full repository audit before any cleanup. The audit should identify code
logic drift, unnecessary coupling, duplicate or obsolete runtime paths, old
experiment outputs, stale docs, unused scripts, and files that should be kept,
archived, ignored, or deleted later.

This is an evidence-first audit. Do not delete, rewrite, or move files during
the audit unless a separate cleanup step is explicitly approved.

## Current Project Baseline

- Active branch: `feat/ctr-gcn-upgrade`
- Locked runtime target: TCN CaucaFall/LE2i controlled mix25 balanced, OP-2
- Current frontend/backend replay stability fix is part of the active worktree.
- The worktree contains many generated CTR-GCN/TCN artifacts from the recent
  sweep and report-prep work. Treat them as potentially useful until classified.

## Audit Rules

1. Preserve user and experiment work. Do not revert existing dirty files.
2. Separate findings into: must-fix, should-clean, archive, ignore, and keep.
3. Mark generated outputs separately from source code.
4. Prefer objective evidence: imports, call sites, git tracking, file size,
   config references, test coverage, and runtime route usage.
5. Avoid cleanup that changes locked offline or online behavior without a test.

## Scope

### Source Code

- Backend API and monitor runtime under `applications/backend`.
- Frontend monitor/settings UI under `applications/frontend/src`.
- ML package under `ml/src/fall_detection`.
- Ops scripts/configs under `ops`.
- QA tests under `qa`.
- Top-level Makefile, Python packaging, requirements, Docker/render configs.

### Artifacts and Data

- `artifacts/`
- `outputs/`
- `data/`
- generated checkpoints and deploy assets
- generated report figures/logs/metrics

### Documentation

- `README.md`
- `docs/reports/notes`
- report drafts
- old candidate notes and CTR-GCN planning notes

## Audit Checklist

### Phase 1 - Inventory

- [ ] Capture git dirty state and classify tracked vs untracked changes.
- [ ] Count files by top-level directory and extension.
- [ ] Find largest files and largest directories.
- [ ] Identify generated outputs currently not covered by `.gitignore`.
- [ ] Identify duplicated config/result families.

### Phase 2 - Runtime Paths

- [ ] Trace monitor replay path from frontend to backend.
- [ ] Trace live camera path separately from replay path.
- [ ] Trace active settings/runtime config selection.
- [ ] Trace deployed model/checkpoint manifest usage.
- [ ] Identify stale modes: `gcn`, `hybrid`, old operating points, dataset select.

### Phase 3 - Code Cleanliness

- [ ] Search for TODO/FIXME/HACK/deprecated/legacy/lite/old markers.
- [ ] Search for duplicate implementations across scripts and package modules.
- [ ] Search for dead exports and unreferenced helpers where static evidence is
      reliable.
- [ ] Identify overly large files/modules that should be split or simplified.
- [ ] Identify coupling hotspots where UI, settings, model runtime, and
      experiment policy are mixed together.

### Phase 4 - Experiment Output Hygiene

- [ ] Classify current locked results.
- [ ] Classify exploratory CTR-GCN/TCN sweep outputs.
- [ ] Identify obsolete wrongbone/lite/preliminary artifacts.
- [ ] Recommend archive/delete/ignore policy for generated outputs.

### Phase 5 - Docs Hygiene

- [ ] Identify current canonical docs.
- [ ] Identify obsolete candidate notes and historical planning docs.
- [ ] Recommend a report-facing experiment evidence index.
- [ ] Recommend an archive structure for old notes.

### Phase 6 - Tests and Safety Nets

- [ ] Map source areas to test coverage.
- [ ] Identify high-risk cleanup areas that need tests first.
- [ ] Verify existing tests after any cleanup proposal.

## Finding Format

Each finding should include:

- Category: must-fix, should-clean, archive, ignore, keep
- Area: backend, frontend, ML, ops, artifacts, docs, tests
- Evidence: file paths, line numbers, command output, or reference count
- Risk: why it matters
- Recommendation: exact action, preferably reversible

## Initial Hypotheses to Validate

- Generated CTR-GCN/TCN sweep YAML and artifact directories are useful for
  report evidence but too noisy to remain in the repo root long-term.
- Several monitor runtime branches still support `gcn` and `hybrid`, while the
  product/runtime direction is now TCN and CTR-GCN only.
- Frontend settings may still expose options that are no longer part of the
  locked deployment story.
- Some old candidate notes and replay eval outputs are historical evidence but
  should move to an archive index.
- `.gitignore` likely needs updates for current generated output families.

## Output Deliverables

- This task document.
- Initial deep audit report:
  `docs/reports/notes/PROJECT_DEEP_AUDIT_INITIAL_REPORT_2026-05-06.md`
- A cleanup proposal split into safe batches.
- Optional follow-up patches only after findings are reviewed.

## Progress Log

- 2026-05-06: Created task document.
- 2026-05-06: Completed initial inventory and code-signal scan.
- 2026-05-06: Wrote initial audit report with 12 findings and 5 proposed
  cleanup batches.
