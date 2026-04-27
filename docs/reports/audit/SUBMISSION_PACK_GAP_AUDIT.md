# Submission Pack Gap Audit

Date: 2026-03-16

Scope:
Audit the project against the stated submission requirements for:
- working software artefact
- code submission snapshot
- user guide
- demo recording

Assessment basis:
- repository contents only
- no new runtime validation performed in this audit

## Overall Verdict

The project appears to satisfy the core technical requirement of having a working software artefact, but the submission pack is not fully closed.

The strongest parts are:
- runnable software system exists
- core end-to-end feature is documented
- README and user guide are present
- demo runbooks are present

The main missing or weak parts are:
- no final tagged markable snapshot yet
- no demo recording artifact yet
- claim/evidence documents required by your own checklist are still placeholders
- no recorded clean-machine signoff yet

## Findings

### 1. No markable release tag exists
Severity: `High`

Evidence:
- `git tag` returned no tags
- `docs/project_targets/SUBMISSION_PACK_INDEX.md`
- `docs/project_targets/FINAL_SUBMISSION_CHECKLIST.md`

Why it matters:
- The brief explicitly asks for a markable snapshot.
- Your own docs still say `git tag <FINAL_TAG>` is to be filled later.

Required action:
- Create one final release tag and record it in the submission pack.

### 2. Demo recording is still missing
Severity: `High`

Evidence:
- `docs/project_targets/SUBMISSION_PACK_INDEX.md` lists demo video as recommended location only
- `docs/project_targets/FINAL_SUBMISSION_CHECKLIST.md` still has demo video unchecked
- `docs/project_targets/DELIVERY_ALIGNMENT_STATUS.md` marks demo recording artifact as `TODO`
- `artifacts/demo/` does not exist

Why it matters:
- The brief requires proof that the software works end-to-end.
- This is one of the most visible assessor-facing artefacts.

Required action:
- Record the 5-minute demo and add either the file or a submission link.

### 3. Claim table is not regenerated
Severity: `High`

Evidence:
- `docs/project_targets/CLAIM_TABLE.md`

Current state:
- file says legacy claims were removed and new evidence must be regenerated

Why it matters:
- Your own readiness and submission docs expect explicit evidence-backed claims.
- Without this, the project is weaker on “clear connection between your design and what you built”.

Required action:
- Regenerate the claim table from the current 33-joint profiles and current metrics artifacts.

### 4. Thesis evidence map is still a placeholder
Severity: `High`

Evidence:
- `docs/project_targets/THESIS_EVIDENCE_MAP.md`

Current state:
- file says legacy evidence entries were removed and must be regenerated

Why it matters:
- Your own checklist requires that the evidence map has no stale or unmapped claim rows.
- This is the main bridge from reported claims to actual artefacts.

Required action:
- Rebuild the evidence map with current artefact paths and reproduce commands.

### 5. Clean-machine dry run is documented as pending, not closed
Severity: `Medium`

Evidence:
- `docs/project_targets/FINAL_SUBMISSION_CHECKLIST.md`
- `docs/project_targets/DELIVERY_ALIGNMENT_STATUS.md`

Why it matters:
- The brief says the software must be understandable and runnable by someone else.
- A pending dry run weakens confidence that the marker can follow the guide successfully.

Required action:
- Perform one clean-machine or clean-user dry run and record the result.

### 6. User guide exists, but sample-input guidance is still light
Severity: `Medium`

Evidence:
- `docs/reports/runbooks/USER_GUIDE.md`
- `docs/project_targets/FINAL_DEMO_WALKTHROUGH.md`

Why it matters:
- The brief asks for test credentials or sample inputs if relevant.
- The current guide explains replay mode, but it does not clearly list exactly which sample clip(s) the marker should use or where they are.

Required action:
- Add one short section naming the exact recommended demo input files or replay cases.

### 7. User guide is acceptable, but installation instructions are split across multiple docs
Severity: `Medium`

Evidence:
- `README.md`
- `docs/reports/runbooks/USER_GUIDE.md`
- `docs/reports/runbooks/DEMO_RUNBOOK.md`

Why it matters:
- The information is present, but the marker may need to jump across several files.
- For assessment, a single authoritative path is safer.

Required action:
- Make `README.md` the single first-stop path and point it to one exact demo route.

## What Already Meets the Brief

### Working software artefact
Status: `Meets`

Evidence:
- `README.md`
- `docs/reports/runbooks/USER_GUIDE.md`
- `docs/reports/runbooks/DEMO_RUNBOOK.md`
- `docs/project_targets/DEPLOYMENT_LOCK.md`

Why:
- The system is more than a description. It includes backend, frontend, runtime profiles, and documented demo flow.
- At least one meaningful end-to-end feature is clearly intended: replay-based fall detection through the monitor UI.

### Readable code and repo overview
Status: `Meets`

Evidence:
- `README.md`
- `src/`, `server/`, `apps/`, `tests/`

Why:
- The repository is structured and documented at a level suitable for marking.

### User guide presence
Status: `Meets, with improvement needed`

Evidence:
- `docs/reports/runbooks/USER_GUIDE.md`
- `.env.example`

Why:
- The guide includes software purpose, core features, setup, run steps, environment config, and limitations.
- It would be stronger with exact sample inputs and one single recommended examiner path.

## Priority Fix Order

1. Create final git tag.
2. Record demo video and link it in submission docs.
3. Regenerate `CLAIM_TABLE.md`.
4. Regenerate `THESIS_EVIDENCE_MAP.md`.
5. Record one clean-machine dry run outcome.
6. Add exact sample replay inputs to the user guide.

## Bottom Line

You are not missing the software artefact itself.

You are mainly missing submission closure items around packaging, evidence traceability, and assessor-facing proof:
- release tag
- demo recording
- current claim table
- current evidence map
- dry-run signoff
