# Research Ops

This directory is the branch-local research control layer for the thesis/report work. It is intentionally separated from the runtime product surface and should stay off `main` unless you explicitly want the research-control layer in the production branch.

## Live control files

- `MEGA_PROMPT_v2.md`
- `PROJECT_CONFIG.yaml`
- `STAGE_PROTOCOL.yaml`
- `QUALITY_GATES.yaml`
- `CLAIMS.yaml`
- `EVIDENCE_INDEX.yaml`
- `PAPER_ALIGNMENT_AUDIT_2026-04-05.md`
- `PROGRESS.md`
- `plans/PLAN_TEMPLATE.md`

## Roles

- `PROJECT_CONFIG.yaml`: project constants, paths, writing constraints, and export assumptions
- `CLAIMS.yaml`: paper-safe claim ledger
- `EVIDENCE_INDEX.yaml`: evidence-side ledger tied to concrete repo artifacts
- `PAPER_ALIGNMENT_AUDIT_2026-04-05.md`: current draft-versus-ledger status
- `PROGRESS.md`: live research state log
- `MEGA_PROMPT_v2.md`, `STAGE_PROTOCOL.yaml`, `QUALITY_GATES.yaml`: execution contract

## Rules

- Treat `CLAIMS.yaml` and `EVIDENCE_INDEX.yaml` as the source of truth for paper-safe wording.
- If an evidence artifact changes, update both ledgers in the same pass.
- Keep this directory on research branches and cherry-pick only code changes into `main`.
- Do not keep stale audit or cleanup notes here; replace them or remove them.
