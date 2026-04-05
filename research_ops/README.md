# Research Ops

This directory is branch-local research control infrastructure.

Intent:
- keep research execution prompts, gates, and claim ledgers off `main`
- let this branch carry paper-planning state without polluting the production code branch
- make later code-only cherry-picks into `main` straightforward

Core files:
- `MEGA_PROMPT_v2.md`
- `PROJECT_CONFIG.yaml`
- `STAGE_PROTOCOL.yaml`
- `QUALITY_GATES.yaml`
- `CLAIMS.yaml`
- `EVIDENCE_INDEX.yaml`
- `PAPER_ALIGNMENT_AUDIT_2026-04-05.md`
- `PROGRESS.md`
- `plans/PLAN_TEMPLATE.md`

Usage model:
- keep these files on the research branch
- fill `PROJECT_CONFIG.yaml` and `CLAIMS.yaml` with project-specific content
- treat `EVIDENCE_INDEX.yaml` as the source of truth for evidence IDs and artifact traces
- update `PROGRESS.md` and `plans/` as stages advance
- do not merge this directory into `main` unless explicitly intended
