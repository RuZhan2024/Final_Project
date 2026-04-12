# Paper Alignment Audit

Date: 2026-04-05

Scope:
- Draft audited: `docs/reports/drafts/PAPER_FINAL_2026-04-11.md`
- Claim ledger: `research_ops/CLAIMS.yaml`
- Evidence ledger: `research_ops/EVIDENCE_INDEX.yaml`

Purpose:
- identify where the current draft is already aligned
- flag wording that is stronger, older, or less precise than the current ledgers allow

## Overall Status

- Claim alignment: aligned
- Evidence alignment: aligned for the current thesis-safe evidence surface
- Immediate risk level: low_to_moderate

Main current position:
- the deployment/runtime section now points to the current `2026-04-02` online replay evidence
- the calibration contribution is explicitly registered as `C5` with evidence `E9`
- the remaining evidence weakness is `C4`, which correctly remains only partially supported

## Aligned Areas

1. `C1` end-to-end system claim is aligned.
   - Draft sections:
     - Abstract
     - 1.5 Contributions
     - 9.5 Deployment and Runtime Results
     - 13. Conclusion
   - Current wording is appropriately bounded to deployment-oriented system evidence.

2. `C2` TCN-vs-GCN comparative claim is aligned.
   - Draft sections:
     - Abstract
     - 9.1 Offline Comparative Results
     - 10.1 Answer to RQ1
     - 13. Conclusion
   - The draft correctly uses directional wording and preserves statistical caution.

3. `C3` cross-dataset limitation claim is aligned.
   - Draft sections:
     - Abstract
     - 9.2 Cross-Dataset Results
     - 11. Limitations
     - 13. Conclusion
   - The draft correctly avoids universal robustness wording.

4. `C5` calibration/operating-point design claim is now aligned at the ledger level and supported by a compact quantitative table in the draft.
   - Draft sections:
     - 7. Calibration and Alert Policy
     - 9.4 Calibration and Alert-Policy Results
     - 10.2 Answer to RQ2
   - This claim previously existed only in prose; it is now explicitly registered in `CLAIMS.yaml`.

## Closed Alignment Issues

1. Replay-matrix references in Section 9.5 were refreshed.
   - Current draft references:
     - `artifacts/reports/online_mc_replay_matrix_20260402.csv`
     - `artifacts/reports/online_mc_replay_matrix_20260402.json`
   - Current wording:
     - on the fixed raw online replay path, MC on/off produced no video-level change across the 12 combinations
   - Status:
     - closed

2. Section 10.3 wording was narrowed.
   - Current wording now separates:
     - the current review preset
     - the strongest bounded replay row
   - Status:
     - closed

3. Calibration wording was upgraded and bounded.
   - The draft includes a compact OP trade-off table.
   - The wording remains tied to current fitted profiles rather than universal OP behavior.
   - Status:
     - closed

## Remaining Live Risk

1. `C4` remains only partially supported.
   - Reason:
     - replay and sample field-validation evidence are still bounded
   - Required stance:
     - keep deployment wording cautious
     - do not present current field evidence as broad real-world closure

## Current Verdict

- The draft is currently aligned with the live claim and evidence ledgers.
- No immediate wording rollback is required.
- The next draft pass should focus on presentation quality, bibliography handling, and final packaging rather than claim repair.
