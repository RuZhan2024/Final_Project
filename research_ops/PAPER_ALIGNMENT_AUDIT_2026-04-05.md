# Paper Alignment Audit

Date: 2026-04-05

Scope:
- Draft audited: `docs/reports/drafts/PHD_FINAL_PROJECT_REPORT_DRAFT_2026-03-29.md`
- Claim ledger: `research_ops/CLAIMS.yaml`
- Evidence ledger: `research_ops/EVIDENCE_INDEX.yaml`

Purpose:
- identify where the current draft is already aligned
- flag wording that is stronger, older, or less precise than the current ledgers allow

## Overall Status

- Claim alignment: mostly aligned
- Evidence alignment: partially aligned
- Immediate risk level: moderate

Main issue:
- the deployment/runtime section had cited an older replay matrix and older MC-dropout interpretation
- the calibration contribution had been present in the draft but missing from the original claim ledger; this is now fixed as `C5` with evidence `E9`

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

## Required Draft Updates

1. Update the replay-matrix artifact references in Section 9.5.
   - Current draft references:
     - `artifacts/reports/mc_replay_matrix_20260401.csv`
     - wording: "MC-on did not improve any bounded replay result and degraded two combinations"
   - Current ledger references:
     - `artifacts/reports/online_mc_replay_matrix_20260402.csv`
     - `artifacts/reports/online_mc_replay_matrix_20260402.json`
   - Required wording change:
     - state that on the current fixed raw online replay path, MC on/off made no difference across the 12 combinations
   - Why:
     - the current evidence ledger is anchored on `E6`, not the older pre-fix matrix

2. Narrow the "especially for the locked TCN OP-2 runtime path" wording in Section 10.3.
   - Current issue:
     - this risks conflating the current review preset (`LE2i + TCN + OP-2`) with the strongest bounded replay row (`caucafall_tcn OP-2`)
   - Safer wording:
     - "Replay-oriented evidence supports bounded practical system use under controlled runtime conditions, with the strongest bounded replay performance observed for `caucafall_tcn OP-2`."

3. Keep calibration wording bounded to the current evidence level.
   - The draft is now stronger because it includes a compact current-profile OP trade-off table.
   - Even so, keep the claim bounded to the current fitted profiles rather than implying universal OP behaviour.

## Suggested Edits by Location

### Section 9.5 Deployment and Runtime Results

Replace:
- references to `mc_replay_matrix_20260401.csv`
- references to "degraded two combinations"

With:
- references to `online_mc_replay_matrix_20260402.csv` and `.json`
- wording that the current fixed raw online replay matrix shows no MC benefit across the 12 combinations

### Section 10.3 Answer to RQ3

Replace:
- "especially for the locked TCN OP-2 runtime path"

With:
- "in bounded controlled replay conditions, with the strongest bounded replay row observed for `caucafall_tcn OP-2`"

### Section 13 Conclusion

Keep:
- bounded deployment wording
- directional TCN-vs-GCN wording
- incomplete field/generalization wording

Update:
- the MC-dropout sentence so it reflects the current `2026-04-02` online replay evidence rather than the older `2026-04-01` matrix

## Recommended Next Action

1. Refresh Section 9.5, Section 10.3, and the relevant sentence in Section 13.
2. Completed on 2026-04-05: add one compact OP-tradeoff table so `C5` can move from `partially_supported` to `supported`.
