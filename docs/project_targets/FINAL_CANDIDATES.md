# Final Candidates

Date: 2026-03-22

Protocol Freeze:
- `Paper Protocol Freeze v1`
- This file is the source of truth for the frozen candidate roots referenced by the current paper-facing docs.

Scope:
- Current 33-joint profiles and artifacts only
- Primary dataset: `CAUCAFall`
- Comparative/generalization dataset: `LE2i`
- Primary architecture comparison scope: `TCN` vs `GCN` under the same frozen protocol

Frozen Seed Set:
- `1337`
- `17`
- `2025`
- `33724876`
- `42`

Rule:
- These candidate roots define the paper-facing model family comparison.
- Do not substitute newer checkpoints, alternate sweeps, or ad hoc single-run winners in the paper unless this file and the linked evidence docs are all updated together.

## Frozen Candidate Roots

| Dataset Role | Dataset | Architecture | Frozen Root | Current Role in Paper |
|---|---|---|---|---|
| Primary benchmark + deployment target | `caucafall` | `TCN` | `outputs/caucafall_tcn_W48S12_r2_train_hneg` | Main positive result and deployment-facing benchmark candidate |
| Primary benchmark + deployment target | `caucafall` | `GCN` | `outputs/caucafall_gcn_W48S12_r2_recallpush_b` | Matched comparison baseline for the main architecture comparison |
| Comparative/generalization | `le2i` | `TCN` | `outputs/le2i_tcn_W48S12_opt33_r2` | Comparative in-domain candidate, not the primary deployment claim |
| Comparative/generalization | `le2i` | `GCN` | `outputs/le2i_gcn_W48S12_opt33_r2` | Comparative in-domain candidate, not the primary deployment claim |

## Frozen Interpretation Rules

- `CAUCAFall` is the primary dataset for benchmark and deployment-oriented claims.
- `LE2i` is mandatory comparative evidence, but not the dataset used to unlock the main deployment claim.
- The current paper-safe architecture result is:
  - TCN trends stronger than GCN under the frozen protocol
  - the statistical conclusion remains cautious at the current `n=5`
- Cross-dataset transfer results are limitations evidence, not support for a universal-robustness claim.

## Required Evidence Alignment

These files must remain aligned with this freeze:
- [CLAIM_TABLE.md](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/docs/project_targets/CLAIM_TABLE.md)
- [THESIS_EVIDENCE_MAP.md](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/docs/project_targets/THESIS_EVIDENCE_MAP.md)
- [RESEARCH_QUESTIONS_MAPPING.md](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/docs/project_targets/RESEARCH_QUESTIONS_MAPPING.md)
- [STABILITY_REPORT.md](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/docs/project_targets/STABILITY_REPORT.md)
- [SIGNIFICANCE_REPORT.md](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/docs/project_targets/SIGNIFICANCE_REPORT.md)

## Change Control

If this file changes, update in the same pass:
1. claim wording in `CLAIM_TABLE.md`
2. evidence rows in `THESIS_EVIDENCE_MAP.md`
3. question status wording in `RESEARCH_QUESTIONS_MAPPING.md`
4. any paper draft text that names the final candidates
