# Report High-Score Restructure Tasks

Date: 2026-05-05
Target source: `docs/reports/drafts/FULL_PROJECT_REPORT_FINAL_2026-04-11.md`
Target output: `docs/reports/drafts/FULL_PROJECT_REPORT_HIGH_SCORE_RESTRUCTURED.md`

## Goal

Restructure the final-year project report from an evidence-heavy project archive into a focused academic report with evidence-rich appendices.

The restructured report must:

- preserve the current factual claims and evidence hierarchy
- improve readability, flow, and scoring potential
- reduce archive-like prose in the main body
- keep appendices as the home for artifact lineage, audit detail, commands, and freeze-state material
- maintain bounded claims without inflating deployment or notification evidence

## Core Transformation

Transform the report from:

- complete project archive with strong evidence discipline

into:

- focused academic report with a clear research problem
- controlled methodology
- strong system architecture chapter
- concise results chapter
- bounded deployment chapter
- direct discussion and conclusion

## Fixed Target Structure

The new report must follow this structure:

1. `Abstract`
2. `Introduction`
3. `Background and Related Work`
4. `Requirements and Research Questions`
5. `System Architecture`
6. `Data and Experimental Protocol`
7. `Model and Alert-Policy Design`
8. `Implementation`
9. `Experimental Results`
10. `System Validation and Deployment Evidence`
11. `Discussion`
12. `Limitations and Future Work`
13. `Conclusion`
14. `References`
15. `Appendices`

## Chapter Mapping

### Keep and compress

- `Abstract` -> rewritten after the full restructure
- `Introduction` -> compress current framing, aims, contributions, and report structure
- `Background and Related Work` -> retain problem-layer organisation, increase literature tone, reduce project self-positioning
- `System Architecture` -> retain as a major strength, add a responsibility table
- `Data and Experimental Protocol` -> keep as a technical protocol chapter

### Merge

- `Project Context and Requirements`
- `Research Questions and Scope`

into:

- `Requirements and Research Questions`

and merge:

- `Model Design and Training`
- `Calibration and Alert Policy`

into:

- `Model and Alert-Policy Design`

### Rewrite as final-system chapter

- `Implementation and Refactoring` -> `Implementation`

The new chapter must describe the final system rather than the full development history.

### Move mostly to appendices

- `Project Management, Iteration, and Risk Control`
- most of `Audit and Code Review Work`
- freeze/handoff details
- long artifact-state inventories
- reproduction command lists
- command-to-output tables
- objective-revisited tables
- risk-control matrices
- issue-to-fix tables
- architecture evolution narrative
- iteration timeline narrative

### Remove as standalone chapter

- `Final Evaluation`

Its strongest conclusions must be merged into `Discussion` and `Conclusion`.

### Merge

- `Limitations`
- `Future Work`

into:

- `Limitations and Future Work`

## Main-Report Content Rules

### Main text should prioritise

- research problem
- methodological control
- architecture clarity
- concise results
- evidence boundaries
- direct RQ answers
- professional final-submission tone

### Main text should reduce

- archive-like artifact references
- long path-heavy tables
- repeated self-explanatory prose
- repeated caveat sentences after every result
- project-diary or management-diary tone

## Table Strategy

### Keep in main report

1. functional requirements and verification summary
2. architecture responsibility table
3. protocol summary table
4. model and alert-policy design table
5. frozen in-domain comparison table
6. cross-dataset transfer table
7. runtime evidence summary table
8. validation interpretation matrix

### Move to appendices

- artifact lineage tables
- command-to-output maps
- risk-control matrices
- issue-to-fix summaries
- freeze and handoff tables
- detailed audit closure tables
- configuration snapshot inventories
- appendix-only figure usage guidance

### Table style constraints

- no full file paths in main-report tables
- no report-anchor columns in main-report tables
- concise row text
- use interpretation columns sparingly
- keep artifact filenames or command families in appendices

## Figure Strategy

### Keep in main report

- Figure 1: system architecture
- Figure 2: dataset/evidence hierarchy
- Figure 3: temporal window contract
- Figure 4: alert-policy flow
- Figure 5: offline stability comparison
- Figure 6: cross-dataset transfer summary
- Figure 7: online replay heatmap
- Figure 8: MC-dropout delta
- Figure 9: runtime UI evidence

### Move to appendices

- iteration timeline
- architecture evolution
- supporting quantitative figures
- design-evolution figures
- diagnostic-only visuals

### Figure cleanup rules

- figures must be embedded with markdown image syntax
- main-report figure sections should not use `Asset:` registry framing
- figure captions must read like final academic report captions
- figure numbering must remain sequential after restructuring

## Language Cleanup Rules

Aggressively reduce repeated phrases such as:

- `This matters because`
- `This distinction matters because`
- `This is important because`
- `This is one of the report's strengths`
- `The report is strongest when`
- `A high-quality report should`

Also reduce repeated overuse of:

- `bounded`
- `evidence`
- `reviewable`
- `coherent`
- `defensible`

Retain bounded-claim discipline, but express it more directly.

## Claim-Control Rules

Do not introduce or strengthen claims beyond the current file.

The restructured report must not claim:

- solved fall detection
- broad real-home robustness
- clinical readiness
- replay as benchmark evidence
- delivery as proof of detection validity
- uncertainty-aware runtime improvement if the replay matrix stayed unchanged
- SMS/phone-call escalation as implemented functionality

The strongest acceptable claims remain:

- a cautious directional TCN advantage under the locked primary protocol
- an integrated monitoring system that joins pose inference, alert policy, persistence, review, and Telegram delivery
- bounded replay/live system evidence rather than field closure

## Reference Cleanup Rules

- remove any preface before `References`
- start directly with the reference list
- keep one consistent style
- do not invent missing metadata
- preserve current references unless obviously irrelevant or duplicated

## Final Checklist

Before closing the restructure:

1. confirm the new report follows the target 12-chapter structure
2. confirm `Final Evaluation` is removed as a standalone chapter
3. confirm `Project Management` is mostly moved to appendices
4. confirm Figures 1 to 9 remain embedded in the main report
5. confirm Results tables are split and readable
6. confirm artifact paths are mostly pushed to appendices
7. confirm draft-like wording is removed
8. confirm conclusion does not introduce new evidence
9. confirm Telegram remains the active notification path
10. confirm replay remains separated from benchmark evidence
11. confirm `CAUCAFall` remains the primary benchmark/deployment-target dataset
12. confirm `LE2i` remains comparative and transfer-boundary evidence
13. confirm `MUVIM` remains secondary and exploratory

## Execution Order

1. audit the source report against the target structure
2. create the task document
3. write the restructured report as a new markdown file
4. perform a final structural and claim audit on the new file
