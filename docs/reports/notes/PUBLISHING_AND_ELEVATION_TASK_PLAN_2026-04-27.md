# Publishing and Elevation Task Plan

Date: 2026-04-27  
Project: Pose-Based Fall Detection System  
Purpose: define a strict 20-day execution plan for strengthening the project where necessary and raising the final report and compact paper to a more defensible research standard.

## 1. Why This Plan Exists

The current project is already strong enough to support a serious final-year submission. It is not yet strong enough to be treated as a stable publication-ready research package without further tightening.

The immediate risk is not that the system does nothing. The system already works. The real risk is that effort is wasted on low-yield feature expansion while the stronger path to a higher-quality report and paper is left underdeveloped.

This plan therefore enforces three priorities:

1. preserve the strongest current framing: an end-to-end fall-detection and monitoring system
2. strengthen only those parts of the project that materially improve evidence quality
3. raise the report and paper through stricter contribution framing, stronger result synthesis, and tighter claim control

## 2. Current Position

### 2.1 What Is Already Strong

1. The project already has a functioning end-to-end system with frontend, backend, monitoring, replay, persistence, and caregiver-facing delivery.
2. The repository already supports a serious project-report story rather than a notebook-only model exercise.
3. The full report already has the structure and depth expected of a substantial final-year report.
4. The compact paper already has a viable research direction, especially if framed as a deployment-oriented applied AI systems study rather than a pure architecture paper.

### 2.2 What Is Not Yet Strong Enough

1. The compact paper is still closer to a compressed dissertation/report than to a focused publishable manuscript.
2. The strongest claims are not yet expressed through a small, disciplined set of headline results.
3. Some evidence exists, but its narrative value has not yet been converted into high-quality synthesis tables, concise comparative interpretation, and clear generalisable lessons.
4. Publication-level novelty is not yet sharp enough if the work is presented mainly as a `TCN vs GCN` comparison.

### 2.3 Working Judgment

At this stage, the project should be treated primarily as:

1. a working software system
2. a deployment-oriented applied AI project
3. a controlled model-comparison study in a supporting role

The report should follow that order of importance. The paper should also follow that order unless stronger new research evidence is produced.

## 3. Strategic Rule

Do not assume that the project must be heavily expanded before the report and paper can improve.

For the full report, the main bottleneck is now expression, synthesis, and evidence discipline rather than raw system functionality.

For the paper, the bottleneck is split:

1. framing and manuscript discipline are immediate weaknesses
2. research-evidence strength may still need targeted reinforcement

Therefore:

1. the report should be upgraded now
2. the paper should be refocused now
3. the project itself should only be extended where the added work directly improves defended evidence

## 4. Non-Negotiable Writing Rules

These rules apply to both the full report and the paper.

1. Every strong claim must map to a live artifact.
2. Replay evidence must never be allowed to replace formal offline evidence.
3. Delivery-path evidence must never be used to strengthen model-comparison claims.
4. Runtime feasibility must be described as bounded feasibility, not broad field closure.
5. The project must not be marketed as solving fall detection in general.
6. The paper must not be written as a shortened version of the report; it must have its own argumentative spine.

## 5. Main Workstreams

The remaining 20-day effort should be split into three controlled workstreams.

### Workstream A. Evidence-Strengthening Work on the Project

This workstream exists only to close evidence gaps that materially affect publishability or high-level report quality.

Allowed tasks:

1. produce one compact quantitative summary table covering the main defended results
2. produce one concise error/failure analysis section from existing artifacts
3. run limited, targeted verification or lightweight analysis if it sharpens a central claim
4. improve claim-to-artifact traceability where the evidence already exists but is not clearly surfaced

Disallowed tasks unless explicitly justified:

1. large new feature additions
2. UI polish that does not change evidence quality
3. new exploratory branches that do not feed the final report or paper
4. broad experiment sprawl without a clear publication-facing question

### Workstream B. Full Report Elevation

This is the highest-priority writing workstream because the full report is already close to a strong submission and can realistically be pushed to a visibly higher standard within the remaining time.

Objectives:

1. make the report read as a research-grade technical report, not a long project diary
2. tighten the contribution framing around the system-and-deployment story
3. ensure every major chapter has a clear argumentative job
4. make the results chapter more decisive and easier to defend orally

### Workstream C. Compact Paper Refocus

This workstream should treat the paper as a separate manuscript problem, not as a formatting derivative of the report.

Objectives:

1. identify one central contribution
2. reduce the number of equal-status claims
3. convert the current manuscript from dissertation-compressed prose into paper-style argumentation
4. decide honestly whether the near-term target is:
   - publishable workshop-style paper
   - high-quality project companion manuscript
   - longer-term post-project paper foundation

## 6. Strict Task List

### Phase 1. Lock the Defended Contribution Story

Priority: immediate

Tasks:

1. Finalise one sentence that defines the project at the highest level.
   Recommended baseline:
   `This project is an end-to-end pose-based fall-detection and monitoring system study, supported by controlled model comparison and deployment-oriented evaluation.`
2. Finalise one sentence that defines the paper contribution more narrowly.
3. Finalise which evidence answers each research question.
4. Remove or soften any claim that does not improve under this framing.

Exit criteria:

1. one stable system-level framing exists
2. one stable paper-level framing exists
3. no chapter is trying to tell a different project story

### Phase 2. Strengthen the Evidence Surface

Priority: immediate

Tasks:

1. Build one compact main-results summary table for both report and paper use.
   Required columns:
   - dataset
   - model
   - operating-point relevance if applicable
   - main metric(s)
   - evidence role
   - artifact anchor
2. Build one compact deployment-evidence summary table.
   Required columns:
   - scenario
   - evidence type
   - defended finding
   - limitation
3. Add one concise error-analysis subsection.
   Candidate themes:
   - cross-dataset asymmetry
   - replay versus realtime behaviour
   - pose-quality and timing sensitivity
4. Build a clean claim-to-artifact checklist for all headline claims in the report and paper.

Exit criteria:

1. the strongest conclusions can be read from one page of tables
2. the report no longer depends on diffuse figure-reading alone
3. the paper has a visible evidence backbone

### Phase 3. Elevate the Full Report

Priority: highest writing priority

Tasks:

1. Tighten the abstract so that it states:
   - what was built
   - what was compared
   - what was found
   - what remains limited
2. Tighten the introduction so that the system problem is foregrounded before the architecture comparison.
3. Rework the contributions section so that system contribution comes first and model comparison is clearly secondary.
4. Rebuild the results chapter around a small number of defended takeaways.
5. Strengthen the discussion so that it interprets the evidence rather than restating it.
6. Make limitations sharp, explicit, and non-defensive.
7. Ensure appendices carry audit and reproducibility value rather than overflow prose.

Exit criteria:

1. the report can be defended as a research-grade technical report
2. the strongest takeaways are easy to identify
3. limitations are clear enough that over-claiming is not a live risk

### Phase 4. Refocus the Compact Paper

Priority: after report structure is stable

Tasks:

1. Decide the manuscript type.
   Preferred current direction:
   deployment-oriented applied AI systems paper
2. Rewrite the abstract for paper style rather than report style.
3. Rewrite the introduction to converge quickly on the paper’s main question.
4. Reduce the contribution list to one primary contribution and at most two secondary contributions.
5. Compress methods to only what is necessary for understanding the defended result.
6. Rebuild results around two or three headline findings only.
7. Remove report-style process detail that belongs in appendices or the full report.
8. Add a discussion paragraph on what generalises beyond this exact repository.

Exit criteria:

1. the paper no longer reads like a compressed dissertation chapter
2. the manuscript has one central claim
3. every included figure or table supports that claim directly

### Phase 5. Publication Gate Check

Priority: final week

Tasks:

1. Judge whether the paper now supports a real near-term submission target.
2. If not, keep the paper honest and position it as a strong project companion manuscript rather than pretending it is submission-ready.
3. Record the remaining evidence gaps explicitly.

Decision options:

1. `submit-quality project paper draft`
2. `workshop-potential draft with more evidence needed`
3. `project companion manuscript only`

Exit criteria:

1. the team has an honest publication-position statement
2. no one is confusing project strength with publication strength

## 7. What Project-Level Strengthening Is Actually Worth Doing

Only a narrow class of project work is worth doing now.

### Worth Doing

1. tighter result synthesis
2. lightweight additional analysis from existing artifacts
3. failure-case interpretation
4. traceability cleanup
5. one or two targeted checks that increase confidence in a headline claim

### Usually Not Worth Doing

1. major frontend redesign
2. large backend refactors unless they fix a result-relevant correctness issue
3. broad new dataset exploration
4. many new model variants
5. adding features that make demos nicer but claims no stronger

## 8. Report-Specific Acceptance Standard

The full report should be judged ready only if all of the following are true:

1. the project framing is stable and consistent throughout
2. the system contribution is clear within the first pages
3. the model-comparison evidence is cautious and defensible
4. the deployment/runtime evidence is explicitly bounded
5. the main results can be summarised quickly without hunting through the text
6. figure and table references are correct
7. appendix material improves traceability and reproducibility

## 9. Paper-Specific Acceptance Standard

The paper should be judged meaningfully improved only if all of the following are true:

1. the paper has one main contribution
2. the first page makes the problem, method, and defended finding clear
3. the results section is compact and selective
4. the contribution is not sold mainly as `TCN vs GCN`
5. the deployment/system framing is visible and coherent
6. limitations and evidence boundaries are explicit

The paper should be judged publishable only if, in addition, the following becomes true:

1. the contribution reads as reusable or generalisable beyond this single project
2. the evidence is strong enough to survive reviewer questions about scope and novelty
3. the manuscript remains convincing after removing any weak or over-precise claim

## 10. 20-Day Execution Order

### Days 1-3

1. lock contribution framing
2. build main-results and deployment-results summary tables
3. complete headline claim-to-artifact review

### Days 4-8

1. revise the full report abstract, introduction, contributions, results, discussion, and limitations
2. insert the new synthesis tables
3. run one final consistency pass on figure, table, and artifact references

### Days 9-13

1. rebuild the paper around one central claim
2. compress methods and restructure results
3. remove dissertation-style material

### Days 14-17

1. perform publication-gate review
2. decide whether additional targeted analysis is required
3. if required, run only high-yield targeted evidence work

### Days 18-20

1. final report polish and build
2. final paper polish
3. supervisor-facing summary and submission packaging

## 11. Supervisor-Facing Decision Questions

These are the decisions the supervisor should help settle, not detailed proofreading questions.

1. Is the project strongest as a system study, a deployment-oriented applied AI project, or a model-comparison paper?
2. Is the current contribution framing sharp enough for the paper, or does it still need a narrower central claim?
3. Is the current evidence base sufficient for a workshop-style paper, or is the stronger near-term goal a high-quality project companion manuscript?
4. If more project strengthening is needed, which single evidence gap matters most?

## 12. Final Rule

The remaining time should be spent increasing defended insight, not increasing activity count.

A smaller number of stronger claims, better synthesised and better bounded, will raise both the report and the paper more than a larger number of weak additions.

## 13. Execution Sheet

This section converts the strategy above into a direct execution plan. Every task here should either produce a concrete file change, a concrete artifact, or a concrete verification result.

### 13.1 Execution Rules

1. Do not start paper polishing before the full report contribution framing and results synthesis are stable.
2. Do not run new experiments unless the output is expected to change a headline claim, a key table, or a supervisor-facing answer.
3. Do not add a section to either document unless its argumentative job is explicit.
4. Every finished task must leave behind one of:
   - a changed draft file
   - a new table or figure
   - a new checklist or audit note
   - a verified build output

### 13.2 Mandatory Deliverables

The following deliverables must exist before final submission or final supervisor review.

1. one tightened full report draft
2. one refocused compact paper draft
3. one main-results synthesis table
4. one deployment-evidence synthesis table
5. one short error-analysis subsection
6. one claim-to-artifact checklist for headline claims
7. one final report PDF build
8. one final report DOCX build if needed for sharing
9. one paper PDF or DOCX build for review circulation

### 13.3 Task Tracking Fields

Every execution task below should be tracked with these fields while working:

1. owner
2. status
3. target file
4. output artifact
5. completion evidence

Suggested status values:

1. `not started`
2. `in progress`
3. `blocked`
4. `done`

## 14. Detailed Task Breakdown

### 14.1 Contribution and Framing Lock

These tasks must be completed first because they constrain all later writing.

#### Task F1. Lock the project-level framing sentence

Goal:
- define the one sentence that describes the whole project consistently across report, paper, README, and supervisor discussion

Target files:
- `docs/reports/drafts/FULL_PROJECT_REPORT_FINAL_2026-04-11.md`
- `docs/reports/drafts/PAPER_FINAL_2026-04-11.md`

Output:
- one final project framing sentence reused in both drafts

Required content:
- system-first
- deployment-aware
- model comparison clearly secondary

Done condition:
- the sentence appears in the abstract and introduction of the full report
- the same framing logic appears in the abstract and introduction of the paper

#### Task F2. Lock the paper-level central claim

Goal:
- define what the paper is actually trying to contribute beyond “we built a project”

Target file:
- `docs/reports/drafts/PAPER_FINAL_2026-04-11.md`

Output:
- one sentence in the introduction stating the manuscript’s main defended contribution

Recommended direction:
- deployment-oriented pose-based monitoring system study

Done condition:
- the paper’s contribution list, abstract, and conclusion all point to the same main claim

#### Task F3. Lock research-question-to-evidence mapping

Goal:
- make sure each research question is answered by the right evidence layer

Target files:
- `docs/reports/drafts/FULL_PROJECT_REPORT_FINAL_2026-04-11.md`
- `docs/reports/drafts/PAPER_FINAL_2026-04-11.md`
- optional supporting checklist under `docs/reports/notes/` or `docs/reports/runbooks/`

Output:
- a short mapping block or checklist showing:
  - RQ1 -> offline model evidence
  - RQ2 -> operating-point and alert-policy evidence
  - RQ3 -> replay/runtime/deployment evidence

Done condition:
- no chapter uses the wrong evidence layer to answer a research question

### 14.2 Results-Synthesis Deliverables

These tasks produce the concrete artifacts that should anchor both documents.

#### Task R1. Build the main-results synthesis table

Goal:
- make the main defended findings visible in one compact table

Target location:
- main insertion into full report results chapter
- condensed insertion into paper results chapter

Potential supporting file:
- a standalone markdown table note under `docs/reports/notes/` for drafting

Required columns:

1. dataset
2. model
3. evidence setting
4. main metric or finding
5. interpretation
6. artifact anchor

Source evidence:
- `artifacts/reports/stability_summary.csv`
- `artifacts/reports/significance_summary.json`
- `artifacts/reports/cross_dataset_summary.csv`
- any final defended deployment summary artifacts actually used in the drafts

Done condition:
- the table can be read on its own and still communicate the main result story
- every row can be traced to a live artifact

#### Task R2. Build the deployment-evidence synthesis table

Goal:
- summarise bounded runtime, replay, and delivery evidence without conflating it with benchmark evidence

Target location:
- deployment/runtime results chapter in the full report
- one compressed version for the paper if still needed

Required columns:

1. scenario
2. evidence type
3. defended finding
4. main limitation
5. artifact anchor

Source evidence:
- `docs/reports/runbooks/FOUR_VIDEO_DELIVERY_PROFILE.md`
- `artifacts/reports/online_mc_replay_matrix_20260402.csv`
- `artifacts/reports/deployment_field_validation_summary.md`
- any final replay/runtime summary artifacts actually cited

Done condition:
- the table makes the bounded nature of deployment evidence immediately obvious

#### Task R3. Build the error-analysis subsection

Goal:
- convert limitations and failure patterns into one analytical subsection rather than scattering them informally

Preferred location:
- full report discussion or limitations chapter

Optional paper location:
- short discussion paragraph only

Candidate themes:

1. cross-dataset asymmetry
2. replay versus realtime mismatch
3. pose-quality sensitivity
4. timing and latency dependence

Done condition:
- the subsection explains at least one meaningful weakness using repository evidence rather than generic caution language

#### Task R4. Build the headline claim-to-artifact checklist

Goal:
- eliminate unsupported headline claims before final polish

Target file:
- create a dedicated checklist under `docs/reports/notes/` or `docs/reports/runbooks/`

Checklist rows should include:

1. claim text
2. evidence layer
3. artifact path
4. safe wording status
5. action needed

Done condition:
- every headline claim in abstract, introduction, contributions, results, and conclusion has been checked

### 14.3 Full Report Upgrade Tasks

#### Task FR1. Rewrite the abstract

Goal:
- make the abstract precise, strong, and bounded

Target file:
- `docs/reports/drafts/FULL_PROJECT_REPORT_FINAL_2026-04-11.md`

Checklist:

1. state the problem as a monitoring problem, not just classification
2. state what system was built
3. state what comparison was performed
4. state the strongest defended findings
5. state what is not being claimed

Done condition:
- the abstract can stand alone as a faithful summary of the whole report

#### Task FR2. Tighten the introduction

Goal:
- foreground the system-level problem before detailed model discussion

Target file:
- `docs/reports/drafts/FULL_PROJECT_REPORT_FINAL_2026-04-11.md`

Checklist:

1. background explains why detection alone is not enough
2. problem framing distinguishes system, policy, and model layers
3. objectives are aligned with final contribution framing
4. contributions are ordered by actual project strength

Done condition:
- the introduction makes the report read like a system study with controlled model comparison support

#### Task FR3. Rebuild the contributions section

Goal:
- stop presenting too many equal-status contributions

Target file:
- `docs/reports/drafts/FULL_PROJECT_REPORT_FINAL_2026-04-11.md`

Checklist:

1. contribution 1: system-level end-to-end artifact
2. contribution 2: controlled TCN vs custom GCN comparison
3. contribution 3: alert-policy and deployment-evidence discipline
4. contribution 4 only if it is truly distinct and defended

Done condition:
- each contribution has a clear evidence basis and no contribution feels decorative

#### Task FR4. Rebuild the results chapter around defended takeaways

Goal:
- turn the results chapter into a small number of argued conclusions

Target file:
- `docs/reports/drafts/FULL_PROJECT_REPORT_FINAL_2026-04-11.md`

Structure target:

1. offline comparative findings
2. cross-dataset findings
3. deployment/runtime findings
4. synthesis paragraph linking back to research questions

Checklist:

1. insert main-results synthesis table
2. insert deployment-evidence synthesis table
3. every figure is interpreted, not just shown
4. remove diffuse repetition

Done condition:
- the chapter can be skimmed quickly and still reveal the main conclusions

#### Task FR5. Strengthen discussion and limitations

Goal:
- make the report sound analytically mature rather than defensive

Target file:
- `docs/reports/drafts/FULL_PROJECT_REPORT_FINAL_2026-04-11.md`

Checklist:

1. discussion explains why the findings matter
2. limitations are concrete and evidence-linked
3. future work follows directly from actual limits
4. no claim remains stronger than its evidence layer

Done condition:
- a reviewer can see both what the project achieved and where its evidence stops

#### Task FR6. Final report consistency pass

Goal:
- remove all reference, numbering, and cross-link inconsistencies

Target file:
- `docs/reports/drafts/FULL_PROJECT_REPORT_FINAL_2026-04-11.md`

Verification list:

1. all figure paths exist
2. all table numbers are unique and ordered
3. all appendix references resolve
4. all code-path references match the final repository structure
5. all exact quantitative claims match current artifacts

Done condition:
- no obvious structural or reference-quality issue remains

### 14.4 Paper Refocus Tasks

#### Task P1. Reclassify the paper type explicitly

Goal:
- decide what kind of paper is actually being written

Target file:
- `docs/reports/drafts/PAPER_FINAL_2026-04-11.md`

Decision target:
- deployment-oriented applied AI systems paper

Done condition:
- title, abstract, introduction, and conclusion all reflect the same paper type

#### Task P2. Rewrite the abstract for manuscript style

Goal:
- shorten and sharpen the abstract so that it behaves like a paper abstract rather than a report summary

Target file:
- `docs/reports/drafts/PAPER_FINAL_2026-04-11.md`

Checklist:

1. one-sentence problem setup
2. one-sentence method/system contribution
3. one-sentence main defended finding
4. one-sentence limitation boundary

Done condition:
- the abstract is compact and publication-facing

#### Task P3. Rewrite the introduction around one main question

Goal:
- reduce narrative spread and converge faster on the manuscript’s central contribution

Target file:
- `docs/reports/drafts/PAPER_FINAL_2026-04-11.md`

Checklist:

1. remove report-like sectioning if too heavy
2. state the gap in existing fall-detection writing
3. state what this manuscript contributes
4. state why deployment-aware framing matters

Done condition:
- a reader can identify the paper’s main purpose within the first page

#### Task P4. Compress methods

Goal:
- keep only methods detail that serves the paper’s central claim

Target file:
- `docs/reports/drafts/PAPER_FINAL_2026-04-11.md`

Checklist:

1. retain enough detail for the controlled comparison
2. retain enough detail for alert-policy explanation
3. remove report-only implementation narration
4. avoid duplicating long repository detail

Done condition:
- methods support the argument without feeling like an appendix dump

#### Task P5. Rebuild the paper results section

Goal:
- reduce the results to two or three headline findings

Target file:
- `docs/reports/drafts/PAPER_FINAL_2026-04-11.md`

Preferred result blocks:

1. controlled offline comparison
2. calibration/alert-policy importance
3. bounded deployment/runtime interpretation

Checklist:

1. reuse the compact synthesis table where appropriate
2. drop weak or secondary observations
3. ensure every included number serves a main claim

Done condition:
- the paper has a small, high-density results section

#### Task P6. Add generalisable discussion

Goal:
- move the paper closer to publication quality by extracting lessons beyond this exact repository

Target file:
- `docs/reports/drafts/PAPER_FINAL_2026-04-11.md`

Candidate discussion themes:

1. why deployment-aware alerting changes evaluation priorities
2. why bounded replay evidence is useful but not equivalent to benchmark evidence
3. why system coherence matters in pose-based monitoring

Done condition:
- the paper makes at least one insight that a reader could transfer beyond this project

#### Task P7. Paper consistency pass

Goal:
- ensure the paper no longer carries report-style drift or unsupported precision

Target file:
- `docs/reports/drafts/PAPER_FINAL_2026-04-11.md`

Verification list:

1. all headline claims checked against artifacts
2. no stale project-structure references
3. no exact number remains unless directly defended
4. figures and tables are minimal and necessary

Done condition:
- the paper reads like a deliberate manuscript rather than a shortened thesis chapter

### 14.5 Build and Verification Tasks

#### Task V1. Full report build verification

Goal:
- confirm the final report can be exported cleanly

Inputs:
- `docs/reports/drafts/FULL_PROJECT_REPORT_FINAL_2026-04-11.md`

Outputs:

1. PDF build
2. DOCX build if required

Done condition:
- the output file exists
- images are embedded correctly where applicable
- no major formatting break is visible

#### Task V2. Paper build verification

Goal:
- produce a shareable review artifact for the paper

Inputs:
- `docs/reports/drafts/PAPER_FINAL_2026-04-11.md`

Outputs:

1. PDF and/or DOCX build

Done condition:
- the paper can be circulated to supervisor without broken media or structure

#### Task V3. Final oral-defence readiness check

Goal:
- make sure the report and paper can be explained consistently in a supervisor meeting

Output:
- one short note answering:
  1. what the project is
  2. what the strongest findings are
  3. what the limitations are
  4. whether the paper is publication-ready or still transitional

Done condition:
- the team can answer those four questions without contradiction

## 15. Suggested Weekly Cadence

### Week 1

Focus:

1. F1
2. F2
3. F3
4. R1
5. R2
6. R4

Required outputs by end of week:

1. locked framing
2. two synthesis tables
3. claim-to-artifact checklist

### Week 2

Focus:

1. R3
2. FR1
3. FR2
4. FR3
5. FR4
6. FR5

Required outputs by end of week:

1. tightened full report core chapters
2. inserted synthesis tables
3. explicit error-analysis subsection

### Week 3

Focus:

1. FR6
2. P1
3. P2
4. P3
5. P4
6. P5
7. P6
8. P7
9. V1
10. V2
11. V3

Required outputs by end of week:

1. stable report build
2. stable paper build
3. honest publication-position statement

## 16. Stop-Doing List

The team should actively avoid the following during the remaining 20 days.

1. adding features only because they look impressive in a demo
2. widening the paper scope while the main claim is still unstable
3. inserting more exact numbers before checking artifact support
4. writing new prose before the framing and tables are locked
5. treating all evidence as equal just because it is available

## 17. Minimum Completion Standard

This plan should be considered successfully executed only if the following become true.

1. the full report is stronger, clearer, and easier to defend than the current draft
2. the paper has one central claim and a visibly tighter structure
3. the strongest project evidence is surfaced through concise synthesis rather than scattered references
4. the team knows exactly which publication claim can and cannot be made

## 18. Gap Ownership: What Can Be Solved by Writing Versus What Requires New Evidence

This section prevents the team from confusing a writing problem with a project-strength problem.

### 18.1 Gaps That Can Be Solved Directly Through Report/Paper Work

These gaps should be treated as immediate execution targets because they do not require major new project development.

#### A. Framing and contribution weakness

Problem:
- the project story is still at risk of sounding split across system, comparison, and deployment lines

Can be solved by:

1. locking one project-level framing sentence
2. locking one paper-level central claim
3. reordering contribution lists
4. rewriting abstract, introduction, and conclusion around the same defended story

Primary tasks:
- `F1`
- `F2`
- `FR1`
- `FR2`
- `FR3`
- `P1`
- `P2`
- `P3`

#### B. Results are not yet synthesised strongly enough

Problem:
- important findings exist, but they are not yet compressed into high-value summary artifacts

Can be solved by:

1. building the main-results synthesis table
2. building the deployment-evidence synthesis table
3. rebuilding the results sections around a small number of takeaways
4. writing a concise error-analysis subsection

Primary tasks:
- `R1`
- `R2`
- `R3`
- `FR4`
- `P5`

#### C. Claim discipline and consistency weakness

Problem:
- unsupported precision, stale references, or unbounded wording can weaken both marks and paper credibility

Can be solved by:

1. building the headline claim-to-artifact checklist
2. softening unsupported exact claims
3. checking figure and table references
4. checking path and artifact alignment
5. running final consistency passes on both documents

Primary tasks:
- `R4`
- `FR6`
- `P7`
- `V1`
- `V2`

#### D. The paper still reads too much like a report derivative

Problem:
- the manuscript currently risks reading like a shortened dissertation chapter

Can be solved by:

1. compressing methods
2. reducing the number of equal-status results
3. inserting a generalisable discussion paragraph
4. removing process-heavy report material

Primary tasks:
- `P4`
- `P5`
- `P6`
- `P7`

### 18.2 Gaps That Usually Require New or Stronger Evidence

These gaps cannot be solved honestly through wording alone. If the team decides to chase them, they must be treated as targeted project-strengthening work.

#### E. Publication-level central novelty is still too soft

Problem:
- a complete project and a careful comparison do not automatically become a publishable research novelty

Usually requires:

1. a sharper reusable methodological claim
2. stronger evidence that the contribution matters beyond this single repository
3. possibly one additional analysis or structured evaluation pass that supports the stronger claim

Writing alone can help:
- express the claim more clearly
- remove misleading novelty inflation

Writing alone cannot do:
- create a genuine new contribution if the evidence does not support one

#### F. Reviewer-proof experimental depth is insufficient

Problem:
- reviewers may question breadth, robustness, or comparative strength

Usually requires:

1. targeted additional analysis
2. selected extra validation or ablation work
3. more robust comparison support if current evidence is too narrow for the intended venue

Writing alone can help:
- bound the claims correctly
- present the existing evidence more convincingly

Writing alone cannot do:
- supply missing robustness evidence

#### G. Generalisation and transfer claims are not yet strong enough

Problem:
- proving that this project works is not the same as proving broader transfer value

Usually requires:

1. stronger cross-setting or cross-dataset interpretation
2. more explicit evidence for what generalises and what does not
3. possibly additional structured analysis of failure boundaries

Writing alone can help:
- extract cleaner lessons from current evidence

Writing alone cannot do:
- turn a bounded project result into a broad generalisation claim

### 18.3 Decision Rule

Use this rule before adding any new work.

1. If the gap is mainly about framing, synthesis, discipline, or consistency:
   solve it in the documents now.
2. If the gap is mainly about novelty strength, robustness, or generalisation:
   decide explicitly whether a targeted evidence task is worth the time.
3. If a new task does not clearly improve either defended marks or defended publication potential:
   do not do it.

### 18.4 Practical Implication

For the remaining 20 days, the team should assume:

1. the path to `90+` is mostly a document-quality and defended-synthesis problem
2. the path to a more publishable paper is partly a document problem and partly an evidence-strength problem
3. therefore, report and paper work should proceed immediately, while project-strengthening work should be added only where it closes a clearly identified publication-facing gap
