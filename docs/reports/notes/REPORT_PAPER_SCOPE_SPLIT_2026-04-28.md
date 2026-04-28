Date: 2026-04-28  
Purpose: define the correct division of labour between the `20-mark` full report and the `10-mark` paper.

# Report–Paper Scope Split

## 1. Priority Rule

The weighting determines the writing strategy:

1. `full report = 20 marks`
2. `paper = 10 marks`

Therefore:

1. the full report must carry the **complete defended project story**
2. the paper must carry the **single strongest publication-facing story**
3. the paper must not be allowed to consume writing time that should go into raising the report to high-mark quality

## 2. Core Principle

The report should show the breadth and coherence of the whole project.

The paper should not try to represent the whole project evenly. It should extract one stronger line and organise the evidence around that line.

Correct split:

1. **report = comprehensive defended technical artifact**
2. **paper = selective, narrower manuscript built from the strongest defended line**

## 3. What the Full Report Should Cover

The full report should explicitly retain all major workstreams that materially demonstrate project scope, technical decision-making, and defended outcomes.

### Keep in the Full Report

1. problem framing and motivation
2. full system architecture
3. dataset roles and protocol
4. TCN versus custom GCN comparison
5. calibration / operating-point fitting
6. replay and runtime evidence
7. live/realtime evidence
8. persistence and caregiver-delivery path
9. cross-dataset limitation evidence
10. retraining / strengthening work
11. testing, audit, and system-validation structure
12. limitations and future work

### How These Should Be Weighted

Not every part should receive equal narrative weight.

Recommended weighting inside the report:

1. primary weight:
   - system contribution
   - results
   - discussion
   - limitations
2. secondary weight:
   - model comparison
   - calibration and alert policy
   - runtime/deployment evidence
3. tertiary/supporting weight:
   - exploratory tracks such as `MUVIM`
   - extended audit/process material
   - detailed reproducibility notes better left to appendix/supporting sections

### Full Report Writing Rule

The report must include the work, but it must not read like a diary or inventory.

Each included workstream should answer one of these questions:

1. what was built?
2. what was tested?
3. what was learned?
4. what remains limited?

If a section does not help answer one of those four questions, compress it or move it down.

## 4. What the Paper Should Focus On

The paper should be built around the strongest narrow thesis currently available:

`This project is best understood as a deployment-oriented pose-based fall-detection system study, supported by controlled offline comparison and bounded runtime analysis.`

### Keep in the Paper

1. problem and deployment-oriented motivation
2. short explanation of pose-based temporal monitoring framing
3. locked TCN versus custom GCN comparison as supporting model evidence
4. calibration / alert-policy layer as a methodological contribution
5. bounded replay/runtime evidence
6. cross-dataset limitation evidence
7. modest retraining strengthening result only insofar as it sharpens the deployment story

### De-Emphasise or Remove from the Paper

1. broad audit/process-control narrative
2. extensive refactoring history
3. too much implementation-detail prose about frontend/backend cleanup
4. exploratory-track detail that does not sharpen the paper thesis
5. large amounts of supplementary operational detail that are valuable in the report but not in the manuscript

### Paper Writing Rule

Every retained section in the paper must strengthen the same main line:

1. why deployment-oriented interpretation matters
2. what controlled evidence supports it
3. where the current boundary still fails

If material is interesting but does not strengthen that line, it belongs in the report, not in the paper.

## 5. Recommended Report Structure Emphasis

Recommended emphasis order for the report:

1. introduction and project framing
2. architecture and method
3. primary offline results
4. calibration / operating-point results
5. deployment / runtime results
6. strengthening results
7. discussion
8. limitations
9. validation / testing / audit
10. appendices / supporting material

Recommended report editing actions:

1. keep the full result breadth, but make the results chapter more synthesised and less fragmented
2. keep retraining strengthening in the main report because it improves the defended project story
3. keep audit/testing, but compress it to what helps the marker trust the system
4. move detailed process and supporting material downward instead of deleting it entirely

## 6. Recommended Paper Structure Emphasis

Recommended emphasis order for the paper:

1. abstract
2. deployment-oriented problem framing
3. locked protocol and system framing
4. strongest offline comparative result
5. alert-policy / operating-point contribution
6. bounded runtime and replay evidence
7. cross-dataset limitation boundary
8. discussion and limitations

Recommended paper editing actions:

1. make the introduction shorter and sharper
2. keep only one central contribution line
3. treat model comparison as support, not as the whole identity of the paper
4. keep strengthening results, but describe them as bounded improvements rather than a new main claim
5. cut or compress sections that mainly prove engineering effort rather than scientific interpretation

## 7. Concrete Content Split

### Content that belongs mainly in the Report

1. full stack architecture walkthrough
2. broader explanation of frontend/backend responsibilities
3. detailed audit and repository-cleanup narrative
4. extended testing matrix
5. broader system-validation discussion
6. richer explanation of persistence, delivery, and review surfaces
7. exploratory track context such as `MUVIM`

### Content that belongs mainly in the Paper

1. concise system framing
2. strongest frozen offline results
3. calibration-aware alerting as a methodological contribution
4. bounded replay/runtime interpretation
5. cross-dataset failure boundary
6. bounded retraining improvement as a deployment-strengthening note

## 8. Current Best Allocation of Effort

Given the current state of the project:

1. raise the full report toward high-mark quality first
2. keep the paper focused and disciplined, but secondary
3. do not keep expanding project work unless a new evidence gap becomes clearly blocking

## 9. Immediate Next-Step Rule

From this point onward:

1. if a change improves both report and paper, do it
2. if a change mainly improves the report, prioritise it
3. if a change mainly improves the paper but costs major report time, defer it
4. if a change adds breadth but weakens focus, keep it in the report and remove it from the paper
