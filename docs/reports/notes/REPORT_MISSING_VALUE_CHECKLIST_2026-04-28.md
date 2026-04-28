Date: 2026-04-28  
Purpose: identify high-value report content that is still worth adding or strengthening, even if the underlying project work is largely complete.

# Report Missing-Value Checklist

## 1. Use Rule

This checklist is not a list of everything that could still be written.

It is a list of content that has **high value for the full report** because it either:

1. improves defended interpretation
2. raises the maturity of discussion and limitations
3. turns existing project work into stronger report evidence
4. increases the marker’s confidence that the team understands what the project actually proved

## 2. Highest-Value Missing or Underdeveloped Content

### A. Deployment Optimisation Analysis

Status:

- partially present in scattered artifacts
- not yet fully synthesised as one report-level analytical thread

Why it matters:

1. this is one of the most publication-like parts of the whole project
2. it shows that deployment performance was improved by understanding the system path, not by random tuning
3. it explains why replay/runtime metrics changed over time

What it should contain:

1. baseline failure pattern
2. direct-window / gate fix
3. motion-support fix
4. online operating-point refit
5. deploy-aware rule design
6. retraining strengthening
7. why each stage mattered
8. what remained unresolved afterwards

Recommended location:

- main body discussion
- or the latter half of deployment/runtime results

Priority:

- **main text**

### B. Kitchen Subset Explanation

Status:

- now partly written into the report and paper
- still worth treating as an explicit runtime limitation theme

Why it matters:

1. it converts a weak-looking subset into a meaningful deployment lesson
2. it shows that the team understands camera geometry and pose observability as first-class variables
3. it turns “why did this stay bad?” into an evidence-backed systems explanation

What it should contain:

1. corridor camera geometry
2. kitchen camera geometry
3. partial-body framing risk at close range
4. skeleton instability
5. geometry shift relative to training scenes
6. special explanation for `kitchen_front_2` as a rapid forward collapse with insufficient persistence under `OP-2`

Recommended location:

- discussion
- limitations

Priority:

- **main text**

### C. Misclassified Clip Analysis

Status:

- the misclassified clips are now identifiable
- but the report does not yet use that information in a compact, high-value way

Why it matters:

1. it makes bounded runtime results much more interpretable
2. it shows exactly what retraining improved and what it did not
3. it supports a stronger error-analysis subsection

What it should contain:

1. persistent false negatives
2. persistent false positives
3. clips corrected by retraining
4. clips not corrected by retraining
5. grouping by mechanism rather than by filename only

Recommended location:

- short discussion paragraph in the main text
- optional detailed table in appendix or note

Priority:

- **main text + supporting appendix/note**

### D. Stronger Evidence-Hierarchy Paragraph

Status:

- present implicitly
- not yet fully exploited as a report-strengthening device

Why it matters:

1. it shows academic control
2. it prevents marker confusion about what each result type does and does not prove
3. it helps the project feel mature rather than over-claimed

What it should contain:

1. offline frozen evidence
2. bounded replay/runtime evidence
3. live/demo evidence
4. what each layer supports
5. what each layer cannot support

Recommended location:

- introduction to results
- discussion

Priority:

- **main text**

## 3. Medium-Value Additions

### E. Retraining Strengthening Rationale

Status:

- the experiments exist
- the selection logic can still be explained more explicitly

Why it matters:

1. it demonstrates targeted technical decision-making
2. it explains why `Candidate A` is the lead strengthening result
3. it clarifies why `Candidate D` is corroboration rather than the new winner

Recommended location:

- discussion
- or a short strengthening paragraph in results

Priority:

- **main text if short; appendix if long**

### F. Policy-Design Insight

Status:

- partly present
- could still be stated more explicitly as a systems lesson

Why it matters:

1. it shows why multi-window persistence is more realistic than single-window triggering
2. it strengthens the project’s deployment-oriented contribution

What it should contain:

1. why `k/n` logic is more appropriate for alert-worthy events
2. why `OP-1` and `OP-2` reflect real operational trade-offs
3. why fast falls can still be lost under stricter persistence logic

Recommended location:

- calibration/alert-policy discussion
- runtime discussion

Priority:

- **main text**

### G. “What Was Not Promoted” Discipline

Status:

- present in practice
- not yet clearly framed as a report-strengthening virtue

Why it matters:

1. it shows evidence discipline
2. it tells the marker that the team did not simply choose the biggest-looking number

What it should contain:

1. old `24/24` delivery-only profile is historical, not current defended evidence
2. replay/runtime lines are bounded, not generalisation evidence
3. exploratory tracks were not all promoted into the final claim set

Recommended location:

- discussion
- limitations

Priority:

- **main text**

## 4. Low-Value or Low-Priority Additions

These are not forbidden, but they should not consume major writing time now.

### H. Large New Experiment Expansion

Reason to de-prioritise:

1. the report gains more from stronger interpretation than from more breadth
2. current evidence already supports a strong defended story

Priority:

- **defer**

### I. Long Engineering Diary Detail

Reason to de-prioritise:

1. it weakens the report’s main line
2. it belongs in working notes rather than in the main defended narrative

Priority:

- **exclude or compress heavily**

## 5. Recommended Main-Text Additions From This Point

If only four remaining additions are made to the full report, the best four are:

1. deployment optimisation analysis
2. kitchen subset / camera-geometry explanation
3. misclassified-clip / error-analysis synthesis
4. stronger evidence-hierarchy paragraph

## 6. Appendix or Supporting-Only Material

Good candidates for appendix/supporting notes rather than prominent main-text space:

1. detailed filename-level clip tables
2. full tuning chains and intermediate reverify snapshots
3. extended exploratory-track detail
4. additional diagnostic figures that support but do not drive the main conclusions

## 7. Immediate Editing Rule

From here onward, prefer:

1. adding interpretation to existing evidence
2. strengthening discussion and limitations
3. making the report more coherent and defensible

Avoid:

1. adding breadth that is not tied to a defended conclusion
2. turning the report into an experiment log
