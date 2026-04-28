Date: 2026-04-28  
Purpose: audit whether the high-value technical work of the project is already represented strongly enough in the full report, and identify any remaining under-covered value areas.

# Report Value Coverage Audit

## 1. Audit Rule

This audit does not ask whether a topic is merely mentioned.

It asks whether a topic is:

1. clearly visible in the report
2. connected to defended evidence
3. interpreted at the right level
4. weighted appropriately for a high-mark full report

Each value area is therefore marked as one of:

1. **Strongly covered**
2. **Partially covered**
3. **Still underdeveloped**

## 2. Strongly Covered Value Areas

### A. Project Framing as an End-to-End Monitoring System

Status:

1. **Strongly covered**

Why:

1. the introduction, aims, requirements, and conclusion all state clearly that the project is more than a classifier
2. model, policy, runtime, persistence, and delivery are repeatedly treated as one integrated artifact
3. the report does not drift back into pure benchmark language

Where it is already strong:

1. Introduction and aims
2. requirements table
3. architecture chapter
4. conclusion

### B. Evidence Hierarchy and Claim Discipline

Status:

1. **Strongly covered**

Why:

1. the report explicitly separates offline evidence, cross-dataset evidence, replay/runtime evidence, live evidence, and delivery evidence
2. the results chapter and validation matrix reinforce that separation
3. the report repeatedly states what replay and live evidence cannot prove

Where it is already strong:

1. introduction and scope
2. evaluation policy
3. runtime-validation matrix
4. discussion and conclusion

### C. Data Protocol, Temporal Contract, and Reproducibility Discipline

Status:

1. **Strongly covered**

Why:

1. dataset roles are clear and well differentiated
2. `FPS`, `W=48`, and `S=12` are explained as methodological constraints, not loose defaults
3. labels, spans, splits, and window metadata are treated as protocol-bearing artifacts

Where it is already strong:

1. dataset roles
2. preprocessing
3. temporal window contract
4. evaluation policy
5. reproducibility appendix

### D. Locked Offline Comparative ML Story

Status:

1. **Strongly covered**

Why:

1. the offline comparison is clearly identified as the formal answer to `RQ1`
2. the TCN-vs-GCN story is controlled, multi-seed, and directionally interpreted rather than overclaimed
3. the report clearly distinguishes primary `CAUCAFall` evidence from secondary `LE2i` evidence

Where it is already strong:

1. model/training chapters
2. offline results section
3. discussion of `RQ1`

### E. Calibration / Operating-Point / Policy Contribution

Status:

1. **Strongly covered**

Why:

1. the report treats calibration and operating-point fitting as methodological work, not cosmetic thresholding
2. `OP-1 / OP-2 / OP-3` are explained as deployable policy intents
3. temporal policy is connected to reviewable alert behaviour rather than raw scores alone

Where it is already strong:

1. literature framing
2. calibration and alert-policy chapter
3. results
4. `RQ2` answer

### F. Frontend / Backend System Contribution

Status:

1. **Strongly covered**

Why:

1. the report now explicitly promotes monitor-path responsibility split, replay/realtime separation, persisted-event path, delivery path, and runtime recovery work
2. frontend and backend are described as evidence-bearing system layers rather than implementation trivia
3. the full-stack architecture clearly supports the report’s main framing

Where it is already strong:

1. system architecture
2. frontend implementation
3. backend implementation
4. report-relevant frontend/backend work section

### G. Audit, Code Review, Freeze, and Defended Snapshot Control

Status:

1. **Strongly covered**

Why:

1. audit and review are treated as technical risk-control work rather than housekeeping
2. freeze-state and artifact selection are tied directly to evidence defensibility
3. appendices reinforce the main-text claims with concrete review categories and reproducibility maps

Where it is already strong:

1. audit/code-review sections
2. freeze and handoff state
3. reproducibility appendix
4. issue-to-fix and verification tables

### H. Runtime Validation and Bounded Deployment Interpretation

Status:

1. **Strongly covered**

Why:

1. replay/runtime evidence is rich, bounded, and properly caveated
2. runtime surfaces are narrated as policy-sensitive trade-offs, not broad success
3. live evidence and delivery evidence are present but not allowed to inflate the benchmark claim

Where it is already strong:

1. deployment and runtime results
2. runtime-validation matrix
3. discussion of `RQ3`
4. limitations

### I. Cross-Dataset Limitation Boundary

Status:

1. **Strongly covered**

Why:

1. transfer asymmetry is now clearly explained
2. the new failure-mode interpretation makes the boundary more concrete
3. the report avoids turning this into a disguised success story

Where it is already strong:

1. cross-dataset results
2. targeted failure analysis references
3. discussion and limitations

### J. Kitchen / Camera-Geometry Deployment Insight

Status:

1. **Strongly covered**

Why:

1. the report now explains the kitchen weakness as a geometry and observability problem rather than as a vague bad subset
2. the distinction between general kitchen misses and `kitchen_front_2` is useful and specific
3. this is one of the more publication-like system insights in the report

Where it is already strong:

1. deployment/runtime results
2. limitations

## 3. Partially Covered Value Areas

### K. Misclassified-Clip Analysis as a Compact Main-Text Result

Status:

1. **Partially covered**

Why:

1. the report now interprets corridor vs kitchen behaviour and explains major mechanisms
2. however, the clip-level improvement story is not yet summarised as compactly as it could be
3. there is still room for one concise “what retraining fixed vs what remained broken” sentence or mini-table

What is still missing:

1. a short main-text synthesis of:
   - persistent `FP`
   - persistent `FN`
   - one corridor clip recovered by strengthening
   - kitchen-heavy residual misses

Priority:

1. useful, but not critical

### L. Hard-Negative / Training-Optimisation Mechanism

Status:

1. **Partially covered**

Why:

1. the new ML-pipeline section explains that selected optimisation work mattered
2. but the report still does not foreground hard-negative mining as clearly as it could
3. the reader can understand that retraining happened, but the “why this continuation was technically plausible” story could still be sharper

What is still missing:

1. one or two direct sentences linking hard-negative continuation to the observed missed-fall weakness

Priority:

1. moderate

### M. Stability / Reliability Track

Status:

1. **Partially covered**

Why:

1. the report clearly uses the frozen multi-seed summaries
2. but the stability work is still more visible as supporting evidence than as an explicit report-strengthening theme
3. this is acceptable, but could be made slightly more explicit in one sentence

What is still missing:

1. a sharper statement that multi-seed stability is part of why the offline claim is defended

Priority:

1. low to moderate

### N. Deployment Optimisation Sequence as One Named Analytical Thread

Status:

1. **Partially covered**

Why:

1. the report clearly discusses gate fixes, motion-support fixes, online refit, and runtime-path recovery in pieces
2. however, these pieces are still somewhat distributed across implementation, runtime results, and audit chapters
3. a reader can reconstruct the sequence, but it is not yet as unified as it could be

What is still missing:

1. one compact paragraph naming the sequence explicitly:
   - baseline replay failure
   - gate fix
   - motion-support fix
   - online refit
   - targeted retraining

Priority:

1. moderate and high-yield if polished

## 4. Still Underdeveloped Value Areas

### O. Submission / Runability / Packaging as a Report-Level Engineering Achievement

Status:

1. **Still underdeveloped**

Why:

1. the report does mention reproducibility and bootstrap paths in appendix material
2. but it does not yet present runability and markable-snapshot discipline as a meaningful engineering outcome of the project
3. for this assignment context, that work has real value

What could be added:

1. a short sentence or short paragraph stating that the project was hardened into a reviewer-runnable software artifact through:
   - bootstrap path
   - canonical test entrypoints
   - final defended snapshot discipline

Priority:

1. useful for assignment framing
2. not essential for the paper-facing story

### P. Configuration Architecture as a Managed Technical Asset

Status:

1. **Still underdeveloped**

Why:

1. the report talks about fitted profiles and artifact roots
2. but it does not yet explicitly explain the value of the config architecture itself:
   - active runtime profiles
   - delivery/repro profiles
   - supporting/diagnostic profiles
   - archived experiment families

What could be added:

1. a brief sentence in architecture or reproducibility discussion noting that the project became configuration-driven rather than ad hoc

Priority:

1. low to moderate

## 5. Overall Judgment

The report already covers most of the genuinely high-value work well.

The strongest covered areas are:

1. end-to-end system framing
2. evidence hierarchy
3. protocol discipline
4. offline comparison
5. policy/ops contribution
6. frontend/backend system work
7. audit/freeze defensibility
8. runtime bounded interpretation
9. cross-dataset limitation analysis
10. kitchen/camera deployment insight

The main remaining opportunities are not major missing categories. They are mostly opportunities to make a few already-present themes more compact and explicit:

1. compress clip-level error-analysis gain into one sharp paragraph or mini-table
2. tighten the hard-negative / strengthening rationale
3. state the deployment-optimisation sequence more explicitly as one named analytical thread
4. optionally mention runability / defended snapshot hardening as a practical engineering outcome

## 6. Bottom Line

There is no large hidden value area that the report has completely missed.

At this stage, the report’s problem is not lack of important material.

It is mainly a question of:

1. whether a few partially covered value areas should be tightened
2. whether some engineering-value threads should be made more explicit
3. whether the report should stop expanding and focus on polishing the strongest existing structure

The current full report already captures the majority of the project’s genuinely high-value technical work.
