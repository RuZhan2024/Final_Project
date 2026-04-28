Date: 2026-04-28  
Purpose: provide a final direct mapping from major high-value project work to its current treatment in the full report.

# Value-to-Report Mapping Final Check

## 1. Use Rule

This note answers one practical question:

`Have the major high-value project contributions actually been carried into the full report?`

For each value area, this note records:

1. whether it is already in the main report
2. where it appears
3. whether the treatment is strong enough
4. whether any extra work is still worth doing

## 2. Final Mapping

| Value area | In main report? | Main location(s) | Coverage judgment | Extra action needed? |
| --- | --- | --- | --- | --- |
| End-to-end monitoring-system framing | Yes | introduction, aims, requirements, conclusion | strong | no |
| Locked offline TCN-vs-GCN comparison | Yes | model/training chapters, offline results, `RQ1` discussion | strong | no |
| Data protocol, labels, splits, FPS, and temporal contract | Yes | dataset roles, preprocessing, temporal window contract, evaluation policy | strong | no |
| Calibration and operating-point fitting | Yes | literature framing, calibration/alert-policy chapter, results, `RQ2` discussion | strong | no |
| Multi-window alert policy (`k/n`, cooldown, persistence) | Yes | calibration/alert-policy chapter, runtime interpretation, `RQ2` discussion | strong | no |
| Frontend/backend responsibility split | Yes | architecture, implementation, report-relevant frontend/backend work | strong | no |
| Replay-vs-realtime semantic separation | Yes | architecture, runtime validation, appendices, audit findings | strong | no |
| Persisted event / dashboard / event-history path | Yes | system architecture, backend implementation, runtime evidence, validation chapter | strong | no |
| Telegram-first delivery path | Yes | notification architecture, deployment/runtime results, appendices | strong and bounded | no |
| Runtime-path debugging and deployment recovery | Yes | report-relevant frontend/backend work, deployment/runtime results, audit sections | now strong | no major action |
| Online replay repair sequence (`gate fix -> motion-support fix -> online refit -> retraining`) | Yes | deployment/runtime results | good and now explicit | optional tiny polish only |
| Cross-dataset transfer boundary | Yes | cross-dataset results, discussion, limitations | strong | no |
| Targeted failure-mode analysis | Yes | cross-dataset results and linked notes/figures | strong | no |
| Targeted retraining strengthening (`Candidate A/D`) | Yes | ML pipeline strategy section, deployment/runtime results, limitations | strong | no |
| Hard-negative optimisation logic | Yes | ML pipeline strategy section | adequate | optional minor sharpening only |
| Misclassified custom-clip/error pattern | Yes | deployment/runtime results | now good | no major action |
| Kitchen/camera-geometry deployment insight | Yes | deployment/runtime results, limitations | strong | no |
| `kitchen_front_2` persistence-failure explanation | Yes | deployment/runtime results, limitations | strong | no |
| Evidence hierarchy / claim discipline | Yes | intro, scope, evaluation policy, validation matrix, discussion, appendices | strong | no |
| Audit / code review / freeze-state defensibility | Yes | dedicated chapters and appendices | strong | no |
| Canonical testing structure | Yes | system validation/testing/audit chapter | strong | no |
| Environment-sensitive validation awareness | Yes | canonical tests, reproducibility appendix | strong | no |
| Reproducibility and defended snapshot discipline | Yes | training protocol, freeze state, appendices A/D/E | strong | no |
| `MUVIM` exploratory breadth | Yes | scope, dataset roles, secondary MUVIM track, appendices | appropriate and bounded | no |
| Stability / multi-seed reliability | Yes | offline results and appendix/supporting figures | adequate | optional one-sentence emphasis only |
| Packaging / runnable submission engineering | Partly | reproducibility appendix, final evaluation, readiness/testing references | present but not foregrounded | optional if assignment-facing emphasis desired |
| Configuration architecture as a managed asset | Partly | ops/profile discussion, appendices A/C | present implicitly | optional only |

## 3. What This Means

### High-confidence conclusion

The major high-value work of the project is already represented in the report.

The report is not omitting any large, high-value contribution category from:

1. ML work
2. frontend/backend system work
3. runtime/deployment work
4. evidence-control work
5. reproducibility and validation work

### What remains imperfect

The remaining gaps are not major omissions. They are mainly:

1. a few areas that are present but not especially foregrounded
2. a few engineering-value threads that could be named more explicitly if desired

The two most optional areas are:

1. packaging / runnable-submission engineering
2. configuration architecture as a managed technical asset

These are useful, but they are not currently serious weaknesses in the report.

## 4. Final Judgment

If the concern is:

`Did we do a lot of important work that the report is now silently ignoring?`

the answer is:

`No, not at the major-value level.`

The report now captures the project’s main high-value work well enough that the main remaining task should be polish, not hunting for missing contribution categories.
