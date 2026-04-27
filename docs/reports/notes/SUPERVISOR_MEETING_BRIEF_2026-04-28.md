Date: 2026-04-28  
Purpose: concise supervisor-facing discussion brief for the current report/paper branch.

# Supervisor Meeting Brief

## 1. Current Project Position

The project is strongest as an **end-to-end, deployment-oriented pose-based fall-detection and monitoring system**, not as a pure architecture-comparison paper.

The current defended structure is:

1. `RQ1`: bounded offline TCN-versus-custom-GCN comparison under a locked protocol
2. `RQ2`: calibration and operating-point fitting as a real alert-policy layer
3. `RQ3`: bounded replay/live deployment evidence with explicit runtime limitations

## 2. Strongest Defended Claims

1. The repository now supports a coherent end-to-end monitoring path:
   - browser-side pose extraction
   - backend temporal inference
   - fitted alert profiles
   - persisted event history
   - Telegram-first caregiver delivery
2. Under the locked offline protocol, the TCN trends stronger than the matched custom GCN on the main `CAUCAFall` line.
3. Cross-dataset transfer is asymmetric and bounded:
   - `CAUCAFall -> LE2i` remains a strong limitation boundary
   - TCN collapses toward missed falls
   - GCN recovers recall only by becoming much less selective
4. Runtime/deployment evidence is real, but bounded rather than uniform:
   - replay and live evidence support practical feasibility in controlled conditions
   - they do not support broad field closure

## 3. New Strengthening Work Since the Earlier Draft

We carried out a targeted project-strengthening retraining pass on the strongest deployment-facing line:

- `CAUCAFall + TCN`

Two continuation candidates were evaluated:

- `Candidate A`
- `Candidate D`

Main bounded-runtime result:

1. canonical four-folder replay improved from `13/24` to `16/24`
2. historical locked 24-clip replay improved from `15/24` to `16/24`
3. false positives did not increase on those surfaces

Current interpretation:

1. this is a real improvement
2. the gain is modest rather than decisive
3. both candidates still rely on confirm-disabled `fit_ops` fallback
4. `Candidate A` is the lead strengthening result
5. `Candidate D` supports the same direction but does not outperform `A`

## 4. Current Boundaries

1. The paper does **not** claim solved deployment performance.
2. The paper does **not** claim broad cross-dataset robustness.
3. The replay/runtime line remains bounded system evidence rather than formal unseen-test evidence.
4. The retraining improvement is useful, but it does not eliminate runtime fragility or the `kitchen` miss pattern.

## 5. Questions to Ask

1. Is the project best framed primarily as:
   - an end-to-end system paper with model comparison support
   - or a model-comparison paper with deployment support?
2. Is the current paper contribution focused enough, or should it be narrowed further around the deployment-oriented systems claim?
3. Is the modest retraining improvement worth presenting in the main paper narrative, or should it be kept as supporting strengthening evidence only?
4. Given the remaining time, should effort go mainly into:
   - final report quality
   - paper focus and structure
   - or another round of project/evidence strengthening?

## 6. Recommended Verbal Summary

Suggested one-minute framing:

`Our project is strongest as a deployment-oriented pose-based fall-detection system rather than as a pure architecture paper. We now have a locked offline comparison, a calibrated alert-policy layer, and bounded replay/live evidence. We also ran a strengthening retraining pass on the main CAUCAFall TCN line, which improved the defended custom replay surfaces from 13/24 or 15/24 to 16/24 without increasing false positives, although the gain is still modest and not enough to claim deployment solved. We want your view on whether the paper should now focus even more strongly on the systems contribution, and whether the current strengthening evidence belongs in the main paper narrative or only as supporting material.`  
