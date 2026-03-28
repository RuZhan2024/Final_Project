# Report-Relevant Change Summary

Date: 2026-03-28

Purpose:
Keep a concise record of engineering changes that should be reflected in the dissertation/report, without turning the report into a commit log.

## Changes That Must Be Reflected In The Report

### 1. Replay And Realtime Must Be Described Separately

- `replay` and `realtime` do not currently share the same operational characteristics.
- `replay` has been used as the primary controlled demonstration path.
- `realtime` remains a separate validation path and should not be implicitly claimed from replay-only evidence.

Report implication:
- Claims about live/on-device monitoring should be supported by dedicated local realtime validation.
- Replay results should be presented as controlled demo/system validation evidence, not as a substitute for full realtime validation.

### 2. Replay Event Persistence Was Extended For Demonstration Validity

- Replay `fall` and replay `uncertain` events were changed to persist into the event history path.
- `realtime` persistence remained more conservative.
- Replay and realtime events are distinguished by source metadata.

Report implication:
- This should be described as a system-level persistence and auditability improvement for demonstration and review workflows.
- It should not be described as a model improvement.

### 3. Cloud Replay And Local Replay Were Not Initially Equivalent

- The deployed frontend and local frontend did not initially produce the same number of `predict_window` calls for the same replay clip.
- This caused the cloud system to receive fewer effective windows, which in turn reduced the chance of progressing from `uncertain` to `fall`.
- The main issue was not only backend thresholding; it was also frontend replay throughput and browser-side pose extraction stability.

Report implication:
- Any discussion of cloud deployment limitations should mention that replay accuracy was sensitive to frontend window production rate and browser-side pose pipeline behaviour.
- Cloud behaviour should not be overgeneralized as model weakness.

### 4. Replay-Specific Engineering Stabilization Was Added

- Replay startup ordering was changed so detector preparation completes before playback begins.
- Replay monitor lifecycle was refactored to reduce MediaPipe/WebGL/WASM teardown-recreate races.
- Replay pacing was adjusted to preserve prediction windows under slower backend response conditions.
- Replay requests can now ask for a compact backend response to reduce per-request overhead.

Report implication:
- These are deployment/runtime stabilization changes.
- They should be described as engineering improvements to the replay delivery path, not as evidence that the classifier itself improved.

### 5. Replay-Only Decision Tuning Was Explored, Then Rolled Back

- A replay-only relaxed alert policy was explored to make a fixed clip pack perform better.
- That tuning was useful for diagnosis and local delivery experiments.
- It was later reverted on the deployment branch to restore the `main`-style backend alert policy baseline (`k=2`, `n=3`).

Report implication:
- This must be written carefully.
- Replay-specific policy tuning on a fixed clip pack should be described, if used at all, as demo/deployment calibration rather than unseen-test evaluation.
- The deployed branch currently uses the main-style replay backend decision baseline, not the earlier replay-relaxed policy.

### 6. Frontend Pose Pipeline Is The Main Remaining Realtime Risk

- The largest remaining practical risk is browser-side pose extraction stability, not backend routing or database persistence.
- Skeleton loss, startup latency, and window underproduction have been more important than raw backend model execution for observed deployment issues.

Report implication:
- Limitations and future work should explicitly identify frontend pose robustness and realtime validation as the main remaining engineering concerns.

## Changes That Usually Do Not Need Detailed Report Coverage

- Toast UI replacements for monitor errors.
- Spacing and control-label tweaks.
- Minor replay dropdown wording changes.
- Internal refactors that do not change system behaviour or evaluation interpretation.

These can stay in code history unless they materially affect the demonstration or validation story.

## Recommended Report Language

Use wording along these lines:

- The replay workflow was treated as a controlled demonstration and deployment-validation path.
- Replay-specific runtime stabilization was introduced to improve frontend pose extraction consistency and reduce lost prediction windows in browser-based execution.
- Claims about realtime on-device monitoring should rely on dedicated local live validation rather than replay evidence alone.
- Cloud deployment behaviour was influenced by frontend/browser runtime conditions and request throughput, not only by backend classifier thresholds.

## Items To Keep Linked When Writing The Report

- Local on-device validation evidence
- Replay demonstration evidence
- Deployment limitations and engineering mitigations
- Event persistence/auditability behaviour
- Separation between model evaluation claims and deployment/demo calibration
