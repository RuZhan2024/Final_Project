# Final Demo Walkthrough

Use this exact sequence during final demo/exam.

## 0) Pre-check

1. Start backend and frontend.
2. Confirm API reachable:
   - `curl -s http://127.0.0.1:8000/api/health`
3. Apply locked profile:
   - `TCN + caucafall + OP-2`
4. Run:
   - `bash tools/run_deployment_lock_validation.sh`

## 1) Settings page checks

1. Dataset selected: `CAUCAFall`.
2. Model selected: `TCN`.
3. Operating point selected: `OP-2`.
4. MC Dropout enabled (`M=10`, `M_confirm=25`).
5. Save success toast appears.

## 2) Monitor replay checks

1. Non-fall clip replay:
   - Expected: no sustained fall event spam.
2. Fall clip replay:
   - Expected: one clear fall event in timeline.
3. Optional: second fall clip to show repeatability.

## 3) Dashboard / Events checks

1. Events list loads without API errors.
2. New replay event appears in timeline/events table.
3. Summary endpoint returns values (no 500).

## 4) Evidence export

1. Ensure report exists:
   - `artifacts/reports/deployment_lock_validation.md`
2. Ensure release snapshot exists:
   - `artifacts/reports/release_snapshot.md`
3. Run bundle check:
   - `python tools/check_release_bundle.py`

## 5) Pass criteria

- Auto checks in deployment lock validation are all green:
  - health yes, predict endpoint yes, TCN, caucafall, OP-2.
- Manual replay checks:
  - non-fall pass, fall pass.
- `tools/check_release_bundle.py` summary passes all checks.
