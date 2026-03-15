# Online Frontend Smoke Checklist

This checklist is for verifying that the latest online operating points are actually active in the frontend monitor flow.

Use it after changing:

- `configs/ops/*.yaml`
- `server/routes/monitor.py`
- `apps/src/pages/monitor/*`

## Preconditions

- Restart the backend after updating online configs or monitor logic.
- Hard refresh the frontend.
- Open the Monitor page.
- Confirm the backend spec list is not empty.

## Expected Runtime Behavior

The frontend monitor should use backend-driven settings:

- dataset selection
- model selection
- operating point selection

The prediction request should be sent through:

- `/api/monitor/predict_window`

with:

- `dataset_code`
- `mode`
- `op_code`

So the effective online behavior must come from the current `configs/ops/*.yaml` files, not from frontend-only threshold logic.

## Quick Checks Before Testing Videos

1. Open Monitor.
2. Confirm available datasets include:
   - `caucafall`
   - `le2i`
3. Confirm available models include:
   - `TCN`
   - `GCN`
4. Confirm OP choices include:
   - `OP1`
   - `OP2`
   - `OP3`
5. Switch between OPs and confirm the UI selection updates correctly.

## Core Smoke Matrix

Run at least these six checks:

1. `caucafall + TCN + OP1`
2. `caucafall + TCN + OP2`
3. `caucafall + TCN + OP3`
4. `caucafall + GCN + OP1`
5. `le2i + TCN + OP2`
6. `le2i + GCN + OP2`

If time allows, expand to the full matrix:

- `caucafall + TCN + OP1/OP2/OP3`
- `caucafall + GCN + OP1/OP2/OP3`
- `le2i + TCN + OP1/OP2/OP3`
- `le2i + GCN + OP1/OP2/OP3`

## What To Observe

For each combination, verify:

1. The selected dataset/model/OP shown in the UI matches what you chose.
2. `Current Prediction` changes when you switch operating points.
3. `p_fall` and `triage_state` move in the same direction as the OP aggressiveness.
4. Timeline markers match the visible triage state.
5. Switching videos does not inherit the previous video state.
6. Replay/video mode still requires `2` consecutive fall windows before showing `fall`.

## Expected OP Ordering

The intended semantic ordering is:

- `OP1`: more aggressive, higher recall preference
- `OP2`: balanced point
- `OP3`: more conservative, lower false-alert preference

On some datasets the replay surface is very flat, so the measured metrics can be identical even when thresholds differ. In that case, focus on:

- trigger timing
- trigger difficulty
- how quickly the triage flips to `fall`

## High-Priority Acceptance Path

If you only need a minimal acceptance pass, use:

### `caucafall + TCN + OP2`

This is the main delivery path.

Use the 24 custom corridor/kitchen videos and verify:

- fall videos trigger correctly
- ADL videos do not trigger
- video mode still requires `2` consecutive fall windows

Reference artifact:

- `artifacts/fall_test_eval_20260315_online_reverify_20260315/tcn_op2_pose_raw_frontend_emulation_final_k2_v2.json`

Expected summary:

- `TP=12`
- `TN=12`
- `FP=0`
- `FN=0`

## If A Check Fails

Use this triage order:

1. Confirm backend was restarted.
2. Confirm frontend was hard refreshed.
3. Confirm the selected `dataset/model/op` in UI is correct.
4. Confirm the issue is reproducible after starting a fresh session.
5. Compare behavior against:
   - `docs/reports/runbooks/ONLINE_OPS_PROFILE_MATRIX.md`

## Related References

- `docs/reports/runbooks/ONLINE_OPS_PROFILE_MATRIX.md`
- `configs/ops/tcn_caucafall.yaml`
- `configs/ops/gcn_caucafall.yaml`
- `configs/ops/tcn_le2i.yaml`
- `configs/ops/gcn_le2i.yaml`
