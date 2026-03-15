# Online Ops Profile Matrix

This document records the current default online operating points for:

- `caucafall + TCN`
- `caucafall + GCN`
- `le2i + TCN`
- `le2i + GCN`

It is intended as a concise reference for frontend/backend integration, demo setup, and viva discussion.

## OP Semantics

The three operating points are now treated as distinct deployment profiles:

- `OP1`: higher recall / earlier triggering
- `OP2`: balanced point, used as the default online deployment profile
- `OP3`: more conservative / lower false-alert preference

Important:

- On some validation sets, the score surface is very flat.
- That means different thresholds can produce the same measured `precision / recall / F1` on the available replay set.
- Even in those cases, the `OP1 -> OP2 -> OP3` ordering is still kept meaningful by threshold aggressiveness.

## Current Default Profiles

### `caucafall + TCN`

Source:

- `configs/ops/tcn_caucafall.yaml`

Profiles:

- `OP1`
  - `tau_high=0.20`
  - `tau_low=0.1560`
  - `ema_alpha=0.0`
  - `k=1`
  - `n=2`
  - intended role: highest recall / fastest trigger
- `OP2`
  - `tau_high=0.92`
  - `tau_low=0.42`
  - `ema_alpha=0.0`
  - `k=2`
  - `n=2`
  - intended role: balanced delivery profile
  - extra online rules:
    - `delivery_gate`
    - `uncertain_promote` for replay/video path
- `OP3`
  - `tau_high=0.94`
  - `tau_low=0.7332`
  - `ema_alpha=0.0`
  - `k=1`
  - `n=2`
  - intended role: most conservative zero-FP point

Embedded metrics:

- `OP1`: `P=1.0 R=0.6 F1=0.75 FA24h=0.0`
- `OP2`: `P=1.0 R=1.0 F1=1.0 FA24h=0.0`
- `OP3`: `P=1.0 R=0.6 F1=0.75 FA24h=0.0`

Notes:

- `OP2` is the current production-style online profile for the four custom corridor/kitchen videos.
- It includes targeted online logic that is not meant to represent a pure offline sweep point.

### `caucafall + GCN`

Source:

- `configs/ops/gcn_caucafall.yaml`

Profiles:

- `OP1`
  - `tau_high=0.26`
  - `tau_low=0.2028`
  - `ema_alpha=0.1`
  - `k=2`
  - `n=2`
  - intended role: more permissive / higher recall side
- `OP2`
  - `tau_high=0.40`
  - `tau_low=0.3120`
  - `ema_alpha=0.0`
  - `k=1`
  - `n=2`
  - intended role: balanced point
- `OP3`
  - `tau_high=0.52`
  - `tau_low=0.4056`
  - `ema_alpha=0.0`
  - `k=1`
  - `n=2`
  - intended role: most conservative point

Embedded metrics:

- `OP1`: `P=1.0 R=0.6 F1=0.75`
- `OP2`: `P=1.0 R=0.6 F1=0.75`
- `OP3`: `P=1.0 R=0.2 F1=0.3333`

Notes:

- On the available online replay set, all useful `caucafall GCN` candidates were already zero-FP.
- The difference between `OP1` and `OP2` is mostly threshold aggressiveness rather than measured F1 separation.

### `le2i + TCN`

Source:

- `configs/ops/tcn_le2i.yaml`

Profiles:

- `OP1`
  - `tau_high=0.32`
  - `tau_low=0.2496`
  - `ema_alpha=0.0`
  - `k=1`
  - `n=2`
- `OP2`
  - `tau_high=0.48`
  - `tau_low=0.3744`
  - `ema_alpha=0.0`
  - `k=1`
  - `n=2`
- `OP3`
  - `tau_high=0.62`
  - `tau_low=0.4836`
  - `ema_alpha=0.0`
  - `k=1`
  - `n=2`

Shared online guard:

- `allow_low_motion_high_conf_bypass=true`
- `low_motion_high_conf_k=2`
- `low_motion_high_conf_max_lying=0.30`

Embedded metrics:

- `OP1`: `P=1.0 R=1.0 F1=1.0`
- `OP2`: `P=1.0 R=1.0 F1=1.0`
- `OP3`: `P=1.0 R=1.0 F1=1.0`

Notes:

- The online fit surface for `le2i TCN` is very flat.
- These three profiles differ by threshold conservativeness, but not by the replay metrics currently embedded in the YAML.

### `le2i + GCN`

Source:

- `configs/ops/gcn_le2i.yaml`

Profiles:

- `OP1`
  - `tau_high=0.32`
  - `tau_low=0.2496`
  - `ema_alpha=0.0`
  - `k=1`
  - `n=2`
- `OP2`
  - `tau_high=0.50`
  - `tau_low=0.3900`
  - `ema_alpha=0.0`
  - `k=1`
  - `n=2`
- `OP3`
  - `tau_high=0.70`
  - `tau_low=0.5460`
  - `ema_alpha=0.0`
  - `k=1`
  - `n=2`

Shared online guard:

- `allow_low_motion_high_conf_bypass=true`
- `low_motion_high_conf_k=2`
- `low_motion_high_conf_max_lying=0.30`

Embedded metrics:

- `OP1`: `P=1.0 R=1.0 F1=1.0`
- `OP2`: `P=1.0 R=1.0 F1=1.0`
- `OP3`: `P=1.0 R=1.0 F1=1.0`

Notes:

- As with `le2i TCN`, the measured online replay surface is flat across a broad threshold range.
- The three profiles are still kept distinct by increasing conservativeness from `OP1` to `OP3`.

## Practical Recommendation

If no special requirement is given, use:

- `caucafall + TCN + OP2`
- `caucafall + GCN + OP2`
- `le2i + TCN + OP2`
- `le2i + GCN + OP2`

If the priority is recall, move to `OP1`.

If the priority is stricter triggering, move to `OP3`.

## Validation References

Relevant artifacts:

- `artifacts/online_ops_fit_20260315/caucafall_gcn.json`
- `artifacts/online_ops_fit_20260315/le2i_tcn.json`
- `artifacts/online_ops_fit_20260315/le2i_gcn.json`
- `artifacts/online_ops_fit_20260315/caucafall_tcn_refit.json`
- `artifacts/fall_test_eval_20260315_online_reverify_20260315/tcn_op2_pose_raw_frontend_emulation_final_k2_v2.json`

The last artifact above is the final frontend-style replay validation for the 24 custom corridor/kitchen videos with:

- `caucafall + TCN + OP2`
- video mode
- `2` consecutive fall windows required in the frontend
- result: `TP=12 TN=12 FP=0 FN=0`
