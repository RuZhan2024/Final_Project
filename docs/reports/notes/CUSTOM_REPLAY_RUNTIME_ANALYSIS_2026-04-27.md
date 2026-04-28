Date: 2026-04-27  
Purpose: targeted deployment-evidence note for the bounded custom replay surface.

# Custom Replay Runtime Analysis

## Why This Note Exists

The repository already preserves aggregate replay evidence, but the publication-facing weakness was that the runtime story could still collapse into broad statements such as “the profile works” or “the replay surface is mixed”.

This note makes the bounded runtime behaviour more explicit by separating:

1. the canonical 10-video online replay matrix
2. the 24-clip custom replay surface
3. folder-level failure structure on the custom surface

The goal is not to upgrade replay evidence into benchmark evidence. The goal is to state exactly what the deployment surface currently shows.

## Source Artifacts

Canonical replay summary:

1. `artifacts/ops_delivery_verify_20260315/online_replay_summary.json`

Custom replay summaries:

1. `artifacts/fall_test_eval_20260315/summary_tcn_caucafall_locked_op2.csv`
2. `artifacts/fall_test_eval_20260315/summary_tcn_caucafall_locked_op1.csv`
3. `artifacts/fall_test_eval_20260315/summary_gcn_caucafall_locked_op2.csv`

Supporting runbook:

1. `docs/reports/runbooks/FOUR_VIDEO_DELIVERY_PROFILE.md`

## A. Canonical 10-Video Replay Matrix

From `online_replay_summary.json`:

- `caucafall_tcn OP-1`: `accuracy=1.0`, `recall=1.0`, `specificity=1.0`
- `caucafall_tcn OP-2`: `accuracy=0.5`, `recall=0.0`, `specificity=1.0`
- `caucafall_tcn OP-3`: `accuracy=1.0`, `recall=1.0`, `specificity=1.0`
- `caucafall_gcn OP-1/OP-2/OP-3`: `accuracy=0.5`, `recall=0.0`, `specificity=1.0`
- `le2i_tcn` and `le2i_gcn` stay at `accuracy=0.2308`, `recall=0.0`, `specificity=1.0` across all three operating points on the same bounded surface

Defended interpretation:

1. The canonical replay surface is **profile-sensitive**, not uniformly favourable to one preset.
2. On this surface, `CAUCAFall + TCN` is the only family that shows meaningful operating-point separation.
3. The same family can look excellent under `OP-1` or `OP-3` while failing badly under `OP-2`, so replay evidence must be narrated as configuration-sensitive runtime evidence rather than as a stable “best row”.

## B. 24-Clip Custom Replay Surface

### TCN `CAUCAFall + OP-2`

From `summary_tcn_caucafall_locked_op2.csv`:

- `TP=5`
- `TN=10`
- `FP=2`
- `FN=7`

Folder breakdown:

- `corridor`: `TP=3`, `FN=3`
- `corridor_adl`: `TN=6`
- `kitchen`: `TP=2`, `FN=4`
- `kitchen_adl`: `FP=2`, `TN=4`

Interpretation:

1. This profile is comparatively selective on ADL clips.
2. Its dominant weakness is **missed falls**, especially on the `kitchen` clips and part of the `corridor` set.
3. The two false alarms are concentrated in `kitchen_adl`, not spread across all ADL folders.

### TCN `CAUCAFall + OP-1`

From `summary_tcn_caucafall_locked_op1.csv`:

- `TP=9`
- `TN=2`
- `FP=10`
- `FN=3`

Folder breakdown:

- `corridor`: `TP=6`
- `corridor_adl`: `FP=5`, `TN=1`
- `kitchen`: `TP=3`, `FN=3`
- `kitchen_adl`: `FP=5`, `TN=1`

Interpretation:

1. `OP-1` recovers additional fall hits compared with `OP-2`.
2. It does so by heavily relaxing selectivity on both ADL folders.
3. This is a runtime trade-off, not a free improvement.

### GCN `CAUCAFall + OP-2`

From `summary_gcn_caucafall_locked_op2.csv`:

- `TP=12`
- `TN=0`
- `FP=12`
- `FN=0`

Folder breakdown:

- `corridor`: `TP=6`
- `corridor_adl`: `FP=6`
- `kitchen`: `TP=6`
- `kitchen_adl`: `FP=6`

Interpretation:

1. The GCN replay profile is not balanced on this custom surface.
2. It effectively classifies all positive folders as falls and all ADL folders as falls.
3. This makes it unusable as a selective runtime preset even though it avoids missed falls on this bounded clip set.

## C. What This Changes in the Defended Story

This custom replay analysis sharpens the runtime argument in three ways.

1. `TCN + OP-2` is best described as a **more selective but miss-prone** bounded demo preset.
2. `TCN + OP-1` is a **higher-recall but much noisier** preset on the same custom surface.
3. `GCN + OP-2` is **non-selective** on the custom surface because it collapses all ADL clips into false alarms.

This is a better runtime story than “one row is best”.

## Safe Wording for Report and Paper

Safe statement:

`On the bounded 24-clip custom replay surface, the active TCN OP-2 preset is more selective than the looser TCN OP-1 and far more selective than the matched GCN profile, but it achieves that selectivity by missing a substantial subset of fall clips.`

Unsafe statement:

`TCN OP-2 is the strongest replay configuration overall.`

Why unsafe:

1. the canonical 10-video replay matrix and the 24-clip custom surface do not point to a single uniformly strongest row
2. `OP-2` is better understood as a bounded deployment trade-off
3. replay evidence remains configuration-sensitive and should not be narrated as broad deployment closure

## Publication-Facing Value

This note is worthwhile project work because it strengthens the runtime evidence layer directly.

1. It converts custom replay evidence from a single aggregate score into a concrete trade-off analysis.
2. It explains why the preferred demo preset is still bounded rather than globally strongest.
3. It gives the report and paper a cleaner deployment lesson:
   runtime usefulness depends on the balance between missed falls and ADL false alarms, not on replay accuracy alone.
