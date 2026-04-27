Date: 2026-04-27  
Purpose: targeted evidence-strengthening note for the publication-facing gap around cross-dataset failure behaviour.

# Targeted Failure Analysis

## Why This Note Exists

The current paper/report already states that cross-dataset transfer is asymmetric and bounded. What was missing was a compact artifact that shows *how* the failure happens rather than only reporting aggregate metric deltas.

This note closes that gap with one focused comparison:

1. in-domain `CAUCAFall + TCN + OP-2`
2. cross-dataset `CAUCAFall -> LE2i` with TCN
3. cross-dataset `CAUCAFall -> LE2i` with GCN

The goal is not to create a new headline result. The goal is to make the existing limitation evidence more explicit and reviewer-readable.

## Source Artifacts

Main metric JSON files:

1. `outputs/metrics/tcn_caucafall_locked.json`
2. `outputs/metrics/cross_tcn_caucafall_r2_train_hneg_to_le2i_frozen_20260409.json`
3. `outputs/metrics/cross_gcn_caucafall_r2_recallpush_b_to_le2i_frozen_20260409.json`

Derived failure-analysis figures:

1. [caucafall_tcn_locked_failure_box.png](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/artifacts/figures/report/caucafall_tcn_locked_failure_box.png)
2. [cross_tcn_caucafall_to_le2i_failure_box.png](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/artifacts/figures/report/cross_tcn_caucafall_to_le2i_failure_box.png)
3. [cross_gcn_caucafall_to_le2i_failure_box.png](/Users/ruzhan/computer_science/Goldsmiths/Final_Project/fall_detection_v2/artifacts/figures/report/cross_gcn_caucafall_to_le2i_failure_box.png)

Generation command:

```bash
python3 ops/scripts/plot_failure_scatter.py --metrics_json outputs/metrics/tcn_caucafall_locked.json --out_fig artifacts/figures/report/caucafall_tcn_locked_failure_box.png --style box
python3 ops/scripts/plot_failure_scatter.py --metrics_json outputs/metrics/cross_tcn_caucafall_r2_train_hneg_to_le2i_frozen_20260409.json --out_fig artifacts/figures/report/cross_tcn_caucafall_to_le2i_failure_box.png --style box
python3 ops/scripts/plot_failure_scatter.py --metrics_json outputs/metrics/cross_gcn_caucafall_r2_recallpush_b_to_le2i_frozen_20260409.json --out_fig artifacts/figures/report/cross_gcn_caucafall_to_le2i_failure_box.png --style box
```

## Per-Video Outcome Summary

### 1. In-domain locked TCN (`CAUCAFall`)

Per-video status counts:

- `TP = 5`
- `FP = 0`
- `FN = 0`
- `TN = 5`

Interpretation:

- The locked in-domain `CAUCAFall + TCN + OP-2` profile is clean at the per-video level on this bounded surface.
- The failures that dominate later sections are therefore not simply “the TCN always fails at runtime” or “the profile is unstable everywhere”.

### 2. Cross-dataset TCN (`CAUCAFall -> LE2i`)

Per-video status counts:

- `TP = 2`
- `FP = 0`
- `FN = 7`
- `TN = 4`

Interpretation:

- The dominant transfer failure mode is **missed falls**, not false alarms.
- Most failed positive videos collapse to `alert_frac = 0.0`, which means the TCN often does not enter a meaningful alert state at all on this transfer surface.
- This is consistent with the aggregate cross-domain result where `F1` and `Recall` collapse to `0.0`.

### 3. Cross-dataset GCN (`CAUCAFall -> LE2i`)

Per-video status counts:

- `TP = 7`
- `FP = 2`
- `FN = 2`
- `TN = 2`

Interpretation:

- The dominant GCN transfer behaviour is different.
- The GCN recovers many event hits, but does so with a much noisier alert surface.
- Two negative videos show extremely large `fa24h` values, including approximately `6428.57` and `10588.24`.
- This explains why event-level recall can improve while the profile still fails as a practical deployment candidate.

## Main Defended Lessons

1. The cross-dataset limitation is not a generic “performance drop”; it is **failure-mode specific**.
2. The TCN transfer failure is primarily a **false-negative collapse**.
3. The GCN transfer behaviour is primarily a **recall-via-false-alert inflation** pattern.
4. This makes the paper’s bounded interpretation stronger:
   the TCN remains the cleaner in-domain system candidate, but cross-domain transfer exposes a real limitation boundary rather than a small degradation.

## Safe Wording for Report and Paper

Safe statement:

`Cross-dataset transfer from CAUCAFall to LE2i fails in different ways for the two model families: the TCN tends toward missed-fall collapse, while the matched GCN recovers more event hits only by becoming far less selective.`

Unsafe statement:

`The TCN simply generalises better than the GCN across datasets.`

Why unsafe:

- the current evidence does not support a simple universal ranking under transfer
- the GCN can recover recall in transfer, but in a practically weak way
- the defended conclusion is about *trade-off structure*, not a flat winner claim

## Publication-Facing Value

This is worthwhile project work because it improves the evidence layer directly:

1. it sharpens the explanation behind the cross-dataset results
2. it turns an abstract limitation into a concrete, inspectable failure pattern
3. it supports a more generalisable discussion point about deployment-oriented evaluation:
   event recall alone is not enough when transfer behaviour can be recovered only by exploding false-alert cost
