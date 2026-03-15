# Artifacts Index

This directory stores generated evidence, evaluation bundles, report support material, and figure outputs.

It is evidence-heavy by design. Most subdirectories are generated outputs rather than hand-maintained source files.

## Report / Thesis Evidence

These subtrees are important for writing and defending results:

- `reports/`
- `baseline/`
- `repro/`
- `registry/`
- `templates/`
- `figures/cross_dataset/`
- `figures/pr_curves/`
- `figures/stability/`
- `figures/latency/`

## Delivery and Online Validation Bundles

These hold the main 2026 delivery verification outputs:

- `fall_test_eval_20260315/`
- `fall_test_eval_20260315_online_reverify_20260315/`
- `fall_side_corridor_eval_20260315/`
- `fall_test_generalization_20260315/`
- `online_ops_fit_20260315/`
- `online_ops_fit_20260315_verify/`
- `online_ops_fit_20260315_verify_le2i_bypass/`
- `ops_delivery_verify_20260315/`
- `ops_reverify_20260315/`
- `ops_reverify_20260315_after_gatefix/`
- `ops_reverify_20260315_after_motionfix/`

These are generated, but they are still relevant evidence and should be archived rather than casually deleted.

## Reports Subtree

`artifacts/reports/` is the main narrative evidence area. It already groups evidence into themes such as:

- `gcn_aug/`
- `gcn_overtake/`
- `hneg_cycle/`
- `tuning/`

Use that subtree when you need:

- comparison tables
- sweep summaries
- command traces
- tuning notes
- release snapshots

## Figures

- `figures/cross_dataset/`: cross-dataset transfer and comparison visuals
- `figures/pr_curves/`: PR / score-curve visuals
- `figures/stability/`: stability analysis figures
- `figures/latency/`: runtime latency visuals

Generated plot scratch outputs should not live here permanently unless they are cited evidence.

## Practical Rule

- If a file is cited or likely to be cited in the report or thesis, keep it.
- If it is just an intermediate scratch output, keep it out of the main evidence folders or ignore it.
