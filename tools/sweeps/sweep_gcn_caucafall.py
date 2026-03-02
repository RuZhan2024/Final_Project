#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Auto-sweep for GCN on CAUCAFALL (native FPS from Makefile)

- Loops over explicit parameter value lists (PARAM_GRID + STRATEGIES)
- Runs: `make train-gcn-caucafall OUT_TAG=... VAR=...`
- Reads: <save_dir>/history.jsonl
- Scores: best val_f1 across epochs (default: val_f1)
- Writes: outputs/sweeps/gcn/caucafall/<exp>/* including best_command.sh
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running as: python3 tools/sweeps/<script>.py from repo root
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from sweep_lib import RunSpec, iter_param_grid, run_sweep


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="gcn_caucafall_sweep", help="Name for this sweep run (output subfolder).")
    ap.add_argument("--metric", default="val_f1", help="Metric key in history.jsonl to maximize.")
    ap.add_argument("--trials", type=int, default=80,
                    help="How many trials to run (random subset of full grid). Use 0 to run ALL.")
    ap.add_argument("--max_trials", type=int, default=None,
                    help="Alias for --trials (kept for backward compatibility).")
    ap.add_argument("--seed", type=int, default=33724876, help="Seed used ONLY for sampling subset when trials>0.")
    ap.add_argument("--no_skip_existing", action="store_true", help="Re-run even if history.jsonl already exists.")
    ap.add_argument("--silent_make", action="store_true", help="Pass -s to make (logs still contain full output).")

    # Stage-2 selection (deployment-style): fit_ops + metrics on top-K trials.
    ap.add_argument("--stage2", action="store_true", help="Run fit_ops.py + metrics.py on top-K trials and pick best by OP targets.")
    ap.add_argument("--stage2_topk", type=int, default=5, help="Number of top stage-1 trials to run stage2 on.")
    ap.add_argument("--stage2_split", choices=["val", "test"], default="val", help="Which split to score on (val recommended to avoid test leakage).")
    ap.add_argument("--stage2_no_windows_eval", action="store_true", help="Do not run make windows-eval-* automatically (assume windows_eval already exists).")
    ap.add_argument("--stage2_op1_target", type=float, default=0.95, help="Recall target used to prioritise OP1 feasibility.")
    ap.add_argument("--stage2_op3_target", type=float, default=1.0, help="FA/24h target used by fit_ops to pick OP3.")
    args = ap.parse_args()

    max_trials = args.trials
    if args.max_trials is not None:
        max_trials = args.max_trials
    if max_trials is not None and max_trials <= 0:
        max_trials = None

    run = RunSpec(
        arch="gcn",
        dataset="caucafall",
        target="train-gcn-caucafall",
        base_out_dir="outputs/caucafall_gcn_W48S12",
    )

    base_overrides = {
    "WIN_W": 48,
    "WIN_S": 12,
    "EPOCHS_GCN": 200,
    "BATCH_GCN": 64,
    "FEAT_USE_MOTION": 1,
    "FEAT_USE_CONF_CHANNEL": 1,
    "FEAT_MOTION_SCALE_BY_FPS": 1,
    "FEAT_CONF_GATE": 0.2,
    "FEAT_USE_PRECOMPUTED_MASK": 1
}

    param_grid = {
    "LR_GCN_caucafall": [
        0.001,
        0.0005,
        0.0003
    ],
    "GCN_DROPOUT": [
        0.2,
        0.35
    ],
    "GCN_HIDDEN": [
        64,
        96,
        128
    ],
    "MASK_JOINT_P": [
        0.05,
        0.1
    ],
    "MASK_FRAME_P": [
        0.02,
        0.05
    ]
}

    strategies = [
    {
        "GCN_LOSS": "bce",
        "GCN_POS_WEIGHT": "auto",
        "GCN_BALANCED_SAMPLER": 0,
        "GCN_TWO_STREAM": 1,
        "GCN_FUSE": "concat"
    },
    {
        "GCN_LOSS": "focal",
        "GCN_FOCAL_ALPHA": 0.25,
        "GCN_FOCAL_GAMMA": 2.0,
        "GCN_POS_WEIGHT": "none",
        "GCN_BALANCED_SAMPLER": 0,
        "GCN_TWO_STREAM": 1,
        "GCN_FUSE": "concat"
    }
]

    grid = list(iter_param_grid(param_grid))
    trials = []
    for g in grid:
        for s in strategies:
            ov = dict(g)
            ov.update(s)
            trials.append(ov)

    results_dir = Path("outputs") / "sweeps"
    run_sweep(
        run=run,
        exp=args.exp,
        metric=args.metric,
        base_overrides=base_overrides,
        grid_overrides=trials,
        results_dir=results_dir,
        max_trials=max_trials,
        seed=args.seed,
        skip_existing=(not args.no_skip_existing),
        silent_make=args.silent_make,
        stage2=args.stage2,
        stage2_topk=args.stage2_topk,
        stage2_split=args.stage2_split,
        stage2_run_windows_eval=(not args.stage2_no_windows_eval),
        stage2_op1_target=args.stage2_op1_target,
        stage2_op3_target=args.stage2_op3_target,
    )

if __name__ == "__main__":
    main()
