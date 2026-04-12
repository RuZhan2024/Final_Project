#!/usr/bin/env python3
"""
Tune fit-ops + eval params automatically and pick TWO best configs per dataset:
  - thesis_best
  - deploy_best

Assumes:
  fit-ops-* writes YAML to:   configs/ops/*.yaml
  eval-*    writes JSON to:   outputs/metrics/*.json

Key fixes vs earlier version:
- Always sets a unique OUT_TAG per trial (unless --no_out_tag). This forces Make to rebuild
  and avoids "up to date" causing missing new YAML/JSON.
- Computes the expected YAML/JSON paths directly (instead of relying only on mtime filtering).
- Adds richer logs: per-trial command, outputs, key metrics, and mismatch warnings.
- Optional --force_make adds `make -B` to rebuild even if filenames collide.

Typical usage:
  # TCN targets

  TCN

python3 tools/tune_fit_eval_profiles.py \
  --datasets le2i caucafall \
  --fit_target "fit-ops-{dataset}" \
  --eval_target "eval-{dataset}" \
  --out_csv outputs/tuning/tcn_profiles.csv \
  --min_recall_thesis 0.90 \
  --min_recall_deploy 0.95 \
  --force_make \
  --log_each

GCN

python3 tools/tune_fit_eval_profiles.py \
  --datasets le2i caucafall \
  --fit_target "fit-ops-gcn-{dataset}" \
  --eval_target "eval-gcn-{dataset}" \
  --out_csv outputs/tuning/gcn_profiles.csv \
  --min_recall_thesis 0.90 \
  --min_recall_deploy 0.95 \
  --force_make \
  --log_each
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import time
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import yaml  # pip install pyyaml
except Exception as e:
    raise SystemExit("Missing dependency: pyyaml. Install with: pip install pyyaml") from e


# -------------------------
# IO helpers
# -------------------------
def run_make(repo_root: Path, target: str, make_args: List[str], verbose: bool, force_make: bool) -> str:
    cmd = ["make"]
    if force_make:
        cmd.append("-B")
    cmd += [target] + make_args

    if verbose:
        print("[run]", " ".join(cmd), flush=True)

    p = subprocess.run(
        cmd,
        cwd=str(repo_root),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    out = p.stdout or ""
    if p.returncode != 0:
        print(out)
        raise RuntimeError(f"make failed: {target} rc={p.returncode}")

    if verbose:
        print("\n".join(out.splitlines()[-25:]), flush=True)

    return out


def load_yaml(path: Path) -> Dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def to_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def get_op(doc: Dict[str, Any], op_name: str) -> Dict[str, Any]:
    # JSON/YAML are typically top-level op1/op2/op3.
    blk = (doc.get(op_name) or {})
    if not blk and isinstance(doc.get("ops"), dict):
        blk = doc["ops"].get(op_name) or {}
    if not blk and isinstance(doc.get("metrics"), dict):
        blk = doc["metrics"].get(op_name) or {}
    return blk


def get_meta(doc: Dict[str, Any]) -> Dict[str, Any]:
    m = doc.get("meta")
    return m if isinstance(m, dict) else {}


def infer_model_prefix(fit_target_tmpl: str, eval_target_tmpl: str) -> str:
    s = (fit_target_tmpl + " " + eval_target_tmpl).lower()
    return "gcn" if "gcn" in s else "tcn"


def ensure_out_tag(tag: str) -> str:
    if not tag:
        return ""
    return tag if tag.startswith("_") else ("_" + tag)


# -------------------------
# Config grid
# -------------------------
@dataclass(frozen=True)
class RunConfig:
    overrides: Dict[str, str]

    def make_args(self, extra_make_vars: List[str]) -> List[str]:
        args = list(extra_make_vars)
        args.extend([f"{k}={v}" for k, v in self.overrides.items()])
        return args


def default_grid_thesis() -> List[RunConfig]:
    """
    Thesis grid (benchmark-friendly):
    - Lower confirm_s and lower min_lying (datasets can end quickly after impact).
    - require_low often off for clip-based evaluation.
    """
    grid = {
        "ALERT_EMA_ALPHA": ["0.2"],
        "ALERT_K": ["2"],
        "ALERT_N": ["3"],
        "ALERT_COOLDOWN_S": ["30"],
        "ALERT_CONFIRM": ["1"],
        "ALERT_CONFIRM_REQUIRE_LOW": ["0"],
        "ALERT_CONFIRM_S": ["0.4", "0.6", "0.8"],
        "ALERT_CONFIRM_MIN_LYING": ["0.10", "0.20", "0.30", "0.40"],
        "ALERT_CONFIRM_MAX_MOTION": ["0.20", "0.25", "0.30"],
        "FIT_TAU_LOW_RATIO": ["0.78"],
        "FIT_OVERLAP_SLACK_S": ["0.5"],  # keep fit/eval consistent
    }
    keys = list(grid.keys())
    out: List[RunConfig] = []
    for vals in product(*[grid[k] for k in keys]):
        out.append(RunConfig({k: v for k, v in zip(keys, vals)}))
    return out


def default_grid_deploy() -> List[RunConfig]:
    """
    Deployment grid (stream-correct):
    - require_low on (re-arm), now safe after your patch.
    - includes stricter confirm_s / min_lying options.
    """
    grid = {
        "ALERT_EMA_ALPHA": ["0.2"],
        "ALERT_K": ["2"],
        "ALERT_N": ["3"],
        "ALERT_COOLDOWN_S": ["30"],
        "ALERT_CONFIRM": ["1"],
        "ALERT_CONFIRM_REQUIRE_LOW": ["1"],
        "ALERT_CONFIRM_S": ["1.0", "1.5", "2.0"],
        "ALERT_CONFIRM_MIN_LYING": ["0.20", "0.30", "0.40", "0.50", "0.65"],
        "ALERT_CONFIRM_MAX_MOTION": ["0.12", "0.20", "0.25"],
        "FIT_TAU_LOW_RATIO": ["0.78"],
        "FIT_OVERLAP_SLACK_S": ["0.5"],
    }
    keys = list(grid.keys())
    out: List[RunConfig] = []
    for vals in product(*[grid[k] for k in keys]):
        out.append(RunConfig({k: v for k, v in zip(keys, vals)}))
    return out


# -------------------------
# Scoring strategy
# -------------------------
def thesis_score(opm: Dict[str, Any], min_recall: float) -> float:
    """
    Thesis score = primarily maximize F1 on benchmark split.
    - Enforce a minimum recall (default 0.80 or 0.90 if you prefer).
    - Small delay penalty to avoid picking overly conservative thresholds on a recall plateau.
    """
    r = to_float(opm.get("recall"))
    f1 = to_float(opm.get("f1"))
    d = to_float(opm.get("mean_delay_s")) or 0.0
    if r is None or f1 is None:
        return -1e9
    if r < min_recall:
        return -1e6 - (min_recall - r) * 1e5
    return f1 - 0.02 * d


def deploy_score(opm: Dict[str, Any], meta: Dict[str, Any], min_recall: float, fa24_max: Optional[float]) -> float:
    """
    Deployment score = safety-first:
    - Enforce high recall (default 0.95).
    - Prefer lower delay.
    - Optionally enforce FA/24h if (and only if) FA duration exists in eval.
    """
    r = to_float(opm.get("recall"))
    f1 = to_float(opm.get("f1"))
    d = to_float(opm.get("mean_delay_s")) or 0.0
    fa = to_float(opm.get("fa_per_24h"))

    if r is None or f1 is None:
        return -1e9
    if r < min_recall:
        return -1e6 - (min_recall - r) * 1e5

    n_fa = meta.get("n_fa_videos", 0) or 0
    dur_fa = meta.get("total_duration_s_fa", 0) or 0
    fa_is_meaningful = (n_fa > 0) and (dur_fa > 0)

    if fa24_max is not None and fa_is_meaningful and fa is not None and fa > fa24_max:
        return -5e5 - (fa - fa24_max) * 1e5

    # Recall dominates; delay matters; tiny bonus for f1 for stability.
    return r - 0.03 * d + 0.05 * f1


# -------------------------
# Main
# -------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", required=True)
    ap.add_argument("--repo_root", default=".")
    ap.add_argument("--ops_dir", default="configs/ops")
    ap.add_argument("--metrics_dir", default="outputs/metrics")
    ap.add_argument("--fit_target", default="fit-ops-{dataset}")
    ap.add_argument("--eval_target", default="eval-{dataset}")
    ap.add_argument("--op", default="op1", choices=["op1", "op2", "op3"])
    ap.add_argument("--make_vars", nargs="*", default=[], help='Extra make vars, e.g. WIN_W=48 WIN_S=12 ...')
    ap.add_argument("--out_csv", default="outputs/tuning/profiles.csv")
    ap.add_argument("--limit_thesis", type=int, default=0, help="0 = no limit")
    ap.add_argument("--limit_deploy", type=int, default=0, help="0 = no limit")
    ap.add_argument("--min_recall_thesis", type=float, default=0.80)
    ap.add_argument("--min_recall_deploy", type=float, default=0.95)
    ap.add_argument("--fa24_max_deploy", type=float, default=-1.0, help="-1 disables FA constraint")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--log_each", action="store_true", help="Print one-line metrics after each trial")
    ap.add_argument("--force_make", action="store_true", help="Use `make -B` for each command (rebuild).")
    ap.add_argument("--out_tag_prefix", default="tune", help="Prefix for unique OUT_TAG per trial.")
    ap.add_argument("--no_out_tag", action="store_true", help="Disable OUT_TAG forcing (not recommended).")
    args = ap.parse_args()

    repo = Path(args.repo_root).resolve()
    ops_dir = (repo / args.ops_dir).resolve()
    metrics_dir = (repo / args.metrics_dir).resolve()
    out_csv = (repo / args.out_csv).resolve()
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    thesis_grid = default_grid_thesis()
    deploy_grid = default_grid_deploy()

    if args.limit_thesis and args.limit_thesis > 0:
        thesis_grid = thesis_grid[: args.limit_thesis]
    if args.limit_deploy and args.limit_deploy > 0:
        deploy_grid = deploy_grid[: args.limit_deploy]

    fa24_max = None if args.fa24_max_deploy < 0 else float(args.fa24_max_deploy)

    model_prefix = infer_model_prefix(args.fit_target, args.eval_target)
    session_id = time.strftime("%Y%m%d_%H%M%S")

    rows: List[Dict[str, Any]] = []
    best_thesis: Dict[str, Tuple[float, Dict[str, Any]]] = {ds: (-1e18, {}) for ds in args.datasets}
    best_deploy: Dict[str, Tuple[float, Dict[str, Any]]] = {ds: (-1e18, {}) for ds in args.datasets}

    def expected_paths(ds: str, out_tag: str) -> Tuple[Path, Path]:
        tag = ensure_out_tag(out_tag)
        y = ops_dir / f"{model_prefix}_{ds}{tag}.yaml"
        j = metrics_dir / f"{model_prefix}_{ds}{tag}.json"
        return y, j

    def eval_one(ds: str, cfg: RunConfig, profile: str, trial_idx: int) -> Dict[str, Any]:
        out_tag = ""
        if not args.no_out_tag:
            out_tag = f"{args.out_tag_prefix}_{session_id}_{model_prefix}_{profile}_{ds}_t{trial_idx:04d}"
        # Ensure leading underscore for Makefile naming convention
        out_tag = ensure_out_tag(out_tag)

        overrides = dict(cfg.overrides)
        if out_tag:
            overrides["OUT_TAG"] = out_tag

        make_args = RunConfig(overrides).make_args(args.make_vars)

        # Expected outputs
        exp_yaml, exp_json = expected_paths(ds, out_tag)

        if args.verbose:
            print(f"\n=== {profile} trial={trial_idx:04d} ds={ds} model={model_prefix} ===")
            print("overrides:", overrides)
            print("expect:", exp_yaml, "and", exp_json)

        # fit-ops
        run_make(repo, args.fit_target.format(dataset=ds), make_args, args.verbose, args.force_make)
        if not exp_yaml.exists():
            raise RuntimeError(
                f"Expected ops YAML not found: {exp_yaml}\n"
                f"Tip: ensure Makefile writes to {args.ops_dir} and uses OUT_TAG in filename."
            )
        ops_doc = load_yaml(exp_yaml)

        # eval
        run_make(repo, args.eval_target.format(dataset=ds), make_args, args.verbose, args.force_make)
        if not exp_json.exists():
            raise RuntimeError(
                f"Expected metrics JSON not found: {exp_json}\n"
                f"Tip: ensure Makefile writes to {args.metrics_dir} and uses OUT_TAG in filename."
            )
        eval_doc = load_json(exp_json)

        opm_eval = get_op(eval_doc, args.op)
        opm_fit = get_op(ops_doc, args.op)
        meta = get_meta(eval_doc)

        th = thesis_score(opm_eval, args.min_recall_thesis)
        dp = deploy_score(opm_eval, meta, args.min_recall_deploy, fa24_max)

        # Quick mismatch checks (helpful debugging)
        slack_meta = meta.get("overlap_slack_s")
        slack_cfg = overrides.get("FIT_OVERLAP_SLACK_S")
        slack_warn = (slack_meta is not None and slack_cfg is not None and str(slack_meta) != str(slack_cfg))

        row: Dict[str, Any] = {
            "profile": profile,
            "dataset": ds,
            "model": model_prefix,
            "op": args.op,
            "ops_yaml": str(exp_yaml),
            "eval_json": str(exp_json),
            **overrides,
            # Fit thresholds (debug)
            "fit_tau_high": opm_fit.get("tau_high"),
            "fit_tau_low": opm_fit.get("tau_low"),
            # Eval metrics (truth)
            "precision": opm_eval.get("precision"),
            "recall": opm_eval.get("recall"),
            "f1": opm_eval.get("f1"),
            "mean_delay_s": opm_eval.get("mean_delay_s"),
            "fa_per_24h": opm_eval.get("fa_per_24h"),
            "n_gt_events": opm_eval.get("n_gt_events"),
            "n_alert_events": opm_eval.get("n_alert_events"),
            "n_true_alerts": opm_eval.get("n_true_alerts"),
            "n_false_alerts": opm_eval.get("n_false_alerts"),
            "score_thesis": th,
            "score_deploy": dp,
            # Meta
            "n_fa_videos": meta.get("n_fa_videos"),
            "total_duration_s_fa": meta.get("total_duration_s_fa"),
            "overlap_slack_s_meta": slack_meta,
            "warn_overlap_slack_mismatch": slack_warn,
        }

        if args.log_each:
            print(
                f"[{profile}] ds={ds} t={trial_idx:04d} "
                f"recall={row['recall']} f1={row['f1']} delay={row['mean_delay_s']} "
                f"fa24={row['fa_per_24h']} fit_tau_high={row['fit_tau_high']} "
                f"{'WARN(slack)' if slack_warn else ''}",
                flush=True,
            )

        return row

    trial_counter = 0

    # Thesis grid
    for cfg in thesis_grid:
        for ds in args.datasets:
            trial_counter += 1
            row = eval_one(ds, cfg, profile="thesis", trial_idx=trial_counter)
            rows.append(row)
            sc = float(row["score_thesis"])
            if sc > best_thesis[ds][0]:
                best_thesis[ds] = (sc, row)

    # Deploy grid
    for cfg in deploy_grid:
        for ds in args.datasets:
            trial_counter += 1
            row = eval_one(ds, cfg, profile="deploy", trial_idx=trial_counter)
            rows.append(row)
            sc = float(row["score_deploy"])
            if sc > best_deploy[ds][0]:
                best_deploy[ds] = (sc, row)

    # Write CSV
    fields = sorted({k for r in rows for k in r.keys()})
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    def print_best(title: str, best_map: Dict[str, Tuple[float, Dict[str, Any]]], score_key: str) -> None:
        print(f"\n{title}")
        for ds, (sc, r) in best_map.items():
            if not r:
                continue
            print(
                f"- {ds}: {score_key}={sc:.4f} "
                f"recall={r.get('recall')} f1={r.get('f1')} delay={r.get('mean_delay_s')} "
                f"fa24={r.get('fa_per_24h')} out_tag={r.get('OUT_TAG')}"
            )
            overrides = {k: r[k] for k in r.keys() if k.isupper()}
            make_line = " ".join(args.make_vars + [f"{k}={overrides[k]}" for k in sorted(overrides.keys())])
            print(f"  make {args.eval_target.format(dataset=ds)} {make_line}")

    print_best("Thesis-best configs:", best_thesis, "score_thesis")
    print_best("Deploy-best configs:", best_deploy, "score_deploy")
    print(f"\nWrote: {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
