#!/usr/bin/env python3
"""Append one experiment record row for reproducible reporting."""

from __future__ import annotations

import argparse
import csv
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def _run_git(args: list[str]) -> str:
    try:
        out = subprocess.check_output(["git", *args], stderr=subprocess.DEVNULL).decode("utf-8").strip()
        return out
    except Exception:
        return ""


def _git_commit() -> str:
    return _run_git(["rev-parse", "--short", "HEAD"]) or "unknown"


def _git_branch() -> str:
    return _run_git(["branch", "--show-current"]) or "unknown"


def _git_dirty() -> str:
    status = _run_git(["status", "--porcelain"])
    return "yes" if status else "no"


def main() -> None:
    ap = argparse.ArgumentParser(description="Append one row to experiment registry.")
    ap.add_argument("--csv", default="artifacts/registry/overfit_experiment_registry.csv")
    ap.add_argument("--exp_id", required=True)
    ap.add_argument("--phase", default="train", choices=["plan", "train", "fit_ops", "eval", "analyze"])
    ap.add_argument("--dataset", required=True, choices=["caucafall", "le2i"])
    ap.add_argument("--arch", required=True, choices=["tcn", "gcn"])
    ap.add_argument("--seed", default="")
    ap.add_argument("--status", default="planned", choices=["planned", "running", "done", "failed", "rejected"])
    ap.add_argument("--changed_params", default="")
    ap.add_argument("--command", default="")
    ap.add_argument("--artifacts", default="")
    ap.add_argument("--metrics_json", default="")
    ap.add_argument("--ops_yaml", default="")
    ap.add_argument("--notes", default="")
    ap.add_argument("--timestamp_utc", default="")
    args = ap.parse_args()

    path = Path(args.csv)
    path.parent.mkdir(parents=True, exist_ok=True)
    headers = [
        "timestamp_utc",
        "exp_id",
        "phase",
        "dataset",
        "arch",
        "seed",
        "status",
        "changed_params",
        "command",
        "artifacts",
        "metrics_json",
        "ops_yaml",
        "git_branch",
        "git_commit",
        "git_dirty",
        "notes",
    ]

    row = {
        "timestamp_utc": args.timestamp_utc or datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "exp_id": args.exp_id,
        "phase": args.phase,
        "dataset": args.dataset,
        "arch": args.arch,
        "seed": args.seed,
        "status": args.status,
        "changed_params": args.changed_params,
        "command": args.command,
        "artifacts": args.artifacts,
        "metrics_json": args.metrics_json,
        "ops_yaml": args.ops_yaml,
        "git_branch": _git_branch(),
        "git_commit": _git_commit(),
        "git_dirty": _git_dirty(),
        "notes": args.notes,
    }

    write_header = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        if write_header:
            writer.writeheader()
        writer.writerow(row)

    print(f"[ok] appended: {path}")
    print(f"      exp_id={args.exp_id} status={args.status} commit={row['git_commit']} dirty={row['git_dirty']}")


if __name__ == "__main__":
    main()
