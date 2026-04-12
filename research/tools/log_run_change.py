#!/usr/bin/env python3
"""Append one experiment run-change record to the tuning changelog CSV."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description="Append a run-change record.")
    ap.add_argument("--log", default="artifacts/reports/tuning/PARAM_CHANGELOG.csv")
    ap.add_argument("--csv", default=None, help="Alias of --log for compatibility.")
    ap.add_argument("--timestamp_utc", default=None, help="Optional override timestamp.")
    ap.add_argument("--exp_id", required=True)
    ap.add_argument("--base_ref", default="")
    ap.add_argument("--changed_params", required=True)
    ap.add_argument("--command", required=True)
    ap.add_argument("--status", default="planned")
    ap.add_argument("--artifacts", default="")
    ap.add_argument("--notes", default="")
    args = ap.parse_args()

    log_path = Path(args.csv or args.log)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    headers = [
        "timestamp_utc",
        "exp_id",
        "base_ref",
        "changed_params",
        "command",
        "status",
        "artifacts",
        "notes",
    ]

    row = {
        "timestamp_utc": args.timestamp_utc
        or datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "exp_id": args.exp_id,
        "base_ref": args.base_ref,
        "changed_params": args.changed_params,
        "command": args.command,
        "status": args.status,
        "artifacts": args.artifacts,
        "notes": args.notes,
    }

    write_header = not log_path.exists()
    with log_path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


if __name__ == "__main__":
    main()
