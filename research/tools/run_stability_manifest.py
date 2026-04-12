#!/usr/bin/env python3
"""Run stability manifest commands sequentially with status updates.

Usage:
  python tools/run_stability_manifest.py \
    --manifest artifacts/registry/stability_manifest.csv \
    --start_status todo
"""

from __future__ import annotations

import argparse
import csv
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List


def _read_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _write_rows(path: Path, rows: List[Dict[str, str]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def _run(cmd: str, env: Dict[str, str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as logf:
        logf.write(f"\n$ {cmd}\n")
        logf.flush()
        p = subprocess.Popen(
            ["/bin/zsh", "-lc", cmd],
            stdout=logf,
            stderr=subprocess.STDOUT,
            env=env,
        )
        return p.wait()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="artifacts/registry/stability_manifest.csv")
    ap.add_argument("--start_status", default="todo")
    ap.add_argument("--stop_on_fail", type=int, default=1)
    ap.add_argument("--sleep_s", type=float, default=0.2)
    args = ap.parse_args()

    manifest = Path(args.manifest)
    rows = _read_rows(manifest)
    if not rows:
        print(f"[err] empty manifest: {manifest}")
        return 1

    env = os.environ.copy()
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")

    runlog = Path("artifacts/reports/stability_runner.log")

    for i, row in enumerate(rows):
        status = (row.get("status") or "").strip().lower()
        if status not in {args.start_status, "retry"}:
            continue

        cid = row.get("candidate_id", "?")
        arch = row.get("arch", "?")
        ds = row.get("dataset", "?")
        seed = row.get("seed", "?")
        tag = row.get("out_tag", "")

        print(f"[run] idx={i} {cid} {arch} {ds} seed={seed} tag={tag}")
        row["status"] = "running"
        _write_rows(manifest, rows)

        for step in ("train_cmd", "fit_ops_cmd", "eval_cmd"):
            cmd = (row.get(step) or "").strip()
            if not cmd:
                row["status"] = "failed"
                _write_rows(manifest, rows)
                print(f"[fail] missing {step} at idx={i}")
                return 1
            rc = _run(cmd, env, runlog)
            if rc != 0:
                row["status"] = "failed"
                row["failed_step"] = step
                row["returncode"] = str(rc)
                _write_rows(manifest, rows)
                print(f"[fail] idx={i} step={step} rc={rc}")
                if int(args.stop_on_fail) == 1:
                    return rc
                break
            time.sleep(float(args.sleep_s))
        else:
            row["status"] = "done"
            row["failed_step"] = ""
            row["returncode"] = "0"
            _write_rows(manifest, rows)
            print(f"[ok] idx={i} done")

    print("[ok] manifest processing complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
