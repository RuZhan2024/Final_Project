#!/usr/bin/env python3
"""Build a reproducibility bundle and optional execution manifest for thesis claims."""

from __future__ import annotations

import argparse
import datetime as dt
import glob
import hashlib
import json
import os
import shlex
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _git(cmd: List[str], default: str = "") -> str:
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL, text=True).strip()
        return out or default
    except Exception:
        return default


def _now_tag() -> str:
    return dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def _stats(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"count": 0, "min": 0.0, "max": 0.0, "mean": 0.0}
    return {
        "count": float(len(values)),
        "min": float(min(values)),
        "max": float(max(values)),
        "mean": float(sum(values) / len(values)),
    }


def _extract_metric_row(path: Path, op: str) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "path": str(path),
        "op": op.lower(),
        "f1": None,
        "recall": None,
        "fa24h": None,
        "ap": None,
    }
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return row
    ops = data.get("ops") or {}
    op_row = ops.get(op.lower()) if isinstance(ops, dict) else {}
    if isinstance(op_row, dict):
        row["f1"] = op_row.get("f1")
        row["recall"] = op_row.get("recall")
        row["fa24h"] = op_row.get("fa24h")
        row["ap"] = op_row.get("ap", data.get("ap"))
    else:
        row["ap"] = data.get("ap")
    return row


def _write_markdown_summary(out_md: Path, metric_rows: List[Dict[str, Any]]) -> None:
    lines = [
        "# Reproducibility Summary",
        "",
        "| metrics_json | op | f1 | recall | fa24h | ap |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in metric_rows:
        lines.append(
            f"| {r['path']} | {r['op']} | {r['f1']} | {r['recall']} | {r['fa24h']} | {r['ap']} |"
        )
    lines.append("")
    out_md.write_text("\n".join(lines), encoding="utf-8")


def _run_command(cmd: str, cwd: Path, log_dir: Path) -> Dict[str, Any]:
    started = time.time()
    ts = dt.datetime.utcnow().isoformat() + "Z"
    log_file = log_dir / f"cmd_{int(started * 1000)}.log"
    p = subprocess.run(
        cmd,
        cwd=str(cwd),
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    log_file.write_text(p.stdout or "", encoding="utf-8")
    ended = time.time()
    return {
        "cmd": cmd,
        "started_utc": ts,
        "duration_s": round(ended - started, 3),
        "returncode": int(p.returncode),
        "log": os.path.relpath(str(log_file), start=str(cwd)),
    }


def build_manifest(
    repo_root: Path,
    dataset: str,
    model: str,
    op: str,
    checkpoint: Path,
    metric_globs: List[str],
    run_results: List[Dict[str, Any]],
) -> Dict[str, Any]:
    def _rel_repo_path(p: Path) -> str:
        rp = p.resolve()
        try:
            return os.path.relpath(str(rp), start=str(repo_root.resolve()))
        except Exception:
            return str(rp)

    metric_files: List[str] = []
    for pat in metric_globs:
        metric_files.extend(sorted(glob.glob(str(repo_root / pat))))
    metric_rows = []
    for p in metric_files:
        row = _extract_metric_row(Path(p), op)
        row["path"] = _rel_repo_path(Path(p))
        metric_rows.append(row)

    ckpt_info: Dict[str, Any] = {"path": _rel_repo_path(checkpoint), "exists": checkpoint.is_file(), "sha256": None}
    if checkpoint.is_file():
        ckpt_info["sha256"] = _sha256(checkpoint)

    return {
        "schema_version": "1.0",
        "generated_utc": dt.datetime.utcnow().isoformat() + "Z",
        "git": {
            "commit": _git(["git", "rev-parse", "HEAD"], default="unknown"),
            "branch": _git(["git", "rev-parse", "--abbrev-ref", "HEAD"], default="unknown"),
            "dirty": bool(_git(["git", "status", "--porcelain"], default="")),
        },
        "claim": {"dataset": dataset, "model": model, "op": op},
        "checkpoint": ckpt_info,
        "commands": run_results,
        "metrics": metric_rows,
        "command_durations": _stats([float(x.get("duration_s", 0.0)) for x in run_results]),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="le2i")
    ap.add_argument("--model", default="tcn", choices=["tcn", "gcn"])
    ap.add_argument("--op", default="op2")
    ap.add_argument("--run", type=int, default=0, help="1 to execute commands, 0 to generate manifest only.")
    ap.add_argument(
        "--cmd",
        action="append",
        default=[],
        help="Command to execute (repeatable). If omitted, canonical defaults are used.",
    )
    ap.add_argument("--out_dir", default="")
    ap.add_argument("--metrics_glob", action="append", default=[])
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    out_dir = Path(args.out_dir) if args.out_dir else repo_root / "artifacts" / "repro" / f"RESULTS_{_now_tag()}"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir = out_dir / "command_logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    default_cmds = [
        f"make pipeline-auto-{args.model}-{args.dataset}",
        f"make audit-all MODEL={args.model}",
    ]
    cmds = list(args.cmd) if args.cmd else default_cmds

    run_results: List[Dict[str, Any]] = []
    if int(args.run) == 1:
        for c in cmds:
            print(f"[run] {c}")
            res = _run_command(c, repo_root, log_dir)
            run_results.append(res)
            if int(res["returncode"]) != 0:
                print(f"[fail] command failed: {c}")
                break
    else:
        run_results = [{"cmd": c, "planned": True} for c in cmds]

    metrics_glob = args.metrics_glob or [f"outputs/metrics/{args.model}_{args.dataset}*.json"]
    ckpt = repo_root / "outputs" / f"{args.dataset}_{args.model}_W48S12" / "best.pt"
    manifest = build_manifest(
        repo_root=repo_root,
        dataset=str(args.dataset),
        model=str(args.model),
        op=str(args.op),
        checkpoint=ckpt,
        metric_globs=metrics_glob,
        run_results=run_results,
    )

    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_markdown_summary(out_dir / "SUMMARY.md", list(manifest.get("metrics", [])))

    print(f"[ok] reproducibility bundle: {out_dir}")
    print(f"[ok] manifest: {manifest_path}")


if __name__ == "__main__":
    main()
