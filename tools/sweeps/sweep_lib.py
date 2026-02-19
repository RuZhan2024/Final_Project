#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tools/sweeps/sweep_lib.py

Dependency-free sweep helper.

Design goals:
- Run `make <target> VAR=...` loops deterministically
- Parse `history.jsonl` from the trainer to score each run
- Save best run as:
  - best_command.sh (re-run exactly)
  - best_overrides.mk (Makefile overrides)
  - results.jsonl / results.csv

Assumptions:
- You run from the repo root (where Makefile lives).
- Each training run writes: <save_dir>/history.jsonl
"""

from __future__ import annotations

import csv
import json
import os
import sys
import random
import re
import shlex
import subprocess
import time
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple


@dataclass(frozen=True)
class RunSpec:
    arch: str          # "tcn" | "gcn"
    dataset: str       # "le2i" | "caucafall"
    target: str        # make target (e.g. train-tcn-le2i)
    base_out_dir: str  # base save_dir without tag (e.g. outputs/le2i_tcn_W48S12)


def repo_root() -> Path:
    here = Path.cwd()
    if not (here / "Makefile").exists():
        raise SystemExit("[err] Run this from your repo root (Makefile not found).")
    return here


def safe_tag(s: str, max_len: int = 120) -> str:
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    if not s:
        s = "run"
    if len(s) > max_len:
        s = s[:max_len]
    if not s.startswith("_"):
        s = "_" + s
    return s


def make_cmd(target: str, overrides: Dict[str, Any], *, silent: bool = False) -> List[str]:
    cmd = ["make"]
    if silent:
        cmd.append("-s")
    cmd.append(target)
    for k, v in overrides.items():
        cmd.append(f"{k}={v}")
    return cmd


def _build_env(env: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    env2 = dict(os.environ)
    if env:
        env2.update(env)
    # Ensure Python prints immediately (helps with live logs)
    env2.setdefault("PYTHONUNBUFFERED", "1")
    return env2


def run_subprocess(cmd: List[str], log_path: Path, *, env: Optional[Dict[str, str]] = None) -> int:
    """Run a command and stream output to BOTH terminal and a log file.

    IMPORTANT: we run the child under a pseudo-TTY on POSIX so tqdm progress bars
    behave normally (single in-place bar) instead of printing a new line per update.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env2 = _build_env(env)

    with log_path.open("w", encoding="utf-8", buffering=1) as f:
        f.write("$ " + " ".join(shlex.quote(x) for x in cmd) + "\n\n")
        f.flush()

        # POSIX: use pty to make tqdm see a real TTY.
        if os.name == "posix":
            import pty
            import select

            master_fd, slave_fd = pty.openpty()
            try:
                p = subprocess.Popen(
                    cmd,
                    stdin=slave_fd,
                    stdout=slave_fd,
                    stderr=slave_fd,
                    env=env2,
                    close_fds=True,
                )
            finally:
                # Parent must close the slave end; child keeps it.
                try:
                    os.close(slave_fd)
                except OSError:
                    pass

            # Read raw bytes so we preserve carriage returns used by tqdm.
            try:
                while True:
                    r, _, _ = select.select([master_fd], [], [], 0.1)
                    if master_fd in r:
                        try:
                            data = os.read(master_fd, 4096)
                        except OSError:
                            data = b""
                        if not data:
                            break
                        s = data.decode("utf-8", errors="replace")
                        f.write(s)
                        sys.stdout.write(s)
                        sys.stdout.flush()
                    # If process ended and nothing left to read, exit.
                    if p.poll() is not None and not r:
                        # There may still be buffered output; loop will drain it.
                        continue
                rc = p.wait()
            finally:
                try:
                    os.close(master_fd)
                except OSError:
                    pass
            return rc

        # Fallback (non-POSIX): PIPE tee.
        p2 = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env2,
            text=True,
            bufsize=1,
        )

        assert p2.stdout is not None
        for line in p2.stdout:
            f.write(line)
            sys.stdout.write(line)
            sys.stdout.flush()
        return p2.wait()


def read_history_best(history_path: Path, metric: str = "monitor_score") -> Tuple[float, Dict[str, Any]]:
    """
    Returns (best_metric_value, best_row) over all epochs in history.jsonl.
    """
    best = float("-inf")
    best_row: Dict[str, Any] = {}
    with history_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            v = row.get(metric, None)
            if v is None:
                continue
            try:
                vv = float(v)
            except Exception:
                continue
            if vv > best:
                best = vv
                best_row = row
    return best, best_row


def dump_overrides_mk(path: Path, overrides: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for k in sorted(overrides.keys()):
        lines.append(f"{k} := {overrides[k]}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def dump_best_command(path: Path, target: str, overrides: Dict[str, Any]) -> None:
    cmd = make_cmd(target, overrides, silent=False)
    script = "#!/usr/bin/env bash\nset -euo pipefail\n\n" + " ".join(shlex.quote(x) for x in cmd) + "\n"
    path.write_text(script, encoding="utf-8")
    os.chmod(path, 0o755)


def write_results_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: List[str] = []
    seen = set()
    for r in rows:
        for k in r.keys():
            if k not in seen:
                keys.append(k)
                seen.add(k)

    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def iter_param_grid(param_values: Dict[str, List[Any]]) -> Iterator[Dict[str, Any]]:
    """
    Deterministic cartesian product over param_values.
    Keys order is stable (sorted by key).
    """
    keys = sorted(param_values.keys())
    lists = [param_values[k] for k in keys]
    for vals in __import__("itertools").product(*lists):
        yield {k: v for k, v in zip(keys, vals)}


def choose_subset(items: List[Dict[str, Any]], *, max_trials: Optional[int], seed: int) -> List[Dict[str, Any]]:
    if max_trials is None or max_trials <= 0 or len(items) <= max_trials:
        return items
    rng = random.Random(int(seed))
    idx = list(range(len(items)))
    rng.shuffle(idx)
    idx = idx[: int(max_trials)]
    idx.sort()
    return [items[i] for i in idx]


def run_sweep(
    *,
    run: RunSpec,
    exp: str,
    metric: str,
    base_overrides: Dict[str, Any],
    grid_overrides: List[Dict[str, Any]],
    results_dir: Path,
    max_trials: Optional[int] = None,
    seed: int = 33724876,
    skip_existing: bool = True,
    silent_make: bool = False,
    # Stage-2: deployment-style selection using fit_ops.py + metrics.py.
    # NOTE: default uses validation split to avoid test leakage.
    stage2: bool = False,
    stage2_topk: int = 5,
    stage2_split: str = "val",  # "val" or "test"
    stage2_run_windows_eval: bool = True,
    stage2_op1_target: float = 0.95,
    stage2_op3_target: float = 1.0,
) -> Path:
    """
    Executes a sweep, returns path to best.json.
    """
    repo_root()

    ts = time.strftime("%Y%m%d_%H%M%S")
    sweep_root = results_dir / run.arch / run.dataset / exp
    sweep_root.mkdir(parents=True, exist_ok=True)

    trials = choose_subset(grid_overrides, max_trials=max_trials, seed=seed)

    results_jsonl = sweep_root / "results.jsonl"
    results_csv = sweep_root / "results.csv"
    top10_txt = sweep_root / "top10.txt"
    best_json = sweep_root / "best.json"
    best_cmd = sweep_root / "best_command.sh"
    best_mk = sweep_root / "best_overrides.mk"

    rows: List[Dict[str, Any]] = []
    best_score = float("-inf")
    best_row: Dict[str, Any] = {}
    best_overrides: Dict[str, Any] = {}

    meta = {
        "arch": run.arch,
        "dataset": run.dataset,
        "target": run.target,
        "base_out_dir": run.base_out_dir,
        "metric": metric,
        "exp": exp,
        "timestamp": ts,
        "n_trials": len(trials),
        "max_trials": max_trials,
        "seed": seed,
        "base_overrides": base_overrides,
    }
    (sweep_root / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    def append_jsonl(obj: Dict[str, Any]) -> None:
        with results_jsonl.open("a", encoding="utf-8") as f:
            f.write(json.dumps(obj) + "\n")

    for i, ov in enumerate(trials, start=1):
        overrides = dict(base_overrides)
        overrides.update(ov)

        tag = safe_tag(f"{exp}_t{i:04d}")
        overrides["OUT_TAG"] = tag

        save_dir = Path(run.base_out_dir + tag)
        hist = save_dir / "history.jsonl"
        log_path = sweep_root / "logs" / f"t{i:04d}.log"

        if skip_existing and hist.exists():
            score, best_ep_row = read_history_best(hist, metric=metric)
            status = "skipped_existing"
            rc = 0
        else:
            cmd = make_cmd(run.target, overrides, silent=silent_make)
            print(f"[{run.arch}/{run.dataset}] t{i:04d} starting tag={tag} log={log_path}", flush=True)
            rc = run_subprocess(cmd, log_path)
            status = "ok" if rc == 0 else "fail"
            if hist.exists():
                score, best_ep_row = read_history_best(hist, metric=metric)
            else:
                score, best_ep_row = float("-inf"), {}

        rec = {
            "trial": i,
            "tag": tag,
            "status": status,
            "returncode": rc,
            "score": float(score),
            "save_dir": str(save_dir),
            "history": str(hist),
            "log": str(log_path),
            "overrides": overrides,
            "best_epoch_row": best_ep_row,
        }
        rows.append(rec)
        append_jsonl(rec)

        if float(score) > best_score:
            best_score = float(score)
            best_row = rec
            best_overrides = overrides

        print(f"[{run.arch}/{run.dataset}] t{i:04d} {status:>15} score={score:.4f} tag={tag}", flush=True)

    rows_sorted = sorted(rows, key=lambda r: float(r.get("score", float("-inf"))), reverse=True)

    lines = []
    for r in rows_sorted[:10]:
        bo = r.get("best_epoch_row", {}) or {}
        ep = bo.get("epoch", "")
        lines.append(f"{r['score']:.6f}\ttrial={r['trial']}\tep={ep}\ttag={r['tag']}\tstatus={r['status']}")
    top10_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")

    write_results_csv(results_csv, [
        {
            "trial": r["trial"],
            "tag": r["tag"],
            "status": r["status"],
            "score": r["score"],
            "save_dir": r["save_dir"],
            "epoch": (r.get("best_epoch_row", {}) or {}).get("epoch", ""),
            "val_f1": (r.get("best_epoch_row", {}) or {}).get("val_f1", ""),
            "val_ap": (r.get("best_epoch_row", {}) or {}).get("val_ap", ""),
            "val_auc": (r.get("best_epoch_row", {}) or {}).get("val_auc", ""),
            "val_loss": (r.get("best_epoch_row", {}) or {}).get("val_loss", ""),
            "train_loss": (r.get("best_epoch_row", {}) or {}).get("train_loss", ""),
            "lr": (r.get("best_epoch_row", {}) or {}).get("lr", ""),
            "monitor": (r.get("best_epoch_row", {}) or {}).get("monitor", ""),
            "overrides_json": json.dumps(r.get("overrides", {}), sort_keys=True),
        }
        for r in rows_sorted
    ])

    best_json.write_text(json.dumps(best_row, indent=2), encoding="utf-8")
    dump_best_command(best_cmd, run.target, best_overrides)
    dump_overrides_mk(best_mk, {k: v for k, v in best_overrides.items() if k != "OUT_TAG"})

    # -------------------------
    # Stage-2 selection (fit_ops + metrics)
    # -------------------------
    if stage2:
        if stage2_split not in {"val", "test"}:
            raise ValueError(f"stage2_split must be 'val' or 'test', got: {stage2_split}")

        stage2_root = sweep_root / "stage2"
        stage2_ops_dir = stage2_root / "ops"
        stage2_reports_dir = stage2_root / "reports"
        stage2_logs_dir = stage2_root / "logs"
        stage2_best_json = stage2_root / "best_stage2.json"
        stage2_best_cmd = stage2_root / "best_stage2_command.sh"
        stage2_best_mk = stage2_root / "best_stage2_overrides.mk"
        stage2_top_txt = stage2_root / "top_stage2.txt"
        stage2_csv = stage2_root / "stage2_results.csv"

        stage2_ops_dir.mkdir(parents=True, exist_ok=True)
        stage2_reports_dir.mkdir(parents=True, exist_ok=True)
        stage2_logs_dir.mkdir(parents=True, exist_ok=True)

        # Candidates: top-K by stage-1 metric.
        candidates = [r for r in rows_sorted if r.get("status") in {"ok", "skipped_existing"}]
        if stage2_topk and stage2_topk > 0:
            candidates = candidates[: int(stage2_topk)]

        def _safe_f(x: Any, default: float = float("nan")) -> float:
            try:
                v = float(x)
                return v if math.isfinite(v) else default
            except Exception:
                return default

        def _rank_key(report: Dict[str, Any]) -> Tuple[Any, ...]:
            ops_eval = report.get("ops_eval", {}) or {}
            op1 = ops_eval.get("op1", {}) or {}
            op2 = ops_eval.get("op2", {}) or {}
            op3 = ops_eval.get("op3", {}) or {}

            r1 = _safe_f(op1.get("micro_event_recall"), default=float("nan"))
            f2 = _safe_f(op2.get("micro_event_f1"), default=float("nan"))
            fa3 = _safe_f(op3.get("fa24h"), default=float("inf"))

            feasible = (math.isfinite(r1) and r1 >= float(stage2_op1_target))
            # Prefer configs meeting recall target; among those minimize FA/24h.
            if feasible:
                return (0, fa3, -(_safe_f(f2, default=float("-inf"))), -(_safe_f(r1, default=float("-inf"))))
            # Otherwise: maximize recall first, then minimize FA/24h.
            return (1, -(_safe_f(r1, default=float("-inf"))), fa3, -(_safe_f(f2, default=float("-inf"))))

        stage2_rows: List[Dict[str, Any]] = []

        for r in candidates:
            tag = str(r["tag"])
            ov = (r.get("overrides") or {})
            W = int(ov.get("WIN_W", base_overrides.get("WIN_W", 48)))
            S = int(ov.get("WIN_S", base_overrides.get("WIN_S", 12)))
            win_eval_dir = Path("data") / "processed" / run.dataset / f"windows_eval_W{W}_S{S}"

            # Ensure eval windows exist (cheap when --skip_existing).
            if stage2_run_windows_eval:
                mk_ov = {"WIN_W": W, "WIN_S": S}
                cmd_win = make_cmd(f"windows-eval-{run.dataset}", mk_ov, silent=silent_make)
                log_win = stage2_logs_dir / f"{tag}.windows_eval.log"
                if not (win_eval_dir / stage2_split).exists():
                    print(f"[stage2] windows-eval {run.dataset} W={W} S={S} tag={tag}", flush=True)
                    rc_win = run_subprocess(cmd_win, log_win)
                    if rc_win != 0:
                        print(f"[stage2] windows-eval failed rc={rc_win} tag={tag} (skipping stage2)", flush=True)
                        continue

            ckpt = Path(r["save_dir"]) / "best.pt"
            if not ckpt.exists():
                print(f"[stage2] missing ckpt: {ckpt} (skipping)", flush=True)
                continue

            ops_yaml = stage2_ops_dir / f"{tag}.yaml"
            report_json = stage2_reports_dir / f"{tag}.{stage2_split}.json"

            # 1) fit_ops (val)
            if not ops_yaml.exists():
                cmd_fit = [
                    sys.executable, "-u", "eval/fit_ops.py",
                    "--arch", run.arch,
                    "--val_dir", str(win_eval_dir / "val"),
                    "--ckpt", str(ckpt),
                    "--out", str(ops_yaml),
                    "--ema_alpha", "0.20",
                    "--k", "2",
                    "--n", "3",
                    "--cooldown_s", "30.0",
                    "--tau_low_ratio", "0.78",
                    "--confirm", "1",
                    "--confirm_s", "2.0",
                    "--confirm_min_lying", "0.65",
                    "--confirm_max_motion", "0.08",
                    "--confirm_require_low", "1",
                    "--thr_min", "0.01",
                    "--thr_max", "0.95",
                    "--thr_step", "0.01",
                    "--time_mode", "center",
                    "--merge_gap_s", "1.0",
                    "--overlap_slack_s", "0.5",
                    "--op1_recall", str(stage2_op1_target),
                    "--op3_fa24h", str(stage2_op3_target),
                ]
                log_fit = stage2_logs_dir / f"{tag}.fit_ops.log"
                print(f"[stage2] fit_ops tag={tag}", flush=True)
                rc_fit = run_subprocess(cmd_fit, log_fit)
                if rc_fit != 0 or not ops_yaml.exists():
                    print(f"[stage2] fit_ops failed rc={rc_fit} tag={tag} (skipping)", flush=True)
                    continue

            # 2) metrics on requested split
            if not report_json.exists():
                cmd_met = [
                    sys.executable, "-u", "eval/metrics.py",
                    "--win_dir", str(win_eval_dir / stage2_split),
                    "--ckpt", str(ckpt),
                    "--ops_yaml", str(ops_yaml),
                    "--out_json", str(report_json),
                    "--thr_min", "0.001",
                    "--thr_max", "0.95",
                    "--thr_step", "0.01",
                ]
                log_met = stage2_logs_dir / f"{tag}.metrics_{stage2_split}.log"
                print(f"[stage2] metrics split={stage2_split} tag={tag}", flush=True)
                rc_met = run_subprocess(cmd_met, log_met)
                if rc_met != 0 or not report_json.exists():
                    print(f"[stage2] metrics failed rc={rc_met} tag={tag} (skipping)", flush=True)
                    continue

            try:
                report = json.loads(report_json.read_text(encoding="utf-8"))
            except Exception:
                print(f"[stage2] cannot read report json: {report_json} (skipping)", flush=True)
                continue

            ops_eval = report.get("ops_eval", {}) or {}
            op1 = ops_eval.get("op1", {}) or {}
            op2 = ops_eval.get("op2", {}) or {}
            op3 = ops_eval.get("op3", {}) or {}

            row2 = {
                "trial": r["trial"],
                "tag": tag,
                "score_stage1": float(r.get("score", float("-inf"))),
                "split": stage2_split,
                "op1_recall": _safe_f(op1.get("micro_event_recall")),
                "op1_fa24h": _safe_f(op1.get("fa24h"), default=float("nan")),
                "op2_f1": _safe_f(op2.get("micro_event_f1")),
                "op2_fa24h": _safe_f(op2.get("fa24h"), default=float("nan")),
                "op3_recall": _safe_f(op3.get("micro_event_recall")),
                "op3_fa24h": _safe_f(op3.get("fa24h"), default=float("inf")),
                "ckpt": str(ckpt),
                "ops_yaml": str(ops_yaml),
                "report_json": str(report_json),
            }
            stage2_rows.append(row2)

        if stage2_rows:
            stage2_rows_sorted = sorted(stage2_rows, key=lambda rr: _rank_key(json.loads(Path(rr["report_json"]).read_text(encoding="utf-8"))))
            best2 = stage2_rows_sorted[0]

            # Save stage2 artifacts.
            stage2_best_json.write_text(json.dumps(best2, indent=2), encoding="utf-8")
            # Reuse the *training* command for rerun; keep exact overrides.
            best2_overrides = None
            for r in rows_sorted:
                if str(r.get("tag")) == str(best2.get("tag")):
                    best2_overrides = r.get("overrides")
                    break
            if best2_overrides:
                dump_best_command(stage2_best_cmd, run.target, best2_overrides)
                dump_overrides_mk(stage2_best_mk, {k: v for k, v in best2_overrides.items() if k != "OUT_TAG"})

            # Human-readable top list.
            lines2 = []
            for rr in stage2_rows_sorted[:10]:
                lines2.append(
                    f"trial={rr['trial']:>4} tag={rr['tag']}  OP1(recall={rr['op1_recall']:.3f})  OP3(fa24h={rr['op3_fa24h']:.3f})  OP2(f1={rr['op2_f1']:.3f})"
                )
            stage2_top_txt.write_text("\n".join(lines2) + "\n", encoding="utf-8")

            write_results_csv(stage2_csv, stage2_rows_sorted)

            print(f"\n[stage2-best] tag={best2.get('tag','')} split={stage2_split}", flush=True)
            print(f"  saved: {stage2_best_json}", flush=True)
            if stage2_best_cmd.exists():
                print(f"  rerun: {stage2_best_cmd}", flush=True)
            if stage2_best_mk.exists():
                print(f"  mk   : {stage2_best_mk}", flush=True)
        else:
            print("[stage2] no valid stage2 results (nothing to rank)", flush=True)

    print(f"\n[best] score={best_score:.6f} tag={best_row.get('tag','')}", flush=True)
    print(f"  saved: {best_json}", flush=True)
    print(f"  rerun: {best_cmd}", flush=True)
    print(f"  mk   : {best_mk}", flush=True)
    return best_json
