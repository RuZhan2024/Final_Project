#!/usr/bin/env python3
"""
Minimal Makefile-driven sweeper (style matches your sweep_tcn_le2i.py),
but with:
  - correct Makefile variable names for each trainer
  - monitor_score parsing (robust across ap/f1 monitors)
  - optional deterministic sampling of a large grid (--max_runs)
  - best_command.sh + best_overrides.json saved automatically
"""

import argparse
import itertools
import json
import os
import random
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple, Optional


def best_metric(history_path: Path, key: str = "monitor_score") -> float:
    """Return max(key) over history.jsonl."""
    best = float("-inf")
    with history_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            v = row.get(key, None)
            if v is None:
                continue
            try:
                best = max(best, float(v))
            except Exception:
                pass
    return best


def run_make(target: str, tag: str, overrides: Dict[str, object], dry_run: bool = False) -> List[str]:
    cmd = ["make", target, f"OUT_TAG={tag}"] + [f"{k}={v}" for k, v in overrides.items()]
    print("\n>>", " ".join(cmd))
    if dry_run:
        return cmd
    subprocess.run(cmd, check=True)
    return cmd


def ensure_exists(p: Path, hint: str) -> None:
    if not p.exists():
        raise FileNotFoundError(f"Missing {p}. {hint}")


def write_best_files(
    out_sweep_dir: Path,
    best_item: Tuple[float, str, Dict[str, object], List[str]],
) -> None:
    score, tag, overrides, cmd = best_item
    out_sweep_dir.mkdir(parents=True, exist_ok=True)

    (out_sweep_dir / "best.json").write_text(
        json.dumps(
            {"score": score, "tag": tag, "overrides": overrides, "cmd": cmd},
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    # Re-run command script
    sh = out_sweep_dir / "best_command.sh"
    sh.write_text("#!/bin/bash\nset -euo pipefail\n" + " ".join(cmd) + "\n", encoding="utf-8")
    os.chmod(sh, 0o755)

    # Simple overrides file (json)
    (out_sweep_dir / "best_overrides.json").write_text(
        json.dumps(overrides, indent=2, sort_keys=True), encoding="utf-8"
    )


def deterministic_sample(items: List[Tuple], seed: int, max_runs: int) -> List[Tuple]:
    if max_runs <= 0 or max_runs >= len(items):
        return items
    rng = random.Random(seed)
    idx = list(range(len(items)))
    rng.shuffle(idx)
    pick = sorted(idx[:max_runs])
    return [items[i] for i in pick]
