#!/usr/bin/env python3
"""Static safety audit: absolute local paths and unsafe np.load flags."""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys


ABS_PATTERNS = (
    re.compile(r"/Users/"),
    re.compile(r"[A-Za-z]:\\"),
)
ALLOW_PICKLE_TRUE = re.compile(r"allow_pickle\s*=\s*True")


def _scan_file(path: Path) -> tuple[list[str], list[str]]:
    text = path.read_text(encoding="utf-8", errors="replace")
    abs_hits: list[str] = []
    pickle_hits: list[str] = []
    for i, line in enumerate(text.splitlines(), start=1):
        if any(p.search(line) for p in ABS_PATTERNS):
            abs_hits.append(f"{path}:{i}: {line.strip()}")
        if ALLOW_PICKLE_TRUE.search(line):
            pickle_hits.append(f"{path}:{i}: {line.strip()}")
    return abs_hits, pickle_hits


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", default="src,scripts,server,configs")
    ap.add_argument("--exclude", default="scripts/audit_static.py")
    args = ap.parse_args()

    roots = [Path(r.strip()) for r in args.roots.split(",") if r.strip()]
    exclude = {Path(p.strip()).as_posix() for p in args.exclude.split(",") if p.strip()}
    targets: list[Path] = []
    for root in roots:
        if root.is_file():
            targets.append(root)
            continue
        if not root.exists():
            continue
        for p in root.rglob("*"):
            if p.suffix.lower() in {".py", ".yaml", ".yml", ".json", ".md", ".txt"} and p.is_file():
                if p.as_posix() in exclude:
                    continue
                targets.append(p)

    abs_hits: list[str] = []
    pickle_hits: list[str] = []
    for p in sorted(set(targets)):
        a, pk = _scan_file(p)
        abs_hits.extend(a)
        pickle_hits.extend(pk)

    failures: list[str] = []
    if abs_hits:
        failures.append(f"absolute-path hits: {len(abs_hits)}")
    if pickle_hits:
        failures.append(f"allow_pickle=True hits: {len(pickle_hits)}")

    if failures:
        print("[fail] static audit failed:")
        for f in failures:
            print(f" - {f}")
        if abs_hits:
            print("[details] absolute paths:")
            for h in abs_hits[:200]:
                print(f"  {h}")
        if pickle_hits:
            print("[details] allow_pickle=True:")
            for h in pickle_hits[:200]:
                print(f"  {h}")
        sys.exit(1)

    print("[ok] static audit passed")


if __name__ == "__main__":
    main()
