#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""core/yamlio.py

Tiny YAML subset loader/dumper (no PyYAML dependency).

Supports:
  key: value
  nested:
    key: value

Values parsed as bool/int/float/str.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Tuple


_RE_INT = re.compile(r"[-+]?\d+\Z")
_RE_FLOAT = re.compile(r"[-+]?(\d+\.\d*|\d*\.\d+)([eE][-+]?\d+)?\Z|[-+]?\d+[eE][-+]?\d+\Z")


def _parse_scalar(s: str) -> Any:
    s = s.strip()
    if not s:
        return ""
    lo = s.lower()
    if lo in {"true", "false"}:
        return lo == "true"
    if lo in {"null", "none"}:
        return None
    if _RE_INT.match(s):
        try:
            return int(s)
        except Exception:
            return s
    if _RE_FLOAT.match(s):
        try:
            return float(s)
        except Exception:
            return s
    # Strip quotes if present
    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        return s[1:-1]
    return s


def yaml_load_simple(path: str) -> Dict[str, Any]:
    """Load a small YAML subset into nested dicts."""
    out: Dict[str, Any] = {}
    stack: List[Tuple[int, Dict[str, Any]]] = [(0, out)]

    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            if not raw.strip() or raw.lstrip().startswith("#"):
                continue
            indent = len(raw) - len(raw.lstrip(" "))
            line = raw.rstrip("\n")
            if ":" not in line:
                continue
            key, val = line.strip().split(":", 1)
            key = key.strip()
            val = val.strip()

            while stack and indent < stack[-1][0]:
                stack.pop()
            cur = stack[-1][1]

            if val == "":
                nxt: Dict[str, Any] = {}
                cur[key] = nxt
                stack.append((indent + 2, nxt))
            else:
                cur[key] = _parse_scalar(val)

    return out


def yaml_dump_simple(obj: Dict[str, Any], path: str) -> None:
    """Write a small YAML subset."""

    def dump_value(v: Any) -> str:
        if v is None:
            return "null"
        if isinstance(v, bool):
            return "true" if v else "false"
        if isinstance(v, (int, float)):
            return str(v)
        # quote strings safely
        return json.dumps(str(v), ensure_ascii=False)

    lines: List[str] = []

    def rec(d: Dict[str, Any], indent: int) -> None:
        for k, v in d.items():
            if isinstance(v, dict):
                lines.append(" " * indent + f"{k}:")
                rec(v, indent + 2)
            else:
                lines.append(" " * indent + f"{k}: {dump_value(v)}")

    rec(obj, 0)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
