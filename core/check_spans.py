#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
core/check_spans.py

Small sanity checker for span-based supervision.

Why this exists
---------------
It's very easy to accidentally produce an EMPTY spans JSON (e.g. wrong parsing mode,
missing per-frame files, wrong class_id). If you then build windows without noticing,
your training/evaluation can silently fall back to clip-level labels and make event
metrics meaningless.

This checker is intentionally conservative:
- With --require_nonempty 1: fail if spans JSON has zero span entries.
- Optionally, you can require a minimum coverage of positive-labeled videos.

Expected inputs
---------------
labels_json: {stem: label} (label can be 0/1, true/false, "fall"/"adl", etc.)
spans_json : {stem: [[start, stop), ...], ...}
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple


_POS_STR = {"1", "true", "fall", "pos", "positive", "yes", "y", "t"}
_NEG_STR = {"0", "false", "adl", "neg", "negative", "no", "n", "f", "nonfall", "nofall", "normal"}


def to_binary_label(v: Any) -> int:
    if isinstance(v, bool):
        return int(v)
    if isinstance(v, (int, float)):
        return 1 if float(v) >= 0.5 else 0
    s = str(v).strip().lower()
    if s in _POS_STR:
        return 1
    if s in _NEG_STR:
        return 0
    # unknown strings default to negative (safe)
    return 0


def normalize_spans(raw: Any) -> Dict[str, List[Tuple[int, int]]]:
    out: Dict[str, List[Tuple[int, int]]] = {}
    if not isinstance(raw, dict):
        return out
    for k, v in raw.items():
        if not v:
            continue
        spans: List[Tuple[int, int]] = []
        if isinstance(v, list):
            for pair in v:
                if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                    continue
                try:
                    a = int(pair[0]); b = int(pair[1])
                except Exception:
                    continue
                if b > a:
                    spans.append((a, b))
        if spans:
            out[str(k)] = spans
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels_json", required=True)
    ap.add_argument("--spans_json", required=True)
    ap.add_argument("--require_nonempty", type=int, default=1,
                    help="If 1: fail when spans contain zero entries.")
    ap.add_argument("--require_pos_coverage", type=int, default=0,
                    help="If 1: require that a minimum fraction of POS stems have spans.")
    ap.add_argument("--min_pos_coverage", type=float, default=0.80,
                    help="Minimum fraction of POS stems that must have spans (only if require_pos_coverage=1).")
    args = ap.parse_args()

    lp = Path(args.labels_json)
    sp = Path(args.spans_json)

    if not lp.exists():
        raise SystemExit(f"[ERR] labels_json not found: {lp}")
    if not sp.exists():
        raise SystemExit(f"[ERR] spans_json not found: {sp}")

    labels_raw = json.loads(lp.read_text(encoding="utf-8"))
    spans_raw = json.loads(sp.read_text(encoding="utf-8"))

    labels = {str(k): to_binary_label(v) for k, v in labels_raw.items()} if isinstance(labels_raw, dict) else {}
    spans = normalize_spans(spans_raw)

    n_pos = sum(1 for v in labels.values() if v == 1)
    n_total = len(labels)
    n_spanned = len(spans)

    if int(args.require_nonempty) == 1 and n_spanned == 0:
        raise SystemExit(
            f"[ERR] spans_json is empty: {sp}\n"
            "      This usually means per-frame parsing failed or the wrong mode/class_id was used.\n"
            "      Fix spans first, or set CAUCA_REQUIRE_SPANS=0 to continue (not recommended)."
        )

    # POS coverage
    if n_pos > 0:
        pos_with_spans = sum(1 for stem, y in labels.items() if y == 1 and stem in spans)
        cov = float(pos_with_spans) / float(n_pos)
    else:
        pos_with_spans = 0
        cov = 1.0

    print(f"[ok] labels={n_total}  pos={n_pos}  stems_with_spans={n_spanned}  pos_with_spans={pos_with_spans}  pos_coverage={cov:.3f}")

    if int(args.require_pos_coverage) == 1 and n_pos > 0 and cov < float(args.min_pos_coverage):
        raise SystemExit(
            f"[ERR] POS span coverage too low: {cov:.3f} < {float(args.min_pos_coverage):.3f}\n"
            f"      pos={n_pos} pos_with_spans={pos_with_spans}.\n"
            "      If your dataset legitimately lacks span info for some clips, lower min_pos_coverage or disable this check."
        )


if __name__ == "__main__":
    main()
