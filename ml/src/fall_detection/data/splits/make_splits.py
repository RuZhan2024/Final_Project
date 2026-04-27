#!/usr/bin/env python3
"""
split/make_splits.py  (upgraded)

Create train/val/test stem lists from a labels JSON: {stem: label}.

Design goals
------------
1) Leak-proof: support group-wise splitting (subject / scene / clip) so that
   all stems from the same group go to the same split.
2) Stratified: keep class proportions roughly stable (binary by default, but
   supports multi-class).
3) Deterministic: controlled by --seed; NO use of Python's built-in hash().
4) Stable sizes: when groups have very different sizes, split to match fractions
   by STEM COUNTS (default), not just group counts.

Outputs (in --out_dir)
----------------------
  <prefix>_train.txt
  <prefix>_val.txt
  <prefix>_test.txt
  <prefix>_split_summary.json  (unless --summary_json specified)

CLI compatibility
-----------------
Keeps old flags used by your Makefile:
  --labels_json --out_dir --prefix --seed --train --val --test
  --group_mode --group_regex

New/Improved:
  --group_mode json --group_json <path>
  --group_mode caucafall_subject   (robust Subject-level grouping)
  --balance_by stems|groups
  --ensure_min_per_class <int>
  --summary_json <path>

Recommended for CAUCAFall:
  --group_mode caucafall_subject
OR
  --group_mode regex --group_regex "(?i)^(subject[._-]?\\d+)"
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# -----------------------------
# Label normalization
# -----------------------------
_POS_STR = {"1", "true", "fall", "pos", "positive", "yes"}
_NEG_STR = {"0", "false", "adl", "neg", "negative", "no", "nonfall", "nofall"}


def norm_label(v: Any) -> str:
    """
    Normalize labels to a string class id.

    Binary-friendly:
      - fall-like => "1"
      - adl-like  => "0"

    Multi-class:
      - any other string/int becomes its own class key
    """
    if v is None:
        return "0"
    s = str(v).strip().lower()
    if s in _POS_STR:
        return "1"
    if s in _NEG_STR:
        return "0"
    return str(v).strip()


# -----------------------------
# Stable hashing / determinism
# -----------------------------
def stable_hash_u32(text: str) -> int:
    """
    Stable 32-bit-ish integer hash for strings across platforms/runs.
    Uses MD5 (fast, stable) and takes first 8 hex chars.
    """
    h = hashlib.md5(text.encode("utf-8")).hexdigest()[:8]
    return int(h, 16)


# -----------------------------
# Grouping helpers
# -----------------------------
_CAUC_SUBJ_RE = re.compile(r"(?i)(subject[._-]?\d+)")


def group_id_for(
    stem: str,
    mode: str,
    regex: Optional[str],
    group_map: Optional[Dict[str, str]],
) -> str:
    """
    Return a group id for a given stem.

    Modes:
      - none: each stem is its own group
      - before_dunder: split on first "__"
      - regex: use regex match (group 1 if exists, else full match)
      - caucafall_subject: extract "SubjectXX"/"Subject.XX"/"Subject_XX" anywhere in stem
      - json: lookup stem in group_map, fallback to stem
    """
    if mode == "none":
        return stem

    if mode == "before_dunder":
        return stem.split("__", 1)[0]

    if mode == "caucafall_subject":
        m = _CAUC_SUBJ_RE.search(stem)
        return m.group(1) if m else stem

    if mode == "regex":
        if not regex:
            return stem
        m = re.search(regex, stem)
        if not m:
            return stem
        return m.group(1) if m.groups() else m.group(0)

    if mode == "json":
        if not group_map:
            return stem
        return group_map.get(stem, stem)

    return stem


def load_group_map(group_json: Optional[str]) -> Optional[Dict[str, str]]:
    """
    Accepts JSON in one of these formats:
      1) { "stem": "group_id", ... }
      2) { "stem": {"group": "group_id", ...}, ... }
    """
    if not group_json:
        return None
    p = Path(group_json)
    if not p.exists():
        raise SystemExit(f"[ERR] group_json not found: {p}")
    raw = json.loads(p.read_text(encoding="utf-8", errors="ignore"))
    out: Dict[str, str] = {}
    if isinstance(raw, dict):
        for k, v in raw.items():
            if isinstance(v, dict) and "group" in v:
                out[str(k)] = str(v["group"])
            else:
                out[str(k)] = str(v)
    return out


# -----------------------------
# Splitting helpers
# -----------------------------
def _validate_fracs(train: float, val: float, test: float) -> None:
    total = train + val + test
    if abs(total - 1.0) > 1e-6:
        raise SystemExit(f"[ERR] train+val+test must sum to 1.0, got {total:.6f}")
    if min(train, val, test) < 0:
        raise SystemExit("[ERR] fractions must be non-negative")


def _targets(n: int, train: float, val: float, test: float) -> Tuple[int, int, int]:
    """
    Turn fractions into counts that sum to n (rounded).
    """
    n_train = int(round(train * n))
    n_val = int(round(val * n))
    n_test = n - n_train - n_val

    # Guard against negative due to rounding:
    if n_test < 0:
        n_test = 0
        over = (n_train + n_val) - n
        if over > 0:
            take = min(over, n_val)
            n_val -= take
            over -= take
            n_train = max(0, n_train - over)

    return n_train, n_val, n_test


def split_groups_to_match_targets(
    group_ids: List[str],
    group_sizes: Dict[str, int],
    train: float,
    val: float,
    test: float,
    rng: random.Random,
    balance_by: str = "stems",
) -> Tuple[List[str], List[str], List[str]]:
    """
    Split group_ids into train/val/test.

    balance_by:
      - "groups": split by number of groups (simple)
      - "stems":  split by total stem count across groups (default; better when group sizes vary)

    Determinism:
      - group_ids are sorted before any seeded randomness
      - for stems balancing, largest groups are placed first (bin-pack style)
    """
    balance_by = balance_by.lower()
    if balance_by not in {"groups", "stems"}:
        raise ValueError("balance_by must be 'groups' or 'stems'")

    gids = sorted(set(group_ids))  # deterministic base order

    if balance_by == "groups":
        rng.shuffle(gids)
        n = len(gids)
        t_tr, t_va, _ = _targets(n, train, val, test)
        tr = gids[:t_tr]
        va = gids[t_tr : t_tr + t_va]
        te = gids[t_tr + t_va :]
        return tr, va, te

    # balance_by == "stems"
    total = sum(group_sizes[g] for g in gids)
    t_tr, t_va, t_te = _targets(total, train, val, test)

    # Place large groups first. Use deterministic random tie-break for equal sizes.
    items = [(g, group_sizes[g], rng.random()) for g in gids]
    items.sort(key=lambda x: (-x[1], x[2]))
    ordered = [g for (g, _, _) in items]

    tr: List[str] = []
    va: List[str] = []
    te: List[str] = []
    c_tr = c_va = c_te = 0

    def deficit(cur: int, target: int) -> int:
        return target - cur

    def rel_fill(cur: int, target: int) -> float:
        # lower is "more underfilled"; if target==0, treat as fully filled
        if target <= 0:
            return 1e9
        return float(cur) / float(target)

    for g in ordered:
        sz = group_sizes[g]

        d_tr = deficit(c_tr, t_tr)
        d_va = deficit(c_va, t_va)
        d_te = deficit(c_te, t_te)

        # Prefer the split with the largest positive deficit.
        choices = [("tr", d_tr), ("va", d_va), ("te", d_te)]
        choices.sort(key=lambda x: x[1], reverse=True)

        if choices[0][1] > 0:
            pick = choices[0][0]
        else:
            # All deficits <= 0: pick the split with smallest relative fill
            # (better than always dumping into test).
            fills = [
                ("tr", rel_fill(c_tr, t_tr)),
                ("va", rel_fill(c_va, t_va)),
                ("te", rel_fill(c_te, t_te)),
            ]
            fills.sort(key=lambda x: x[1])
            pick = fills[0][0]

        if pick == "tr":
            tr.append(g)
            c_tr += sz
        elif pick == "va":
            va.append(g)
            c_va += sz
        else:
            te.append(g)
            c_te += sz

    return tr, va, te


def enforce_min_per_class(
    class_to_split_gids: Dict[str, Dict[str, List[str]]],
    min_per_class: int,
) -> None:
    """
    Best-effort: ensure each class has at least min_per_class groups in each split,
    if possible. This helps tiny datasets.

    class_to_split_gids[class]["tr"/"va"/"te"] = [group ids]
    """
    if min_per_class <= 0:
        return

    splits = ("tr", "va", "te")

    for cls, d in class_to_split_gids.items():
        total = sum(len(d[s]) for s in splits)
        if total == 0:
            continue

        # If total groups is too small to satisfy all splits, do best-effort only.
        # (e.g., 2 groups can never cover 3 splits.)
        for s in splits:
            if len(d[s]) >= min_per_class:
                continue

            need = min_per_class - len(d[s])
            # Donors sorted by available surplus (deterministic)
            donors = sorted(
                [x for x in splits if x != s],
                key=lambda x: (len(d[x]), x),
                reverse=True,
            )

            for donor in donors:
                if need <= 0:
                    break
                surplus = len(d[donor]) - min_per_class
                if surplus <= 0:
                    continue

                take = min(need, surplus)
                # Move deterministically from the front (lists are deterministic from split logic)
                moved = d[donor][:take]
                d[donor] = d[donor][take:]
                d[s].extend(moved)
                need -= take


def expand_groups(groups: Dict[str, List[str]], gids: List[str]) -> List[str]:
    out: List[str] = []
    for gid in gids:
        out.extend(groups[gid])
    return sorted(out)


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels_json", required=True, help="Path to labels JSON {stem: label}")
    ap.add_argument("--out_dir", default="configs/splits", help="Output directory for split txt files")
    ap.add_argument("--prefix", default="dataset", help="Prefix for output files (e.g., muvim)")
    ap.add_argument("--seed", type=int, default=33724876)

    ap.add_argument("--train", type=float, default=0.80)
    ap.add_argument("--val", type=float, default=0.10)
    ap.add_argument("--test", type=float, default=0.10)

    ap.add_argument(
        "--group_mode",
        choices=["none", "before_dunder", "regex", "caucafall_subject", "json"],
        default="none",
        help="How to group stems to avoid leakage (group-wise split).",
    )
    ap.add_argument("--group_regex", default=None, help="Regex for group_mode=regex.")
    ap.add_argument("--group_json", default=None, help="JSON mapping stem->group_id for group_mode=json.")

    ap.add_argument(
        "--balance_by",
        choices=["stems", "groups"],
        default="stems",
        help="When group_mode != none, balance split sizes by stem counts (default) or group counts.",
    )
    ap.add_argument(
        "--ensure_min_per_class",
        type=int,
        default=1,
        help="Best-effort minimum number of GROUPS per class per split (helps tiny datasets).",
    )
    ap.add_argument("--summary_json", default=None, help="Optional path to write a JSON split summary.")
    args = ap.parse_args()

    _validate_fracs(args.train, args.val, args.test)

    labels_path = Path(args.labels_json)
    if not labels_path.exists():
        raise SystemExit(f"[ERR] labels_json not found: {labels_path}")

    raw = json.loads(labels_path.read_text(encoding="utf-8", errors="ignore"))
    labels: Dict[str, str] = {str(stem): norm_label(lab) for stem, lab in raw.items()}
    if not labels:
        raise SystemExit("[ERR] labels_json is empty")

    group_map = load_group_map(args.group_json) if args.group_mode == "json" else None

    # Build groups deterministically
    groups: Dict[str, List[str]] = defaultdict(list)
    for stem in sorted(labels.keys()):
        gid = group_id_for(stem, args.group_mode, args.group_regex, group_map)
        groups[gid].append(stem)

    # Group sizes (stems)
    group_sizes = {gid: len(stems) for gid, stems in groups.items()}

    # Assign group label:
    # - binary: group is positive if ANY positive exists (safer for fall detection)
    # - multi-class: majority vote (stable tie-break)
    classes = sorted(set(labels.values()))
    is_binary_01 = set(classes) <= {"0", "1"}

    group_class: Dict[str, str] = {}
    for gid, stems in groups.items():
        labs = [labels[s] for s in stems]
        if is_binary_01:
            group_class[gid] = "1" if any(l == "1" for l in labs) else "0"
        else:
            cnt = Counter(labs)
            best_n = max(cnt.values())
            # stable tie-break: smallest label string
            best_labels = sorted([k for k, v in cnt.items() if v == best_n])
            group_class[gid] = best_labels[0]

    # Split per class at GROUP level
    class_to_gids: Dict[str, List[str]] = defaultdict(list)
    for gid, cls in group_class.items():
        class_to_gids[cls].append(gid)

    class_to_split_gids: Dict[str, Dict[str, List[str]]] = {}

    for cls, gids in sorted(class_to_gids.items(), key=lambda kv: kv[0]):
        # Stable per-class RNG (NO Python hash()).
        cls_offset = stable_hash_u32(cls) % 1_000_000_000
        rng_cls = random.Random(args.seed + cls_offset)

        tr, va, te = split_groups_to_match_targets(
            gids,
            group_sizes,
            args.train,
            args.val,
            args.test,
            rng_cls,
            balance_by=args.balance_by,
        )
        class_to_split_gids[cls] = {"tr": tr, "va": va, "te": te}

    # Optional: ensure at least N groups per class per split
    enforce_min_per_class(class_to_split_gids, args.ensure_min_per_class)

    # Combine groups from all classes
    tr_gids: List[str] = []
    va_gids: List[str] = []
    te_gids: List[str] = []

    for cls in sorted(class_to_split_gids.keys()):
        tr_gids.extend(class_to_split_gids[cls]["tr"])
        va_gids.extend(class_to_split_gids[cls]["va"])
        te_gids.extend(class_to_split_gids[cls]["te"])

    train_stems = expand_groups(groups, tr_gids)
    val_stems = expand_groups(groups, va_gids)
    test_stems = expand_groups(groups, te_gids)

    # Safety checks (no overlaps; no missing)
    set_tr, set_va, set_te = set(train_stems), set(val_stems), set(test_stems)
    if (set_tr & set_va) or (set_tr & set_te) or (set_va & set_te):
        raise SystemExit("[ERR] split overlap detected (grouping bug or duplicate stems).")

    if len(set_tr) + len(set_va) + len(set_te) != len(labels):
        missing = set(labels) - (set_tr | set_va | set_te)
        raise SystemExit(
            f"[ERR] missing stems in splits: {len(missing)} "
            f"(example: {next(iter(missing)) if missing else 'n/a'})"
        )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def write_txt(name: str, stems: List[str]) -> Path:
        p = out_dir / name
        p.write_text("\n".join(stems) + ("\n" if stems else ""), encoding="utf-8")
        return p

    p_tr = write_txt(f"{args.prefix}_train.txt", train_stems)
    p_va = write_txt(f"{args.prefix}_val.txt", val_stems)
    p_te = write_txt(f"{args.prefix}_test.txt", test_stems)

    # Summary helpers
    def count_by_class(stems: List[str]) -> Dict[str, int]:
        c: Dict[str, int] = defaultdict(int)
        for s in stems:
            c[labels[s]] += 1
        return dict(sorted(c.items(), key=lambda kv: kv[0]))

    summary = {
        "labels_json": str(labels_path),
        "prefix": args.prefix,
        "seed": args.seed,
        "fractions": {"train": args.train, "val": args.val, "test": args.test},
        "grouping": {
            "mode": args.group_mode,
            "regex": args.group_regex,
            "group_json": args.group_json,
            "balance_by": args.balance_by,
        },
        "counts": {
            "total": len(labels),
            "train": len(train_stems),
            "val": len(val_stems),
            "test": len(test_stems),
        },
        "class_counts_total": count_by_class(list(labels.keys())),
        "class_counts_train": count_by_class(train_stems),
        "class_counts_val": count_by_class(val_stems),
        "class_counts_test": count_by_class(test_stems),
        "groups": {
            "total_groups": len(groups),
            "groups_by_class": dict(sorted(Counter(group_class.values()).items(), key=lambda kv: kv[0])),
        },
        "files": {"train": str(p_tr), "val": str(p_va), "test": str(p_te)},
    }

    if args.summary_json:
        sp = Path(args.summary_json)
    else:
        sp = out_dir / f"{args.prefix}_split_summary.json"

    sp.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print("[OK] wrote splits:")
    print(" ", p_tr, len(train_stems))
    print(" ", p_va, len(val_stems))
    print(" ", p_te, len(test_stems))
    print("[OK] wrote summary:", sp)
    print("[class total]", summary["class_counts_total"])
    print("[class train]", summary["class_counts_train"])
    print("[class val]  ", summary["class_counts_val"])
    print("[class test] ", summary["class_counts_test"])
    print(f"[group] mode={args.group_mode} groups={len(groups)} balance_by={args.balance_by}")


if __name__ == "__main__":
    main()
