#!/usr/bin/env python3
"""Promote a validated TCN profile into Makefile defaults and canonical ops.

Usage:
  python tools/promote_tcn_defaults.py \
    --profile_json artifacts/reports/tuning/tcn_promotion_profile.json
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path


KEY_MAP = {
    "dropout": "TCN_DROPOUT",
    "mask_joint_p": "TCN_MASK_JOINT_P",
    "mask_frame_p": "TCN_MASK_FRAME_P",
    "weight_decay": "TCN_WEIGHT_DECAY",
    "label_smoothing": "TCN_LABEL_SMOOTHING",
    "hard_neg_mult": "TCN_HARD_NEG_MULT",
    "hard_neg_list": "TCN_HARD_NEG_LIST",
    "resume": "TCN_RESUME",
}


def _replace_make_var(text: str, key: str, value: str) -> str:
    pat = re.compile(rf"^({re.escape(key)}\s*\?=\s*).*$", re.MULTILINE)
    if pat.search(text):
        return pat.sub(lambda m: f"{m.group(1)}{value}", text, count=1)
    return text + f"\n{key} ?= {value}\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile_json", required=True)
    ap.add_argument("--makefile", default="Makefile")
    ap.add_argument("--canonical_ops", default="configs/ops/tcn_caucafall.yaml")
    args = ap.parse_args()

    profile = json.loads(Path(args.profile_json).read_text(encoding="utf-8"))
    tcn = profile.get("tcn_defaults", {})
    ops_yaml = profile.get("ops_yaml")
    if not isinstance(tcn, dict):
        raise SystemExit("profile.tcn_defaults must be an object")
    if not ops_yaml:
        raise SystemExit("profile.ops_yaml is required")

    mk_path = Path(args.makefile)
    mk_text = mk_path.read_text(encoding="utf-8")

    changed = []
    for src_k, mk_k in KEY_MAP.items():
        if src_k not in tcn:
            continue
        v = tcn[src_k]
        if isinstance(v, str):
            value = v
        elif isinstance(v, (int, float)):
            value = str(v)
        elif v is None:
            value = ""
        else:
            raise SystemExit(f"unsupported value type for {src_k}: {type(v)}")
        mk_text = _replace_make_var(mk_text, mk_k, value)
        changed.append((mk_k, value))

    mk_path.write_text(mk_text, encoding="utf-8")

    src_ops = Path(ops_yaml)
    dst_ops = Path(args.canonical_ops)
    if not src_ops.exists():
        raise SystemExit(f"ops_yaml not found: {src_ops}")
    dst_ops.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src_ops, dst_ops)

    print("[ok] promoted TCN defaults in Makefile:")
    for k, v in changed:
        print(f"  - {k}={v}")
    print(f"[ok] canonical ops updated: {dst_ops} <- {src_ops}")


if __name__ == "__main__":
    main()
