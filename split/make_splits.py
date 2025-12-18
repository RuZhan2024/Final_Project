
"""
Stratified train/val/test split from a labels JSON.

Input
-----
--labels_json : path to JSON {stem: label}
                where label is e.g. "fall", "adl", "nonfall", 0/1, etc.
--out_dir     : where to write split txt files (default: configs/splits)
--seed        : RNG seed (default: 33724876)
--train/--val/--test : fractions; if --test omitted, it's 1 - train - val
--prefix      : filename prefix for outputs. If omitted, inferred from labels_json
                e.g. labels/le2i.json -> 'le2i'  → writes:
                    out_dir/le2i_train.txt, le2i_val.txt, le2i_test.txt

Notes
-----
- Only uses two classes: "fall" treated as positive, everything else as negative.
- Ensures stratified split (same fractions for pos/neg lists separately).
- Works for LE2I, URFD, CAUCAFall, MUVIM as long as labels_json exists.
"""

import argparse, json, os, random, pathlib, sys


def infer_prefix_from_path(p: str) -> str:
    name = pathlib.Path(p).stem
    # handle common names like 'urfd_auto' → 'urfd'
    if name.endswith("_auto"):
        name = name[:-5]
    return name


def split_list(lst, f_train, f_val, f_test):
    n = len(lst)
    i = int(f_train * n)
    j = int((f_train + f_val) * n)
    # remainder goes to test
    return lst[:i], lst[i:j], lst[j:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels_json", required=True)
    ap.add_argument("--out_dir", default="configs/splits")
    ap.add_argument("--seed", type=int, default=33724876)
    ap.add_argument("--train", type=float, default=0.8)
    ap.add_argument("--val",   type=float, default=0.1)
    ap.add_argument("--test",  type=float, default=None,
                    help="If omitted, computed as 1 - train - val")
    ap.add_argument("--prefix", type=str, default=None,
                    help="Prefix for output filenames (e.g., 'urfd', 'muvim').")
    args = ap.parse_args()

    if args.test is None:
        args.test = 1.0 - args.train - args.val

    if args.train < 0 or args.val < 0 or args.test < 0:
        sys.exit("[ERR] Negative split fraction. Check train/val/test.")

    tot = args.train + args.val + args.test
    if abs(tot - 1.0) > 1e-6:
        sys.exit(f"[ERR] Fractions must sum to 1.0 (got {tot:.4f}).")

    with open(args.labels_json, "r") as f:
        labels = json.load(f)

    # separate positives/negatives
    pos, neg = [], []
    for stem, lab in labels.items():
        lab_s = str(lab).lower().strip()
        # Treat anything clearly "fall-like" as positive
        if lab_s in {"fall", "1", "true", "pos", "positive"}:
            pos.append(stem)
        else:
            # "adl", "nonfall", "0", "false", etc. → negative
            neg.append(stem)

    random.seed(args.seed)
    random.shuffle(pos)
    random.shuffle(neg)

    tr_p, va_p, te_p = split_list(pos, args.train, args.val, args.test)
    tr_n, va_n, te_n = split_list(neg, args.train, args.val, args.test)

    train_stems = tr_p + tr_n
    val_stems   = va_p + va_n
    test_stems  = te_p + te_n

    # shuffle within splits to mix pos/neg
    random.shuffle(train_stems)
    random.shuffle(val_stems)
    random.shuffle(test_stems)

    os.makedirs(args.out_dir, exist_ok=True)
    prefix = args.prefix or infer_prefix_from_path(args.labels_json)

    out_train = os.path.join(args.out_dir, f"{prefix}_train.txt")
    out_val   = os.path.join(args.out_dir, f"{prefix}_val.txt")
    out_test  = os.path.join(args.out_dir, f"{prefix}_test.txt")

    with open(out_train, "w") as f: f.write("\n".join(train_stems) + "\n")
    with open(out_val,   "w") as f: f.write("\n".join(val_stems)   + "\n")
    with open(out_test,  "w") as f: f.write("\n".join(test_stems)  + "\n")

    print(f"[OK] wrote splits to {args.out_dir}")
    print(f"  prefix: {prefix}")
    print(f"  pos={len(pos)} neg={len(neg)} total={len(labels)}")
    print(f"  train={len(train_stems)} val={len(val_stems)} test={len(test_stems)}")


if __name__ == "__main__":
    main()
