
import argparse, glob, os, pathlib, re

def norm(s: str) -> str:
    # case-insensitive; collapse spaces/underscores so "Lecture room" == "lecture_room"
    return re.sub(r'[\s_]+', '_', s.lower())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz_dir", required=True, help="Root of LE2i NPZs (from extract_2d.py)")
    ap.add_argument("--out", required=True, help="Output txt with one NPZ stem per line")
    ap.add_argument("--scenes", nargs="+", required=True, help='Scene names to match, e.g. Office "Lecture room"')
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.npz_dir, "**", "*.npz"), recursive=True))
    if not files:
        raise SystemExit(f"[ERR] No NPZ under {args.npz_dir}")

    scene_keys = [norm(s) for s in args.scenes]
    picked = []
    for p in files:
        pkey = norm(p)  # full path normalized
        if any(sk in pkey for sk in scene_keys):
            picked.append(pathlib.Path(p).stem)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        f.write("\n".join(picked) + ("\n" if picked else ""))

    print(f"[OK] wrote {len(picked)} stems → {args.out}")
    if picked[:10]:
        print("[sample]", ", ".join(picked[:10]))

if __name__ == "__main__":
    main()
