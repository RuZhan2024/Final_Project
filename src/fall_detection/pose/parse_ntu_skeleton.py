
import os, argparse, glob, numpy as np

NTU_J = 25


def _read_body_block(fp):
    """Read one NTU body block and return (arr[25,3], score)."""
    # NTU variants in the wild encode body metadata as either:
    # - one line with 10 fields (canonical)
    # - ten lines (legacy/custom exports)
    meta0 = next(fp).strip().split()
    if len(meta0) == 1:
        # Legacy style: 10 metadata lines, one scalar each.
        for _ in range(9):
            next(fp)
    joints = int(next(fp).strip())

    arr = np.zeros((NTU_J, 3), np.float32)
    upto = min(int(joints), NTU_J)
    for ji in range(int(joints)):
        parts = next(fp).split()
        if ji >= upto:
            continue
        try:
            arr[ji, 0] = float(parts[0])
            arr[ji, 1] = float(parts[1])
            arr[ji, 2] = float(parts[2])
        except Exception:
            # Leave zeros for malformed rows.
            pass
    # Prefer non-empty body tracks when multiple bodies exist.
    score = float(np.sum(np.abs(arr[:upto])))
    return arr, score


def read_ntu_skeleton(path):
    with open(path, 'r') as f:
        frames = int(next(f).strip())
        seq = np.zeros((frames, NTU_J, 3), np.float32)
        for fi in range(frames):
            bodies = int(next(f).strip())
            if bodies == 0:
                continue
            best_arr = None
            best_score = -1.0
            for _ in range(int(bodies)):
                arr_b, score_b = _read_body_block(f)
                if score_b > best_score:
                    best_arr = arr_b
                    best_score = score_b
            if best_arr is not None:
                seq[fi] = best_arr
    return seq  # [T,25,3]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--skeletons_glob", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--drop_z", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    for p in glob.glob(args.skeletons_glob):
        arr = read_ntu_skeleton(p)
        if args.drop_z:
            xy = arr[..., :2]
            conf = np.ones((xy.shape[0], NTU_J), np.float32)
            out = dict(xy=xy, conf=conf)
        else:
            xyz = arr
            conf = np.ones((xyz.shape[0], NTU_J), np.float32)
            out = dict(xyz=xyz, conf=conf)
        fn = os.path.splitext(os.path.basename(p))[0]
        np.savez_compressed(os.path.join(args.out_dir, f"{fn}.npz"), **out)
        print("[ntu]", p)
