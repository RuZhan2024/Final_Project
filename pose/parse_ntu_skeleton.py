#!/usr/bin/env python3
import os, argparse, glob, numpy as np

NTU_J = 25

def read_ntu_skeleton(path):
    with open(path, 'r') as f:
        lines = [l.strip() for l in f]
    idx = 0
    frames = int(lines[idx]); idx += 1
    seq = []
    for _ in range(frames):
        bodies = int(lines[idx]); idx += 1
        if bodies == 0:
            seq.append(np.zeros((NTU_J, 3), np.float32))
            continue
        idx += 10  # skip body metadata
        joints = int(lines[idx]); idx += 1
        xyz = []
        for _ in range(joints):
            parts = list(map(float, lines[idx].split()))
            x, y, z = parts[0], parts[1], parts[2]
            xyz.append([x, y, z]); idx += 1
        arr = np.array(xyz, dtype=np.float32)
        if arr.shape[0] != NTU_J:
            if arr.shape[0] < NTU_J:
                pad = np.zeros((NTU_J - arr.shape[0], 3), np.float32)
                arr = np.concatenate([arr, pad], 0)
            else:
                arr = arr[:NTU_J]
        seq.append(arr)
    return np.stack(seq)  # [T,25,3]

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
