#!/usr/bin/env python3
import argparse, itertools, json
from pathlib import Path

from sweep_lib_min import best_metric, run_make, ensure_exists, deterministic_sample, write_best_files

def run_one(target: str, save_dir_prefix: str, tag: str, overrides: dict, metric_key: str, dry_run: bool) -> tuple:
    cmd = run_make(target, tag, overrides, dry_run=dry_run)
    save_dir = Path(f"{save_dir_prefix}{tag}")
    hist = save_dir / "history.jsonl"
    if dry_run:
        return float("-inf"), cmd
    ensure_exists(hist, "Training didn’t write history.jsonl? Check your trainer save_dir.")
    score = best_metric(hist, key=metric_key)
    print(f"[result] {tag} best_{metric_key}={score:.4f}")
    return score, cmd

def main():
    ap = argparse.ArgumentParser(description="Grid/sampled sweep: TCN on CAUCAFall (Makefile-driven).")
    ap.add_argument("--exp", default="tcn_caucafall", help="Experiment name (used under outputs/sweeps).")
    ap.add_argument("--metric_key", default="monitor_score", help="history.jsonl key to maximize.")
    ap.add_argument("--max_runs", type=int, default=100, help="Deterministically sample this many runs (0 = full grid).")
    ap.add_argument("--seed", type=int, default=33724876, help="Sampling seed.")
    ap.add_argument("--dry_run", action="store_true", help="Print commands only; don't run training.")
    args = ap.parse_args()

    target = "train-tcn-caucafall"
    save_dir_prefix = "outputs/caucafall_tcn_fps_W48S12"

    # ---- Tuned parameter value lists (edit freely) ----
    lrs = [1e-3, 7e-4, 5e-4, 3e-4]
    dropouts = [0.20, 0.30, 0.40]
    hiddens = [128, 192, 256]
    num_blocks = [4, 5]
    kernels = [3, 5]
    mask_joint_ps = [0.10, 0.15, 0.20]
    mask_frame_ps = [0.05, 0.10]
    losses = ["bce", "focal"]
    focal_gammas = [2.0, 3.0]
    balanced_sampler = [0, 1]
    pos_weight = ["auto", "none"]  # try none if sampler is used (sometimes stabilizes)

    grid = []
    for lr, do, hid, nb, k, mj, mf, loss, fg, bs, pw in itertools.product(
        lrs, dropouts, hiddens, num_blocks, kernels, mask_joint_ps, mask_frame_ps,
        losses, focal_gammas, balanced_sampler, pos_weight
    ):
        if loss != "focal" and fg != focal_gammas[0]:
            continue
        # mild pruning to avoid redundant/odd combos
        if bs == 0 and pw == "none":
            continue
        tag = f"_lr{lr:g}_do{do:g}_h{hid}_b{nb}_k{k}_mj{mj:g}_mf{mf:g}_{loss}_bs{bs}_pw{pw}"
        overrides = {
            "LR_TCN_CAUC": lr,
            "TCN_DROPOUT": do,
            "TCN_HIDDEN": hid,
            "TCN_NUM_BLOCKS": nb,
            "TCN_KERNEL": k,
            "TCN_MASK_JOINT_P": mj,
            "TCN_MASK_FRAME_P": mf,
            "TCN_LOSS": loss,
            "TCN_FOCAL_GAMMA": fg,
            "TCN_BALANCED_SAMPLER": bs,
            "TCN_POS_WEIGHT": pw,
        }
        grid.append((tag, overrides))

    runs = deterministic_sample(grid, seed=args.seed, max_runs=args.max_runs)
    print(f"[info] total_grid={len(grid)}  running={len(runs)}  target={target}  metric={args.metric_key}")

    results = []
    for i, (tag, overrides) in enumerate(runs, 1):
        try:
            print(f"[run {i}/{len(runs)}]")
            score, cmd = run_one(target, save_dir_prefix, tag, overrides, args.metric_key, args.dry_run)
            results.append((score, tag, overrides, cmd))
        except Exception as e:
            print(f"[fail] {tag}: {e}")
            results.append((float("-inf"), tag, overrides, ["make", target, f"OUT_TAG={tag}"] + [f"{k}={v}" for k, v in overrides.items()]))

    results.sort(reverse=True, key=lambda x: x[0])

    print("\n=== TOP 10 ===")
    for score, tag, ov, _cmd in results[:10]:
        print(f"{score:8.4f}  {tag}  {json.dumps(ov, sort_keys=True)}")

    out = Path("outputs/sweeps") / "tcn" / "caucafall" / args.exp
    out.mkdir(parents=True, exist_ok=True)

    tsv = out / "results.tsv"
    with tsv.open("w", encoding="utf-8") as f:
        f.write("score\ttag\toverrides_json\n")
        for score, tag, ov, _cmd in results:
            f.write(f"{score}\t{tag}\t{json.dumps(ov, sort_keys=True)}\n")
    print(f"[done] Wrote: {tsv}")

    if results and results[0][0] != float("-inf"):
        write_best_files(out, results[0])
        print(f"[done] Best rerun: {out / 'best_command.sh'}")
    else:
        print("[warn] No successful runs to save as best.")

if __name__ == "__main__":
    main()
