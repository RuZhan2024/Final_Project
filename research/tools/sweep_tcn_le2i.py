#!/usr/bin/env python3
import itertools, json, subprocess
from pathlib import Path

def best_val_ap(history_path: Path) -> float:
    best = float("-inf")
    with history_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            # train_tcn.py logs val_ap each epoch
            best = max(best, float(row.get("val_ap", float("-inf"))))
    return best

def run_one(tag: str, overrides: dict) -> float:
    cmd = ["make", "train-tcn-le2i", f"OUT_TAG={tag}"] + [f"{k}={v}" for k, v in overrides.items()]
    print("\n>>", " ".join(cmd))
    subprocess.run(cmd, check=True)

    save_dir = Path(f"outputs/le2i_tcn_W48S12{tag}")
    hist = save_dir / "history.jsonl"
    if not hist.exists():
        raise FileNotFoundError(f"Missing {hist} (training didn’t write history?)")
    score = best_val_ap(hist)
    print(f"[result] {tag} best_val_ap={score:.4f}")
    return score

def main():
    lrs = [1e-3, 5e-4, 3e-4]
    dropouts = [0.10, 0.20, 0.30]
    hiddens = [128, 256]
    mask_joint_ps = [0.10, 0.15]
    mask_frame_ps = [0.05, 0.10]

    results = []
    for lr, do, hid, mj, mf in itertools.product(lrs, dropouts, hiddens, mask_joint_ps, mask_frame_ps):
        tag = f"_lr{lr:g}_do{do:g}_h{hid}_mj{mj:g}_mf{mf:g}"
        overrides = {
            "LR_TCN_LE2I": lr,
            "TCN_DROPOUT": do,
            "TCN_HIDDEN": hid,
            "MASK_JOINT_P": mj,
            "MASK_FRAME_P": mf,
        }
        try:
            score = run_one(tag, overrides)
            results.append((score, tag, overrides))
        except subprocess.CalledProcessError:
            results.append((float("-inf"), tag, overrides))

    results.sort(reverse=True, key=lambda x: x[0])
    print("\n=== TOP 10 (by best val_ap) ===")
    for score, tag, ov in results[:10]:
        print(f"{score:8.4f}  {tag}  {ov}")

    # Save a CSV-ish summary
    out = Path("outputs/sweeps")
    out.mkdir(parents=True, exist_ok=True)
    summary = out / "sweep_tcn_le2i_results.tsv"
    with summary.open("w", encoding="utf-8") as f:
        f.write("best_val_ap\ttag\toverrides\n")
        for score, tag, ov in results:
            f.write(f"{score}\t{tag}\t{json.dumps(ov, sort_keys=True)}\n")
    print(f"\nWrote: {summary}")

if __name__ == "__main__":
    main()
