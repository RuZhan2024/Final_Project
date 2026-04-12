# Native-FPS sweep scripts (Option A)

These sweeps are compatible with the **Makefile you attached** (no resampling / no `*-30` targets).
They run the existing targets:

- `train-tcn-le2i`, `train-gcn-le2i`
- `train-tcn-caucafall`, `train-gcn-caucafall`

Each trial sets a unique `OUT_TAG`, so runs never overwrite.

## 0) Run once: build windows (optional but recommended)
From repo root:

```bash
make pipeline-le2i-noextract
make pipeline-caucafall-noextract
```

(If you *haven’t* extracted raw datasets yet, use `pipeline-le2i` / `pipeline-caucafall`.)

## 1) Run sweeps
From repo root:

```bash
python3 tools/sweeps/sweep_tcn_le2i.py --exp tcn_le2i_s1 --trials 60
python3 tools/sweeps/sweep_gcn_le2i.py --exp gcn_le2i_s1 --trials 80

python3 tools/sweeps/sweep_tcn_caucafall.py --exp tcn_cauc_s1 --trials 60
python3 tools/sweeps/sweep_gcn_caucafall.py --exp gcn_cauc_s1 --trials 80
```

### What `--exp` means
Just a name for the output subfolder under `outputs/sweeps/...`. It does **not** change training.

### What `--trials 60` means
Run **60 training runs** (60 different hyperparameter configs).  
It does **not** mean 60 epochs. Epochs are controlled by Makefile variables (`EPOCHS`, `EPOCHS_GCN`).

If you pass `--trials 0`, the script runs **the full grid** (can be very large).

## 2) Where results go
Example:

```
outputs/sweeps/tcn/le2i/tcn_le2i_s1/
  results.jsonl
  results.csv
  top10.txt
  best.json
  best_command.sh
  best_overrides.mk
  logs/t0001.log ...
```

Re-run the best command:

```bash
bash outputs/sweeps/tcn/le2i/tcn_le2i_s1/best_command.sh
```

## Notes
- Scoring uses `monitor_score` from `history.jsonl` by default. This matches how the trainer selects `best.pt`.
- LE2i TCN typically monitors AP (`TCN_MONITOR_LE2I=ap`), CAUCAFall monitors F1.

python3 tools/sweeps/sweep_tcn_le2i.py --exp tcn_le2i_s2 --trials 60 --stage2 --stage2_topk 5
python3 tools/sweeps/sweep_tcn_caucafall.py --exp tcn_cauc_s2 --trials 60 --stage2 --stage2_topk 5
python3 tools/sweeps/sweep_gcn_le2i.py --exp gcn_le2i_s2 --trials 80 --stage2 --stage2_topk 5
python3 tools/sweeps/sweep_gcn_caucafall.py --exp gcn_cauc_s2 --trials 80 --stage2 --stage2_topk 5


