# TCN Round-2 Results (CAUCAFall)

Date: 2026-03-04

## Compared runs

| run | AP | TP | FP | Event Recall | Event Precision | Event F1 | FA/24h | Mean Delay (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `r1_augreg` (baseline) | 0.9691 | 5 | 0 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 4.1739 |
| `r2_train_hneg` | 0.9680 | 4 | 0 | 0.8000 | 1.0000 | 0.8889 | 0.0000 | 4.5652 |
| `r2_train_hneg_plus` | 0.9693 | 4 | 0 | 0.8000 | 1.0000 | 0.8889 | 0.0000 | 4.5652 |

## Outcome

- Round-2 did **not** beat the baseline on deployment-facing event recall/F1.
- Both Round-2 variants missed 1 of 5 events on test (`TP=4`), despite keeping `FP=0`.
- Best operational choice remains:
  - `outputs/caucafall_tcn_W48S12_r1_augreg/best.pt`
  - `configs/ops/tcn_caucafall_r1_augreg.yaml` (or promoted OP profile)

## Decision

- Mark `r2_train_hneg` and `r2_train_hneg_plus` as **rejected for promotion**.
- Keep these runs as evidence that train-only hard-negative replay + stronger regularization can trade recall down on CAUCAFall.
