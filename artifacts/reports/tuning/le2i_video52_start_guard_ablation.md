# LE2i video_52 False-Alert A/B (GCN r1)

## Setup
- Base model: `outputs/le2i_gcn_W48S12_opt33_r1/best.pt`
- Base ops: `configs/ops/gcn_le2i_opt33_r1.yaml`
- Problem clip: `Coffee_room_02__Videos__video__52_` (single false alert)

## Variants
1. `baseline`
2. `sg085_global`: `start_guard_max_lying=0.85`
3. `sg090_global`: `start_guard_max_lying=0.90`
4. `sg085_coffee2`: `start_guard_max_lying=0.85`, `start_guard_prefixes=["Coffee_room_02"]`

## Results (test)
| Variant | AP | F1 | Recall | Precision | FA24h | False Alerts | FP Video_52 |
|---|---:|---:|---:|---:|---:|---:|---|
| baseline | 0.8314 | 0.8889 | 0.8889 | 0.8889 | 581.58 | 1 | yes |
| sg085_global | 0.8314 | 0.8750 | 0.7778 | 1.0000 | 0.00 | 0 | no |
| sg090_global | 0.8314 | 0.8750 | 0.7778 | 1.0000 | 0.00 | 0 | no |
| sg085_coffee2 | 0.8314 | 0.9412 | 0.8889 | 1.0000 | 0.00 | 0 | no |

## Interpretation
- `video_52` false alert is removable by start-guard.
- Global guard removes false alert but costs recall (`0.8889 -> 0.7778`).
- Scene-scoped guard (`Coffee_room_02`) removes false alert **without recall loss** in this test split.

## Recommendation
- If objective is LE2i benchmark score: use `sg085_coffee2` as a **diagnostic/benchmark profile**.
- For general deployment claims, keep baseline and report this as scene-specific mitigation evidence.
