# EXPERIMENT_MATRIX

Date: 2026-03-02
Constraint: no expensive full sweeps until P0 gates pass.

## 1) Matrix

| Exp ID | Toggle / change | Datasets | Primary metrics | Secondary metrics | Compute budget | Expected interaction |
|---|---|---|---|---|---|---|
| E00 | Refit ops with `ALERT_CONFIRM=0` control (no model retrain) | LE2i, CAUCAFall | event recall, event F1 | FA/24h, delay | very low | tests whether current collapse is policy-only |
| E01 | Refit ops with confirm on but relaxed thresholds grid | LE2i, CAUCAFall | event recall, FA/24h | precision, delay | low | isolates confirm-threshold sensitivity |
| E02 | Add fit-ops sanity fail condition (degenerate sweep guard) | LE2i, CAUCAFall | pass/fail gate | sweep finite-F1 coverage | low | blocks unusable OP configs |
| E03 | Numeric fingerprint CI gate enabled | LE2i, CAUCAFall | gate pass rate | mean/std/p99 deltas | low | catches adapter drift before training |
| E04 | Temporal stride gate enabled | LE2i, CAUCAFall | gate pass rate | window/stride seconds deviation | low | stabilizes cross-dataset semantics |
| E10 | TCN + TSM flag on | LE2i first, then CAUCAFall | AP, event recall | FA/24h, p95 latency | medium | likely boosts onset sensitivity at low cost |
| E11 | GCN adaptive adjacency (A+B+C) flag on | LE2i first, then CAUCAFall | AP, event recall | latency, model size | medium-high | may improve harder views but overfit risk |
| E12 | CTR-GCN-lite refinement on top of E11 | LE2i then CAUCAFall | AP, event F1 | latency | high | incremental after adaptive adjacency only |
| E20 | Cost-sensitive OP objective (`c_fn >> c_fp`) | LE2i, CAUCAFall | recall@FA target | coverage, precision | low | converts good AP into deploy-appropriate decisions |
| E21 | Hard-negative mining loop (1 pass) | LE2i then CAUCAFall | FA/24h reduction | recall retention | medium-high (I/O) | best after stable policy gates |
| E22 | On-device profile budget gate (`profile-infer`) | LE2i model variants | p95 latency | memory, load time | low | final deployment filter |

## 2) Recommended Run Order (cheapest/highest-signal first)
1. E00
2. E01
3. E02
4. E03
5. E04
6. E10
7. E11
8. E20
9. E21
10. E22
11. E12 (only if E11 shows clear gain)

## 3) Stop Conditions
- Stop policy tuning and move to architecture upgrades when:
  - non-degenerate ops sweep exists,
  - event recall > 0 with controllable FA/24h,
  - numeric/temporal gates pass consistently.
- Stop architecture tuning when:
  - AP gain < 0.5 pt across 2 consecutive changes and FA/24h worsens,
  - or p95 latency budget exceeded on target profile.
- Stop hard-negative iterations when:
  - FA/24h reduction plateaus (<5% relative gain) across two loops,
  - or recall drops beyond agreed tolerance.

## 4) Standard Reporting Template Per Experiment
- dataset/model/checkpoint
- window AP/AUC
- selected OP + thresholds
- event precision/recall/F1
- FA/24h
- delay stats
- latency (if deploy-facing)
- pass/fail for numeric + temporal + portability gates
