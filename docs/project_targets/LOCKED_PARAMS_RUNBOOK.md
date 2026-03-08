# Locked Parameters Runbook

This document lists the current locked training/evaluation profiles used for
reproducible results in this repository.

Scope rule:
- These locked targets are dataset-specific.
- They do not change the global default knobs for other datasets/experiments.

## 1) CAUCAFall Locked Profiles

### 1.1 Locked training
- TCN:
  - Target: `train-best-tcn-caucafall`
  - Output ckpt dir: `outputs/repro/caucafall_tcn_r1_augreg`
- GCN:
  - Target: `train-best-gcn-caucafall`
  - Output ckpt dir: `outputs/repro/caucafall_gcn_r2_recallpush_b`
- Combined:
  - Target: `train-best-caucafall`

Run:

```bash
make train-best-caucafall ADAPTER_USE=1
```

### 1.2 Locked evaluation/ops reproduction
- TCN ops: `configs/ops/tcn_caucafall_r1_ctrl.yaml`
- GCN ops: `configs/ops/gcn_caucafall_locked.yaml`
- TCN metrics: `outputs/metrics/tcn_caucafall_locked.json`
- GCN metrics: `outputs/metrics/gcn_caucafall_locked.json`

Run:

```bash
make repro-best-caucafall ADAPTER_USE=1
```

Optional promotion into canonical deploy files:

```bash
make apply-locked-ops-caucafall ADAPTER_USE=1
```

## 2) LE2i Locked Paper-Comparison Profile (GCN)

### 2.1 Locked training
- Target: `train-best-gcn-le2i-paper`
- Output ckpt dir: `outputs/le2i_gcn_W48S12_opt33_r8_dataside_noise`
- Resume base: `outputs/le2i_gcn_W48S12_opt33_r4_recallpush_promoted/best.pt`

Run:

```bash
make train-best-gcn-le2i-paper ADAPTER_USE=1
```

### 2.2 Locked evaluation reproduction
- Ops file: `configs/ops/gcn_le2i_paper_profile.yaml`
- Metrics file: `outputs/metrics/gcn_le2i_opt33_r8_dataside_noise_paperops.json`

Run:

```bash
make repro-best-gcn-le2i-paper ADAPTER_USE=1
```

Current locked LE2i paper-profile metrics snapshot:
- AP `0.8451`
- Recall `0.8889`
- Precision `1.0000`
- F1 `0.9412`
- FA24h `0.0`

## 3) Notes

- Locked targets are encoded via `LOCK_*` variables at the top of `Makefile`.
- To update a locked profile in future:
  1. Produce better metrics with full logs recorded in `artifacts/reports/tuning/PARAM_CHANGELOG.csv`.
  2. Update only the relevant `LOCK_*` variables and locked targets.
  3. Re-run locked reproduce commands and verify outputs.
