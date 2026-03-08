# LE2i GCN r9/r10 Follow-up Summary

## r9 (Positive-focused focal)
- Config: `focal_alpha=0.8`, `focal_gamma=0.5`, resume from r8.
- Observation: non-finite loss emerged in mid-epochs.
- Safety behavior: non-finite guard correctly skipped optimizer steps, run did not crash.
- Result (partial eval): `AP=0.8427`, `Recall=0.8889`, `F1=0.9412`, `FA24h=0.0`.
- Decision: reject.

## r10 (BCE + label smoothing)
- Config: `loss=bce`, `label_smoothing=0.03`, resume from r8.
- Training was stable.
- Result (partial eval): `AP=0.8441`, `Recall=0.8889`, `F1=0.9412`, `FA24h=0.0`.
- Decision: reject (still below r8 AP 0.8451).

## Current LE2i promoted best (unchanged)
- Checkpoint: `outputs/le2i_gcn_W48S12_opt33_r8_dataside_noise/best.pt`
- Policy: `configs/ops/gcn_le2i_paper_profile.yaml`
- Locked metrics: `AP=0.8451`, `Recall=0.8889`, `Precision=1.0000`, `F1=0.9412`, `FA24h=0.0`
- Reproduce: `make repro-best-gcn-le2i-paper ADAPTER_USE=1`
