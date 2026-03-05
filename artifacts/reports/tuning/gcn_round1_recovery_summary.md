# GCN CAUCAFall Round-1 Recovery Summary

|exp|AP|AUC|OP2 F1|OP2 Recall|OP2 Precision|OP2 FA/24h|tau_high|
|-|-:|-:|-:|-:|-:|-:|-:|
|stb_s17|0.9701|0.9783|0.8889|0.8000|1.0000|0.0000|0.860|
|r1_augreg|0.9702|0.9787|0.6667|1.0000|0.5000|4704.5455|0.230|
|r1_recovery|0.9740|0.9830|0.6667|1.0000|0.5000|4704.5455|0.230|
|r1_recovery_min40|0.9740|0.9830|0.8889|0.8000|1.0000|0.0000|0.420|

## Verdict
- `r1_recovery` raw fit_ops selected a low OP2 threshold and caused high FA/24h.
- Recalibration with `min_tau_high=0.40` (`r1_recovery_min40`) restores deploy-safe OP2 metrics: Recall=0.8, F1=0.889, FA/24h=0.
- Compared with `stb_s17`, deploy OP2 metrics are matched while AP is improved (0.974 vs 0.970).
- Recommendation: for 33-joint GCN deployment, use `gcn_caucafall_r1_recovery_min40.yaml`.