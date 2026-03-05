# TCN CAUCAFall Round-3 Summary

|exp|AP|AUC|OP2 F1|OP2 Recall|OP2 Precision|OP2 FA/24h|tau_high|
|-|-:|-:|-:|-:|-:|-:|-:|
|r1_augreg|0.9691|0.9790|1.0000|1.0000|1.0000|0.0000|0.710|
|r3_mild_hneg|0.9824|0.9905|0.7500|0.6000|1.0000|0.0000|0.850|
|r3_mild_nohneg|0.9827|0.9903|0.8889|0.8000|1.0000|0.0000|0.880|

## Verdict
- Round-3 improved AP/AUC but reduced event recall/F1 under OP2 vs `r1_augreg`.
- Recommendation: keep `r1_augreg` as production default; keep Round-3 as logged negative result.
- Next tuning should target recall recovery (lower tau/high-risk policy or balanced hard-neg replay).