# SIGNIFICANCE_REPORT

Date: 2026-03-22

## Scope
Paired 5-seed significance analysis for frozen final candidates (TCN vs GCN) using current stability artifacts.

Protocol Decision:
- `Paper Protocol Freeze v1`
- Day 2 decision: `keep n=5 and narrow claim`
- The paper-safe interpretation is:
  - report directional advantage for TCN under the frozen protocol
  - do not claim definitive superiority over GCN
  - treat Wilcoxon as the primary inferential result

Artifacts used:
- `artifacts/reports/stability_summary.json`
- `artifacts/reports/significance_summary.json`

Primary test: two-sided Wilcoxon signed-rank (small-n paired setting).
Secondary test: paired t-test (reported for reference only).
Alpha: `0.05` (uncorrected, exploratory).

## Hypotheses
1. H1: On `CAUCAFall`, TCN trends stronger than GCN on event metrics (`F1`, `Recall`) at matched seed settings under the frozen protocol.
2. H2: On `LE2i`, TCN trends stronger than GCN on event metrics (`F1`, `Recall`) at matched seed settings under the frozen protocol.
3. H3: AP differences are directionally consistent with event metrics.
4. H4: `FA24h` differs between architectures under fixed protocol.

## Results Table
| Dataset | Metric | n | TCN mean | GCN mean | Diff (TCN-GCN) | Wilcoxon p | Paired t-test p | Effect (Cohen dz) | Decision (alpha=0.05, Wilcoxon) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| caucafall | f1 | 5 | 0.8611 | 0.5873 | 0.2738 | 0.1250 | 0.0402 | 1.3391 | Fail to reject H0 |
| caucafall | recall | 5 | 0.7600 | 0.4400 | 0.3200 | 0.1250 | 0.0349 | 1.4033 | Fail to reject H0 |
| caucafall | fa24h | 5 | 0.0000 | 0.0000 | 0.0000 | NA | NA | NA | NA (degenerate) |
| caucafall | ap | 5 | 0.9819 | 0.9706 | 0.0113 | 0.0625 | 0.0906 | 0.9927 | Fail to reject H0 |
| le2i | f1 | 5 | 0.8235 | 0.7500 | 0.0735 | 0.0625 | 0.0000 | NA | Fail to reject H0 |
| le2i | recall | 5 | 0.7778 | 0.6667 | 0.1111 | 0.0625 | 0.0000 | NA | Fail to reject H0 |
| le2i | fa24h | 5 | 581.5843 | 581.5843 | 0.0000 | NA | NA | NA | NA (degenerate) |
| le2i | ap | 5 | 0.8389 | 0.7471 | 0.0918 | 0.0625 | 0.0097 | 2.0785 | Fail to reject H0 |

## Interpretation
- Decision for paper text:
  - keep the current `n=5`
  - narrow the architecture-comparison claim rather than extending compute for a stronger significance claim
- CAUCAFall: TCN has higher mean `F1/Recall/AP` than GCN, but with `n=5` seeds, Wilcoxon does not cross `0.05` for primary metrics (`p=0.125` for `F1/Recall`, `p=0.0625` for `AP`).
- LE2i: TCN mean is higher for `F1/Recall/AP`; Wilcoxon `p=0.0625` (borderline but not below `0.05`).
- `FA24h`: no architecture difference detected in either dataset under current OP selection (`diff=0`, degenerate test).
- Effect-size direction favors TCN; however, small sample size limits formal significance under non-parametric testing.
- Paper-safe wording:
  - `TCN trends stronger than GCN under the frozen protocol`
  - not `TCN significantly outperforms GCN`

## Limitations
- Seed count is minimal (`n=5`) for statistical power.
- Some metric deltas are near-constant across seeds, causing degenerate variance for t-test statistics.
- P-values are uncorrected for multiple comparisons; treat as exploratory evidence.
- The primary inferential outcome is therefore directional and bounded, not definitive.

## Next Actions
1. Keep Wilcoxon as primary test in thesis/paper text; report effect sizes and confidence intervals.
2. Keep the architecture-comparison claim narrow in the paper draft.
3. Prioritize deployment metrics (`Recall`, `FA24h`, `delay_p95`) over AP-only conclusions.

## Reproduce
```bash
python - <<'PY'
import json, math
from pathlib import Path
import numpy as np
from scipy import stats
# Source: artifacts/reports/stability_summary.json
# Output: artifacts/reports/significance_summary.json
PY
```
