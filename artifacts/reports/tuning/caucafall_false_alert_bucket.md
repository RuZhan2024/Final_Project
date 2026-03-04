# CAUCAFall False-Alert Bucket (Midplateau Policy)

## Scope
- Source files: `outputs/metrics/tcn_caucafall_stb_s*_midplateau.json`
- Seeds reviewed: `1337, 17, 2025, 33724876, 42`

## Findings
- False alerts are concentrated in one ADL class:
  - `Pick up object`
- Video-level concentration:
  - `Subject.6/Pick up object`
- Count summary:
  - total false alerts observed across high-risk seeds: `4`
  - all 4 from `Subject.6/Pick up object`

## Implication
- This is not broad ADL confusion; it is a narrow failure mode.
- Next experiments should be targeted:
  - hard-negative replay with `hard_neg_prefixes=\"Pick up object\"`
  - optional confidence-gate tightening (`conf_gate=0.30`) to suppress noisy pose spikes
