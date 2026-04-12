MUVIM archived operating-point configs.

This directory stores MUVIM quick-search and later exploratory branches that are not part of the active runtime surface.

Archived here:
- `*muvim_quick*`
- `*muvim_r3*`

Kept at `configs/ops/` root:
- `tcn_muvim.yaml`
- `gcn_muvim.yaml`
- labels-oriented MUVIM files that still describe the main labels track

Rationale:
- reduce accidental use of experimental MUVIM ops in the live runtime surface
- preserve reproducibility for the MUVIM side track without mixing it into the main deploy config layer
