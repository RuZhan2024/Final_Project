Current replay/runtime status tags for the 24 custom corridor/kitchen clips.

Authoritative current outputs:
- `online_replay_matrix_20260402.csv`
- `online_replay_matrix_20260402.json`
- `online_replay_le2i_finalfix_20260402.json`
- `online_replay_caucafall_after_replayfpsfix_20260402.json`

Meaning:
- These files reflect the current post-fix runtime interpretation.
- Use them for current discussion of replay behavior after the replay-FPS fix.

Legacy or intermediate outputs to keep but not treat as current config truth:
- `mc_replay_matrix_20260401.csv`
- `mc_replay_matrix_20260401.json`
- `mc_replay_matrix_20260402.csv`
- `mc_replay_matrix_20260402.json`
- `online_replay_le2i_after_historyfix_20260402.json`
- `online_replay_le2i_after_replayfpsfix_20260402.json`
- `online_replay_le2i_after_opsalign_20260402.json`

Interpretation:
- `mc_replay_matrix_*` files are precomputed-window matrix outputs, not the final raw online replay fix check.
- `after_historyfix` is a rejected intermediate experiment.
- `after_replayfpsfix` and `after_opsalign` are intermediate checkpoints that informed the final runtime state.
