# Replay + Live Acceptance Lock

## Objective
- Lock a stable replay baseline first.
- Then validate live monitor on high-performance machines against the replay baseline.

## Fixed Baseline Profile (Current)
- Dataset: `caucafall`
- Model mode: `TCN` (primary), `GCN` (secondary)
- OP: `OP-2`
- Inference transport: `WebSocket only`
- Frontend source:
  - Replay mode for baseline acceptance
  - Realtime mode for deployment acceptance
- Capture resolution options:
  - `480p`, `540p`, `720p`, `1080p` (select in Monitor Controls)

## Why This Lock
- Replay is deterministic and closest to offline evaluation behavior.
- Live is sensitive to camera quality, FPS jitter, and frontend load.
- Using the same `dataset + model + OP` in both modes keeps policy alignment clear.

## Acceptance Process

1. Start backend and frontend.
2. Set monitor profile:
   - `dataset = caucafall`
   - `model = TCN`
   - `op = OP-2`
3. Run one-command acceptance report:
```bash
bash tools/run_replay_live_acceptance.sh
```
4. Fill manual rows in:
   - `artifacts/reports/replay_live_acceptance.md`
5. Repeat step 2-4 for `GCN`.

## Pass Criteria
- Replay (TCN/GCN):
  - Non-fall clip does not trigger repeated fall events.
  - Fall clip triggers a clear fall event.
- Live (high-performance machine):
  - ADL does not produce repeated false fall events.
  - Controlled fall is detected.
- Auto checks:
  - API health reachable.
  - Deploy specs available.
  - OP `live_guard` exists for TCN/GCN OP1/OP2/OP3.
  - Monitor page does not poll `/api/summary`.

## Notes
- If live recall is too low, tune `ops.<OP>.live_guard` first, not model weights.
- If live false alerts are high, tighten `OP-2 live_guard` incrementally.
- Keep replay clips fixed across runs for fair comparison.
