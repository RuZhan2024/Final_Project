# Mode Verification Report (LE2i / CAUCAFall × TCN / GCN / HYBRID)

## Scope
- Verify end-to-end mode availability across:
  - datasets: `le2i`, `caucafall`
  - modes: `tcn`, `gcn`, `hybrid`
- Focus on integration readiness (frontend/backend/ML wiring), not final model quality.

## Validation Commands

### 1) Deploy specs availability
```bash
python - <<'PY'
from server.deploy_runtime import get_specs
specs=get_specs()
print('spec_keys=',sorted(specs.keys()))
for k in sorted(specs):
    s=specs[k]
    print(k,'arch',s.arch,'dataset',s.dataset,'ckpt',s.ckpt)
PY
```

### 2) Backend 6-combo smoke (predict_window)
```bash
source .venv/bin/activate
PYTHONPATH="$(pwd)/src:$(pwd)" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python - <<'PY'
from server.core import MonitorPredictPayload
from server.routes.monitor import predict_window

def mk_payload(mode,dataset,fps):
    t=[0,40,80,120,160,200]
    xy=[[[0.5,0.5] for _ in range(33)] for _ in t]
    conf=[[0.95 for _ in range(33)] for _ in t]
    return {
        'session_id': f'smoke-{dataset}-{mode}',
        'resident_id': 1,
        'mode': mode,
        'dataset_code': dataset,
        'op_code': 'OP-2',
        'target_T': 48,
        'target_fps': fps,
        'raw_t_ms': t,
        'raw_xy': xy,
        'raw_conf': conf,
        'persist': False,
        'use_mc': True,
        'mc_M': 4,
    }

combos=[('caucafall','tcn',23),('caucafall','gcn',23),('caucafall','hybrid',23),
        ('le2i','tcn',25),('le2i','gcn',25),('le2i','hybrid',25)]
for ds,mode,fps in combos:
    p=MonitorPredictPayload(**mk_payload(mode,ds,fps))
    out=predict_window(p)
    print(ds,mode,'effective=',out.get('effective_mode'),'models=',sorted((out.get('models') or {}).keys()))
PY
```

### 3) Frontend model-mode mapping checks
```bash
rg -n "HYBRID|hybrid|\\[\"TCN\", \"GCN\", \"HYBRID\"\\]|normModeFromCode|prettyModelTag" \
  apps/src/lib/modelCodes.js apps/src/pages/monitor/utils.js \
  apps/src/pages/settings/SettingsPage.js apps/src/pages/Monitor.js
```

## Results

### A) Deploy specs
- Found:
  - `caucafall_tcn`
  - `caucafall_gcn`
  - `le2i_tcn`
  - `le2i_gcn`
- Status: `PASS`

### B) 6-combo backend smoke
- `caucafall + tcn`: `PASS` (`effective_mode=tcn`, models contain `tcn`)
- `caucafall + gcn`: `PASS` (`effective_mode=gcn`, models contain `gcn`)
- `caucafall + hybrid`: `PASS` (`effective_mode=hybrid`, models contain `tcn,gcn`)
- `le2i + tcn`: `PASS` (`effective_mode=tcn`, models contain `tcn`)
- `le2i + gcn`: `PASS` (`effective_mode=gcn`, models contain `gcn`)
- `le2i + hybrid`: `PASS` (`effective_mode=hybrid`, models contain `tcn,gcn`)
- Status: `PASS`

### C) Frontend wiring
- Settings model radio includes `HYBRID`
- Mode normalization handles `HYBRID -> hybrid`
- Monitor effective mode supports `hybrid` when both model IDs exist
- Status: `PASS`

## Interpretation
- Integration chain for all six combinations is operational.
- Hybrid mode is now a real dual-model path (not aliasing to TCN).
- Remaining risk is model quality/threshold behavior (especially GCN alert quality), not wiring.

## Current deployment recommendation
- Default profile: `TCN + caucafall + OP-2` (stable auto-alert).
- `HYBRID`: enabled for demo/feature validation and dual-channel analysis.
- `GCN` alone: available, but do not use as primary auto-alert until training-side quality improves.

## Acceptance Summary
- Frontend mode selection: `PASS`
- Backend predict route by mode/dataset: `PASS`
- ML runtime model loading by dataset/arch: `PASS`
- Overall integration readiness (6 combos): `PASS`
