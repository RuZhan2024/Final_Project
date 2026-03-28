# Clean Dry Run Report

Status: `PASS`
Date: 2026-03-22
Operator: Codex agent

## Scope

- Protocol: `Paper Protocol Freeze v1`
- Dry-run path: `docs/project_targets/CLEAN_DRY_RUN_MINIMUM_PATH.md`

## Environment

- OS: macOS-style local environment (exact host not auto-captured in this run)
- Python: `3.10.9`
- Node: `v22.20.0`
- Repo ref: working tree, dirty
- Clean environment type:
  - fresh virtualenv

## Commands Executed

```bash
python3 --version
node --version
source .venv/bin/activate && python --version
source .venv/bin/activate && PYTHONPATH="$(pwd)/src:$(pwd)" python3 scripts/smoke_api_contract.py
source .venv/bin/activate && PYTHONPATH="$(pwd)/src:$(pwd)" uvicorn server.app:app --host 127.0.0.1 --port 8000
curl -sS http://127.0.0.1:8000/api/health
curl -sS http://127.0.0.1:8000/api/spec
cd apps && REACT_APP_API_BASE=http://localhost:8000 npm start
source .venv/bin/activate && PYTHONPATH="$(pwd)/src:$(pwd)" python3 - <<'PY'
import json
from urllib.request import Request, urlopen
xy=[[[0.0,0.0] for _ in range(33)] for _ in range(48)]
conf=[[1.0 for _ in range(33)] for _ in range(48)]
payload={
  'session_id':'dryrun-1',
  'mode':'tcn',
  'dataset_code':'caucafall',
  'op_code':'OP-2',
  'target_T':48,
  'xy':xy,
  'conf':conf,
  'persist':False
}
data=json.dumps(payload).encode('utf-8')
req=Request('http://127.0.0.1:8000/api/monitor/predict_window', data=data, headers={'Content-Type':'application/json'})
with urlopen(req, timeout=60) as r:
    print(r.status)
    print(r.read().decode('utf-8'))
PY
```

## Results

- Environment setup:
  - existing `.venv` reused successfully
  - Python and Node versions detected as expected
- Backend startup:
  - `PASS`
  - backend started successfully under `uvicorn`
- Frontend startup:
  - `PASS with warning`
  - CRA dev server compiled successfully with one non-blocking ESLint warning in `apps/src/pages/monitor/hooks/usePoseMonitor.js`
- `/api/health`:
  - `PASS`
  - returned `{"ok":true,...}`
- `/api/spec`:
  - `PASS`
  - non-empty spec payload returned for the locked deployment profiles
- Replay non-fall check:
  - `PASS`
  - recommended case:
    - `Office__video__27_`
  - monitor page opened: yes
  - replay loaded: yes
  - observed state: skeleton rendered and no fall alert observed during replay ADL check
  - result matches expectation: yes
- Replay fall check:
  - `PASS`
  - recommended case:
    - `Subject.1__Fall_forward__80e1655b`
  - monitor page opened: yes
  - replay loaded: yes
  - observed state: skeleton rendered and fall event detected during replay fall-event check
  - result matches expectation: yes
- Headless inference request:
  - `PASS`
  - `POST /api/monitor/predict_window` returned HTTP `200`
  - response included `triage_state`, model payload, latency, and active policy fields

## Outcome

- Overall result:
  - `PASS`

## Failures / Workarounds

- Initial sandboxed API smoke failed with OpenMP shared-memory error:
  - `OMP: Error #179: Function Can't open SHM2 failed`
- The same smoke command passed outside the sandbox, so this is recorded as an environment/sandbox limitation rather than a project failure.
- Sandboxed `curl` could not reach the backend process started outside the sandbox; non-sandbox `curl` succeeded.
- A first headless inference check using `requests` failed because `requests` was not installed in `.venv`; retried successfully using Python standard library `urllib`.

## Notes For Final Submission Checklist

- Root `README.md` quickstart re-tested:
  - not fully closed in this run
- One clean-machine dry run completed:
  - yes, with browser replay checks completed
- Known caveats documented:
  - yes
