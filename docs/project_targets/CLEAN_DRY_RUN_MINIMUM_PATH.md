# Clean Dry-Run Minimum Path

Date: 2026-03-22

Purpose:
Define the minimum external-review path that should be executed once and recorded as the clean reproducibility check for the paper/submission package.

## Goal

Prove that an external reviewer can:
- set up the environment
- start the backend
- verify health/spec endpoints
- start the frontend
- follow the locked replay path

This dry run is not meant to retrain models or regenerate the full experimental suite.

## Minimum Required Path

### Step 1: Environment setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.lock.txt
pip install -r requirements_server.txt
pip install -e . --no-build-isolation
cd apps && npm install && cd ..
```

### Step 2: API smoke

```bash
source .venv/bin/activate
PYTHONPATH="$(pwd)/src:$(pwd)" python3 scripts/smoke_api_contract.py
```

Expected:
- script exits successfully
- API contract smoke passes

### Step 3: Start backend

```bash
source .venv/bin/activate
PYTHONPATH="$(pwd)/src:$(pwd)" uvicorn server.app:app --host 0.0.0.0 --port 8000
```

### Step 4: Health/spec checks

```bash
curl -sS http://localhost:8000/api/health
curl -sS http://localhost:8000/api/spec
```

Expected:
- `/api/health` returns `ok`
- `/api/spec` is non-empty for the locked demo profile

### Step 5: Start frontend

```bash
cd apps
REACT_APP_API_BASE=http://localhost:8000 npm start
```

### Step 6: Locked replay check

Use the locked replay path from:
- `docs/reports/runbooks/USER_GUIDE.md`
- `docs/project_targets/DEPLOYMENT_LOCK.md`

Minimum reviewer-facing checks:
- one non-fall replay
- one fall replay
- visible model/policy output in the monitor flow

## Pass Criteria

Record the dry run as `PASS` only if all are true:
- environment setup completed without undocumented fixes
- backend started successfully
- frontend started successfully
- health and spec checks passed
- locked replay path behaved as documented

If anything fails:
- still record the run
- write the exact failure and workaround
- do not silently mark the dry run complete

## Output Record

Fill this artifact after the run:
- `artifacts/reports/clean_dry_run_report.md`

Then update:
- `docs/project_targets/FINAL_SUBMISSION_CHECKLIST.md`
- `docs/reports/readiness/READINESS_REPORT.md`
