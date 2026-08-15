# Safe Guard API

FastAPI service for the clean rebuild.

The API exposes:

- `GET /health`
- `GET /api/v1/runtime`
- `POST /api/v1/predictions`

The service imports `safe_guard_ml` through the monorepo development install:

```powershell
python -m pip install -r ../../requirements-rebuild-dev.txt
```

Run locally from the repository root:

```powershell
$env:PYTHONPATH = "$(Resolve-Path packages/ml/src);$(Resolve-Path applications/api/src)"
python -m uvicorn safe_guard_api.app:app --reload
```
