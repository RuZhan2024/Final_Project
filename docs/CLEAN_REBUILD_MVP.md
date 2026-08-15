# Clean Rebuild MVP

This branch introduces a new implementation path beside the legacy project.

The rebuilt code lives in:

```text
packages/ml/        Framework-light prediction core
applications/api/   FastAPI HTTP boundary
applications/web/   Next.js monitoring console
```

Legacy folders remain available as reference material, but the rebuilt system should not import from them.

## Local Setup

Install Python dependencies from the repository root:

```powershell
python -m pip install -r requirements-rebuild-dev.txt
```

Install frontend dependencies:

```powershell
cd applications/web
npm install
```

## Run

Start the API:

```powershell
$env:PYTHONPATH = "$(Resolve-Path packages/ml/src);$(Resolve-Path applications/api/src)"
python -m uvicorn safe_guard_api.app:app --reload
```

Start the web app:

```powershell
cd applications/web
npm run dev
```

The web app calls `http://localhost:8000` by default. Set `NEXT_PUBLIC_API_BASE_URL` when the API runs elsewhere.

## Verify

```powershell
.\scripts\verify_rebuild.ps1
```

The verification script runs:

- ML unit tests
- API unit tests
- Next.js type check
- Next.js production build

## Current Scope

The MVP uses a deterministic heuristic predictor. It is intentionally simple and testable. The next model-backed runtime should implement the same predictor boundary instead of changing the API or frontend contracts.
