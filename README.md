# Fall Detection Full-Stack (TCN / GCN / HYBRID)

This repo contains:

- **Backend (FastAPI)**: `server/`
- **Frontend (React + MediaPipe Pose)**: `apps/`
- **Model/pipeline code** (training/eval/etc): root folders like `core/`, `models/`, `eval/`, etc.

## 1) Database (MySQL)

1. Create a MySQL database (e.g. `fallguard`).
2. Run the schema:

```sql
SOURCE create_db.sql;
```

> Note: `create_db.sql` includes an `event_metadata` column (JSON).  
> If you already created tables earlier, the backend will also try a **best-effort migration** on startup.

### Recommended seed rows (model codes)

The backend uses stable **model codes**: `TCN`, `GCN`, `HYBRID`.  
You can insert them once (optional — backend will auto-create rows if missing):

```sql
INSERT INTO models (code, name, family)
VALUES ('TCN','TCN','TCN'),('GCN','GCN','GCN'),('HYBRID','HYBRID','Hybrid');
```

## 2) Backend (FastAPI)

### Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r server/requirements.txt
```

### Configure env

Set MySQL connection via env vars (see `server/db.py`):

```bash
export DB_HOST=127.0.0.1
export DB_PORT=3306
export DB_NAME=fallguard
export DB_USER=root
export DB_PASS=your_password
```

### Run

```bash
# Preferred:
uvicorn server.app:app --reload --port 8000

# Or, via the root wrapper (same server):
uvicorn app:app --reload --port 8000
```

Health check:

- `GET http://localhost:8000/api/health`

## 3) Frontend (React)

```bash
cd apps
npm install
npm start
```

(Optional) point frontend to backend:

```bash
export REACT_APP_API_BASE=http://localhost:8000
```

Open:

- `http://localhost:3000`

## 4) Live Monitor Modes

Frontend **Monitor** supports:

- **TCN**
- **GCN**
- **TCN + GCN (dual / HYBRID)**

### DB storage

When `Persist events to DB` is enabled:

- `models.code` is stored as: `TCN` / `GCN` / `HYBRID`
- The *runner/spec id* (e.g. `muvim_gcn_W48S12`) is stored in `events.event_metadata`:

Examples:

```json
{ "spec_id": "muvim_gcn_W48S12", "mode": "gcn", "triage": {...} }
```

```json
{ "spec_tcn": "muvim_tcn_W48S12", "spec_gcn": "muvim_gcn_W48S12", "mode": "dual", "triage": {...} }
```

## 5) Latency targets

Triage timing settings are in:

- `configs/deploy_modes.yaml`

These targets control how quickly the state machine emits **POSSIBLE** and **CONFIRMED** alerts.
