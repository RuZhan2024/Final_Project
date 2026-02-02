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
source .venv/bin/activate
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





Here’s the **exact step-by-step workflow** using the new Makefile. I’ll show it as a sequence you can run in Terminal, and what each step produces.

> Tip: pick one dataset first (recommend **le2i**) so you don’t overwhelm your disk.

---

## 0) Check what targets exist

```bash
make help
```

---

## 1) Build the dataset into windows (the “data pipeline”)

### Step 1.1 Extract pose (raw videos → pose_npz_raw)

```bash
make extract-le2i
```

Output folder:

* `data/interim/le2i/pose_npz_raw/`

### Step 1.2 Preprocess pose (pose_npz_raw → pose_npz)

Resample to 30 FPS + One-Euro smoothing + masks.

```bash
make preprocess-le2i
```

Output folder:

* `data/interim/le2i/pose_npz/`

### Step 1.3 Build labels/spans

```bash
make labels-le2i
```

Outputs:

* `configs/labels/le2i.json`
* `configs/labels/le2i_spans.json`

### Step 1.4 Make train/val/test splits

```bash
make splits-le2i
```

Outputs:

* `configs/splits/le2i_train.txt`
* `configs/splits/le2i_val.txt`
* `configs/splits/le2i_test.txt`

### Step 1.5 Build windows (proc_npz + splits → windows_W48_S12)

```bash
make windows-le2i
```

Outputs:

* `data/processed/le2i/windows_W48_S12/train/`
* `data/processed/le2i/windows_W48_S12/val/`
* `data/processed/le2i/windows_W48_S12/test/`

### Optional sanity check

```bash
make check-windows-le2i
```

---

## 2) Train models (TCN / GCN)

### Step 2.1 Train TCN

```bash
make train-tcn-le2i
```

Output:

* `outputs/le2i_tcn_W48S12/best.pt`

### Step 2.2 Train GCN

```bash
make train-gcn-le2i
```

Output:

* `outputs/le2i_gcn_W48S12/best.pt`

---

## 3) Calibrate temperature (optional but recommended)

### Step 3.1 Calibrate TCN temperature on val

```bash
make calibrate-tcn-le2i
```

Output:

* `outputs/calibration/tcn_le2i.yaml`

### Step 3.2 Calibrate GCN temperature on val

```bash
make calibrate-gcn-le2i
```

Output:

* `outputs/calibration/gcn_le2i.yaml`

---

## 4) Fit operating points OP-1/OP-2/OP-3

### Step 4.1 (LE2i only) Build unlabeled windows for FA/day estimation

This creates a “normal-life” unlabeled test stream.

```bash
make unlabeled-le2i
```

Output:

* `data/processed/le2i/windows_W48_S12/test_unlabeled/`

### Step 4.2 Fit OPs for TCN

Uses unlabeled stream automatically (since LE2i has it).

```bash
make fit-ops-tcn-le2i FITOPS_FA=auto
```

Output:

* `configs/ops/tcn_le2i.yaml`

### Step 4.3 Fit OPs for GCN

```bash
make fit-ops-gcn-le2i FITOPS_FA=auto
```

Output:

* `configs/ops/gcn_le2i.yaml`

---

## 5) Evaluate (window + event metrics) on test split

### Step 5.1 Evaluate TCN

```bash
make eval-tcn-le2i OP=op2
```

Output:

* `outputs/metrics/tcn_le2i.json`

### Step 5.2 Evaluate GCN

```bash
make eval-gcn-le2i OP=op2
```

Output:

* `outputs/metrics/gcn_le2i.json`

---

## 6) Replay eval (deployment-like streaming validation)

### Step 6.1 Replay TCN

```bash
make replay-tcn-le2i OP=op2
```

### Step 6.2 Replay GCN

```bash
make replay-gcn-le2i OP=op2
```

Outputs:

* `outputs/metrics/replay_tcn_le2i.json`
* `outputs/metrics/replay_gcn_le2i.json`

---

## 7) Score false alarms on unlabeled stream (LE2i only)

```bash
make score-unlabeled-tcn-le2i OP=op2
make score-unlabeled-gcn-le2i OP=op2
```

Outputs:

* `outputs/metrics/unlabeled_tcn_le2i.json`
* `outputs/metrics/unlabeled_gcn_le2i.json`

---

## 8) Plot the ops curves

```bash
make plot-ops-tcn-le2i
make plot-ops-gcn-le2i
```

Outputs:

* `outputs/plots/*_recall_vs_fa.png`
* `outputs/plots/*_f1_vs_tau.png`

---

## 9) Deploy simulation runner (optional)

This replays windows and prints “POSSIBLE/CONFIRMED/RESOLVED”.

```bash
make deploy-dual-le2i
```

(or `deploy-tcn-le2i`, `deploy-gcn-le2i`)

---

# The “all-in-one” shortcuts

If you want the whole pipeline:

### Data pipeline only

```bash
make pipeline-le2i
```

### Full workflow (TCN)

```bash
make workflow-tcn-le2i
```

### Full workflow (GCN)

```bash
make workflow-gcn-le2i
```

---

If you tell me which dataset you’re running **right now** (le2i/urfd/caucafall/muvim) and whether you’re using **W=48 S=12**, I can give you the exact “copy/paste” command block with no extras.
