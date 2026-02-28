# Fall Detection Full-Stack (TCN / GCN / HYBRID)

This repo contains:

- **Backend (FastAPI)**: `server/`
- **Frontend (React + MediaPipe Pose)**: `apps/`
- **Model/pipeline code** (training/eval/etc): root folders like `core/`, `models/`, `eval/`, etc.

## 1) Database (MySQL)

1. Create a MySQL database (e.g. `fallguard`).
2. Run the schema:

```sql
SOURCE server/create_db.sql;
```

> Note: `server/create_db.sql` uses a `meta` JSON column on `events`.  
> If you already created tables earlier, the backend will also try a **best-effort migration** on startup.

### Recommended seed rows (model codes)

The backend uses stable **model codes**: `TCN`, `GCN`, `HYBRID`.  
You can insert them once (optional — backend will auto-create rows if missing):

```sql
INSERT INTO models (code, name)
VALUES ('TCN','TCN'),('GCN','GCN'),('HYBRID','Hybrid');
```

## 2) Backend (FastAPI)

### Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements_server.txt
```

### Configure env

Create local env file, then adjust values:

```bash
cp .env.example .env
```

Optional inference/runtime optimization flags:

- `FD_DYNAMIC_QUANT_LINEAR=1`  
  Enable CPU dynamic quantization for `nn.Linear` layers at deploy load time.
- `FD_TORCH_COMPILE=1`  
  Enable `torch.compile(..., mode="reduce-overhead")` for steady-state inference.

Both are best-effort: if unsupported by the local PyTorch/runtime, server falls back automatically.

### Run

```bash
# Preferred:
uvicorn server.app:app --reload --port 8000

# Smoke test
curl http://localhost:8000/api/health
```

Health check:

- `GET http://localhost:8000/api/health`

Backend tests:

```bash
make install-dev
make test-server
make test-server-cov
```

Override coverage gate (default `70`) when needed:

```bash
make test-server-cov COVERAGE_MIN=75
```

`make test-server-cov` also writes `coverage.xml` (used by CI artifacts/reporting).

CI coverage policy:
- Pull requests and non-`main` branches require at least `75%`.
- Pushes to `main` require at least `81%`.

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
- The *runner/spec id* (e.g. `muvim_gcn_W48S12`) is stored in `events.meta`:

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

## 6) Pose Preprocess Robustness Flags

`pose/preprocess_pose_npz.py` now supports extra guards for noisy skeleton extraction:

- `--clip_xy --clip_xy_min 0.0 --clip_xy_max 1.0`  
  Clip finite coordinates to a bounded range before downstream processing.
- `--max_step 0.0` (disabled by default)  
  Limit per-frame joint displacement to suppress extraction spikes.

These options are recorded in each output NPZ `preprocess` metadata JSON.

## 7) Uncertainty Telemetry (Live Monitor)

`/api/monitor/predict_window` accepts optional:

- `mc_sigma_tol` (positive float): adaptive MC-dropout early-stop tolerance.
- `mc_se_tol` (positive float): adaptive MC-dropout standard-error early-stop tolerance.

Response now includes:

- `mc_n_used`: per-model MC samples actually consumed (useful to quantify adaptive compute savings).





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

Below is the **resolved running-order (dependency chain)** printed for **all four datasets**. I’m showing the same structure for each dataset:

* **`pipeline-data-<ds>`** (data prep to training windows)
* **`pipeline-<ds>`** (TCN end-to-end)
* **`pipeline-gcn-<ds>`** (GCN end-to-end)
* **Unlabeled** (only actually meaningful for **LE2i** in your setup)

---

## le2i

### `pipeline-data-le2i`

1. `extract-le2i`
2. `preprocess-le2i`
3. `labels-le2i`
4. `splits-le2i`
5. `windows-le2i`

### `pipeline-le2i` (TCN)

1. `pipeline-data-le2i` (chain above)
2. `train-tcn-le2i`
3. `windows-eval-le2i`
4. `fit-ops-le2i` *(+ `fa-windows-le2i` only if `FITOPS_USE_FA=1`)*
5. `eval-le2i`
6. `plot-le2i`

### `pipeline-gcn-le2i` (GCN)

1. `pipeline-data-le2i`
2. `train-gcn-le2i`
3. `windows-eval-le2i`
4. `fit-ops-gcn-le2i` *(+ `fa-windows-le2i` only if `FITOPS_USE_FA=1`)*
5. `eval-gcn-le2i`
6. `plot-gcn-le2i`

### Unlabeled (LE2i)

1. `preprocess-only-le2i` *(or `preprocess-le2i`)*
2. `splits-unlabeled-le2i`
3. `windows-unlabeled-le2i`

---

## urfd

### `pipeline-data-urfd`

1. `extract-urfd`
2. `preprocess-urfd`
3. `labels-urfd`
4. `splits-urfd`
5. `windows-urfd`

### `pipeline-urfd` (TCN)

1. `pipeline-data-urfd`
2. `train-tcn-urfd`
3. `windows-eval-urfd`
4. `fit-ops-urfd` *(+ `fa-windows-urfd` if `FITOPS_USE_FA=1`)*
5. `eval-urfd`
6. `plot-urfd`

### `pipeline-gcn-urfd` (GCN)

1. `pipeline-data-urfd`
2. `train-gcn-urfd`
3. `windows-eval-urfd`
4. `fit-ops-gcn-urfd` *(+ `fa-windows-urfd` if `FITOPS_USE_FA=1`)*
5. `eval-gcn-urfd`
6. `plot-gcn-urfd`

### Unlabeled (URFD)

* You *can* run `splits-unlabeled-urfd` → `windows-unlabeled-urfd`, but with empty/default scenes it’s effectively a no-op unless you configure `UNLABELED_SCENES_urfd`.

---

## caucafall

*(Has the extra spans sanity requirement before any windows build.)*

### `pipeline-data-caucafall`

1. `extract-caucafall`
2. `preprocess-caucafall`
3. `labels-caucafall`
4. `splits-caucafall`
5. `check-spans-caucafall`
6. `windows-caucafall`

### `pipeline-caucafall` (TCN)

1. `pipeline-data-caucafall`
2. `train-tcn-caucafall`
3. `windows-eval-caucafall` *(also depends on `check-spans-caucafall`)*
4. `fit-ops-caucafall` *(+ `fa-windows-caucafall` if `FITOPS_USE_FA=1`)*
5. `eval-caucafall`
6. `plot-caucafall`

### `pipeline-gcn-caucafall` (GCN)

1. `pipeline-data-caucafall`
2. `train-gcn-caucafall`
3. `windows-eval-caucafall`
4. `fit-ops-gcn-caucafall` *(+ `fa-windows-caucafall` if `FITOPS_USE_FA=1`)*
5. `eval-gcn-caucafall`
6. `plot-gcn-caucafall`

### Unlabeled (CAUCAFall)

* Same note as URFD: possible, but only useful if you define `UNLABELED_SCENES_caucafall` (you haven’t).

---

## muvim

### `pipeline-data-muvim`

1. `extract-muvim`
2. `preprocess-muvim`
3. `labels-muvim`
4. `splits-muvim`
5. `windows-muvim`

### `pipeline-muvim` (TCN)

1. `pipeline-data-muvim`
2. `train-tcn-muvim`
3. `windows-eval-muvim` *(has the eval fallback fix you added)*
4. `fit-ops-muvim` *(+ `fa-windows-muvim` if `FITOPS_USE_FA=1`)*
5. `eval-muvim`
6. `plot-muvim`

### `pipeline-gcn-muvim` (GCN)

1. `pipeline-data-muvim`
2. `train-gcn-muvim`
3. `windows-eval-muvim`
4. `fit-ops-gcn-muvim` *(+ `fa-windows-muvim` if `FITOPS_USE_FA=1`)*
5. `eval-gcn-muvim`
6. `plot-gcn-muvim`

### Unlabeled (MUVIM)

* Same as URFD/CAUCAFall: only meaningful if you define `UNLABELED_SCENES_muvim` (not defined).

---

### Bonus: “All datasets” one-liners (parallel-safe)

* TCN end-to-end: `make -j pipeline-all`
* GCN end-to-end: `make -j pipeline-all-gcn`
* Data-only: `make -j $(addprefix pipeline-data-,$(DATASETS))` (or just call each explicitly)
