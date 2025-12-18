# FallGuard — Privacy‑Preserving On‑Device Fall Detection (Skeleton‑Only)

> **TL;DR**: This repo builds a real‑time fall detection system that runs **on‑device**, uses **2D pose (skeleton) only** (no cloud video), and optimises for **safety (Recall)** and **usability (FA/24h)** via uncertainty‑ & cost‑aware decision rules. It includes dataset prep, pose extraction, windowing, training (TCN baseline → GCN variants), operating‑point fitting, FA/24h evaluation, and a lightweight mobile client.

---

## 0) Features & Goals

* **Privacy‑first**: RGB used transiently to extract pose; only **keypoints** are stored/processed.
* **On‑device**: Edge‑friendly models (TCN/GCN) with target **≤ 200 ms** inference per clip.
* **Uncertainty‑aware**: MC Dropout / evidential proxies inform **adaptive thresholds**.
* **Operational metrics**: Report **Recall, F1, detection delay**, and **False Alarms per 24h (FA/24h)**.
* **Reproducible**: Pinned envs, configs, and deterministic splits.

---

## 1) Repository Structure

```
fallguard/
├─ apps/
│  └─ mobile/                  # React Native (TypeScript) client (stub)
├─ configs/
│  ├─ datasets.yaml            # Dataset locations & pose model
│  ├─ ops.yaml                 # Operating points (exported after fitting)
│  └─ splits/                  # Train/val/test split lists per dataset
├─ data/                       # (gitignored) raw/interim/processed assets
│  ├─ raw/                     # Original videos/files (do not commit)
│  ├─ interim/                 # Pose npz per video
│  └─ processed/               # Windowed npz with labels/meta
├─ docs/
│  └─ schema.md                # Data schema and conventions
├─ eval/
│  ├─ metrics.py               # Recall/F1/FA24h/delay utilities
│  ├─ fit_ops.py               # Fit OP‑1/2/3 thresholds from validation
│  └─ plot_fa_recall.py        # Slide‑ready FA/24h vs Recall figure
├─ models/
│  ├─ train_tcn.py             # Baseline trainer (TCN)
│  └─ backbones/               # (future) ST‑GCN/TCN variants
├─ pose/
│  ├─ extract_2d.py            # MediaPipe BlazePose → npz per video
│  └─ make_windows.py          # Windowing & label baking
├─ scripts/                    # Automation helpers (optional)
├─ server/                     # (stub) Python API for live inference/logging
├─ .gitignore
└─ README.md
```

> **Note**: `data/` is **gitignored** by default. Do not commit videos or npz files.

---

## 2) Quickstart (10–15 minutes)

### Prerequisites

* macOS/Linux with Python **3.12** (kept consistent with our machines)
* Node.js **≥ 18** (for the mobile client later)
* FFmpeg recommended (video decoding stability)

### 2.1 Create the project & environment

```bash
# 1) Create skeleton
mkdir -p fallguard && cd fallguard
mkdir -p apps/mobile server models/ backbones/ pose eval docs configs/splits scripts \
         data/{raw,interim,processed} docs

# 2) Git + ignore
git init
cat > .gitignore <<'EOF'
.venv/
__pycache__/
.DS_Store
node_modules/
*.mp4
*.avi
*.mov
*.zip
*.tar.gz
*.pt
outputs/
data/
EOF

# 3) Python env
python3 -m venv .venv && source .venv/bin/activate
python -m pip install --upgrade pip wheel pip-tools
cat > requirements.in <<'EOF'
numpy
pandas
opencv-python
mediapipe
pyyaml
scikit-learn
matplotlib
torch
EOF
pip-compile requirements.in
pip install -r requirements.txt
```

### 2.2 Register datasets (single source of truth)

Create `configs/datasets.yaml` (edit local paths as needed):

```yaml
root: "./data/raw"
datasets:
  le2i:
    path: "./data/raw/le2i/videos"   # place RGB videos here
    fps: 25
  urfd:
    path: "./data/raw/urfd/videos"
    fps: 30
  caucafall:
    path: "./data/raw/caucafall/videos"
    fps: 30
  muvim:
    path: "./data/raw/muvim/videos"
    fps: 25
pretrain:
  ntu60:
    note: "Optional: 3D skeleton pretraining"
pose_model:
  name: "BlazePose-FullBody-33"
  out_joints: 33
  format: "xyc"   # [x,y,confidence]
standard_schema:
  frame: {J: 33, order: "mediapipe_blazepose_33"}
  window: {W: 48, stride: 12}
```

Add minimal split lists (example):

```
configs/splits/le2i_train.txt
configs/splits/le2i_val.txt
configs/splits/le2i_test.txt
```

Each file contains **video IDs (filenames without extension)**, one per line.

Create label maps (example): `configs/labels/le2i.json`

```json
{
  "home_01": "fall",
  "home_02": "adl",
  "office_07": "adl"
}
```

### 2.3 Pose extraction → per‑video npz

```bash
source .venv/bin/activate
python pose/extract_2d.py \
  --videos_glob "data/raw/le2i/videos/*.mp4" \
  --out_dir    "data/interim/le2i/pose_npz"

# repeat for the other datasets as you add them
```

**Outputs:** one `*.npz` per video with arrays: `xy: [T,33,2]`, `conf: [T,33]`.

### 2.4 Windowing + baking labels

```bash
python pose/make_windows.py \
  --npz_dir      data/interim/le2i/pose_npz \
  --labels_json  configs/labels/le2i.json \
  --out_dir      data/processed/le2i/windows_W48_S12 \
  --W 48 --stride 12
```

**Outputs:** many window files under `data/processed/...`, each with:
`xy, conf, start, end, label, video_id`.

### 2.5 Train the baseline (TCN)

```bash
python models/train_tcn.py \
  --train_dir data/processed/le2i/windows_W48_S12/train \
  --val_dir   data/processed/le2i/windows_W48_S12/val \
  --epochs 50 --batch 128 --lr 1e-3 --seed 33724876 \
  --save_dir outputs/le2i_tcn_W48S12
```

**Targets (first run)**: Recall ≥ 0.90, F1 ≥ 0.85 on Le2i‑val (OP‑1).

### 2.6 Fit Operating Points (OP‑1/2/3)

```bash
python eval/fit_ops.py \
  --val_dir data/processed/le2i/windows_W48_S12/val \
  --ckpt outputs/le2i_tcn_W48S12/best.pt \
  --out  configs/ops.yaml
```

This exports three thresholds for: **OP‑1: high‑recall**, **OP‑2: balanced**, **OP‑3: low‑alarm**.

### 2.7 Cross‑dataset & FA/24h evaluation

```bash
# Cross‑dataset: URFD
python eval/metrics.py --eval_dir data/processed/urfd/windows_W48_S12/test \
  --ckpt outputs/le2i_tcn_W48S12/best.pt --ops configs/ops.yaml \
  --report outputs/reports/urfd_cross.json

# FA/24h on long negatives (Le2i ADL + CAUCA + MUVIM negatives)
python eval/metrics.py --eval_dir data/processed/neg_long/ \
  --ckpt outputs/le2i_tcn_W48S12/best.pt --ops configs/ops.yaml \
  --report outputs/reports/fa24h.json
```

### 2.8 Make the Slide‑ready FA/24h vs Recall chart

```bash
python eval/plot_fa_recall.py \
  --reports outputs/reports/fa24h.json \
  --title "Innovation Impact: Reducing Alarm Fatigue at Target Recall (H2)" \
  --subtitle "FA/24h vs. Recall (OP‑3 Validation)" \
  --out_fig outputs/figures/slide6_fa_recall.png
```

The plotting script avoids overlapped tags and uses clean label placement.

---

## 3) Datasets We Use & Why

**Primary training & evaluation**

* **Le2i / ImViA** → realistic domestic scenes; great for pose‑only training/eval.

**Cross‑dataset generalisation**

* **UR Fall Detection (URFD)** → standard benchmark; short sequences stress generalisation.

**Stress/robustness**

* **CAUCAFall** → varied lighting/occlusion/distance; home‑like conditions.
* **MUVIM (RGB only)** → night/low‑light slices; keep pipeline identical (2D pose).

**Optional pretraining**

* **NTU RGB+D 60** (3D skeleton) → pretrain temporal backbones; later fine‑tune on Le2i.

> We store only **2D keypoints + confidences** downstream. Depth/IR are not required.

---

## 4) Data Schema (docs/schema.md — summary)

* **Per‑frame**: `xy [33,2]` in normalised image coords; `conf [33]` per joint.
* **Per‑window**: `W=48` frames, `stride=12`; label ∈ `{fall, adl}`; metadata: `video_id, start, end, source`.
* **Conventions**: joint order = `mediapipe_blazepose_33`; missing joints → `NaN` with low confidence.
* **File format**: compressed `npz` for portability and speed.

---

## 5) Models

* **Baseline**: **TCN** over flattened per‑frame features ([33×2] → temporal convs → sigmoid).
* **Variants (later)**: lightweight ST‑GCN/CTR‑GCN; evidential heads.
* **Latency goal**: ≤ 200 ms per 2‑second window on a laptop/edge CPU.

---

## 6) Operating Points (Deployment Views)

* **OP‑1 — High‑Recall**: maximise Recall (safety priority). Target ≥ 0.95 when feasible.
* **OP‑2 — Balanced**: maximise F1 under latency constraint.
* **OP‑3 — Low‑Alarm**: minimise FA/24h subject to a safety floor (Recall ≥ 0.90).

`eval/fit_ops.py` sweeps thresholds on validation to export `configs/ops.yaml`.

---

## 7) Uncertainty‑ & Cost‑Aware Policy

* **MC Dropout**: keep dropout **on** at inference; `T=20` stochastic passes → mean `p` & variance `u`.
* **Decision rule** (initial): adjust threshold by `Δ(p_thr)` up/down when `u` crosses bands `[u_lo, u_hi]`.
* **Goal**: move the **FA/24h vs Recall** curve **left** of the fixed‑threshold baseline.

---

## 8) Metrics & Protocols

* **Recall, Precision, F1** per window and per event.
* **Detection delay**: time between fall onset and first positive window.
* **FA/24h**: (\text{FA/24h} = \frac{\text{false alarms}}{\text{negative hours}}\times 24). Compute negative hours from total duration of **non‑fall** footage.
* **Reporting**: JSON reports in `outputs/reports/`, figures in `outputs/figures/`.

---

## 9) Live Inference (stub)

* `server/` exposes a simple API for camera frames → pose → decision state.
* `apps/mobile/` receives alerts; confirmation/escalation workflow is planned.

---

## 10) Reproducibility Checklist

* [ ] Python 3.12 env created; `requirements.txt` installed
* [ ] `configs/datasets.yaml` filled with correct local paths
* [ ] Splits in `configs/splits/` and labels JSON created
* [ ] Pose extraction run for each dataset
* [ ] Windowing run; `data/processed/...` populated
* [ ] Baseline trained; checkpoint saved to `outputs/.../best.pt`
* [ ] OPs fitted & exported to `configs/ops.yaml`
* [ ] Cross‑dataset and FA/24h reports generated
* [ ] Slide 6 figure exported without overlapping labels

---

## 11) Troubleshooting

* **pip-tools / pip mismatch** (error: `AttributeError: 'InstallRequirement' object has no attribute 'use_pep517'`)

  * *Fix (recommended)*

    ```bash
    source .venv/bin/activate
    python -m pip install --upgrade "pip<25" setuptools wheel
    python -m pip install --upgrade "pip-tools>=7.4"
    pip-compile requirements.in
    pip install -r requirements.txt
    ```
  * *Fix (pin a compatible pair)*

    ```bash
    source .venv/bin/activate
    python -m pip install "pip==24.0" "pip-tools==7.3.0"
    pip-compile requirements.in
    pip install -r requirements.txt
    ```
  * *Bypass pip-tools (quick start)*

    ```bash
    pip install -r requirements.in
    pip freeze > requirements.txt
    ```
* **MediaPipe on macOS**: ensure `opencv-python` is installed; if camera reading fails, install FFmpeg and prefer `.mp4` (H.264/AAC).
* **Torch CPU vs MPS**: default to CPU for determinism; if using MPS, pin a Torch version that supports your macOS & hardware.
* **NaN joints**: extraction sets `NaN` with zero confidence; models should mask or impute.
* **Long videos**: consider downsampling (fps) at extraction time for stability.

---

## 12) Roadmap

* [ ] Event‑level evaluation & smoothing across windows
* [ ] Lightweight ST‑GCN backbone
* [ ] Evidential head for uncertainty
* [ ] React Native alert UI & caregiver flow

---

## 13) License & Ethics

* Use datasets within their licenses and consent terms. This project emphasises **privacy**: store/process skeletons only; avoid keeping raw video when not required.

---

## 14) Citation

If you use this project, please cite the repo and credit the original datasets.

```bibtex
@software{fallguard_2025,
  title  = {FallGuard: Privacy-Preserving On-Device Fall Detection},
  author = {Ru, Zhan and Collaborators},
  year   = {2025},
  url    = {https://example.com/fallguard}
}
```

---

### Appendix A — Script Interfaces (Stubs)

**pose/extract_2d.py**

```
usage: extract_2d.py --videos_glob GLOB --out_dir DIR
outputs: DIR/<video_id>.npz with {xy:[T,33,2], conf:[T,33]}
```

**pose/make_windows.py**

```
usage: make_windows.py --npz_dir DIR --labels_json FILE --out_dir DIR [--W 48 --stride 12]
outputs: windowed npz with baked labels and metadata
```

**models/train_tcn.py**

```
usage: train_tcn.py --train_dir DIR --val_dir DIR --epochs N --batch B --lr LR --save_dir DIR
```

**eval/fit_ops.py**

```
usage: fit_ops.py --val_dir DIR --ckpt FILE --out configs/ops.yaml
```

**eval/metrics.py**

```
usage: metrics.py --eval_dir DIR --ckpt FILE --ops configs/ops.yaml --report FILE
```

**eval/plot_fa_recall.py**

```
usage: plot_fa_recall.py --reports FILE --title STR --subtitle STR --out_fig FILE
```

uvicorn server.app:app --reload --host 0.0.0.0 --port 8000

Nice, this is a good moment to freeze the “runbook” for your examiner 👍
I’ll give you a **clear step-by-step order**, then a **short per-dataset mini-recipe**.

---

## 0. Pre-requisite

Before anything:

* Put raw datasets here (already done in your project):

  * `data/raw/LE2i/...`
  * `data/raw/UR_Fall/...`
  * `data/raw/CAUCAFall/CAUCAFall/...`
  * `data/raw/MUVIM/ZED_RGB/{ADL,Fall}/...`

---

## Global run order (full project, all 4 datasets)

### 1️⃣ Install environment

```bash
make install
```

Creates `.venv`, compiles `requirements.txt`, installs deps.

---

### 2️⃣ Extract pose → NPZ (per dataset)

**LE2i (videos → NPZ)**

```bash
make extract-le2i
```

**URFD (image sequences → NPZ)**

```bash
make extract-urfd
```

**CAUCAFall (image sequences → NPZ)**

```bash
make extract-caucafall
```

**MUVIM (ZED_RGB ADL/Fall → NPZ)**

```bash
make extract-muvim
```

Output: `data/interim/<ds>/pose_npz/*.npz`

---

### 3️⃣ Build labels & unlabeled list

**LE2i labels + spans**

```bash
make labels-le2i
```

**URFD labels (from stems)**

```bash
make labels-urfd
```

**CAUCAFall labels**

* Already written during `extract-caucafall` as:

  * `configs/labels/caucafall_auto.json`
    (so no extra command needed)

**MUVIM labels**

* Also written during `extract-muvim` (using your updated script with `--dataset muvim`).

**Optional LE2i unlabeled office/lecture list**

```bash
make le2i-unlabeled-list
```

---

### 4️⃣ Train/val/test splits

Make stratified (or random, for MUVIM) splits:

```bash
make splits-le2i
make splits-urfd
make splits-caucafall
make splits-muvim
# or all at once:
make splits-all
```

Outputs: `configs/splits/*.txt`

---

### 5️⃣ Windowing (sequence → overlapping windows)

Create sliding windows for each dataset:

```bash
make windows-le2i
make windows-urfd
make windows-caucafall
make windows-muvim
```

Optional unlabeled windows (office / lecture scenes):

```bash
make windows-le2i-unlabeled
```

Outputs:
`data/processed/<ds>/windows_W48_S12/{train,val,test}/...`

---

### 6️⃣ Sanity checks

Quick check that window counts and label balance look reasonable:

```bash
make check-le2i
make check-urfd
make check-caucafall
make check-muvim
```

---

### 7️⃣ Train TCN baselines

Train one TCN per dataset:

```bash
make train-le2i      # alias: make train
make train-urfd
make train-caucafall
make train-muvim
```

Outputs checkpoints:

* `outputs/le2i_tcn_W48S12/best.pt`
* `outputs/urfd_tcn_W48S12/best.pt`
* `outputs/caucafall_tcn_W48S12/best.pt`
* `outputs/muvim_tcn_W48S12/best.pt`

---

### 8️⃣ Fit OP1/OP2/OP3 thresholds (TCN)

Fit 3 operating points per dataset on **validation** windows:

```bash
make fit-ops-le2i
make fit-ops-urfd
make fit-ops-caucafall
make fit-ops-muvim

# alias used in pipeline (just LE2i)
make fit-ops   # == fit-ops-le2i
```

Outputs:

* `configs/ops_le2i.yaml`
* `configs/ops_urfd.yaml`
* `configs/ops_caucafall.yaml`
* `configs/ops_muvim.yaml`

---

### 9️⃣ Evaluate TCN models

In-domain & cross-dataset (for your hypotheses):

```bash
# LE2i test (in-domain)
make eval-le2i

# LE2i model on URFD test (cross, your H2 / innovation impact)
make eval-le2i-on-urfd   # alias: make eval-urfd

# CAUCAFall test (in-domain)
make eval-caucafall

# CAUCAFall model on URFD test (cross)
make eval-caucafall-on-urfd

# MUVIM test (in-domain)
make eval-muvim
```

Outputs (JSON reports with FA/24h etc.):

* `outputs/reports/le2i_in_domain.json`
* `outputs/reports/urfd_cross.json`
* `outputs/reports/caucafall_in_domain.json`
* `outputs/reports/caucafall_on_urfd.json`
* `outputs/reports/muvim_in_domain.json`

---

### 🔟 Plot TCN FA/24h vs recall curves

```bash
make plot-le2i
make plot-urfd-cross
make plot-caucafall
make plot-caucafall-on-urfd
make plot-muvim

# alias
make plot   # == plot-urfd-cross
```

Outputs PNGs under `outputs/figures/`.

---

### 1️⃣1️⃣ Train GCN models (with in-domain reports)

```bash
make train-gcn-le2i
make train-gcn-urfd
make train-gcn-caucafall
make train-gcn-muvim
# or all:
make train-gcn
```

Outputs:

* checkpoints: `outputs/<ds>_gcn_W48S12/best.pt`
* reports: `outputs/reports/<ds>_gcn_in_domain.json`
  (with OP1/2/3 style thresholds & F1 for your thesis comparison)

---

### 1️⃣2️⃣ Plot GCN FA/24h vs recall curves

```bash
make plot-le2i-gcn
make plot-urfd-gcn
make plot-caucafall-gcn
make plot-muvim-gcn
```

Outputs PNGs under `outputs/figures/` for GCN vs TCN comparison.

---

### 1️⃣3️⃣ Shadow-deploy on unlabeled office/lecture (optional)

For “alarm fatigue in normal days” on LE2i unlabeled scenes:

```bash
make score-unlabeled-le2i
# or override threshold / fps / cooldown:
make score-unlabeled-le2i THR=0.38 FPS=25 COOL=3
```

Outputs CSV:

* `outputs/reports/office_lecture_unlabeled_scores.csv`

---

## Short “recipes” per dataset

### LE2i (main in-domain + cross-URFD)

```bash
make extract-le2i
make labels-le2i
make splits-le2i
make windows-le2i
make check-le2i

make train-le2i           # TCN
make fit-ops-le2i
make eval-le2i            # in-domain
make eval-le2i-on-urfd    # cross → URFD
make plot-le2i
make plot-urfd-cross

make train-gcn-le2i       # GCN + report
make plot-le2i-gcn
```

### URFD

```bash
make extract-urfd
make labels-urfd
make splits-urfd
make windows-urfd
make check-urfd

make train-urfd
make fit-ops-urfd
make eval-muvim   # (for URFD we mostly care about cross with LE2i/CAUCAFall)
make plot-urfd-gcn (after train-gcn-urfd)
```

### CAUCAFall

```bash
make extract-caucafall
make splits-caucafall
make windows-caucafall
make check-caucafall

make train-caucafall
make fit-ops-caucafall
make eval-caucafall
make eval-caucafall-on-urfd
make plot-caucafall
make plot-caucafall-on-urfd

make train-gcn-caucafall
make plot-caucafall-gcn
```

### MUVIM

```bash
make extract-muvim
make splits-muvim
make windows-muvim
make check-muvim

make train-muvim
make fit-ops-muvim
make eval-muvim
make plot-muvim

make train-gcn-muvim
make plot-muvim-gcn
```


1. Extract pose         → unlabeled NPZ sequences
2. Attach labels        → labels JSON (per-sequence)
3. Split sequences      → train/val/test .txt lists
4. Make window NPZs     → each with xy, conf, y
5. Train models


How to run the server: 
```
source .venv/bin/activate
uvicorn server.app:app --reload --host 0.0.0.0 --port 8000
```