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
