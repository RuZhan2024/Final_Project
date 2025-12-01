from pathlib import Path
import json
import numpy as np
import torch
import yaml
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from models.train_tcn import TCN   # your tiny TCN class

# -------------------------
# Paths / config
# -------------------------
WIN_W = 48
WIN_S = 12

# window dirs
WIN_DIR_LE2I = Path(f"data/processed/le2i/windows_W{WIN_W}_S{WIN_S}")
WIN_DIR_URFD = Path(f"data/processed/urfd/windows_W{WIN_W}_S{WIN_S}")
WIN_DIR_CAUCA = Path(f"data/processed/caucafall/windows_W{WIN_W}_S{WIN_S}")

# checkpoints
CKPT_LE2I = Path(f"outputs/le2i_tcn_W{WIN_W}S{WIN_S}/best.pt")
CKPT_URFD = Path(f"outputs/urfd_tcn_W{WIN_W}S{WIN_S}/best.pt")
CKPT_CAUCA = Path(f"outputs/caucafall_tcn_W{WIN_W}S{WIN_S}/best.pt")

# fit-ops yaml (per dataset)
OPS_LE2I = Path("configs/ops_le2i.yaml")
OPS_URFD = Path("configs/ops_urfd.yaml")
OPS_CAUCA = Path("configs/ops_caucafall.yaml")

# metrics reports from your eval commands
REPORT_LE2I = Path("outputs/reports/le2i_in_domain.json")
REPORT_URFD_CROSS = Path("outputs/reports/urfd_cross.json")                  # LE2I → URFD
REPORT_CAUCA_IN_DOMAIN = Path("outputs/reports/caucafall_in_domain.json")
REPORT_CAUCA_ON_URFD = Path("outputs/reports/caucafall_on_urfd.json")


app = FastAPI()

# Allow React dev server to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------
# Model loading
# -------------------------
def load_tcn_from_ckpt(ckpt_path: Path):
    """
    Load a TCN from one of your best.pt files.

    best.pt was saved as:
      {"model": state_dict, "in_ch": C, "best_thr": thr}
    """
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if not isinstance(ckpt, dict) or "model" not in ckpt:
        raise RuntimeError(f"Unexpected checkpoint format: {ckpt_path}")

    state_dict = ckpt["model"]
    in_ch = ckpt.get("in_ch", None)
    best_thr = float(ckpt.get("best_thr", 0.5))

    if in_ch is None:
        # Try to infer from first Conv1d weight
        for name, w in state_dict.items():
            if "net.0.weight" in name and w.ndim == 3:
                in_ch = w.shape[1]
                break
        if in_ch is None:
            raise RuntimeError("Could not infer in_ch; set it manually")

    model = TCN(in_ch=in_ch)
    model.load_state_dict(state_dict)
    model.eval()
    return model, best_thr

model_le2i, thr_ckpt_le2i = load_tcn_from_ckpt(CKPT_LE2I)
model_urfd, thr_ckpt_urfd = load_tcn_from_ckpt(CKPT_URFD)
model_cauca, thr_ckpt_cauca = load_tcn_from_ckpt(CKPT_CAUCA)


def load_ops_thr(path: Path, default_thr: float):
    if not path.exists():
        return default_thr
    with open(path, "r") as f:
        ops = yaml.safe_load(f)
    # try to use OP3_low_alarm, else just fallback
    if isinstance(ops, dict) and "OP3_low_alarm" in ops:
        return float(ops["OP3_low_alarm"].get("thr", default_thr))
    return default_thr

THR_LE2I = load_ops_thr(OPS_LE2I, thr_ckpt_le2i)
THR_URFD = load_ops_thr(OPS_URFD, thr_ckpt_urfd)
THR_CAUCA = load_ops_thr(OPS_CAUCA, thr_ckpt_cauca)

def load_report(path: Path):
    if not path.exists():
        return None
    with open(path, "r") as f:
        return json.load(f)

def summarise_report(rep):
    """
    Convert the metrics report (whatever its exact structure) into
    a list of operating points with basic metrics.

    Expected format (based on plot_fa_recall.py):
      - either: {"ops": {"OP1": {...}, "OP3_low_alarm": {...}}}
      - or: {"OP1": {...}, "OP3_low_alarm": {...}}
    """
    if rep is None or not isinstance(rep, dict):
        return []

    if "ops" in rep and isinstance(rep["ops"], dict):
        ops_dict = rep["ops"]
    else:
        ops_dict = rep

    summary = []
    for name, d in ops_dict.items():
        if not isinstance(d, dict):
            continue
        summary.append(
            {
                "name": name,
                "thr": d.get("thr"),
                "precision": d.get("precision"),
                "recall": d.get("recall"),
                "fa24h": d.get("fa24h"),
                # optional, in case F1 is present
                "F1": d.get("F1"),
            }
        )
    return summary


# -------------------------
# Feature extraction for windows (same as WindowNPZ)
# -------------------------
def make_window_features(npz_path: Path) -> np.ndarray:
    """
    Re-creates the x features exactly like WindowNPZ.__getitem__:
      xy: [W,33,2], conf: [W,33] -> x: [W, 33*2]
    """
    d = np.load(npz_path, allow_pickle=False)
    xy = d["xy"].astype(np.float32)    # [W,33,2]
    conf = d["conf"].astype(np.float32)  # [W,33]

    xy = np.nan_to_num(xy, nan=0.0, posinf=0.0, neginf=0.0)
    x = xy * conf[..., None]           # [W,33,2]
    x = x.reshape(x.shape[0], -1)      # [W, 33*2] = [T, C]
    return x


# -------------------------
# Demo window loaders
# -------------------------
def _load_demo_windows(root_dir: Path, max_windows: int = 40) -> np.ndarray:
    """
    Generic helper: load up to max_windows NPZ windows from root_dir/test
    and build features X with shape [N, T, C].
    """
    root = root_dir / "test"
    npz_files = sorted(root.glob("*.npz"))
    if not npz_files:
        raise RuntimeError(f"No .npz windows found in {root}")

    xs = []
    for f in npz_files[:max_windows]:
        xs.append(make_window_features(f))

    X = np.stack(xs, axis=0)  # [N, T, C]
    return X


def load_demo_windows_from_le2i(max_windows: int = 40) -> np.ndarray:
    return _load_demo_windows(WIN_DIR_LE2I, max_windows)


def load_demo_windows_from_urfd(max_windows: int = 40) -> np.ndarray:
    return _load_demo_windows(WIN_DIR_URFD, max_windows)


def load_demo_windows_from_caucafall(max_windows: int = 40) -> np.ndarray:
    return _load_demo_windows(WIN_DIR_CAUCA, max_windows)


def run_tcn_demo(model: TCN, X: np.ndarray, thr: float):
    """
    X: [N, T, C] → returns list of points:
      [{t: int, p_fall: float, fall: bool}, ...]
    We treat each window (sample) as one timestep in the UI.
    """
    with torch.no_grad():
        inp = torch.from_numpy(X).float()  # [N, T, C]
        logits = model(inp).squeeze(-1)    # [N]
        probs = torch.sigmoid(logits).cpu().numpy()  # [N]

    points = []
    for i, p in enumerate(probs):
        points.append(
            {
                "t": int(i),
                "p_fall": float(p),
                "fall": bool(p >= thr),
            }
        )
    return points

@app.get("/api/models/summary")
def models_summary():
    """
    Returns a summary of all three TCN models:
      - which dataset they were trained on
      - best_thr from checkpoint / OPs
      - metrics reports (in-domain + cross) if available
    """
    le2i_rep = summarise_report(load_report(REPORT_LE2I))
    urfd_cross_rep = summarise_report(load_report(REPORT_URFD_CROSS))
    cauca_in_rep = summarise_report(load_report(REPORT_CAUCA_IN_DOMAIN))
    cauca_on_urfd_rep = summarise_report(load_report(REPORT_CAUCA_ON_URFD))

    return {
        "models": [
            {
                "id": "le2i",
                "label": "TCN trained on LE2I",
                "dataset": "LE2I",
                "ckpt": str(CKPT_LE2I),
                "best_thr_from_ckpt": thr_ckpt_le2i,
                "demo_thr": THR_LE2I,
                "reports": {
                    "in_domain": le2i_rep,
                    "cross": urfd_cross_rep,  # LE2I → URFD
                },
            },
            {
                "id": "urfd",
                "label": "TCN trained on URFD",
                "dataset": "URFD",
                "ckpt": str(CKPT_URFD),
                "best_thr_from_ckpt": thr_ckpt_urfd,
                "demo_thr": THR_URFD,
                # if you later run eval-urfd-in-domain, you can plug it here
                "reports": {
                    "in_domain": [],  # placeholder
                    "cross": [],      # placeholder
                },
            },
            {
                "id": "caucafall",
                "label": "TCN trained on CAUCAFall",
                "dataset": "CAUCAFall",
                "ckpt": str(CKPT_CAUCA),
                "best_thr_from_ckpt": thr_ckpt_cauca,
                "demo_thr": THR_CAUCA,
                "reports": {
                    "in_domain": cauca_in_rep,
                    "cross": cauca_on_urfd_rep,  # CAUCA → URFD
                },
            },
        ]
    }



# -------------------------
# API routes
# -------------------------
@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.post("/api/demo/le2i_fall")
def demo_le2i_fall():
    """
    Demo endpoint for LE2I model.
    """
    X = load_demo_windows_from_le2i(max_windows=40)
    points = run_tcn_demo(model_le2i, X, THR_LE2I)

    return {
        "model": "le2i",
        "fps": 5,                # UI playback speed hint
        "threshold": THR_LE2I,   # fall vs safe threshold
        "points": points,
    }


@app.post("/api/demo/urfd_fall")
def demo_urfd_fall():
    """
    Demo endpoint for URFD model.
    """
    X = load_demo_windows_from_urfd(max_windows=40)
    points = run_tcn_demo(model_urfd, X, THR_URFD)

    return {
        "model": "urfd",
        "fps": 5,
        "threshold": THR_URFD,
        "points": points,
    }


@app.post("/api/demo/caucafall_fall")
def demo_caucafall_fall():
    """
    Demo endpoint for CAUCAFall model.
    """
    X = load_demo_windows_from_caucafall(max_windows=40)
    points = run_tcn_demo(model_cauca, X, THR_CAUCA)

    return {
        "model": "caucafall",
        "fps": 5,
        "threshold": THR_CAUCA,
        "points": points,
    }


