#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""eval/fit_ops.py

Fit operating points (OP1/OP2/OP3) for the *deployment alert policy*.

Key behaviours (repo contracts):
  - Contract B: build features via core.features (no hard-coded slicing).
  - Contract C: calibration is temperature scaling only (scalar T).

This script:
  1) Loads a validation window dir + model checkpoint
  2) Runs inference to obtain logits
  3) Fits temperature T on validation logits
  4) Sweeps tau_high under the alert policy (incl. confirm gating)
  5) Picks OPs and writes ops YAML (embedding calibration)

Optionally, provide a separate FA-only window dir to estimate FA/24h from a
more realistic negative stream (recommended).

Notes
  - The sweep curve can have large plateaus (many thresholds with identical
    best F1 / FA/24h). By default we pick *conservative* thresholds on such
    plateaus (highest tau_high), which tends to be more deployable.
"""

from __future__ import annotations

import argparse
import os
import json
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from fall_detection.core.alerting import AlertCfg, pick_ops_from_sweep, sweep_alert_policy_from_windows
from fall_detection.core.calibration import fit_temperature, sigmoid
from fall_detection.core.ckpt import load_ckpt
from fall_detection.core.confirm import confirm_scores_window
from fall_detection.core.features import FeatCfg, build_canonical_input, build_tcn_input, read_window_npz, split_gcn_two_stream
from fall_detection.core.models import build_model
from fall_detection.core.yamlio import yaml_dump_simple, yaml_load_simple


def _strip(s: str) -> str:
    return str(s).strip()

def _strip_lower(s: str) -> str:
    return str(s).strip().lower()

def _int_or_default(default: int):
    def _f(x):
        xs = str(x).strip()
        return default if xs == "" else int(xs)
    return _f


def logits_1d(x: torch.Tensor) -> torch.Tensor:
    """Force logits into shape [B]."""
    if x.ndim == 2 and x.shape[1] == 1:
        return x[:, 0]
    if x.ndim == 1:
        return x
    return x.reshape(x.shape[0], -1)[:, 0]


def _sanitize_json(x: Any) -> Any:
    """Convert numpy/scalars to JSON-safe types and replace NaN/Inf with null."""
    if isinstance(x, dict):
        return {str(k): _sanitize_json(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_sanitize_json(v) for v in x]
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        f = float(x)
        return None if not math.isfinite(f) else f
    if isinstance(x, float):
        return None if not math.isfinite(x) else x
    if isinstance(x, (int, str, bool)) or x is None:
        return x
    # fallback: string representation
    return str(x)


def _to_np(x: Any, *, dtype=np.float32) -> np.ndarray:
    """Convert a sweep list (possibly containing None) to a numpy array."""
    if isinstance(x, np.ndarray):
        return x.astype(dtype, copy=False)
    if x is None:
        return np.asarray([], dtype=dtype)
    # Replace None with nan
    if isinstance(x, (list, tuple)):
        out = [np.nan if v is None else v for v in x]
        return np.asarray(out, dtype=dtype)
    return np.asarray(x, dtype=dtype)


def _pick_index_by_thr(cand: np.ndarray, thr: np.ndarray, *, tie_break: str) -> int:
    """Pick an index from candidate indices based on thr tie-break."""
    if cand.size == 0:
        raise ValueError("empty candidate set")
    if cand.size == 1:
        return int(cand[0])
    if tie_break == "min_thr":
        return int(cand[np.nanargmin(thr[cand])])
    # default: max_thr
    return int(cand[np.nanargmax(thr[cand])])


def _path_ref_for_output(target_path: str, out_yaml_path: str, *, emit_absolute_paths: bool) -> str:
    """Encode path in output YAML as absolute or out-relative."""
    tgt = os.path.abspath(str(target_path))
    if emit_absolute_paths:
        return tgt
    base_dir = os.path.dirname(os.path.abspath(str(out_yaml_path)))
    return os.path.relpath(tgt, start=base_dir)


def _sweep_has_non_degenerate_alert_region(sweep: Dict[str, Any], *, eps: float = 1e-12) -> Tuple[bool, str]:
    """Check whether sweep has at least one meaningful alert point."""
    rec = _to_np(sweep.get("recall"), dtype=np.float32)
    f1 = _to_np(sweep.get("f1"), dtype=np.float32)
    n_alert = _to_np(sweep.get("n_alert_events"), dtype=np.float32)

    has_pos_recall = bool(np.isfinite(rec).any() and np.nanmax(rec) > eps)
    has_pos_f1 = bool(np.isfinite(f1).any() and np.nanmax(f1) > eps)
    has_alert_events = bool(np.isfinite(n_alert).any() and np.nanmax(n_alert) > eps)

    if has_pos_recall or has_pos_f1 or has_alert_events:
        return True, ""

    return (
        False,
        "degenerate sweep: no positive recall, no positive F1, and no alert events across thresholds. "
        "Try refitting with ALERT_CONFIRM=0 first, or relax confirm gating/thresholds.",
    )


def pick_ops_from_sweep_conservative(
    sweep: Dict[str, Any],
    *,
    op1_recall: float,
    op3_fa24h: float,
    op2_objective: str = "f1",
    cost_fn: float = 5.0,
    cost_fp: float = 1.0,
    tie_break: str = "max_thr",
    min_tau_high: float = 0.0,
    eps: float = 1e-9,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Pick OPs from sweep dict with conservative tie-breaks.

    Expected sweep keys (from fall_detection.core.alerting.sweep_alert_policy_from_windows):
      thr, tau_low, precision, recall, f1, fa24h, mean_delay_s, ...

    Strategy
      OP2: maximize F1, tie-break by thr (default: max_thr).
      OP1: among points with recall>=op1_recall, maximize F1, tie-break by thr.
           If none meet recall, fall back to max recall (tie-break by thr).
      OP3: among points with fa24h<=op3_fa24h, maximize recall, then minimize
           fa24h, then tie-break by thr. If none meet fa, fall back to minimum
           fa24h (then max recall), then tie-break by thr.
    """

    thr = _to_np(sweep.get("thr"), dtype=np.float32)
    tau_low = _to_np(sweep.get("tau_low"), dtype=np.float32)
    prec = _to_np(sweep.get("precision"), dtype=np.float32)
    rec = _to_np(sweep.get("recall"), dtype=np.float32)
    f1 = _to_np(sweep.get("f1"), dtype=np.float32)
    fa24h = _to_np(sweep.get("fa24h"), dtype=np.float32)
    mean_delay = _to_np(sweep.get("mean_delay_s"), dtype=np.float32)
    med_delay = _to_np(sweep.get("median_delay_s"), dtype=np.float32)
    n_gt = _to_np(sweep.get("n_gt_events"), dtype=np.float32)
    n_alert = _to_np(sweep.get("n_alert_events"), dtype=np.float32)
    n_true = _to_np(sweep.get("n_true_alerts"), dtype=np.float32)
    n_false = _to_np(sweep.get("n_false_alerts"), dtype=np.float32)

    if thr.size == 0:
        raise ValueError("sweep.thr is empty")

    # Optional lower bound on tau_high for deployability.
    ok_thr = np.isfinite(thr) & (thr >= float(min_tau_high))

    def _row(i: int) -> Dict[str, Any]:
        i = int(i)
        return {
            "tau_high": float(thr[i]),
            "tau_low": float(tau_low[i]) if tau_low.size == thr.size else float(0.0),
            "precision": None if not np.isfinite(prec[i]) else float(prec[i]),
            "recall": None if not np.isfinite(rec[i]) else float(rec[i]),
            "f1": None if not np.isfinite(f1[i]) else float(f1[i]),
            "fa24h": None if not np.isfinite(fa24h[i]) else float(fa24h[i]),
            "mean_delay_s": None if not np.isfinite(mean_delay[i]) else float(mean_delay[i]),
            "median_delay_s": None if not np.isfinite(med_delay[i]) else float(med_delay[i]),
            "n_gt_events": int(n_gt[i]) if np.isfinite(n_gt[i]) else None,
            "n_alert_events": int(n_alert[i]) if np.isfinite(n_alert[i]) else None,
            "n_true_alerts": int(n_true[i]) if np.isfinite(n_true[i]) else None,
            "n_false_alerts": int(n_false[i]) if np.isfinite(n_false[i]) else None,
        }

    # ---------- OP2 (objective) ----------
    op2_obj = str(op2_objective).strip().lower()
    if op2_obj == "cost_sensitive":
        ng_ok = np.isfinite(n_gt) & ok_thr
        nt_ok = np.isfinite(n_true) & ok_thr
        nf_ok = np.isfinite(n_false) & ok_thr
        cost_ok = ng_ok & nt_ok & nf_ok
        if cost_ok.any():
            fn = np.maximum(0.0, n_gt - n_true)
            cost = float(cost_fn) * fn + float(cost_fp) * n_false
            best_cost = float(np.nanmin(cost[cost_ok]))
            cand = np.where(cost_ok & (cost <= best_cost + eps))[0]
            i2 = _pick_index_by_thr(cand, thr, tie_break=tie_break)
        else:
            op2_obj = "f1"  # fallback
    if op2_obj == "f1":
        f1_ok = np.isfinite(f1) & ok_thr
        if not f1_ok.any():
            # fallback to max recall
            rec_ok = np.isfinite(rec) & ok_thr
            i2 = int(np.nanargmax(rec[rec_ok]))
            cand = np.where(rec_ok & (rec >= np.nanmax(rec[rec_ok]) - eps))[0]
            i2 = _pick_index_by_thr(cand, thr, tie_break=tie_break)
        else:
            best = float(np.nanmax(f1[f1_ok]))
            cand = np.where(f1_ok & (f1 >= best - eps))[0]
            i2 = _pick_index_by_thr(cand, thr, tie_break=tie_break)

    # ---------- OP1 (recall target) ----------
    rec_ok = np.isfinite(rec) & ok_thr
    meet = rec_ok & (rec >= float(op1_recall) - eps)
    if meet.any():
        # maximize F1 among those meeting recall
        f1_meet_ok = np.isfinite(f1) & meet
        if f1_meet_ok.any():
            best = float(np.nanmax(f1[f1_meet_ok]))
            cand = np.where(f1_meet_ok & (f1 >= best - eps))[0]
            i1 = _pick_index_by_thr(cand, thr, tie_break=tie_break)
        else:
            # no finite F1 -> pick max thr among recall-meeting
            cand = np.where(meet)[0]
            i1 = _pick_index_by_thr(cand, thr, tie_break=tie_break)
    elif rec_ok.any():
        best_rec = float(np.nanmax(rec[rec_ok]))
        cand = np.where(rec_ok & (rec >= best_rec - eps))[0]
        i1 = _pick_index_by_thr(cand, thr, tie_break=tie_break)
    else:
        i1 = i2

    # ---------- OP3 (FA constraint) ----------
    fa_ok = np.isfinite(fa24h) & ok_thr
    meet_fa = fa_ok & (fa24h <= float(op3_fa24h) + eps)
    if meet_fa.any() and np.isfinite(rec).any():
        # maximize recall, then minimize fa24h
        rec_meet = rec.copy()
        rec_meet[~meet_fa] = -np.inf
        best_rec = float(np.nanmax(rec_meet))
        cand = np.where(meet_fa & (rec >= best_rec - eps))[0]
        if cand.size > 1:
            # minimize fa first
            fa_c = fa24h[cand]
            min_fa = float(np.nanmin(fa_c))
            cand2 = cand[np.where(fa_c <= min_fa + eps)[0]]
            cand = cand2
        i3 = _pick_index_by_thr(cand, thr, tie_break=tie_break)
    elif fa_ok.any():
        # fallback: minimize fa24h, then max recall
        min_fa = float(np.nanmin(fa24h[fa_ok]))
        cand = np.where(fa_ok & (fa24h <= min_fa + eps))[0]
        if cand.size > 1 and np.isfinite(rec).any():
            rec_c = rec[cand]
            best_rec = float(np.nanmax(rec_c))
            cand = cand[np.where(rec_c >= best_rec - eps)[0]]
        i3 = _pick_index_by_thr(cand, thr, tie_break=tie_break)
    else:
        i3 = i2

    ops = {"OP1": _row(i1), "OP2": _row(i2), "OP3": _row(i3)}
    meta = {
        "picker": "conservative",
        "tie_break": str(tie_break),
        "min_tau_high": float(min_tau_high),
        "op1_recall": float(op1_recall),
        "op3_fa24h": float(op3_fa24h),
        "op2_objective": str(op2_obj),
        "cost_fn": float(cost_fn),
        "cost_fp": float(cost_fp),
        "idx": {"OP1": int(i1), "OP2": int(i2), "OP3": int(i3)},
    }
    return ops, meta


@dataclass
class MetaRow:
    video_id: str
    w_start: int
    w_end: int
    fps: float
    y: int
    lying_score: float
    motion_score: float


class WindowDirDataset(Dataset):
    def __init__(
        self,
        root: str,
        *,
        feat_cfg: FeatCfg,
        arch: str,
        two_stream: bool,
        fps_default: float,
        recursive: bool = False,
    ) -> None:
        self.root = str(root)
        self.paths: List[str] = []
        if recursive:
            for dp, _dns, fns in os.walk(self.root):
                for fn in fns:
                    if fn.startswith("."):
                        continue
                    if fn.endswith(".npz"):
                        self.paths.append(os.path.join(dp, fn))
        else:
            for fn in os.listdir(self.root):
                if fn.startswith("."):
                    continue
                if fn.endswith(".npz"):
                    self.paths.append(os.path.join(self.root, fn))
        self.paths = sorted(self.paths)
        if not self.paths:
            raise SystemExit(f"[err] no .npz files under: {self.root}")

        self.feat_cfg = feat_cfg
        self.arch = str(arch).lower()
        self.two_stream = bool(two_stream)
        self.fps_default = float(fps_default)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        path = self.paths[idx]
        joints_xy, motion_xy, conf, mask, fps, meta = read_window_npz(path, fps_default=self.fps_default)
        fps = float(fps) if fps and fps > 0 else float(self.fps_default)

        X, m = build_canonical_input(
            joints_xy=joints_xy,
            motion_xy=motion_xy,
            conf=conf,
            mask=mask,
            fps=fps,
            feat_cfg=self.feat_cfg,
        )
        ls, ms = confirm_scores_window(joints_xy, m, fps, tail_s=1.0)
        # Preserve unavailable confirm signals as NaN.
        # Alert policy treats NaN confirm heuristics as "missing" (no veto).
        ls = float(ls) if np.isfinite(ls) else float("nan")
        ms = float(ms) if np.isfinite(ms) else float("nan")

        y = int(meta.y) if meta.y is not None else -1
        row = MetaRow(
            video_id=str(meta.video_id or ""),
            w_start=int(meta.w_start),
            w_end=int(meta.w_end),
            fps=float(fps),
            y=y,
            lying_score=float(ls),
            motion_score=float(ms),
        )

        if self.arch == "gcn":
            if self.two_stream:
                xj, xm = split_gcn_two_stream(X, self.feat_cfg)
                return (
                    torch.from_numpy(xj).to(torch.float32),
                    torch.from_numpy(xm).to(torch.float32),
                    torch.tensor(float(0.0 if y < 0 else y), dtype=torch.float32),
                    row,
                )
            return (
                torch.from_numpy(X).to(torch.float32),
                torch.tensor(float(0.0 if y < 0 else y), dtype=torch.float32),
                row,
            )

        # tcn
        x = build_tcn_input(X, self.feat_cfg)
        return (
            torch.from_numpy(x).to(torch.float32),
            torch.tensor(float(0.0 if y < 0 else y), dtype=torch.float32),
            row,
        )


def _collate(batch):
    # Keep MetaRow objects as a list.
    if len(batch[0]) == 4:
        xj, xm, y, meta = zip(*batch)
        return torch.stack(list(xj)), torch.stack(list(xm)), torch.stack(list(y)), list(meta)
    xb, y, meta = zip(*batch)
    return torch.stack(list(xb)), torch.stack(list(y)), list(meta)


@torch.no_grad()
def infer_logits(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    arch: str,
    two_stream: bool,
) -> Tuple[np.ndarray, np.ndarray, List[MetaRow]]:
    model.eval()
    logits_all: List[np.ndarray] = []
    y_all: List[np.ndarray] = []
    metas_all: List[MetaRow] = []

    for batch in tqdm(loader, desc="infer", leave=False):
        if arch == "gcn" and two_stream:
            xj, xm, yb, meta = batch
            xj = xj.to(device).float()
            xm = xm.to(device).float()
            yb = yb.to(device).view(-1)
            logits = logits_1d(model(xj, xm)).detach().cpu().numpy()
        else:
            xb, yb, meta = batch
            xb = xb.to(device).float()
            yb = yb.to(device).view(-1)
            logits = logits_1d(model(xb)).detach().cpu().numpy()

        logits_all.append(logits.astype(np.float32, copy=False))
        y_all.append(yb.detach().cpu().numpy().astype(np.float32, copy=False))
        metas_all.extend(list(meta))

    return np.concatenate(logits_all), np.concatenate(y_all), metas_all


def _preflight_input_contract(
    *,
    arch: str,
    model_cfg: Dict[str, Any],
    ds: WindowDirDataset,
    two_stream: bool,
    source_dir: str,
) -> None:
    """Fail fast when checkpoint input contract and window features do not match."""
    if len(ds) == 0:
        return
    kind = str(arch).lower()
    sample = ds[0]
    if kind == "tcn":
        x = sample[0]
        got_in = int(x.shape[-1])
        exp_in = int(model_cfg.get("in_ch", got_in))
        if got_in != exp_in:
            raise RuntimeError(
                f"Input feature mismatch for fit_ops: ckpt expects in_ch={exp_in}, "
                f"but windows from '{source_dir}' produce in_ch={got_in}. "
                "This usually means a stale checkpoint built from a different window contract "
                "(for example 33-joint vs 17-joint adapter mode). Rebuild training + eval windows "
                "and retrain before fit_ops (e.g., make -B ADAPTER_USE=1 train-tcn-<dataset>)."
            )
        return

    if kind == "gcn":
        if two_stream:
            xj, xm = sample[0], sample[1]
            got_j = int(xj.shape[1])
            got_fj = int(xj.shape[-1])
            got_fm = int(xm.shape[-1])
            exp_j = int(model_cfg.get("num_joints", got_j))
            exp_fj = int(model_cfg.get("in_feats_joint", model_cfg.get("in_feats", got_fj)))
            exp_fm = int(model_cfg.get("in_feats_motion", model_cfg.get("in_feats", got_fm)))
            if got_j != exp_j or got_fj != exp_fj or got_fm != exp_fm:
                raise RuntimeError(
                    "Input feature mismatch for fit_ops (GCN two-stream): "
                    f"ckpt expects num_joints={exp_j}, in_feats_joint={exp_fj}, in_feats_motion={exp_fm}; "
                    f"windows from '{source_dir}' produced num_joints={got_j}, "
                    f"in_feats_joint={got_fj}, in_feats_motion={got_fm}. "
                    "Rebuild windows and retrain with a consistent feature contract."
                )
        else:
            x = sample[0]
            got_j = int(x.shape[1])
            got_f = int(x.shape[-1])
            exp_j = int(model_cfg.get("num_joints", got_j))
            exp_f = int(model_cfg.get("in_feats", got_f))
            if got_j != exp_j or got_f != exp_f:
                raise RuntimeError(
                    "Input feature mismatch for fit_ops (GCN): "
                    f"ckpt expects num_joints={exp_j}, in_feats={exp_f}; "
                    f"windows from '{source_dir}' produced num_joints={got_j}, in_feats={got_f}. "
                    "Rebuild windows and retrain with a consistent feature contract."
                )


def _extract_arrays(metas: Sequence[MetaRow]) -> Dict[str, np.ndarray]:
    vids = np.asarray([m.video_id for m in metas], dtype=object)
    ws = np.asarray([m.w_start for m in metas], dtype=np.int32)
    we = np.asarray([m.w_end for m in metas], dtype=np.int32)
    fps = np.asarray([m.fps for m in metas], dtype=np.float32)
    y = np.asarray([m.y for m in metas], dtype=np.int32)
    ls = np.asarray([m.lying_score for m in metas], dtype=np.float32)
    ms = np.asarray([m.motion_score for m in metas], dtype=np.float32)
    return {"video_id": vids, "w_start": ws, "w_end": we, "fps": fps, "y": y, "lying": ls, "motion": ms}


def _override_feat_cfg(base: FeatCfg, args: argparse.Namespace) -> FeatCfg:
    d = base.to_dict()
    # Only override if user explicitly passed the flag.
    if args.use_bone is not None:
        d["use_bone"] = bool(int(args.use_bone))
    if args.use_bone_length is not None:
        d["use_bone_length"] = bool(int(args.use_bone_length))
    if args.use_motion is not None:
        d["use_motion"] = bool(int(args.use_motion))
    if args.use_conf_channel is not None:
        d["use_conf_channel"] = bool(int(args.use_conf_channel))
    if args.center is not None:
        d["center"] = str(args.center)
    return FeatCfg.from_dict(d)


def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument("--arch", type=str, required=True, choices=["tcn", "gcn"])
    ap.add_argument("--val_dir", type=str, required=True)
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--out", type=str, required=True)

    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--num_workers", type=int, default=0)

    ap.add_argument("--fps_default", type=float, default=30.0)

    # Feature overrides (optional). If not provided, use feat_cfg stored in ckpt.
    ap.add_argument("--center", type=str, default=None)
    ap.add_argument("--use_motion", type=int, default=None)
    ap.add_argument("--use_conf_channel", type=int, default=None)
    ap.add_argument("--use_bone", type=int, default=None)
    ap.add_argument("--use_bone_length", type=int, default=None)

    # Calibration
    ap.add_argument("--calibration_yaml", type=str, default="")
    ap.add_argument("--no_fit_calibration", action="store_true")

    # Optional FA window set
    ap.add_argument("--fa_dir", type=str, default="")

    # Persist sweep curves as JSON (recommended; YAML dumper is scalar-only)
    ap.add_argument("--save_sweep_json", type=int, default=1)

    # OP picker behaviour
    ap.add_argument("--ops_picker", type=_strip_lower, default="conservative", choices=["conservative", "core"])
    ap.add_argument("--op_tie_break", type=_strip_lower, default="max_thr", choices=["max_thr", "min_thr"])
    ap.add_argument("--min_tau_high", type=float, default=0.0, help="Optional lower bound for tau_high during OP selection")
    ap.add_argument("--tie_eps", type=float, default=1e-9, help="Tolerance for plateau tie-breaking")

    ap.add_argument("--recursive", type=int, default=0, help="Recursively scan val_dir/fa_dir for .npz")
    ap.add_argument(
        "--emit_absolute_paths",
        type=int,
        default=0,
        help="If 1, emit absolute paths in ops YAML; if 0, emit paths relative to --out",
    )
    ap.add_argument(
        "--allow_degenerate_sweep",
        type=int,
        default=0,
        help="If 1, allow writing ops even when sweep has no meaningful alert region",
    )

    # Sweep range
    ap.add_argument("--thr_min", type=float, default=0.05)
    ap.add_argument("--thr_max", type=float, default=0.95)
    ap.add_argument("--thr_step", type=float, default=0.01)
    ap.add_argument("--tau_low_ratio", type=float, default=0.8)

    # OP selection targets
    ap.add_argument("--op1_recall", type=float, default=0.95)
    ap.add_argument("--op3_fa24h", type=float, default=2.0)
    ap.add_argument("--op2_objective", type=_strip_lower, default="f1", choices=["f1", "cost_sensitive"])
    ap.add_argument("--cost_fn", type=float, default=5.0)
    ap.add_argument("--cost_fp", type=float, default=1.0)

    # Alert policy knobs
    ap.add_argument("--ema_alpha", type=float, default=0.0)
    ap.add_argument("--k", type=int, default=1)
    ap.add_argument("--n", type=int, default=1)
    ap.add_argument("--cooldown_s", type=float, default=0.0)

    ap.add_argument("--confirm", type=int, default=0)
    ap.add_argument("--confirm_s", type=float, default=1.0)
    ap.add_argument("--confirm_min_lying", type=float, default=0.7)
    ap.add_argument("--confirm_max_motion", type=float, default=0.08)
    ap.add_argument("--confirm_require_low", type=_int_or_default(1), default=1)
    ap.add_argument("--start_guard_max_lying", type=float, default=-1.0)
    ap.add_argument("--start_guard_prefixes", type=str, default="")

    # Event metrics details
    ap.add_argument("--merge_gap_s", type=float, default=None)
    ap.add_argument("--overlap_slack_s", type=float, default=0.0)
    ap.add_argument("--time_mode", type=_strip_lower, default="center", choices=["start", "center", "end"])

    args = ap.parse_args()

    device = torch.device(args.device)

    bundle = load_ckpt(args.ckpt, map_location="cpu")
    model_cfg = bundle.get("model_cfg", {})
    feat_cfg = FeatCfg.from_dict(bundle.get("feat_cfg", {}))
    feat_cfg = _override_feat_cfg(feat_cfg, args)

    # Determine two-stream for GCN
    two_stream = bool(model_cfg.get("two_stream", False))

    model = build_model(args.arch, model_cfg, feat_cfg.to_dict(), fps_default=float(args.fps_default))
    model.load_state_dict(bundle["state_dict"], strict=False)
    model.to(device)

    print(f"[info] arch={args.arch} two_stream={two_stream} feat={feat_cfg}")

    val_ds = WindowDirDataset(
        args.val_dir,
        feat_cfg=feat_cfg,
        arch=args.arch,
        two_stream=two_stream,
        fps_default=float(args.fps_default),
        recursive=bool(int(args.recursive)),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(args.batch),
        shuffle=False,
        num_workers=max(0, int(args.num_workers)),
        pin_memory=torch.cuda.is_available(),
        collate_fn=_collate,
    )
    _preflight_input_contract(
        arch=args.arch,
        model_cfg=model_cfg,
        ds=val_ds,
        two_stream=two_stream,
        source_dir=args.val_dir,
    )

    logits_val, y_val_float, metas_val = infer_logits(model, val_loader, device, arch=args.arch, two_stream=two_stream)
    arr_val = _extract_arrays(metas_val)

    # Fit temperature T on labeled windows only.
    y_val = arr_val["y"].astype(np.int32)
    labeled = (y_val == 0) | (y_val == 1)
    if not labeled.any():
        raise SystemExit("[err] validation set has no labeled windows (y in {0,1}).")

    if args.no_fit_calibration:
        T = 1.0
        cal = {
            "method": "temperature",
            "T": float(T),
            "n_val": int(labeled.sum()),
            "note": "--no_fit_calibration enabled",
        }
        # If user supplied a calibration yaml and it exists, try to load T
        if args.calibration_yaml and os.path.exists(args.calibration_yaml):
            loaded = yaml_load_simple(args.calibration_yaml) or {}
            T_loaded = float(((loaded.get("calibration") or loaded).get("T", 1.0)))
            if np.isfinite(T_loaded) and T_loaded > 0:
                T = float(T_loaded)
                cal["T"] = float(T)
                cal["note"] = "loaded existing calibration yaml"
    else:
        fit = fit_temperature(logits_val[labeled], y_val[labeled])
        T = float(fit.T)
        cal = {
            "method": "temperature",
            "T": float(T),
            "n_val": int(fit.n_val),
            "nll_before": float(fit.nll_before),
            "nll_after": float(fit.nll_after),
            "ece_before": float(fit.ece_before),
            "ece_after": float(fit.ece_after),
        }

    # Apply calibration: p = sigmoid(logits/T)
    probs_val = sigmoid(logits_val / max(1e-6, T)).astype(np.float32)

    # Optional FA stream
    fa_payload = {}
    if args.fa_dir:
        fa_ds = WindowDirDataset(
            args.fa_dir,
            feat_cfg=feat_cfg,
            arch=args.arch,
            two_stream=two_stream,
            fps_default=float(args.fps_default),
            recursive=bool(int(args.recursive)),
        )
        fa_loader = DataLoader(
            fa_ds,
            batch_size=int(args.batch),
            shuffle=False,
            num_workers=max(0, int(args.num_workers)),
            pin_memory=torch.cuda.is_available(),
            collate_fn=_collate,
        )
        logits_fa, _y_fa_float, metas_fa = infer_logits(model, fa_loader, device, arch=args.arch, two_stream=two_stream)
        probs_fa = sigmoid(logits_fa / max(1e-6, T)).astype(np.float32)
        arr_fa = _extract_arrays(metas_fa)
        fa_payload = {
            "fa_probs": probs_fa,
            "fa_video_ids": arr_fa["video_id"],
            "fa_w_start": arr_fa["w_start"],
            "fa_w_end": arr_fa["w_end"],
            "fa_fps": arr_fa["fps"],
            "fa_lying_score": arr_fa["lying"],
            "fa_motion_score": arr_fa["motion"],
        }

    alert_base = AlertCfg(
        ema_alpha=float(args.ema_alpha),
        k=int(args.k),
        n=int(args.n),
        tau_high=0.5,
        tau_low=0.4,
        cooldown_s=float(args.cooldown_s),
        confirm=bool(int(args.confirm)),
        confirm_s=float(args.confirm_s),
        confirm_min_lying=float(args.confirm_min_lying),
        confirm_max_motion=float(args.confirm_max_motion),
        confirm_require_low=bool(int(args.confirm_require_low)),
        start_guard_max_lying=(None if float(args.start_guard_max_lying) < 0.0 else float(args.start_guard_max_lying)),
        start_guard_prefixes=([x.strip() for x in str(args.start_guard_prefixes).split(",") if x.strip()] or None),
    )

    sweep, meta = sweep_alert_policy_from_windows(
        probs_val,
        y_val,
        arr_val["video_id"],
        arr_val["w_start"],
        arr_val["w_end"],
        arr_val["fps"],
        alert_base=alert_base,
        thr_min=float(args.thr_min),
        thr_max=float(args.thr_max),
        thr_step=float(args.thr_step),
        tau_low_ratio=float(args.tau_low_ratio),
        merge_gap_s=args.merge_gap_s,
        overlap_slack_s=float(args.overlap_slack_s),
        time_mode=str(args.time_mode),
        fps_default=float(args.fps_default),
        lying_score=arr_val["lying"],
        motion_score=arr_val["motion"],
        **fa_payload,
    )

    sweep_ok, failure_reason = _sweep_has_non_degenerate_alert_region(sweep)
    # If confirm-stage causes a fully degenerate sweep, retry once with confirm disabled.
    # This keeps downstream fit/eval targets usable on heavily occluded streams while
    # preserving the same model/checkpoint and core ML logic.
    used_confirm_fallback = False
    if (not sweep_ok) and bool(alert_base.confirm):
        print(f"[warn] {failure_reason}")
        print("[warn] retrying sweep with confirm disabled (fallback for NaN/occlusion-heavy windows)")
        alert_base = AlertCfg(
            ema_alpha=float(alert_base.ema_alpha),
            k=int(alert_base.k),
            n=int(alert_base.n),
            tau_high=float(alert_base.tau_high),
            tau_low=float(alert_base.tau_low),
            cooldown_s=float(alert_base.cooldown_s),
            confirm=False,
            confirm_s=float(alert_base.confirm_s),
            confirm_min_lying=float(alert_base.confirm_min_lying),
            confirm_max_motion=float(alert_base.confirm_max_motion),
            confirm_require_low=bool(alert_base.confirm_require_low),
            start_guard_max_lying=alert_base.start_guard_max_lying,
            start_guard_prefixes=alert_base.start_guard_prefixes,
        )
        sweep, meta = sweep_alert_policy_from_windows(
            probs_val,
            y_val,
            arr_val["video_id"],
            arr_val["w_start"],
            arr_val["w_end"],
            arr_val["fps"],
            alert_base=alert_base,
            thr_min=float(args.thr_min),
            thr_max=float(args.thr_max),
            thr_step=float(args.thr_step),
            tau_low_ratio=float(args.tau_low_ratio),
            merge_gap_s=args.merge_gap_s,
            overlap_slack_s=float(args.overlap_slack_s),
            time_mode=str(args.time_mode),
            fps_default=float(args.fps_default),
            lying_score=arr_val["lying"],
            motion_score=arr_val["motion"],
            **fa_payload,
        )
        sweep_ok, failure_reason = _sweep_has_non_degenerate_alert_region(sweep)
        used_confirm_fallback = True

    ops_meta: Dict[str, Any] = {}
    if str(args.ops_picker) == "core":
        # Keep legacy behaviour.
        ops = pick_ops_from_sweep(sweep, op1_recall=float(args.op1_recall), op3_fa24h=float(args.op3_fa24h))
        ops_meta = {"picker": "core"}
    else:
        ops, ops_meta = pick_ops_from_sweep_conservative(
            sweep,
            op1_recall=float(args.op1_recall),
            op3_fa24h=float(args.op3_fa24h),
            op2_objective=str(args.op2_objective),
            cost_fn=float(args.cost_fn),
            cost_fp=float(args.cost_fp),
            tie_break=str(args.op_tie_break),
            min_tau_high=float(args.min_tau_high),
            eps=float(args.tie_eps),
        )
    if used_confirm_fallback:
        ops_meta["confirm_fallback"] = "disabled_confirm_after_degenerate_sweep"

    if (not sweep_ok):
        ops_meta["failure_reason"] = str(failure_reason)
        print(f"[warn] {failure_reason}")

    sweep_json_path = ""
    sweep_json_ref = ""
    out_abs = os.path.abspath(args.out)
    out_dir_abs = os.path.dirname(out_abs)
    out_stem = os.path.splitext(os.path.basename(out_abs))[0]
    sweep_json_abs = os.path.join(out_dir_abs, out_stem + ".sweep.json")

    if int(args.save_sweep_json):
        sweep_json_path = sweep_json_abs
        sweep_json_ref = _path_ref_for_output(
            sweep_json_abs,
            args.out,
            emit_absolute_paths=bool(int(args.emit_absolute_paths)),
        )
        payload = _sanitize_json(
            {
                "sweep": sweep,
                "sweep_meta": meta,
                "sweep_cfg": {
                    "thr_min": float(args.thr_min),
                    "thr_max": float(args.thr_max),
                    "thr_step": float(args.thr_step),
                    "tau_low_ratio": float(args.tau_low_ratio),
                },
                "ops": ops,
                "ops_meta": ops_meta,
            }
        )
        with open(sweep_json_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"[ok] wrote sweep json: {sweep_json_path}")

    if (not sweep_ok) and (not bool(int(args.allow_degenerate_sweep))):
        raise SystemExit(f"[err] {failure_reason}")

    sweep_cfg = {
        "thr_min": float(args.thr_min),
        "thr_max": float(args.thr_max),
        "thr_step": float(args.thr_step),
        "tau_low_ratio": float(args.tau_low_ratio),
    }

    out = {
        "ops": ops,
        "ops_meta": ops_meta,
        "alert_cfg": alert_base.to_dict(),
        "alert_base": alert_base.to_dict(),  # legacy alias
        "sweep_meta": meta,
        "sweep_cfg": sweep_cfg,
        "sweep_json": sweep_json_ref if sweep_json_ref else "",
        "calibration": cal,
        "model": {
            "arch": str(args.arch),
            "ckpt": _path_ref_for_output(
                args.ckpt,
                args.out,
                emit_absolute_paths=bool(int(args.emit_absolute_paths)),
            ),
            "model_cfg": model_cfg,
            "feat_cfg": feat_cfg.to_dict(),
        },
    }

    os.makedirs(out_dir_abs, exist_ok=True)
    yaml_dump_simple(_sanitize_json(out), args.out)
    print(f"[ok] wrote ops: {args.out}")

    if args.calibration_yaml:
        os.makedirs(os.path.dirname(os.path.abspath(args.calibration_yaml)), exist_ok=True)
        yaml_dump_simple(_sanitize_json({"calibration": cal}), args.calibration_yaml)
        print(f"[ok] wrote calibration: {args.calibration_yaml}")


if __name__ == "__main__":
    main()
