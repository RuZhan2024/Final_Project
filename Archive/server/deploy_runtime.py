#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""server/deploy_runtime.py

Runtime model discovery + cached inference for the FastAPI backend.

Design goals
------------
- Use *real* trained checkpoints (outputs/*/best.pt), never heuristics.
- Fix TCN input-dim mismatch by deriving feature config from checkpoint/reports.
- Provide per-dataset, per-arch thresholds via outputs/reports/*.json (ops_eval).
- Support MC Dropout uncertainty (mu, sigma) using core/uncertainty.py.

This module is import-safe: if torch isn't available, callers can surface a
clear error at request time.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


@dataclass
class DeploySpec:
    key: str              # e.g. "muvim_tcn"
    dataset: str           # le2i|urfd|caucafall|muvim
    arch: str              # tcn|gcn
    ckpt: str              # path to best.pt
    feat_cfg: Dict[str, Any]
    model_cfg: Dict[str, Any]
    data_cfg: Dict[str, Any]
    ops: Dict[str, Dict[str, float]]  # OP-1/2/3 -> {tau_low, tau_high}


def _safe_float(x: Any, default: float) -> float:
    try:
        v = float(x)
        if v != v:  # NaN
            return default
        return v
    except Exception:
        return default


def _repo_root() -> Path:
    # server/ is inside the project root.
    return Path(__file__).resolve().parents[1]


def _load_json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _extract_ops(report: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    ops_eval = report.get("ops_eval") or {}
    out: Dict[str, Dict[str, float]] = {}
    for k, op_code in [("op1", "OP-1"), ("op2", "OP-2"), ("op3", "OP-3")]:
        o = ops_eval.get(k) or {}
        alert = o.get("alert_cfg") or {}
        tau_low = _safe_float(alert.get("tau_low"), 0.5)
        tau_high = _safe_float(alert.get("tau_high"), 0.85)
        out[op_code] = {"tau_low": tau_low, "tau_high": tau_high}
    return out


def discover_specs() -> Dict[str, DeploySpec]:
    """Discover deployable specs from outputs/reports/*.json.

    Preference rules:
    - If <dataset>_<arch>.json exists, it defines thresholds for that spec.
    - If missing (e.g. le2i_tcn.json), fall back to le2i_on_urfd_tcn.json.
    """
    root = _repo_root()
    reports_dir = root / "outputs" / "reports"
    specs: Dict[str, DeploySpec] = {}
    if not reports_dir.exists():
        return specs

    # Collect candidates, then resolve collisions by priority.
    candidates: Dict[str, Tuple[int, Path]] = {}
    for p in sorted(reports_dir.glob("*.json")):
        name = p.stem.lower()
        r = _load_json(p)
        arch = (r.get("arch") or "").lower().strip()
        ckpt = r.get("ckpt") or r.get("ckpt_path") or ""
        if arch not in {"tcn", "gcn"}:
            continue
        if not isinstance(ckpt, str) or ckpt.strip() == "":
            continue

        # Normalise dataset/spec key.
        # Supported filenames:
        #   muvim_tcn.json, urfd_gcn.json, le2i_on_urfd_tcn.json
        if "_on_" in name:
            # e.g. le2i_on_urfd_tcn -> le2i_tcn
            parts = name.split("_on_")
            left = parts[0]
            # left contains dataset name
            dataset = left
            spec_key = f"{dataset}_{arch}"
            priority = 10  # fallback
        else:
            # expected: <dataset>_<arch>
            if not name.endswith(f"_{arch}"):
                continue
            dataset = name[: -(len(arch) + 1)]
            spec_key = f"{dataset}_{arch}"
            priority = 0  # primary

        # Keep best (lowest) priority
        prev = candidates.get(spec_key)
        if prev is None or priority < prev[0]:
            candidates[spec_key] = (priority, p)

    # Build specs.
    for spec_key, (_prio, path) in candidates.items():
        rep = _load_json(path)
        arch = (rep.get("arch") or "").lower().strip()
        ckpt = str(rep.get("ckpt") or rep.get("ckpt_path") or "").strip()
        dataset = spec_key.split("_")[0]
        ckpt_path = (root / ckpt).resolve() if not os.path.isabs(ckpt) else Path(ckpt)
        if not ckpt_path.exists():
            # Skip broken entries
            continue

        # Prefer feat_cfg/model_cfg from checkpoint; reports are a fallback.
        feat_cfg: Dict[str, Any] = rep.get("feat_cfg") or {}
        model_cfg: Dict[str, Any] = rep.get("model_cfg") or {}
        data_cfg: Dict[str, Any] = rep.get("data_cfg") or {}

        ops = _extract_ops(rep)

        specs[spec_key] = DeploySpec(
            key=spec_key,
            dataset=dataset,
            arch=arch,
            ckpt=str(ckpt_path),
            feat_cfg=feat_cfg,
            model_cfg=model_cfg,
            data_cfg=data_cfg,
            ops=ops,
        )

    return specs


_SPECS: Optional[Dict[str, DeploySpec]] = None


def get_specs() -> Dict[str, DeploySpec]:
    global _SPECS
    if _SPECS is None:
        _SPECS = discover_specs()
    return _SPECS


def _torch() -> Any:
    try:
        import torch  # type: ignore
        return torch
    except Exception as e:
        raise RuntimeError("PyTorch is required for real model inference.") from e


def _pick_device(torch: Any) -> Any:
    # Keep this conservative: CPU is always safe.
    # Users can switch to GPU/MPS in their own deployment if desired.
    return torch.device("cpu")


_MODEL_CACHE: Dict[str, Dict[str, Any]] = {}


def _load_model_and_cfg(spec: DeploySpec) -> Dict[str, Any]:
    if spec.key in _MODEL_CACHE:
        return _MODEL_CACHE[spec.key]

    torch = _torch()
    device = _pick_device(torch)

    from core.ckpt import load_ckpt
    from core.models import build_model
    from core.features import FeatConfig

    bundle = load_ckpt(spec.ckpt, map_location=str(device))
    # Merge report cfg as fallback, but checkpoint is the source of truth.
    feat_cfg_dict = bundle.get("feat_cfg") or spec.feat_cfg or {}
    model_cfg_dict = bundle.get("model_cfg") or spec.model_cfg or {}
    data_cfg_dict = bundle.get("data_cfg") or spec.data_cfg or {}
    feat_cfg = FeatConfig.from_dict(feat_cfg_dict)

    model, _in_dims = build_model(spec.arch, model_cfg=model_cfg_dict, feat_cfg=feat_cfg_dict)
    model.load_state_dict(bundle["state_dict"], strict=True)
    model.to(device)
    model.eval()

    cached = {
        "model": model,
        "device": device,
        "feat_cfg": feat_cfg,
        "feat_cfg_dict": feat_cfg_dict,
        "model_cfg": model_cfg_dict,
        "data_cfg": data_cfg_dict,
    }
    _MODEL_CACHE[spec.key] = cached
    return cached


def _match_in_ch_tcn(torch: Any, model: Any, x: Any) -> Any:
    # x: [B,T,C]
    try:
        expected = int(getattr(model, "conv_in")[0].in_channels)  # type: ignore[index]
    except Exception:
        return x
    c = int(x.shape[-1])
    if c == expected:
        return x
    if c > expected:
        return x[..., :expected]
    # pad zeros
    pad = torch.zeros((x.shape[0], x.shape[1], expected - c), device=x.device, dtype=x.dtype)
    return torch.cat([x, pad], dim=-1)


def predict_spec(
    *,
    spec_key: str,
    joints_xy: Any,
    conf: Any,
    fps: float,
    target_T: int,
    op_code: str = "OP-2",
    use_mc: bool = True,
    mc_M: int = 20,
) -> Dict[str, Any]:
    """Run inference for one spec_key on a resampled window.

    joints_xy/conf are assumed already resampled to [T,J,2] and [T,J].
    """
    specs = get_specs()
    if spec_key not in specs:
        raise KeyError(f"Unknown spec_key: {spec_key}")
    spec = specs[spec_key]

    torch = _torch()
    cached = _load_model_and_cfg(spec)
    model = cached["model"]
    device = cached["device"]
    feat_cfg = cached["feat_cfg"]

    import numpy as np
    from core.features import build_tcn_input, build_gcn_inputs
    from core.uncertainty import mc_predict_mu_sigma

    xy = np.asarray(joints_xy, dtype=np.float32)
    cf = np.asarray(conf, dtype=np.float32) if conf is not None else np.ones((xy.shape[0], xy.shape[1]), dtype=np.float32)

    # Align length
    if xy.shape[0] != int(target_T):
        # Best effort: truncate/pad by repeating edge frames.
        T = int(target_T)
        if xy.shape[0] > T:
            xy = xy[-T:]
            cf = cf[-T:]
        else:
            pad_n = T - xy.shape[0]
            if xy.shape[0] >= 1:
                xy_pad = np.repeat(xy[:1], pad_n, axis=0)
                cf_pad = np.repeat(cf[:1], pad_n, axis=0)
                xy = np.concatenate([xy_pad, xy], axis=0)
                cf = np.concatenate([cf_pad, cf], axis=0)

    tau = (spec.ops.get(op_code) or spec.ops.get("OP-2") or {"tau_low": 0.5, "tau_high": 0.85})
    tau_low = float(tau["tau_low"])
    tau_high = float(tau["tau_high"])

    def triage_from_p(p: float) -> str:
        if p >= tau_high:
            return "fall"
        if p >= tau_low:
            return "uncertain"
        return "not_fall"

    if spec.arch == "tcn":
        x = build_tcn_input(xy, None, cf, None, float(fps), feat_cfg)
        xt = torch.from_numpy(x).to(device)
        xt = _match_in_ch_tcn(torch, model, xt)

        def fwd():
            return model(xt)

        with torch.no_grad():
            logit = fwd()
            p_det = float(torch.sigmoid(logit).item())

        if use_mc:
            mu, sig = mc_predict_mu_sigma(model, fwd, M=int(mc_M))
            mu_f, sig_f = float(mu), float(sig)
        else:
            mu_f, sig_f = p_det, 0.0

        p_for_triage = mu_f

    else:
        xj_np, xm_np = build_gcn_inputs(xy, None, cf, None, float(fps), feat_cfg)
        xj = torch.from_numpy(xj_np).to(device)
        xm = torch.from_numpy(xm_np).to(device)

        def fwd():
            return model(xj, xm)

        with torch.no_grad():
            logit = fwd()
            p_det = float(torch.sigmoid(logit).item())

        if use_mc:
            mu, sig = mc_predict_mu_sigma(model, fwd, M=int(mc_M))
            mu_f, sig_f = float(mu), float(sig)
        else:
            mu_f, sig_f = p_det, 0.0

        p_for_triage = mu_f

    return {
        "spec_key": spec.key,
        "dataset": spec.dataset,
        "arch": spec.arch,
        "fps": float(fps),
        "target_T": int(target_T),
        "p_det": float(p_det),
        "mu": float(mu_f),
        "sigma": float(sig_f),
        "triage": {
            "tau_low": float(tau_low),
            "tau_high": float(tau_high),
            "state": triage_from_p(float(p_for_triage)),
            "op_code": str(op_code),
        },
    }


def fuse_hybrid(
    *,
    tri_tcn: str,
    tri_gcn: str,
) -> str:
    """Hybrid fusion rule requested by the user.

    - NOT_FALL only if both are NOT_FALL
    - FALL if one is FALL and the other is FALL or UNCERTAIN
    - otherwise UNCERTAIN (covers FALL+NOT_FALL disagreement)
    """
    a = (tri_tcn or "").lower()
    b = (tri_gcn or "").lower()
    if a == "not_fall" and b == "not_fall":
        return "not_fall"
    if (a == "fall" and b in {"fall", "uncertain"}) or (b == "fall" and a in {"fall", "uncertain"}):
        return "fall"
    return "uncertain"
