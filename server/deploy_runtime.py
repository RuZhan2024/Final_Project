#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""server/deploy_runtime.py

Runtime model discovery + cached inference for the FastAPI backend.

This module is responsible for **loading real trained checkpoints** and running
inference for the live monitor endpoint.

Source of truth
---------------
For deployment behaviour, we treat **configs/ops/*.yaml** as the source of
truth (not DB rows and not heuristics). These YAML files contain:
- feat_cfg (how windows were built during training/eval)
- alert_cfg (EMA smoothing, k-of-n persistence, cooldown, etc.)
- ops (OP-1/OP-2/OP-3 thresholds)

The FastAPI app applies the alerting policy online; this module focuses on
inference + exposing the params.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import yaml


@dataclass
class DeploySpec:
    key: str                     # e.g. "muvim_tcn"
    dataset: str                  # le2i|urfd|caucafall|muvim
    arch: str                     # tcn|gcn
    ckpt: str                     # absolute path to best.pt
    feat_cfg: Dict[str, Any]      # from configs/ops/*.yaml
    model_cfg: Dict[str, Any]     # from reports/ckpt (fallback)
    data_cfg: Dict[str, Any]      # from reports/ckpt (fallback)
    alert_cfg: Dict[str, Any]     # from configs/ops/*.yaml
    ops: Dict[str, Dict[str, Any]]  # OP-1/2/3 -> op dict (includes tau_low/high)
    ops_path: str = ""            # where this spec came from


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


def _load_yaml(path: Path) -> Dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _norm_op_code(code: str) -> str:
    c = (code or "").strip().upper().replace("_", "-")
    if c in {"OP1", "OP-1", "1"}:
        return "OP-1"
    if c in {"OP2", "OP-2", "2"}:
        return "OP-2"
    if c in {"OP3", "OP-3", "3"}:
        return "OP-3"
    # fallback
    return "OP-2"


def _standardise_ops(ops: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Convert yaml ops keys (op1/op2/op3) into OP-1/2/3."""
    out: Dict[str, Dict[str, Any]] = {}
    if not isinstance(ops, dict):
        return out

    for k, v in ops.items():
        if not isinstance(v, dict):
            continue
        kk = str(k).lower().strip()
        if kk in {"op1", "op-1"}:
            code = "OP-1"
        elif kk in {"op2", "op-2"}:
            code = "OP-2"
        elif kk in {"op3", "op-3"}:
            code = "OP-3"
        else:
            continue

        vv = dict(v)
        # Ensure tau are floats and exist.
        vv["tau_low"] = _safe_float(vv.get("tau_low"), _safe_float(vv.get("tau_low_conf"), 0.5))
        vv["tau_high"] = _safe_float(vv.get("tau_high"), _safe_float(vv.get("tau_high_conf"), 0.85))
        out[code] = vv

    return out


def _discover_from_ops_yaml(root: Path) -> Dict[str, DeploySpec]:
    specs: Dict[str, DeploySpec] = {}
    ops_dir = root / "configs" / "ops"
    if not ops_dir.exists():
        return specs

    for p in sorted(list(ops_dir.glob("*.yaml")) + list(ops_dir.glob("*.yml"))):
        data = _load_yaml(p)
        if not data:
            continue

        # Filename pattern: <arch>_<dataset>.yaml (e.g. gcn_muvim.yaml)
        stem = p.stem.lower().strip()
        parts = stem.split("_")
        if len(parts) < 2:
            continue
        arch_guess, dataset_guess = parts[0], "_".join(parts[1:])

        arch = str(data.get("arch") or arch_guess).lower().strip()
        dataset = dataset_guess.lower().strip()
        if arch not in {"tcn", "gcn"}:
            continue
        if dataset not in {"le2i", "urfd", "caucafall", "muvim"}:
            # allow other datasets, but keep the key stable
            dataset = dataset_guess.lower().strip()

        ckpt_rel = str(data.get("ckpt") or "").strip()
        if not ckpt_rel:
            continue
        ckpt_path = (root / ckpt_rel).resolve() if not os.path.isabs(ckpt_rel) else Path(ckpt_rel)
        if not ckpt_path.exists():
            # Skip broken configs
            continue

        spec_key = f"{dataset}_{arch}"

        feat_cfg = data.get("feat_cfg") or {}
        alert_cfg = data.get("alert_cfg") or {}
        ops = _standardise_ops(data.get("ops") or {})

        specs[spec_key] = DeploySpec(
            key=spec_key,
            dataset=dataset,
            arch=arch,
            ckpt=str(ckpt_path),
            feat_cfg=feat_cfg if isinstance(feat_cfg, dict) else {},
            model_cfg={},
            data_cfg={},
            alert_cfg=alert_cfg if isinstance(alert_cfg, dict) else {},
            ops=ops,
            ops_path=str(p),
        )

    return specs


def _extract_ops_from_report(report: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Fallback when configs/ops/*.yaml is missing."""
    ops_eval = report.get("ops_eval") or {}
    out: Dict[str, Dict[str, Any]] = {}
    for k, op_code in [("op1", "OP-1"), ("op2", "OP-2"), ("op3", "OP-3")]:
        o = ops_eval.get(k) or {}
        alert = o.get("alert_cfg") or {}
        tau_low = _safe_float(alert.get("tau_low"), 0.5)
        tau_high = _safe_float(alert.get("tau_high"), 0.85)
        out[op_code] = {"tau_low": tau_low, "tau_high": tau_high}
    return out


def _discover_from_reports(root: Path) -> Dict[str, DeploySpec]:
    """Legacy discovery from outputs/reports/*.json."""
    reports_dir = root / "outputs" / "reports"
    specs: Dict[str, DeploySpec] = {}
    if not reports_dir.exists():
        return specs

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

        if "_on_" in name:
            dataset = name.split("_on_")[0]
            spec_key = f"{dataset}_{arch}"
            priority = 10
        else:
            if not name.endswith(f"_{arch}"):
                continue
            dataset = name[: -(len(arch) + 1)]
            spec_key = f"{dataset}_{arch}"
            priority = 0

        prev = candidates.get(spec_key)
        if prev is None or priority < prev[0]:
            candidates[spec_key] = (priority, p)

    for spec_key, (_prio, path) in candidates.items():
        rep = _load_json(path)
        arch = (rep.get("arch") or "").lower().strip()
        ckpt_rel = str(rep.get("ckpt") or rep.get("ckpt_path") or "").strip()
        dataset = spec_key.split("_")[0]
        ckpt_path = (root / ckpt_rel).resolve() if not os.path.isabs(ckpt_rel) else Path(ckpt_rel)
        if not ckpt_path.exists():
            continue

        feat_cfg: Dict[str, Any] = rep.get("feat_cfg") or {}
        model_cfg: Dict[str, Any] = rep.get("model_cfg") or {}
        data_cfg: Dict[str, Any] = rep.get("data_cfg") or {}
        ops = _extract_ops_from_report(rep)

        specs[spec_key] = DeploySpec(
            key=spec_key,
            dataset=dataset,
            arch=arch,
            ckpt=str(ckpt_path),
            feat_cfg=feat_cfg,
            model_cfg=model_cfg,
            data_cfg=data_cfg,
            alert_cfg={},
            ops=ops,
            ops_path=str(path),
        )

    return specs


def discover_specs() -> Dict[str, DeploySpec]:
    """Discover deployable specs.

    Preference:
    1) configs/ops/*.yaml (real deploy params)
    2) outputs/reports/*.json (fallback)
    """
    root = _repo_root()
    specs = _discover_from_ops_yaml(root)
    if specs:
        return specs
    return _discover_from_reports(root)


_SPECS: Optional[Dict[str, DeploySpec]] = None


def get_specs() -> Dict[str, DeploySpec]:
    global _SPECS
    if _SPECS is None:
        _SPECS = discover_specs()
    return _SPECS


def get_alert_cfg(spec_key: str, op_code: str = "OP-2") -> Dict[str, Any]:
    """Return the effective alert_cfg dict for (spec, op_code)."""
    specs = get_specs()
    spec = specs.get(spec_key)
    if not spec:
        return {}
    op = _norm_op_code(op_code)
    base = dict(spec.alert_cfg or {})
    # Override tau from ops if present
    op_entry = (spec.ops or {}).get(op) or {}
    if isinstance(op_entry, dict):
        if "tau_low" in op_entry:
            base["tau_low"] = _safe_float(op_entry.get("tau_low"), _safe_float(base.get("tau_low"), 0.5))
        if "tau_high" in op_entry:
            base["tau_high"] = _safe_float(op_entry.get("tau_high"), _safe_float(base.get("tau_high"), 0.85))
    return base


def get_op_taus(spec_key: str, op_code: str = "OP-2") -> Tuple[float, float]:
    cfg = get_alert_cfg(spec_key, op_code)
    tau_low = _safe_float(cfg.get("tau_low"), 0.5)
    tau_high = _safe_float(cfg.get("tau_high"), 0.85)
    return tau_low, tau_high


def _torch() -> Any:
    try:
        import torch  # type: ignore
        return torch
    except Exception as e:
        raise RuntimeError("PyTorch is required for real model inference.") from e


def _pick_device(torch: Any) -> Any:
    # Prefer acceleration when available, but keep a safe CPU fallback.
    try:
        if hasattr(torch, "cuda") and torch.cuda.is_available():
            return torch.device("cuda")
    except Exception:
        pass
    try:
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
    except Exception:
        pass
    return torch.device("cpu")


_MODEL_CACHE: Dict[str, Dict[str, Any]] = {}
_MC_PREDICT_FN: Any = None
_MC_PREDICT_MOD: Any = None


def _mc_predict_fn() -> Any:
    global _MC_PREDICT_FN, _MC_PREDICT_MOD
    if _MC_PREDICT_FN is not None and _MC_PREDICT_MOD is not None:
        try:
            import sys as _sys
            mod_live = _sys.modules.get("core.uncertainty")
            if mod_live is _MC_PREDICT_MOD and getattr(mod_live, "mc_predict_mu_sigma", None) is _MC_PREDICT_FN:
                return _MC_PREDICT_FN
        except Exception:
            pass
    try:
        import sys as _sys
        _unc = _sys.modules.get("core.uncertainty")
        if _unc is None:
            import core.uncertainty as _unc  # type: ignore
        _MC_PREDICT_MOD = _unc
        _MC_PREDICT_FN = getattr(_unc, "mc_predict_mu_sigma")
        return _MC_PREDICT_FN
    except Exception:
        _MC_PREDICT_MOD = None
        _MC_PREDICT_FN = None
        raise


def _sample_fp(a: Optional[np.ndarray]) -> tuple[Optional[float], Optional[float], Optional[float]]:
    if a is None:
        return None, None, None
    n = int(a.size)
    if n < 1:
        return 0.0, 0.0, 0.0
    flat = a.ravel()
    v0 = float(flat[0])
    v1 = float(flat[-1])
    vm = float(flat[n // 2])
    return v0, v1, vm


def _load_model_and_cfg(spec: DeploySpec) -> Dict[str, Any]:
    """Load model + cfg, cached by spec.key."""
    if spec.key in _MODEL_CACHE:
        return _MODEL_CACHE[spec.key]

    torch = _torch()
    device = _pick_device(torch)

    try:
        from core.ckpt import load_ckpt
        from core.models import build_model, fuse_conv_bn_eval_inplace
        from core.features import FeatCfg
    except Exception as e:
        raise RuntimeError(
            "Missing ML runtime package 'core'. "
            "Make sure you run the server from the repo root (so 'core/' is on PYTHONPATH), "
            "or install the ML package into this environment."
        ) from e

    bundle = load_ckpt(spec.ckpt, map_location=str(device))

    # Checkpoint is the base; YAML feat_cfg overrides to ensure we use the real deploy params.
    ckpt_feat_cfg = bundle.get("feat_cfg") or {}
    feat_cfg_dict = {**(ckpt_feat_cfg if isinstance(ckpt_feat_cfg, dict) else {}), **(spec.feat_cfg or {})}

    model_cfg_dict = bundle.get("model_cfg") or spec.model_cfg or {}
    data_cfg_dict = bundle.get("data_cfg") or spec.data_cfg or {}

    feat_cfg = FeatCfg.from_dict(feat_cfg_dict)

    model = build_model(spec.arch, model_cfg=model_cfg_dict, feat_cfg=feat_cfg_dict)
    model.load_state_dict(bundle["state_dict"], strict=True)
    model.to(device)
    model.eval()
    # Optional CPU-only dynamic quantization for on-device latency/memory.
    # Disabled by default; enable with FD_DYNAMIC_QUANT_LINEAR=1.
    if str(os.getenv("FD_DYNAMIC_QUANT_LINEAR", "0")).strip().lower() in {"1", "true", "yes", "on"}:
        try:
            if getattr(device, "type", "") == "cpu":
                model = torch.ao.quantization.quantize_dynamic(  # type: ignore[attr-defined]
                    model,
                    {torch.nn.Linear},
                    dtype=torch.qint8,
                )
                model.eval()
        except Exception:
            pass
    # Runtime-only optimization: fuse Conv1d+BN pairs for lower inference latency.
    try:
        fuse_conv_bn_eval_inplace(model)
    except Exception:
        pass
    # Optional graph compile for steady-state on-device latency.
    # Disabled by default; enable with FD_TORCH_COMPILE=1.
    if str(os.getenv("FD_TORCH_COMPILE", "0")).strip().lower() in {"1", "true", "yes", "on"}:
        try:
            model = torch.compile(model, mode="reduce-overhead")  # type: ignore[attr-defined]
            model.eval()
        except Exception:
            pass

    cached = {
        "model": model,
        "device": device,
        "feat_cfg": feat_cfg,
        "feat_cfg_dict": feat_cfg_dict,
        "feat_cfg_sig": (
            str(getattr(feat_cfg, "center", "pelvis")),
            bool(getattr(feat_cfg, "use_motion", True)),
            bool(getattr(feat_cfg, "use_bone", False)),
            bool(getattr(feat_cfg, "use_bone_length", False)),
            bool(getattr(feat_cfg, "use_conf_channel", True)),
            bool(getattr(feat_cfg, "motion_scale_by_fps", True)),
            float(getattr(feat_cfg, "conf_gate", 0.2)),
            bool(getattr(feat_cfg, "use_precomputed_mask", True)),
        ),
        "model_cfg": model_cfg_dict,
        "data_cfg": data_cfg_dict,
        "is_two_stream": ("twostream" in model.__class__.__name__.lower()) or bool(getattr(model, "two_stream", False)),
    }
    _MODEL_CACHE[spec.key] = cached
    return cached


def _match_in_ch_tcn(torch: Any, model: Any, x: Any) -> Any:
    """Pad/trim to match conv_in channels (TCN)."""
    try:
        expected = int(getattr(model, "conv_in")[0].in_channels)  # type: ignore[index]
    except Exception:
        return x
    c = int(x.shape[-1])
    if c == expected:
        return x
    if c > expected:
        return x[..., :expected]
    pad = torch.zeros((x.shape[0], x.shape[1], expected - c), device=x.device, dtype=x.dtype)
    return torch.cat([x, pad], dim=-1)


def _make_forward_fn(
    *,
    torch: Any,
    model: Any,
    device: Any,
    arch: str,
    Xg: Any,
    feat_cfg: Any,
    model_cfg: Optional[Dict[str, Any]] = None,
    is_two_stream: bool = False,
):
    """Build a single forward callable that returns probability tensor [B]."""
    from core.features import build_tcn_input, split_gcn_two_stream
    use_non_blocking = hasattr(device, "type") and getattr(device, "type", "") in {"cuda", "mps"}
    cfg = model_cfg if isinstance(model_cfg, dict) else {}

    def _as_batch1_f32(x_np: Any) -> Any:
        if isinstance(x_np, np.ndarray) and x_np.dtype == np.float32 and x_np.flags.c_contiguous:
            t = torch.from_numpy(x_np)
        else:
            t = torch.as_tensor(x_np, dtype=torch.float32)
        return t.unsqueeze(0).to(device=device, non_blocking=use_non_blocking)

    if str(arch).lower() == "tcn":
        Xt = build_tcn_input(Xg, feat_cfg)
        exp_in_ch = int(cfg.get("in_ch", 0) or 0)
        if exp_in_ch > 0 and int(Xt.shape[1]) != exp_in_ch:
            raise RuntimeError(
                f"TCN input dimension mismatch: built C={int(Xt.shape[1])}, checkpoint in_ch={exp_in_ch}. "
                "Likely feat_cfg/model_cfg drift between training and deploy."
            )
        x_t = _as_batch1_f32(Xt)  # [1,T,C]
        x_t = _match_in_ch_tcn(torch, model, x_t)

        def forward_fn():
            logits = model(x_t)
            return torch.sigmoid(logits).view(-1)

        return forward_fn

    if is_two_stream:
        xj_np, xm_np = split_gcn_two_stream(Xg, feat_cfg)
        exp_j = int(cfg.get("in_feats_j", 0) or 0)
        exp_m = int(cfg.get("in_feats_m", 0) or 0)
        if exp_j > 0 and int(xj_np.shape[2]) != exp_j:
            raise RuntimeError(
                f"Two-stream GCN joint feature mismatch: built F_j={int(xj_np.shape[2])}, checkpoint in_feats_j={exp_j}. "
                "Likely feat_cfg/model_cfg drift between training and deploy."
            )
        if exp_m > 0 and int(xm_np.shape[2]) != exp_m:
            raise RuntimeError(
                f"Two-stream GCN motion feature mismatch: built F_m={int(xm_np.shape[2])}, checkpoint in_feats_m={exp_m}. "
                "Likely feat_cfg/model_cfg drift between training and deploy."
            )
        xj_t = _as_batch1_f32(xj_np)
        xm_t = _as_batch1_f32(xm_np)

        def forward_fn():
            logits = model(xj_t, xm_t)
            return torch.sigmoid(logits).view(-1)

        return forward_fn

    arr_g = np.asarray(Xg)
    exp_f = int(cfg.get("in_feats", 0) or 0)
    if exp_f > 0 and int(arr_g.shape[2]) != exp_f:
        raise RuntimeError(
            f"GCN feature mismatch: built F={int(arr_g.shape[2])}, checkpoint in_feats={exp_f}. "
            "Likely feat_cfg/model_cfg drift between training and deploy."
        )
    xb = _as_batch1_f32(arr_g)  # [1,T,V,F]

    def forward_fn():
        logits = model(xb)
        return torch.sigmoid(logits).view(-1)

    return forward_fn


def predict_spec(
    *,
    spec_key: str,
    joints_xy: Any,
    conf: Any,
    fps: float,
    target_T: int,
    op_code: str = "OP-2",
    use_mc: bool = True,
    mc_M: int = 16,
    mc_sigma_tol: Optional[float] = None,
    mc_se_tol: Optional[float] = None,
    feature_cache: Optional[Dict[Any, Any]] = None,
    assume_sanitized_inputs: bool = False,
) -> Dict[str, Any]:
    """Run inference for one deploy spec and return p/mu/sigma.

    Alerting state (EMA/k-of-n/cooldown) is handled by server/app.py.
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

    # NOTE: `joints_xy` is expected to already be a fixed-length window (server/app.py
    # resamples/pads to target_T). Our feature builders operate on that window.
    from core.features import build_canonical_input

    if isinstance(joints_xy, np.ndarray) and joints_xy.dtype == np.float32:
        j = joints_xy
    else:
        j = np.asarray(joints_xy, dtype=np.float32)
    if j.ndim != 3 or int(j.shape[2]) < 2:
        raise ValueError(f"joints_xy must be shaped [T,J,2], got {tuple(j.shape)}")
    j = j[..., :2]
    if not assume_sanitized_inputs:
        j = np.nan_to_num(j, nan=0.0, posinf=0.0, neginf=0.0, copy=False)
        np.clip(j, 0.0, 1.0, out=j)
    t_win = int(j.shape[0])
    if int(target_T) > 0 and t_win != int(target_T):
        raise ValueError(f"joints_xy time length mismatch ({t_win}); expected target_T={int(target_T)}")

    if conf is None:
        c = None
    elif isinstance(conf, np.ndarray) and conf.dtype == np.float32:
        c = conf
    else:
        c = np.asarray(conf, dtype=np.float32)
    if c is not None:
        if c.ndim != 2:
            raise ValueError(f"conf must be shaped [T,J], got {tuple(c.shape)}")
        j_joints = int(j.shape[1])
        if c.shape != (t_win, j_joints):
            if int(c.size) == int(t_win * j_joints):
                c = c.reshape(t_win, j_joints)
            else:
                raise ValueError(f"conf shape mismatch {tuple(c.shape)} for joints {(t_win, j_joints)}")
        if not assume_sanitized_inputs:
            c = np.nan_to_num(c, nan=0.0, posinf=0.0, neginf=0.0, copy=False)
            np.clip(c, 0.0, 1.0, out=c)

    def _scalar_first(t: Any) -> float:
        return float(t.view(-1)[0].item())

    # Prepare inputs (optionally reuse per-request cache across dual-model inference).
    cache_key = None
    if feature_cache is not None:
        # Request-local cache only lives for one predict_window request where
        # joints/conf/fps are shared across model variants.
        try:
            j_ptr = int(j.__array_interface__["data"][0])
        except Exception:
            j_ptr = id(j)
        if c is None:
            c_ptr = None
        else:
            try:
                c_ptr = int(c.__array_interface__["data"][0])
            except Exception:
                c_ptr = id(c)
        j_fp = _sample_fp(j)
        c_fp = _sample_fp(c)
        feat_cfg_sig = cached.get("feat_cfg_sig")
        if not isinstance(feat_cfg_sig, tuple):
            feat_cfg_sig = (
                str(getattr(feat_cfg, "center", "pelvis")),
                bool(getattr(feat_cfg, "use_motion", True)),
                bool(getattr(feat_cfg, "use_bone", False)),
                bool(getattr(feat_cfg, "use_bone_length", False)),
                bool(getattr(feat_cfg, "use_conf_channel", True)),
                bool(getattr(feat_cfg, "motion_scale_by_fps", True)),
                float(getattr(feat_cfg, "conf_gate", 0.2)),
                bool(getattr(feat_cfg, "use_precomputed_mask", True)),
            )
        cache_key = (*feat_cfg_sig, float(fps), int(target_T), j_ptr, c_ptr, *j_fp, *c_fp)
        cached_xg = feature_cache.get(cache_key)
    else:
        cached_xg = None

    if cached_xg is None:
        Xg, _mask = build_canonical_input(
            joints_xy=j,
            motion_xy=None,
            conf=c,
            mask=None,
            fps=float(fps),
            feat_cfg=feat_cfg,
            assume_finite_xy=True,
            assume_finite_conf=(c is not None),
        )
        if feature_cache is not None and cache_key is not None:
            feature_cache[cache_key] = Xg
    else:
        Xg = cached_xg

    forward_fn = _make_forward_fn(
        torch=torch,
        model=model,
        device=device,
        arch=spec.arch,
        Xg=Xg,
        feat_cfg=feat_cfg,
        model_cfg=(cached.get("model_cfg") if isinstance(cached, dict) else None),
        is_two_stream=bool(cached.get("is_two_stream", False)),
    )

    p_det: Optional[float] = None
    mu = 0.0
    sigma = 0.0
    mc_n_used: Optional[int] = None
    if use_mc:
        try:
            mc_predict_mu_sigma = _mc_predict_fn()

            mc_kwargs: Dict[str, Any] = {"M": int(mc_M)}
            if mc_sigma_tol is not None:
                try:
                    sigma_tol = float(mc_sigma_tol)
                    if sigma_tol > 0.0:
                        mc_kwargs["max_sigma_for_early_stop"] = sigma_tol
                        mc_kwargs["return_n_used"] = True
                except Exception:
                    pass
            if mc_se_tol is not None:
                try:
                    se_tol = float(mc_se_tol)
                    if se_tol > 0.0:
                        mc_kwargs["max_se_for_early_stop"] = se_tol
                        mc_kwargs["return_n_used"] = True
                except Exception:
                    pass
            try:
                mc_out = mc_predict_mu_sigma(model, forward_fn=forward_fn, **mc_kwargs)
            except TypeError:
                mc_kwargs.pop("max_sigma_for_early_stop", None)
                mc_kwargs.pop("max_se_for_early_stop", None)
                mc_kwargs.pop("return_n_used", None)
                mc_out = mc_predict_mu_sigma(model, forward_fn=forward_fn, **mc_kwargs)

            if isinstance(mc_out, tuple) and len(mc_out) >= 3 and isinstance(mc_out[2], (int, np.integer)):
                mu_t = mc_out[0]
                sig_t = mc_out[1]
                mc_n_used = int(mc_out[2])
            else:
                mu_t = mc_out[0]
                sig_t = mc_out[1]
                mc_n_used = int(mc_M)
            mu = _scalar_first(mu_t)
            sigma = _scalar_first(sig_t)
            # In MC mode, use predictive mean as display/telemetry probability.
            p_det = float(mu)
        except Exception:
            with torch.inference_mode():
                p_det = _scalar_first(forward_fn())
            mu = float(p_det)
            sigma = 0.0
            mc_n_used = 1
    else:
        with torch.inference_mode():
            p_det = _scalar_first(forward_fn())
        mu = float(p_det)
        sigma = 0.0

    op_norm = _norm_op_code(op_code)
    alert_cfg_eff: Dict[str, Any] = dict(spec.alert_cfg or {})
    op_entry = (spec.ops or {}).get(op_norm) or {}
    if isinstance(op_entry, dict):
        if "tau_low" in op_entry:
            alert_cfg_eff["tau_low"] = _safe_float(op_entry.get("tau_low"), _safe_float(alert_cfg_eff.get("tau_low"), 0.5))
        if "tau_high" in op_entry:
            alert_cfg_eff["tau_high"] = _safe_float(op_entry.get("tau_high"), _safe_float(alert_cfg_eff.get("tau_high"), 0.85))
    tau_low = _safe_float(alert_cfg_eff.get("tau_low"), 0.5)
    tau_high = _safe_float(alert_cfg_eff.get("tau_high"), 0.85)

    return {
        "spec_key": spec_key,
        "dataset": spec.dataset,
        "arch": spec.arch,
        "p_det": float(p_det),
        "mu": float(mu),
        "sigma": float(sigma),
        "mc_n_used": int(mc_n_used) if mc_n_used is not None else None,
        "tau_low": float(tau_low),
        "tau_high": float(tau_high),
        "ops": spec.ops,
        "alert_cfg": alert_cfg_eff,
    }


def fuse_hybrid(tri_tcn: str, tri_gcn: str) -> str:
    """Fuse two triage states using the requested policy.

    Policy:
    - NOT_FALL only if both NOT_FALL
    - FALL if (one is FALL) and (the other is FALL or UNCERTAIN)
    - otherwise UNCERTAIN
    """
    t = (tri_tcn or "not_fall").lower()
    g = (tri_gcn or "not_fall").lower()

    if t == "not_fall" and g == "not_fall":
        return "not_fall"

    # "one fall + one uncertain also counts as fall"
    if (t == "fall" and g in {"fall", "uncertain"}) or (g == "fall" and t in {"fall", "uncertain"}):
        return "fall"

    return "uncertain"
