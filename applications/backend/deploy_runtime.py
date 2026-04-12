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

from fall_detection.pose.preprocess_config import get_pose_preprocess_cfg_from_data_cfg

from .services.monitor_uncertainty_service import resolve_uncertainty_cfg, should_run_mc

SUPPORTED_DATASETS = {"caucafall", "le2i"}


@dataclass
class DeploySpec:
    key: str                     # e.g. "caucafall_tcn"
    dataset: str                  # le2i|caucafall
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
    except (TypeError, ValueError):
        return default


def _repo_root() -> Path:
    # server/ is inside the project root.
    return Path(__file__).resolve().parents[1]


def _load_json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return {}


def _load_yaml(path: Path) -> Dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, yaml.YAMLError, UnicodeDecodeError):
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
        model_block = data.get("model") if isinstance(data.get("model"), dict) else {}

        # Filename pattern: <arch>_<dataset>.yaml (e.g. gcn_muvim.yaml)
        stem = p.stem.lower().strip()
        parts = stem.split("_")
        if len(parts) < 2:
            continue
        arch_guess, dataset_guess = parts[0], "_".join(parts[1:])

        arch = str(data.get("arch") or model_block.get("arch") or arch_guess).lower().strip()
        dataset = dataset_guess.lower().strip()
        if arch not in {"tcn", "gcn"}:
            continue
        ckpt_rel = str(data.get("ckpt") or model_block.get("ckpt") or "").strip()
        if not ckpt_rel:
            continue
        if os.path.isabs(ckpt_rel):
            ckpt_path = Path(ckpt_rel)
        else:
            ckpt_from_yaml = (p.parent / ckpt_rel).resolve()
            ckpt_from_root = (root / ckpt_rel).resolve()
            ckpt_path = ckpt_from_yaml if ckpt_from_yaml.exists() else ckpt_from_root
        if not ckpt_path.exists():
            # Skip broken configs
            continue

        spec_key = f"{dataset}_{arch}"

        feat_cfg = data.get("feat_cfg") or model_block.get("feat_cfg") or {}
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
        if dataset not in SUPPORTED_DATASETS:
            continue
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
_POSE_PREPROCESS_CACHE: Dict[str, Dict[str, Any]] = {}


def get_specs() -> Dict[str, DeploySpec]:
    global _SPECS
    if _SPECS is None:
        _SPECS = discover_specs()
    return _SPECS


def get_pose_preprocess_cfg(spec_key: str) -> Dict[str, Any]:
    """Resolve pose-preprocess cfg for a deploy spec with checkpoint precedence."""
    if spec_key in _POSE_PREPROCESS_CACHE:
        return dict(_POSE_PREPROCESS_CACHE[spec_key])

    specs = get_specs()
    spec = specs.get(spec_key)
    if spec is None:
        raise KeyError(f"Unknown spec_key: {spec_key}")

    spec_data_cfg = spec.data_cfg if isinstance(spec.data_cfg, dict) else {}
    cfg = get_pose_preprocess_cfg_from_data_cfg(spec_data_cfg)

    runtime = _get_ml_runtime()
    load_ckpt = runtime["load_ckpt"]
    try:
        bundle = load_ckpt(spec.ckpt, map_location="cpu")
    except Exception:
        bundle = {}

    bundle_data_cfg = bundle.get("data_cfg") if isinstance(bundle, dict) else {}
    cfg = get_pose_preprocess_cfg_from_data_cfg(bundle_data_cfg, fallback=cfg)
    _POSE_PREPROCESS_CACHE[spec_key] = dict(cfg)
    return dict(cfg)


def get_alert_cfg(spec_key: str, op_code: str = "OP-2") -> Dict[str, Any]:
    """Return the effective alert_cfg dict for (spec, op_code)."""
    specs = get_specs()
    spec = specs.get(spec_key)
    if not spec:
        return {}
    op = _norm_op_code(op_code)
    base = dict(spec.alert_cfg or {})
    # Override alert params from ops if present.
    op_entry = (spec.ops or {}).get(op) or {}
    if isinstance(op_entry, dict):
        if "tau_low" in op_entry:
            base["tau_low"] = _safe_float(op_entry.get("tau_low"), _safe_float(base.get("tau_low"), 0.5))
        if "tau_high" in op_entry:
            base["tau_high"] = _safe_float(op_entry.get("tau_high"), _safe_float(base.get("tau_high"), 0.85))
        # Optional per-OP tracker/confirm overrides (deployment policy tuning).
        for key in (
            "ema_alpha",
            "k",
            "n",
            "cooldown_s",
            "confirm",
            "confirm_s",
            "confirm_min_lying",
            "confirm_max_motion",
            "confirm_require_low",
            "start_guard_max_lying",
            "start_guard_prefixes",
        ):
            if key in op_entry:
                base[key] = op_entry.get(key)
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
    except (ImportError, ModuleNotFoundError) as e:
        raise RuntimeError("PyTorch is required for real model inference.") from e


def _pick_device(torch: Any) -> Any:
    # Prefer acceleration when available, but keep a safe CPU fallback.
    try:
        if hasattr(torch, "cuda") and torch.cuda.is_available():
            return torch.device("cuda")
    except (AttributeError, RuntimeError, TypeError):
        pass
    # NOTE: MPS can be unstable for some GCN runtime ops in long-running API process.
    # Keep CPU as default for service stability; allow opt-in via env.
    allow_mps = str(os.environ.get("SERVER_ALLOW_MPS", "0")).strip().lower() in {"1", "true", "yes"}
    if allow_mps:
        try:
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return torch.device("mps")
        except (AttributeError, RuntimeError, TypeError):
            pass
    return torch.device("cpu")


_MODEL_CACHE: Dict[str, Dict[str, Any]] = {}
_ML_RUNTIME: Optional[Dict[str, Any]] = None


def _default_torch_threads() -> int:
    cpu_count = os.cpu_count() or 1
    return max(1, min(4, int(cpu_count)))


def _env_int(name: str, default: int) -> int:
    try:
        value = int(str(os.environ.get(name, default)).strip())
        return value if value > 0 else default
    except (TypeError, ValueError):
        return default


def _get_ml_runtime() -> Dict[str, Any]:
    global _ML_RUNTIME
    if _ML_RUNTIME is not None:
        return _ML_RUNTIME

    torch = _torch()
    try:
        torch.set_num_threads(_env_int("SERVER_TORCH_NUM_THREADS", _default_torch_threads()))
    except (AttributeError, RuntimeError, TypeError, ValueError):
        pass
    try:
        torch.set_num_interop_threads(_env_int("SERVER_TORCH_NUM_INTEROP_THREADS", 1))
    except (AttributeError, RuntimeError, TypeError, ValueError):
        pass

    try:
        from fall_detection.core.ckpt import load_ckpt
        from fall_detection.core.models import build_model
        from fall_detection.core.features import (
            FeatCfg,
            build_canonical_input,
            build_tcn_input,
            split_gcn_two_stream,
        )
        from fall_detection.core.uncertainty import mc_predict_mu_sigma
    except (ImportError, ModuleNotFoundError) as e:
        raise RuntimeError(
            "Missing ML runtime package 'fall_detection'. "
            "Make sure the package is installed (e.g. pip install -e .), "
            "or install the ML package into this environment."
        ) from e

    _ML_RUNTIME = {
        "torch": torch,
        "load_ckpt": load_ckpt,
        "build_model": build_model,
        "FeatCfg": FeatCfg,
        "build_canonical_input": build_canonical_input,
        "build_tcn_input": build_tcn_input,
        "split_gcn_two_stream": split_gcn_two_stream,
        "mc_predict_mu_sigma": mc_predict_mu_sigma,
    }
    return _ML_RUNTIME


def _infer_model_num_joints(model: Any) -> Optional[int]:
    try:
        if hasattr(model, "encoder") and hasattr(model.encoder, "A_hat"):
            return int(model.encoder.A_hat.shape[0])
    except (AttributeError, TypeError, ValueError, IndexError):
        pass
    try:
        if hasattr(model, "j_enc") and hasattr(model.j_enc, "A_hat"):
            return int(model.j_enc.A_hat.shape[0])
    except (AttributeError, TypeError, ValueError, IndexError):
        pass
    return None


def _align_joint_count(
    joints_xy: Any,
    conf: Any,
    *,
    expected_v: Optional[int],
) -> Tuple[Any, Any]:
    """Align runtime joint count to model expectation."""
    import numpy as np

    j = np.asarray(joints_xy, dtype=np.float32)
    c = np.asarray(conf, dtype=np.float32) if conf is not None else None
    if j.ndim != 3 or j.shape[-1] != 2 or expected_v is None:
        return j, c

    cur_v = int(j.shape[1])
    exp_v = int(expected_v)
    if cur_v == exp_v:
        return j, c

    # Preferred semantic remap for MediaPipe33 -> internal17.
    if cur_v >= 29 and exp_v == 17:
        try:
            from fall_detection.data.adapters.base import map_mp33_to_internal17

            j17, c17, _m17 = map_mp33_to_internal17(j, conf=c, mask=None)
            return j17, c17
        except (ImportError, ModuleNotFoundError, ValueError, TypeError, RuntimeError):
            pass

    # Fallback: truncate/pad in joint dimension.
    if cur_v > exp_v:
        j2 = j[:, :exp_v, :]
        c2 = c[:, :exp_v] if c is not None and c.ndim == 2 else c
        return j2, c2

    pad_v = exp_v - cur_v
    jpad = np.zeros((j.shape[0], pad_v, 2), dtype=j.dtype)
    j2 = np.concatenate([j, jpad], axis=1)
    if c is not None and c.ndim == 2:
        cpad = np.zeros((c.shape[0], pad_v), dtype=c.dtype)
        c2 = np.concatenate([c, cpad], axis=1)
    else:
        c2 = c
    return j2, c2


def _load_model_and_cfg(spec: DeploySpec) -> Dict[str, Any]:
    """Load model + cfg, cached by spec.key."""
    if spec.key in _MODEL_CACHE:
        return _MODEL_CACHE[spec.key]

    runtime = _get_ml_runtime()
    torch = runtime["torch"]
    device = _pick_device(torch)
    load_ckpt = runtime["load_ckpt"]
    build_model = runtime["build_model"]
    FeatCfg = runtime["FeatCfg"]

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
    """Pad/trim to match conv_in channels (TCN)."""
    try:
        expected = int(getattr(model, "conv_in")[0].in_channels)  # type: ignore[index]
    except (AttributeError, IndexError, KeyError, TypeError, ValueError):
        return x
    c = int(x.shape[-1])
    if c == expected:
        return x
    if c > expected:
        return x[..., :expected]
    pad = torch.zeros((x.shape[0], x.shape[1], expected - c), device=x.device, dtype=x.dtype)
    return torch.cat([x, pad], dim=-1)


def _get_optimized_infer_model(
    *,
    spec: DeploySpec,
    cached: Dict[str, Any],
    prepared: Dict[str, Any],
) -> Any:
    model = cached["model"]
    if spec.arch != "tcn":
        return model

    device = cached["device"]
    if str(device) != "cpu":
        return model

    x_t = prepared.get("x_t")
    if x_t is None:
        return model

    optimized_cache = cached.setdefault("optimized_models", {})
    shape_key = tuple(int(v) for v in x_t.shape)
    infer_model = optimized_cache.get(shape_key)
    if infer_model is not None:
        return infer_model

    runtime = _get_ml_runtime()
    torch = runtime["torch"]
    try:
        with torch.inference_mode():
            traced = torch.jit.trace(model, x_t, check_trace=False, strict=False)
            infer_model = torch.jit.optimize_for_inference(traced)
    except Exception:
        infer_model = model
    optimized_cache[shape_key] = infer_model
    return infer_model


def _prepare_features(
    *,
    spec: DeploySpec,
    model: Any,
    device: Any,
    feat_cfg: Any,
    model_cfg: Dict[str, Any],
    joints_xy: Any,
    conf: Any,
    fps: float,
) -> Dict[str, Any]:
    runtime = _get_ml_runtime()
    build_canonical_input = runtime["build_canonical_input"]
    build_tcn_input = runtime["build_tcn_input"]
    split_gcn_two_stream = runtime["split_gcn_two_stream"]
    torch = runtime["torch"]

    expected_v = None
    try:
        expected_v = (
            int(model_cfg.get("num_joints"))
            if isinstance(model_cfg, dict) and model_cfg.get("num_joints") is not None
            else None
        )
    except (TypeError, ValueError):
        expected_v = None
    if expected_v is None:
        expected_v = _infer_model_num_joints(model)

    joints_xy_aligned, conf_aligned = _align_joint_count(joints_xy, conf, expected_v=expected_v)
    Xg, _mask = build_canonical_input(
        joints_xy=joints_xy_aligned,
        motion_xy=None,
        conf=conf_aligned,
        mask=None,
        fps=float(fps),
        feat_cfg=feat_cfg,
    )

    if spec.arch == "tcn":
        Xt = build_tcn_input(Xg, feat_cfg)
        x_t = torch.from_numpy(np.asarray(Xt, dtype=np.float32)).to(device=device, dtype=torch.float32).unsqueeze(0)
        x_t = _match_in_ch_tcn(torch, model, x_t)
        return {"kind": "tcn", "x_t": x_t}

    xb = torch.from_numpy(np.asarray(Xg, dtype=np.float32)).to(device=device, dtype=torch.float32).unsqueeze(0)
    is_two_stream = ("twostream" in model.__class__.__name__.lower()) or bool(getattr(model, "two_stream", False))
    if is_two_stream:
        xj_np, xm_np = split_gcn_two_stream(Xg, feat_cfg)
        xj_t = torch.from_numpy(np.asarray(xj_np, dtype=np.float32)).to(device=device, dtype=torch.float32).unsqueeze(0)
        xm_t = torch.from_numpy(np.asarray(xm_np, dtype=np.float32)).to(device=device, dtype=torch.float32).unsqueeze(0)
        return {"kind": "gcn_two_stream", "xj_t": xj_t, "xm_t": xm_t}
    return {"kind": "gcn_single_stream", "xb": xb}


def _forward_prob(model: Any, prepared: Dict[str, Any]) -> Any:
    kind = prepared["kind"]
    if kind == "tcn":
        logits = model(prepared["x_t"])
    elif kind == "gcn_two_stream":
        logits = model(prepared["xj_t"], prepared["xm_t"])
    else:
        logits = model(prepared["xb"])
    return logits.sigmoid().view(-1)


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
) -> Dict[str, Any]:
    """Run inference for one deploy spec and return p/mu/sigma.

    Alerting state (EMA/k-of-n/cooldown) is handled by server/app.py.
    """
    specs = get_specs()
    if spec_key not in specs:
        raise KeyError(f"Unknown spec_key: {spec_key}")
    spec = specs[spec_key]

    runtime = _get_ml_runtime()
    torch = runtime["torch"]
    cached = _load_model_and_cfg(spec)
    model = cached["model"]
    device = cached["device"]
    feat_cfg = cached["feat_cfg"]
    model_cfg = cached.get("model_cfg") or {}
    mc_predict_mu_sigma = runtime["mc_predict_mu_sigma"]

    prepared = _prepare_features(
        spec=spec,
        model=model,
        device=device,
        feat_cfg=feat_cfg,
        model_cfg=model_cfg,
        joints_xy=joints_xy,
        conf=conf,
        fps=float(fps),
    )
    infer_model = _get_optimized_infer_model(spec=spec, cached=cached, prepared=prepared)

    def forward_fn():
        with torch.inference_mode():
            return _forward_prob(infer_model, prepared)

    p_det = float(forward_fn().detach().cpu().view(-1)[0].item())

    tau_low, tau_high = get_op_taus(spec_key, op_code)
    alert_cfg = get_alert_cfg(spec_key, op_code)
    op_entry = (spec.ops or {}).get(_norm_op_code(op_code)) or {}
    uncertainty_cfg = resolve_uncertainty_cfg(alert_cfg, op_entry if isinstance(op_entry, dict) else None)

    mu, sigma = float(p_det), 0.0
    mc_applied, mc_reason = should_run_mc(
        use_mc=bool(use_mc),
        p_det=float(p_det),
        tau_low=float(tau_low),
        tau_high=float(tau_high),
        uncertainty_cfg=uncertainty_cfg,
    )
    if mc_applied:
        try:
            mu_t, sig_t = mc_predict_mu_sigma(model, forward_fn=forward_fn, M=int(mc_M))
            mu = float(mu_t.detach().cpu().view(-1)[0].item())
            sigma = float(sig_t.detach().cpu().view(-1)[0].item())
        except (ImportError, ModuleNotFoundError, AttributeError, RuntimeError, TypeError, ValueError):
            mu, sigma = float(p_det), 0.0
            mc_applied = False
            mc_reason = "mc_failed"

    return {
        "spec_key": spec_key,
        "dataset": spec.dataset,
        "arch": spec.arch,
        "p_det": float(p_det),
        "mu": float(mu),
        "sigma": float(sigma),
        "mc_applied": bool(mc_applied),
        "mc_reason": str(mc_reason),
        "uncertainty_gate": {
            "enabled": bool(uncertainty_cfg.get("enabled", True)),
            "boundary_margin": float(uncertainty_cfg.get("boundary_margin", 0.08)),
            "sigma_fall_max": float(uncertainty_cfg.get("sigma_fall_max", 0.08)),
        },
        "tau_low": float(tau_low),
        "tau_high": float(tau_high),
        "ops": spec.ops,
        "alert_cfg": alert_cfg,
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
