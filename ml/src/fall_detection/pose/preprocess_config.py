from __future__ import annotations

from typing import Any, Dict, Mapping, Optional


DEFAULT_POSE_PREPROCESS_CFG: Dict[str, Any] = {
    "conf_thr": 0.2,
    "smooth_k": 5,
    "max_gap": 4,
    "fill_conf": "thr",
    "normalize": "torso",
    "pelvis_fill": "nearest",
    "rotate": "none",
}

_FILL_CONF_CHOICES = {"keep", "thr", "min_neighbors", "linear"}
_NORMALIZE_CHOICES = {"none", "torso", "shoulder"}
_PELVIS_FILL_CHOICES = {"nearest", "forward", "zero"}
_ROTATE_CHOICES = {"none", "shoulders"}


def normalize_pose_preprocess_cfg(
    cfg: Optional[Mapping[str, Any]],
    *,
    fallback: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Return a validated pose-preprocess config with stable defaults."""
    base = dict(DEFAULT_POSE_PREPROCESS_CFG)
    if isinstance(fallback, Mapping):
        base.update(dict(fallback))

    if not isinstance(cfg, Mapping):
        return base

    out = dict(base)

    try:
        out["conf_thr"] = float(cfg.get("conf_thr", out["conf_thr"]))
    except (TypeError, ValueError):
        pass

    try:
        out["smooth_k"] = max(1, int(cfg.get("smooth_k", out["smooth_k"])))
    except (TypeError, ValueError):
        pass

    try:
        out["max_gap"] = max(0, int(cfg.get("max_gap", out["max_gap"])))
    except (TypeError, ValueError):
        pass

    fill_conf = str(cfg.get("fill_conf", out["fill_conf"])).strip().lower()
    if fill_conf in _FILL_CONF_CHOICES:
        out["fill_conf"] = fill_conf

    normalize = str(cfg.get("normalize", out["normalize"])).strip().lower()
    if normalize in _NORMALIZE_CHOICES:
        out["normalize"] = normalize

    pelvis_fill = str(cfg.get("pelvis_fill", out["pelvis_fill"])).strip().lower()
    if pelvis_fill in _PELVIS_FILL_CHOICES:
        out["pelvis_fill"] = pelvis_fill

    rotate = str(cfg.get("rotate", out["rotate"])).strip().lower()
    if rotate in _ROTATE_CHOICES:
        out["rotate"] = rotate

    return out


def get_pose_preprocess_cfg_from_data_cfg(
    data_cfg: Optional[Mapping[str, Any]],
    *,
    fallback: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    if not isinstance(data_cfg, Mapping):
        return normalize_pose_preprocess_cfg(None, fallback=fallback)
    return normalize_pose_preprocess_cfg(data_cfg.get("pose_preprocess"), fallback=fallback)
