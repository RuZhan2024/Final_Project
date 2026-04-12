#!/usr/bin/env python3
"""
core/ckpt.py

Checkpoint helpers for Fall Detection v2.

Goals
- Make training/eval robust to drift: store arch + model_cfg + feat_cfg + data_cfg together.
- Be backward compatible with older checkpoints that saved only a state_dict.

Public API
- save_ckpt(path, bundle_or_state_dict=None, **fields)
- load_ckpt(path, map_location="cpu") -> dict bundle
- get_cfg(bundle, key=None, default=None)

Notes
- `get_cfg(bundle)` (no key) returns a 4-tuple: (arch, model_cfg, feat_cfg, data_cfg)
  for backward compatibility with earlier scripts.
- `get_cfg(bundle, "model_cfg", default={})` returns a single stored field.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Union

import os
import tempfile
import torch


Bundle = Dict[str, Any]
CKPT_VERSION = 2


def _normalize_bundle(bundle: Bundle) -> Bundle:
    """Normalize bundle fields to a stable schema."""
    b = dict(bundle or {})
    v = int(b.get("ckpt_version", b.get("version", CKPT_VERSION)))
    b["ckpt_version"] = v
    b["version"] = v  # keep legacy alias for older readers
    b["arch"] = str(b.get("arch", "unknown"))
    b["model_cfg"] = dict(b.get("model_cfg", {}) or {})
    b["feat_cfg"] = dict(b.get("feat_cfg", {}) or {})
    b["data_cfg"] = dict(b.get("data_cfg", {}) or {})
    b["meta"] = dict(b.get("meta", {}) or {})
    if "best" in b and b["best"] is None:
        b["best"] = {}
    return b


def _as_bundle(obj: Any) -> Bundle:
    """
    Convert different checkpoint formats into a canonical bundle dict.

    Supported inputs:
    - dict with 'state_dict' (new bundle)
    - plain state_dict (older torch.save(model.state_dict()))
    - dict that looks like a trainer snapshot (may contain 'model' key etc.)
    """
    if isinstance(obj, dict):
        # New style: already a bundle
        if "state_dict" in obj and isinstance(obj["state_dict"], dict):
            return _normalize_bundle(dict(obj))

        # Some trainers saved {"model": state_dict, ...}
        if "model" in obj and isinstance(obj["model"], dict):
            b = {
                "ckpt_version": 1,
                "arch": obj.get("arch", obj.get("model_arch", "unknown")),
                "state_dict": obj["model"],
                "model_cfg": obj.get("model_cfg", obj.get("cfg", {})) or {},
                "feat_cfg": obj.get("feat_cfg", {}) or {},
                "data_cfg": obj.get("data_cfg", {}) or {},
                "meta": obj.get("meta", {}) or {},
            }
            return _normalize_bundle(b)

        # It might itself be a raw state_dict (parameter tensors)
        if obj and all(isinstance(k, str) for k in obj.keys()):
            # Heuristic: state_dict keys often contain dots and end with weight/bias
            sample_keys = list(obj.keys())[:5]
            if any("." in k for k in sample_keys) or any(k.endswith(("weight", "bias")) for k in sample_keys):
                return {
                    "ckpt_version": 0,
                    "arch": "unknown",
                    "state_dict": obj,
                    "model_cfg": {},
                    "feat_cfg": {},
                    "data_cfg": {},
                    "meta": {},
                }

    raise TypeError(f"Unsupported checkpoint object type: {type(obj)}")


def save_ckpt(
    path: str,
    bundle_or_state_dict: Optional[Union[Bundle, Dict[str, Any]]] = None,
    **fields: Any,
) -> None:
    """
    Save a checkpoint.

    You can call it in two equivalent ways:

    1) save_ckpt(path, bundle)
       where bundle is a dict that already contains at least 'state_dict'.

    2) save_ckpt(path, state_dict=..., arch=..., model_cfg=..., feat_cfg=..., data_cfg=..., meta=...)
       (or save_ckpt(path, state_dict_dict, arch=..., ...))

    This flexibility prevents call-site drift.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    if bundle_or_state_dict is None:
        bundle: Bundle = dict(fields)
    elif isinstance(bundle_or_state_dict, dict) and "state_dict" in bundle_or_state_dict:
        bundle = dict(bundle_or_state_dict)
        bundle.update(fields)
    elif isinstance(bundle_or_state_dict, dict):
        # assume it's a raw state_dict
        bundle = {"state_dict": bundle_or_state_dict}
        bundle.update(fields)
    else:
        raise TypeError("bundle_or_state_dict must be a dict or None")

    bundle = _normalize_bundle(bundle)

    if "state_dict" not in bundle or not isinstance(bundle["state_dict"], dict):
        raise ValueError("Checkpoint bundle must contain a dict 'state_dict'.")

    # Atomic write: write to temp file in the same directory then replace.
    out_dir = os.path.dirname(path) or "."
    base = os.path.basename(path)
    fd, tmp_path = tempfile.mkstemp(prefix=f".{base}.", suffix=".tmp", dir=out_dir)
    os.close(fd)
    try:
        torch.save(bundle, tmp_path)
        os.replace(tmp_path, path)
    except Exception:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except OSError:
            pass
        raise


def load_ckpt(path: str, map_location: Union[str, torch.device] = "cpu") -> Bundle:
    """
    Load a checkpoint and return a canonical bundle dict.
    """
    obj = torch.load(path, map_location=map_location)
    return _normalize_bundle(_as_bundle(obj))


def get_cfg(bundle: Bundle, key: Optional[str] = None, default: Any = None) -> Any:
    """Access checkpoint config with backward compatibility.

    Two supported call styles:

    1) Tuple-style (legacy):
        arch, model_cfg, feat_cfg, data_cfg = get_cfg(bundle)

    2) Key-style (preferred):
        model_cfg = get_cfg(bundle, "model_cfg", default={})

    For missing keys, returns `default`.
    """
    if not isinstance(bundle, dict):
        if key is None:
            return "unknown", {}, {}, {}
        return default

    # Legacy tuple-style request.
    if key is None:
        arch = bundle.get("arch")
        model_cfg = bundle.get("model_cfg")
        feat_cfg = bundle.get("feat_cfg")
        data_cfg = bundle.get("data_cfg")

        # Some older bundles may store nested "cfg".
        cfg = bundle.get("cfg")
        if isinstance(cfg, dict):
            arch = arch if arch is not None else cfg.get("arch")
            model_cfg = model_cfg if model_cfg is not None else cfg.get("model_cfg")
            feat_cfg = feat_cfg if feat_cfg is not None else cfg.get("feat_cfg")
            data_cfg = data_cfg if data_cfg is not None else cfg.get("data_cfg")

        return (
            str(arch or "unknown"),
            dict(model_cfg or {}),
            dict(feat_cfg or {}),
            dict(data_cfg or {}),
        )

    # Key-style request.
    if key in bundle:
        v = bundle.get(key)
        return default if v is None else v

    cfg = bundle.get("cfg")
    if isinstance(cfg, dict) and key in cfg:
        v = cfg.get(key)
        return default if v is None else v

    return default
