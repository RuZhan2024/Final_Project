#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
core/ckpt.py

Checkpoint helpers (bundle format) for this repo.

Why we use a "bundle" instead of saving only state_dict
-------------------------------------------------------
If you only save model weights, you lose critical info needed to rebuild the exact
same model later:
- which architecture was used?  (tcn vs gcn)
- what model hyperparameters?   (hidden, dropout, two_stream, ...)
- what feature flags?           (use_motion, use_conf_channel, conf_gate, ...)
- what data config?             (fps_default, W/S, dataset name, ...)

So we save a single dict "bundle" that includes BOTH weights and config.

Canonical bundle fields
-----------------------
Required:
  - state_dict : dict[str, Tensor]  (PyTorch model weights)

Recommended:
  - arch       : "tcn" or "gcn"
  - model_cfg  : dict (model hyperparameters)
  - feat_cfg   : dict (feature flags used to build inputs)
  - data_cfg   : dict (data-related settings)
  - meta       : dict (extra info: git hash, notes, monitor metric, ...)
  - best       : dict (best val score summary)
Optional:
  - ema_state_dict : dict[str, Tensor] (EMA-smoothed weights)
  - ema_decay      : float
  - version        : int (bundle schema version)

Public API (used across your repo)
----------------------------------
- save_ckpt(path, bundle_or_state_dict=None, **fields)
- load_ckpt(path, map_location="cpu") -> bundle dict
- get_cfg(bundle, key=None, default=None)
- get_state_dict(bundle, prefer_ema=True)

This API is intentionally flexible so older scripts won't break.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Union

import os
from pathlib import Path

import torch


# A checkpoint "bundle" is a normal Python dict.
Bundle = Dict[str, Any]


# ============================================================
# 1) Helpers: identify/normalize checkpoint formats
# ============================================================
def _is_state_dict_like(d: Dict[str, Any]) -> bool:
    """
    Heuristic: decide whether a dict looks like a raw PyTorch state_dict.

    Typical state_dict keys look like:
      "layer.weight", "layer.bias", "encoder.blocks.0.conv.weight", ...
    """
    if not d:
        return False
    if not all(isinstance(k, str) for k in d.keys()):
        return False
    sample = list(d.keys())[:8]
    return any("." in k for k in sample) or any(k.endswith(("weight", "bias")) for k in sample)


def _as_bundle(obj: Any) -> Bundle:
    """
    Convert different saved checkpoint formats into ONE canonical bundle dict.

    Supported inputs:
    1) New bundle format:
         {"state_dict": ..., "arch": ..., "model_cfg": ..., ...}

    2) Trainer snapshot format:
         {"model": state_dict, "arch": ..., "cfg": ...}

    3) Old format: raw state_dict saved directly:
         torch.save(model.state_dict(), "x.pt")
    """
    if isinstance(obj, dict):
        # Case 1: already a bundle
        if "state_dict" in obj and isinstance(obj["state_dict"], dict):
            b = dict(obj)
            b.setdefault("version", 2)
            b.setdefault("arch", b.get("arch", "unknown"))
            b.setdefault("model_cfg", b.get("model_cfg", {}) or {})
            b.setdefault("feat_cfg", b.get("feat_cfg", {}) or {})
            b.setdefault("data_cfg", b.get("data_cfg", {}) or {})
            b.setdefault("meta", b.get("meta", {}) or {})
            return b

        # Case 2: trainer snapshot that saved weights under "model"
        if "model" in obj and isinstance(obj["model"], dict):
            return {
                "version": 1,
                "arch": obj.get("arch", obj.get("model_arch", "unknown")),
                "state_dict": obj["model"],
                "model_cfg": obj.get("model_cfg", obj.get("cfg", {})) or {},
                "feat_cfg": obj.get("feat_cfg", {}) or {},
                "data_cfg": obj.get("data_cfg", {}) or {},
                "meta": obj.get("meta", {}) or {},
            }

        # Case 3: dict itself is likely a state_dict
        if _is_state_dict_like(obj):
            return {
                "version": 0,
                "arch": "unknown",
                "state_dict": obj,
                "model_cfg": {},
                "feat_cfg": {},
                "data_cfg": {},
                "meta": {},
            }

    raise TypeError(f"Unsupported checkpoint object type: {type(obj)}")


# ============================================================
# 2) Atomic save
# ============================================================
def _atomic_torch_save(obj: Any, path: str) -> None:
    """
    Atomic write:
      1) save to a temporary file in the same folder
      2) rename to the final destination

    Why:
    - If training crashes mid-save, you avoid corrupted checkpoints.
    - os.replace is atomic on most filesystems.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    tmp = p.with_suffix(p.suffix + ".tmp")
    torch.save(obj, str(tmp))
    os.replace(str(tmp), str(p))


# ============================================================
# 3) Public API: save/load
# ============================================================
def save_ckpt(
    path: str,
    bundle_or_state_dict: Optional[Union[Bundle, Dict[str, Any]]] = None,
    **fields: Any,
) -> None:
    """
    Save a checkpoint bundle.

    Two main calling styles (both supported):

    Style A (recommended):
      save_ckpt(path,
        arch="tcn",
        state_dict=model.state_dict(),
        model_cfg=...,
        feat_cfg=...,
        data_cfg=...,
        ema_state_dict=...,
        best=...,
        meta=...
      )

    Style B:
      save_ckpt(path, bundle_dict)

    bundle_or_state_dict:
      - None            => use only **fields** as the bundle
      - dict with state_dict => merge into it
      - raw state_dict  => wrap it as {"state_dict": ...} then merge fields
    """
    # 1) Build the bundle dict from inputs
    if bundle_or_state_dict is None:
        bundle: Bundle = dict(fields)

    elif isinstance(bundle_or_state_dict, dict) and "state_dict" in bundle_or_state_dict:
        # It's already a bundle; merge extra fields
        bundle = dict(bundle_or_state_dict)
        bundle.update(fields)

    elif isinstance(bundle_or_state_dict, dict):
        # Assume it's a raw state_dict
        bundle = {"state_dict": bundle_or_state_dict}
        bundle.update(fields)

    else:
        raise TypeError("bundle_or_state_dict must be a dict or None")

    # 2) Normalize important fields so other scripts can rely on them
    bundle.setdefault("version", 2)
    bundle.setdefault("arch", bundle.get("arch", "unknown"))
    bundle.setdefault("model_cfg", bundle.get("model_cfg", {}) or {})
    bundle.setdefault("feat_cfg", bundle.get("feat_cfg", {}) or {})
    bundle.setdefault("data_cfg", bundle.get("data_cfg", {}) or {})
    bundle.setdefault("meta", bundle.get("meta", {}) or {})

    # 3) Validate required fields
    if "state_dict" not in bundle or not isinstance(bundle["state_dict"], dict):
        raise ValueError("Checkpoint bundle must contain a dict 'state_dict'.")

    # 4) Save atomically
    _atomic_torch_save(bundle, path)


def load_ckpt(path: str, map_location: Union[str, torch.device] = "cpu") -> Bundle:
    """
    Load a checkpoint file and return a canonical bundle dict.

    Why we do a compatibility try/except:
    - Some torch versions add/remove torch.load parameters (e.g., weights_only).
    - This makes loading robust across environments.
    """
    # torch.load signature varies across torch versions.
    try:
        obj = torch.load(path, map_location=map_location, weights_only=False)  # type: ignore[arg-type]
    except TypeError:
        obj = torch.load(path, map_location=map_location)

    return _as_bundle(obj)


# ============================================================
# 4) Public API: get_cfg / get_state_dict
# ============================================================
def get_cfg(bundle: Bundle, key: Optional[str] = None, default: Any = None) -> Any:
    """
    Access checkpoint config with backward compatibility.

    Two supported call styles:

    1) Tuple-style (legacy):
        arch, model_cfg, feat_cfg, data_cfg = get_cfg(bundle)

    2) Key-style (preferred):
        model_cfg = get_cfg(bundle, "model_cfg", default={})

    Also supports older nesting:
      bundle["cfg"] may contain {"model_cfg":..., "feat_cfg":..., ...}
    """
    if not isinstance(bundle, dict):
        if key is None:
            return "unknown", {}, {}, {}
        return default

    # Legacy tuple request
    if key is None:
        arch = bundle.get("arch")
        model_cfg = bundle.get("model_cfg")
        feat_cfg = bundle.get("feat_cfg")
        data_cfg = bundle.get("data_cfg")

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

    # Key-style request
    if key in bundle:
        v = bundle.get(key)
        return default if v is None else v

    cfg = bundle.get("cfg")
    if isinstance(cfg, dict) and key in cfg:
        v = cfg.get(key)
        return default if v is None else v

    return default


def get_state_dict(bundle: Bundle, prefer_ema: bool = True) -> Dict[str, Any]:
    """
    Return the model state_dict from a checkpoint bundle.

    prefer_ema:
      - True: if ema_state_dict exists, return it (better for evaluation stability)
      - False: always return state_dict
    """
    if not isinstance(bundle, dict):
        raise TypeError("bundle must be a dict")

    if prefer_ema and isinstance(bundle.get("ema_state_dict"), dict):
        return bundle["ema_state_dict"]

    sd = bundle.get("state_dict")
    if not isinstance(sd, dict):
        raise ValueError("bundle missing state_dict")
    return sd
