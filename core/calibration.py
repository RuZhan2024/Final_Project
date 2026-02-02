#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
core/calibration.py

Temperature scaling helpers.

What temperature scaling does
-----------------------------
Your model usually outputs LOGITS (unbounded real values).
If the model is over-confident or under-confident, we can calibrate it by scaling:

    logits_calibrated = logits / T

- T > 1.0  => softer (less confident) probabilities
- T < 1.0  => sharper (more confident) probabilities
- T = 1.0  => no calibration

Then we convert calibrated logits to probabilities using sigmoid:

    p = sigmoid(logits_calibrated)

Important: This does NOT change ranking much; it mainly changes confidence.
That’s why it's commonly used for threshold-based deployment systems.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from core.yamlio import yaml_load_simple


# ============================================================
# 1) Numeric helpers
# ============================================================
def _safe_float(x: Any, default: float) -> float:
    """
    Convert x -> float safely.

    If conversion fails, or result is NaN/inf, fall back to default.
    """
    try:
        v = float(x)
        if not np.isfinite(v):
            return float(default)
        return v
    except Exception:
        return float(default)


def sigmoid(x: np.ndarray) -> np.ndarray:
    """
    Stable sigmoid for numpy arrays.

    Why "stable"?
    - np.exp(large_number) overflows to inf.
    - Clipping the input avoids overflow while preserving behavior.

    Output is float32 in [0, 1].
    """
    x = np.asarray(x, dtype=np.float32)
    x = np.clip(x, -80.0, 80.0)  # prevents overflow in exp
    return (1.0 / (1.0 + np.exp(-x))).astype(np.float32)


# ============================================================
# 2) Temperature application
# ============================================================
def apply_temperature(logits: np.ndarray, temperature: float) -> np.ndarray:
    """
    Apply temperature scaling to logits.

    Parameters
    ----------
    logits:
      numpy array of any shape (e.g. [B], [N], [T], [B,1])

    temperature:
      T > 0, usually around 0.5 ~ 5.0
      If invalid (NaN/inf/<=0), we treat it as 1.0.

    Returns
    -------
    logits_scaled:
      logits / T (float32)
    """
    # Convert temperature to a safe positive float.
    T = _safe_float(temperature, default=1.0)
    if T <= 0.0:
        T = 1.0

    # Ensure logits is a float32 numpy array so downstream math is consistent.
    z = np.asarray(logits, dtype=np.float32)

    # The actual calibration rule.
    return (z / float(T)).astype(np.float32)


def calibrated_prob_from_logits(logits: np.ndarray, temperature: float) -> np.ndarray:
    """
    Convenience helper: logits -> calibrated probability.

    This is often what you want in evaluation or deployment logic:
      p = sigmoid(logits / T)
    """
    return sigmoid(apply_temperature(logits, temperature))


# ============================================================
# 3) Load temperature from YAML
# ============================================================
def load_temperature(calibration_yaml: Optional[str], default: float = 1.0) -> float:
    """
    Load temperature from a YAML file.

    Supported schemas (we accept several to be tolerant):
      temperature: 1.23

      calibration:
        temperature: 1.23

      calibration:
        T: 1.23

    If anything is missing/invalid, returns `default`.

    Why tolerate multiple schemas?
    - Different scripts/experiments may write slightly different keys.
    - Being tolerant prevents your deploy pipeline from breaking.
    """
    # If user didn't provide a file path, return default right away.
    if not calibration_yaml:
        return float(default)

    # Read yaml using our lightweight YAML loader.
    try:
        d = yaml_load_simple(calibration_yaml)
    except Exception:
        return float(default)

    if not isinstance(d, dict):
        return float(default)

    # 1) Top-level temperature
    if "temperature" in d:
        T = _safe_float(d.get("temperature"), default)
        return float(T) if T > 0.0 else float(default)

    # 2) Nested under calibration
    cal = d.get("calibration")
    if isinstance(cal, dict):
        if "temperature" in cal:
            T = _safe_float(cal.get("temperature"), default)
            return float(T) if T > 0.0 else float(default)
        if "T" in cal:
            T = _safe_float(cal.get("T"), default)
            return float(T) if T > 0.0 else float(default)

    # Nothing found
    return float(default)
