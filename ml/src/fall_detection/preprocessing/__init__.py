"""Preprocessing compatibility exports for refactored data modules."""

from fall_detection.core.features import FeatCfg, build_canonical_input, read_window_npz

__all__ = ["FeatCfg", "build_canonical_input", "read_window_npz"]
