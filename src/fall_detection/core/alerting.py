#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
core/alerting.py

Real-time alert policy + event-level metrics.

Alert policy:
  1) Convert logits -> probabilities (caller does sigmoid).
  2) Smooth with EMA.
  3) Persistence: k-of-n "high" windows required to START an alert.
  4) Hysteresis: stay active while p >= tau_low; end when p < tau_low.
  5) Cooldown: after an alert ends, block new starts for cooldown_s.

Event metrics (per video, then aggregated):
  - event recall: #GT events detected / #GT events
  - false alerts per hour/day: #alert events NOT overlapping any GT / duration
  - detection delay: first alert start - GT start (seconds)
  - event precision + event F1 (computed using alert events overlap with GT)
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


# ---------------- config + structs ----------------

@dataclass(frozen=True)
class AlertCfg:
    ema_alpha: float = 0.20
    k: int = 2
    n: int = 3
    tau_high: float = 0.90
    tau_low: float = 0.70
    cooldown_s: float = 30.0

    # Optional "confirmation" stage.
    #
    # If enabled, a k-of-n trigger is treated as a *candidate* alert start.
    # We then wait up to confirm_s seconds for a confirmation condition.
    #
    # In this repo, we don't have RGB or depth at deploy-time; we only have
    # skeleton windows. So confirmation (if used) should be treated as a
    # lightweight heuristic.
    confirm: bool = False
    confirm_s: float = 2.0
    confirm_min_lying: float = 0.65
    confirm_max_motion: float = 0.08
    confirm_require_low: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "AlertCfg":
        d = dict(d or {})
        return AlertCfg(
            ema_alpha=float(d.get("ema_alpha", 0.20)),
            k=int(d.get("k", 2)),
            n=int(d.get("n", 3)),
            tau_high=float(d.get("tau_high", 0.90)),
            tau_low=float(d.get("tau_low", 0.70)),
            cooldown_s=float(d.get("cooldown_s", 30.0)),

            confirm=bool(d.get("confirm", False)),
            confirm_s=float(d.get("confirm_s", 2.0)),
            confirm_min_lying=float(d.get("confirm_min_lying", 0.65)),
            confirm_max_motion=float(d.get("confirm_max_motion", 0.08)),
            confirm_require_low=bool(d.get("confirm_require_low", True)),
        )


@dataclass
class AlertEvent:
    start_idx: int
    end_idx: int
    start_time_s: float
    end_time_s: float
    peak_p: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "start_idx": int(self.start_idx),
            "end_idx": int(self.end_idx),
            "start_time_s": float(self.start_time_s),
            "end_time_s": float(self.end_time_s),
            "peak_p": float(self.peak_p),
        }


@dataclass
class EventMetrics:
    # core
    event_recall: float
    false_alerts_per_hour: float
    false_alerts_per_day: float
    mean_delay_s: float
    median_delay_s: float
    # counts
    n_gt_events: int
    n_alert_events: int
    n_matched_gt: int
    n_true_alerts: int
    n_false_alerts: int
    # extra
    event_precision: float
    event_f1: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------- helpers ----------------

def _as_1d_f32(x: Sequence[float] | np.ndarray) -> np.ndarray:
    if isinstance(x, np.ndarray):
        if x.ndim == 1 and x.dtype == np.float32:
            return x
        return np.asarray(x, dtype=np.float32).reshape(-1)
    return np.asarray(x, dtype=np.float32).reshape(-1)


def _as_1d_i32(x: Sequence[int] | np.ndarray) -> np.ndarray:
    if isinstance(x, np.ndarray):
        if x.ndim == 1 and x.dtype == np.int32:
            return x
        return np.asarray(x, dtype=np.int32).reshape(-1)
    return np.asarray(x, dtype=np.int32).reshape(-1)


def _build_confirm_gate_mask(
    n: int,
    lying_score: Optional[np.ndarray],
    motion_score: Optional[np.ndarray],
    *,
    confirm_min_lying: float,
    confirm_max_motion: float,
) -> np.ndarray:
    """Build boolean confirm gate mask from optional lying/motion scores.

    Non-finite score entries are treated as unknown and left permissive.
    """
    gate_ok = np.ones((n,), dtype=bool)
    if lying_score is not None:
        ls_fin = np.isfinite(lying_score)
        gate_ok[ls_fin] &= (lying_score[ls_fin] >= confirm_min_lying)
    if motion_score is not None:
        ms_fin = np.isfinite(motion_score)
        gate_ok[ms_fin] &= (motion_score[ms_fin] <= confirm_max_motion)
    return gate_ok


def ema_smooth(x: np.ndarray, alpha: float) -> np.ndarray:
    x = _as_1d_f32(x)
    if x.size == 0:
        return x
    a = float(alpha)
    if not (0.0 <= a <= 1.0):
        raise ValueError("ema_alpha must be in [0,1]")
    if a == 0.0:
        return x
    y = np.empty_like(x)
    y[0] = x[0]
    for i in range(1, x.size):
        y[i] = a * x[i] + (1.0 - a) * y[i - 1]
    return y


def _ema_smooth_matrix(x: np.ndarray, alpha: float) -> np.ndarray:
    """Row-wise EMA for x[B,T] using a single time loop vectorized over rows."""
    arr = np.asarray(x, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError("x must be 2D [B,T]")
    B, T = arr.shape
    if B == 0 or T == 0:
        return np.zeros((B, T), dtype=np.float32)
    a = float(alpha)
    if not (0.0 <= a <= 1.0):
        raise ValueError("ema_alpha must be in [0,1]")
    if a == 0.0:
        return arr.copy()
    y = np.empty_like(arr)
    y[:, 0] = arr[:, 0]
    one_minus = 1.0 - a
    for i in range(1, T):
        y[:, i] = (a * arr[:, i]) + (one_minus * y[:, i - 1])
    return y


def _ema_precompute_by_groups(
    probs: np.ndarray,
    groups: List[Tuple[str, np.ndarray]],
    alpha: float,
) -> Dict[str, np.ndarray]:
    """Precompute EMA per video, batching videos with identical lengths."""
    out: Dict[str, np.ndarray] = {}
    if not groups:
        return out
    buckets: Dict[int, List[Tuple[str, np.ndarray]]] = {}
    for v, idx in groups:
        L = int(idx.size)
        if L < 1:
            continue
        buckets.setdefault(L, []).append((v, idx))

    for L, items in buckets.items():
        mat = np.empty((len(items), L), dtype=np.float32)
        for r, (_v, idx) in enumerate(items):
            mat[r] = probs[idx]
        sm = _ema_smooth_matrix(mat, alpha)
        for r, (v, _idx) in enumerate(items):
            out[v] = sm[r]
    return out


def _k_of_n(high01: np.ndarray, k: int, n: int) -> np.ndarray:
    """Return a boolean array where index i is True if >=k of the last n samples are 1."""
    if isinstance(high01, np.ndarray) and high01.ndim == 1:
        pass
    else:
        high01 = np.asarray(high01).reshape(-1)
    k = int(k); n = int(n)
    if high01.size == 0:
        return high01.astype(bool, copy=False)
    if n <= 1:
        return (high01 >= max(1, k)).astype(bool, copy=False)
    n = min(n, high01.size)
    # Window sums for trailing n using one cumulative pass.
    csum = np.cumsum(high01, dtype=np.int32)
    win = np.empty_like(csum)
    win[:n] = csum[:n]
    win[n:] = csum[n:] - csum[:-n]
    return (win >= max(1, k))


def _k_of_n_matrix(high01: np.ndarray, k: int, n: int) -> np.ndarray:
    """Row-wise k-of-n for 2D boolean/int arrays shaped [B,T]."""
    x = np.asarray(high01)
    if x.ndim != 2:
        raise ValueError("high01 must be 2D [B,T]")
    B, T = x.shape
    if T == 0:
        return np.zeros((B, 0), dtype=bool)
    k = int(k)
    n = int(n)
    if n <= 1:
        return (x >= max(1, k)).astype(bool, copy=False)
    n = min(n, T)
    # Let cumsum cast directly to int32 to avoid an explicit astype() temporary.
    csum = np.cumsum(x, axis=1, dtype=np.int32)
    win = np.empty_like(csum)
    win[:, :n] = csum[:, :n]
    win[:, n:] = csum[:, n:] - csum[:, :-n]
    return win >= max(1, k)


def _build_video_groups(vids: np.ndarray, ws: np.ndarray) -> List[Tuple[str, np.ndarray]]:
    """Precompute per-video, w_start-sorted indices once for sweep loops."""
    vids_arr = np.asarray(vids).reshape(-1)
    ws_arr = np.asarray(ws)
    if vids_arr.size < 1:
        return []
    if ws_arr.size != vids_arr.size:
        raise ValueError("vids and ws must have the same length")
    # Fast path: already grouped by video (no reappearing IDs) and each group
    # has non-decreasing w_start. This is common for pre-windowed datasets and
    # avoids the global argsort over object/string arrays.
    cuts_fast = np.flatnonzero(vids_arr[1:] != vids_arr[:-1]) + 1
    starts_fast = np.empty((cuts_fast.size + 1,), dtype=np.int64)
    starts_fast[0] = 0
    starts_fast[1:] = cuts_fast
    ends_fast = np.empty((cuts_fast.size + 1,), dtype=np.int64)
    ends_fast[:-1] = cuts_fast
    ends_fast[-1] = vids_arr.size

    seen_videos = set()
    fast_ok = True
    for a, b in zip(starts_fast, ends_fast):
        v = vids_arr[a]
        if v in seen_videos:
            fast_ok = False
            break
        seen_videos.add(v)
        ws_g = ws_arr[a:b]
        if ws_g.size > 1 and not bool(np.all(ws_g[1:] >= ws_g[:-1])):
            fast_ok = False
            break
    if fast_ok:
        out_fast: List[Tuple[str, np.ndarray]] = []
        idx_all = np.arange(vids_arr.size, dtype=np.int64)
        for a, b in zip(starts_fast, ends_fast):
            out_fast.append((str(vids_arr[a]), idx_all[a:b]))
        return out_fast

    # General path: group globally and then restore first-seen video ordering.
    order = np.argsort(vids_arr, kind="mergesort")
    vids_sorted = vids_arr[order]
    cuts = np.flatnonzero(vids_sorted[1:] != vids_sorted[:-1]) + 1
    starts = np.empty((cuts.size + 1,), dtype=np.int64)
    starts[0] = 0
    starts[1:] = cuts
    ends = np.empty((cuts.size + 1,), dtype=np.int64)
    ends[:-1] = cuts
    ends[-1] = order.size

    groups_with_pos: List[Tuple[int, str, np.ndarray]] = []
    for a, b in zip(starts, ends):
        idx = order[a:b]
        # Stable mergesort keeps equal-video entries in original order.
        first_pos = int(idx[0])
        ws_g = ws_arr[idx]
        if ws_g.size > 1 and not bool(np.all(ws_g[1:] >= ws_g[:-1])):
            sort_idx = np.argsort(ws_g, kind="mergesort")
            idx = idx[sort_idx]
        groups_with_pos.append((first_pos, str(vids_sorted[a]), idx))

    groups_with_pos.sort(key=lambda x: x[0])
    return [(v, idx) for _p, v, idx in groups_with_pos]


def _precompute_video_threshold_items(
    ps_v: np.ndarray,
    t_v: np.ndarray,
    thr_values: np.ndarray,
    tau_low_values: np.ndarray,
    *,
    k: int,
    n: int,
    use_confirm: bool,
    cooldown_s: float,
    low_mask_max_elems: int,
    compute_non_confirm_counts: bool,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    List[Optional[np.ndarray]],
    Sequence[Optional[np.ndarray]],
    List[Optional[np.ndarray]],
    Sequence[Optional[np.ndarray]],
    Sequence[Optional[np.ndarray]],
    Optional[np.ndarray],
]:
    """Precompute per-threshold persistence/low-index caches for one video."""
    n_thr = int(thr_values.size)
    T_v = int(ps_v.size)
    pers_template = np.zeros((T_v,), dtype=bool)

    n_cand = int(np.searchsorted(thr_values, np.max(ps_v), side="right")) if T_v > 0 else 0
    active_thr = np.asarray([], dtype=np.intp)
    if n_cand > 0:
        high_by_thr = (ps_v[None, :] >= thr_values[:n_cand, None])
        pers_c = _k_of_n_matrix(high_by_thr, k, n)
        pers_any_by_thr = np.any(pers_c, axis=1)
        active_thr = np.flatnonzero(pers_any_by_thr)

    low_by_thr = None
    if n_cand > 0 and (n_cand * T_v) <= low_mask_max_elems:
        low_by_thr = (ps_v[None, :] < tau_low_values[:n_cand, None])

    pers_idx_by_thr: List[Optional[np.ndarray]] = [None] * n_thr
    none_by_thr: Tuple[Optional[np.ndarray], ...] = (None,) * n_thr
    if use_confirm:
        pers_t_by_thr: Sequence[Optional[np.ndarray]] = none_by_thr
        end_for_pers_by_thr: Sequence[Optional[np.ndarray]] = none_by_thr
        has_low_after_by_thr: Sequence[Optional[np.ndarray]] = none_by_thr
    else:
        pers_t_by_thr = [None] * n_thr
        end_for_pers_by_thr = [None] * n_thr
        has_low_after_by_thr = [None] * n_thr
    low_idx_by_thr: List[Optional[np.ndarray]] = [None] * n_thr

    for ii_raw in active_thr:
        ii = ii_raw
        pers_idx_i = np.flatnonzero(pers_c[ii])
        if low_by_thr is not None:
            low_idx_i = np.flatnonzero(low_by_thr[ii])
        else:
            low_idx_i = np.flatnonzero(ps_v < tau_low_values[ii])
        pers_idx_by_thr[ii] = pers_idx_i
        if not use_confirm:
            # Active thresholds are defined by at least one persistence hit.
            pers_t_by_thr[ii] = t_v[pers_idx_i]  # type: ignore[index]
        low_idx_by_thr[ii] = low_idx_i
        if (not use_confirm) and pers_idx_i.shape[0] > 1 and low_idx_i.shape[0] > 0:
            low_ptr_i = low_idx_i.searchsorted(pers_idx_i, side="left")
            has_low_i = (low_ptr_i < low_idx_i.size)
            safe_ptr_i = np.minimum(low_ptr_i, max(0, low_idx_i.size - 1))
            end_for_pers_by_thr[ii] = np.where(  # type: ignore[index]
                has_low_i,
                low_idx_i[safe_ptr_i],
                (ps_v.size - 1),
            )
            has_low_after_by_thr[ii] = has_low_i  # type: ignore[index]

    counts_by_thr = None
    if compute_non_confirm_counts and (not use_confirm):
        counts = np.zeros((n_thr,), dtype=np.int32)
        for ii in active_thr:
            pers_idx_i = pers_idx_by_thr[ii]
            pers_t_i = pers_t_by_thr[ii]  # type: ignore[index]
            low_idx_i = low_idx_by_thr[ii]
            end_for_pers_i = end_for_pers_by_thr[ii]  # type: ignore[index]
            has_low_after_i = has_low_after_by_thr[ii]  # type: ignore[index]
            n_pers_i = pers_idx_i.shape[0]
            if n_pers_i == 1:
                counts[ii] = 1
                continue
            n_low_i = low_idx_i.shape[0]
            if n_low_i == 0:
                counts[ii] = 1
                continue
            counts[ii] = _count_non_confirm_segments_from_pers(
                ps_v,
                t_v,
                tau_low=tau_low_values[ii],
                cooldown_s=cooldown_s,
                pers=pers_template,
                low_idx=low_idx_i,
                pers_idx=pers_idx_i,
                pers_t=pers_t_i,
                end_for_pers=end_for_pers_i,
                has_low_after=has_low_after_i,
                assume_idx_prepared=True,
            )
        counts_by_thr = counts

    return (
        pers_template,
        active_thr,
        pers_idx_by_thr,
        pers_t_by_thr,
        low_idx_by_thr,
        end_for_pers_by_thr,
        has_low_after_by_thr,
        counts_by_thr,
    )


def times_from_windows(ws: Sequence[int], we: Sequence[int], fps: float, *, mode: str = "center") -> np.ndarray:
    """Convert window frame indices to timestamps.

    mode:
      - "start": w_start / fps
      - "end":   w_end / fps
      - "center": (w_start + w_end)/2 / fps
    """
    ws = np.asarray(ws, dtype=np.float32).reshape(-1)
    we = np.asarray(we, dtype=np.float32).reshape(-1)
    f = float(fps)
    if not (f > 0.0):
        f = 30.0
    if mode == "start":
        return ws / f
    if mode == "end":
        return we / f
    return (ws + we) * 0.5 / f


def _robust_video_fps(fps_slice: np.ndarray, fps_default: float) -> float:
    """Fast path for common constant-FPS windows; median fallback for mixed FPS."""
    fps_default_f = float(fps_default)
    fps_arr = np.asarray(fps_slice, dtype=np.float32).reshape(-1)
    if fps_arr.size < 1:
        return fps_default_f
    fin = np.isfinite(fps_arr)
    if not fin.any():
        return fps_default_f
    fps_fin = fps_arr[fin]
    f0 = float(fps_fin[0])
    if fps_fin.size == 1:
        return f0 if f0 > 0 else fps_default_f
    if np.all(np.abs(fps_fin - f0) <= 1e-6):
        return f0 if f0 > 0 else fps_default_f
    fps_v = float(np.median(fps_fin))
    return fps_v if fps_v > 0 else fps_default_f


def _robust_time_step_s(t: np.ndarray) -> Optional[float]:
    """Estimate typical step in a sorted timestamp vector with fast constant-step path."""
    tt = np.asarray(t, dtype=np.float32).reshape(-1)
    if tt.size < 2:
        return None
    d = np.diff(tt)
    if d.size == 0:
        return None
    d0 = float(d[0])
    if d.size == 1:
        return d0
    # Float32 timestamp construction can introduce tiny drift; keep tolerance
    # loose enough to preserve the constant-step fast path.
    tol = max(1e-6, 1e-4 * max(1.0, abs(d0)))
    if np.all(np.abs(d - d0) <= tol):
        return d0
    return float(np.median(d))


# ---------------- alert detection ----------------

def detect_alert_events(
    probs: Sequence[float],
    times_s: Sequence[float],
    cfg: AlertCfg,
    *,
    lying_score: Optional[Sequence[float]] = None,
    motion_score: Optional[Sequence[float]] = None,
) -> Tuple[np.ndarray, List[AlertEvent]]:
    """Run alert policy on a probability sequence.

    Returns:
      active_mask: bool array where True indicates "ALERT active".
      events: list of AlertEvent intervals.

    Two thresholds are used:
      - tau_high: START threshold (after k-of-n)
      - tau_low:  END threshold (hysteresis)

    Optional confirm stage:
      If cfg.confirm is True, an alert only becomes active if a confirmation
      condition is met within cfg.confirm_s seconds after the k-of-n trigger.
      If lying_score/motion_score are provided (both optional), we use them as
      heuristics; otherwise we fall back to a probability-only confirmation.
    """
    p = np.asarray(probs, dtype=np.float32).reshape(-1)
    if p.size == 0:
        return np.asarray([], dtype=bool), []
    ps = ema_smooth(p, cfg.ema_alpha)
    return _detect_alert_events_from_smoothed(
        ps,
        times_s,
        cfg,
        lying_score=lying_score,
        motion_score=motion_score,
    )


def detect_alert_events_from_smoothed(
    ps: Sequence[float],
    times_s: Sequence[float],
    cfg: AlertCfg,
    *,
    lying_score: Optional[Sequence[float]] = None,
    motion_score: Optional[Sequence[float]] = None,
) -> Tuple[np.ndarray, List[AlertEvent]]:
    """Run alert policy given pre-smoothed probabilities."""
    return _detect_alert_events_from_smoothed(
        ps,
        times_s,
        cfg,
        lying_score=lying_score,
        motion_score=motion_score,
    )


def _detect_alert_events_from_smoothed(
    ps: np.ndarray,
    times_s: Sequence[float],
    cfg: AlertCfg,
    *,
    lying_score: Optional[Sequence[float]] = None,
    motion_score: Optional[Sequence[float]] = None,
    tau_high_override: Optional[float] = None,
    tau_low_override: Optional[float] = None,
    pers_override: Optional[Sequence[bool]] = None,
    return_active: bool = True,
    prepared_inputs: bool = False,
    gate_ok_override: Optional[np.ndarray] = None,
) -> Tuple[Optional[np.ndarray], List[AlertEvent]]:
    """Run alert policy using pre-smoothed probabilities."""
    if prepared_inputs:
        ps = np.asarray(ps)
        t = np.asarray(times_s)
    else:
        ps = _as_1d_f32(ps)
        t = _as_1d_f32(times_s)
    if ps.size != t.size:
        raise ValueError("ps and times_s must have the same length")
    if ps.size == 0:
        return np.asarray([], dtype=bool), []

    tau_low = float(cfg.tau_low if tau_low_override is None else tau_low_override)
    tau_high = float(cfg.tau_high if tau_high_override is None else tau_high_override)
    use_confirm = bool(cfg.confirm)

    if pers_override is None:
        # Fast path: no sample can trigger a start condition.
        if float(np.max(ps)) < tau_high:
            return (np.zeros(ps.shape, dtype=bool) if return_active else None), []
        # persistence trigger uses tau_high
        high = (ps >= tau_high)
        pers = _k_of_n(high, cfg.k, cfg.n)
    else:
        pers = np.asarray(pers_override, dtype=bool).reshape(-1)
        if pers.size != ps.size:
            raise ValueError("pers_override must have same length as ps")
        if not pers.any():
            return (np.zeros(ps.shape, dtype=bool) if return_active else None), []

    active = np.zeros_like(pers, dtype=bool) if return_active else None
    events: List[AlertEvent] = []

    cooldown_until = -1e9  # time (seconds)
    in_event = False
    start_i = 0
    peak = 0.0

    # confirmation state
    pending = False
    pending_start_i = 0
    pending_deadline_s = -1e9
    pending_peak = 0.0

    # If confirm_require_low is enabled, we only allow a new start when "armed".
    # We (re)arm once the smoothed prob goes below tau_low.
    armed = True
    cooldown_s = float(cfg.cooldown_s)

    if not use_confirm:
        return _detect_events_non_confirm_from_pers(
            ps,
            t,
            tau_low=tau_low,
            cooldown_s=cooldown_s,
            pers=pers,
            return_active=return_active,
        )

    ls = None
    ms = None
    gate_ok = None
    has_confirm_scores = False
    if use_confirm and gate_ok_override is not None:
        gate_ok_arr = np.asarray(gate_ok_override, dtype=bool).reshape(-1)
        if gate_ok_arr.size != ps.size:
            raise ValueError("gate_ok_override must have same length as ps")
        gate_ok = gate_ok_arr
        has_confirm_scores = True
    elif use_confirm:
        if lying_score is not None:
            ls = np.asarray(lying_score) if prepared_inputs else _as_1d_f32(lying_score)
        if motion_score is not None:
            ms = np.asarray(motion_score) if prepared_inputs else _as_1d_f32(motion_score)
        has_confirm_scores = (ls is not None or ms is not None)
    require_low = has_confirm_scores and bool(cfg.confirm_require_low)
    confirm_min_lying = float(cfg.confirm_min_lying)
    confirm_max_motion = float(cfg.confirm_max_motion)
    confirm_s = float(cfg.confirm_s)
    if has_confirm_scores and gate_ok is None:
        gate_ok = _build_confirm_gate_mask(
            ps.size,
            ls,
            ms,
            confirm_min_lying=confirm_min_lying,
            confirm_max_motion=confirm_max_motion,
        )

    if active is None:
        return None, _detect_events_confirm_no_active_from_pers(
            ps,
            t,
            tau_low=tau_low,
            cooldown_s=cooldown_s,
            pers=pers,
            confirm_s=confirm_s,
            require_low=require_low,
            has_confirm_scores=has_confirm_scores,
            gate_ok=gate_ok,
        )

    for i in range(pers.size):
        ti = float(t[i])
        ps_i = float(ps[i])

        # Re-arm when we are clearly below tau_low (hysteresis low band).
        if require_low and ps_i < tau_low:
            armed = True

        if in_event:
            # stay active while ps >= tau_low
            if active is not None:
                active[i] = True
            if ps_i > peak:
                peak = ps_i
            if ps_i < tau_low:
                # end event
                in_event = False
                end_i = i
                events.append(
                    AlertEvent(
                        start_idx=int(start_i),
                        end_idx=int(end_i),
                        start_time_s=float(t[start_i]),
                        end_time_s=float(t[end_i]),
                        peak_p=float(peak),
                    )
                )
                cooldown_until = float(t[end_i]) + cooldown_s
            continue

        # confirmation pending (candidate alert start)
        if pending:
            # We are in the uncertain band: no hard alert yet.
            # Promote to a real alert if confirm condition satisfied.
            if ps_i > pending_peak:
                pending_peak = ps_i

            # If we drop below tau_low, abandon the pending start.
            if ps_i < tau_low:
                pending = False
                if require_low:
                    armed = True
                continue

            # Timeout: if we didn't confirm in time, keep it as uncertain (no alert).
            if ti > pending_deadline_s:
                pending = False
                continue

            ok_prob = ps_i >= tau_low
            if ok_prob and (gate_ok[i] if gate_ok is not None else True):
                # Promote pending -> event
                in_event = True
                pending = False
                start_i = int(pending_start_i)
                peak = float(pending_peak)
                if active is not None:
                    active[i] = True
            continue

        # not in event: can we start?
        if ti < cooldown_until:
            continue

        if pers[i]:
            # Optional "require low": only allow a start when we are re-armed.
            # This prevents immediate re-triggers while the score is still high, but does NOT
            # accidentally suppress all starts when using k-of-n persistence.
            if require_low:
                if not armed:
                    continue

            if has_confirm_scores:
                armed = False
                pending = True
                pending_start_i = i
                pending_deadline_s = float(t[i]) + confirm_s
                pending_peak = ps_i
                continue

            in_event = True
            armed = False
            start_i = i
            peak = ps_i
            if active is not None:
                active[i] = True

    if in_event:
        end_i = ps.size - 1
        events.append(
            AlertEvent(
                start_idx=int(start_i),
                end_idx=int(end_i),
                start_time_s=float(t[start_i]),
                end_time_s=float(t[end_i]),
                peak_p=float(peak),
            )
        )

    return active, events


def _detect_events_non_confirm_from_pers(
    ps: np.ndarray,
    t: np.ndarray,
    *,
    tau_low: float,
    cooldown_s: float,
    pers: np.ndarray,
    return_active: bool,
    low_idx: Optional[np.ndarray] = None,
) -> Tuple[Optional[np.ndarray], List[AlertEvent]]:
    """Fast detector for the confirm=False case using precomputed persistence."""
    starts, ends = _detect_non_confirm_segments_from_pers(
        ps,
        t,
        tau_low=tau_low,
        cooldown_s=cooldown_s,
        pers=pers,
        low_idx=low_idx,
    )
    active = np.zeros_like(pers, dtype=bool) if return_active else None
    events: List[AlertEvent] = []
    if starts.size == 0:
        return active, events
    if active is not None:
        for i, j in zip(starts, ends):
            active[int(i) : int(j) + 1] = True
    for i, j in zip(starts, ends):
        ii = int(i)
        jj = int(j)
        peak = float(np.max(ps[ii : jj + 1]))
        events.append(
            AlertEvent(
                start_idx=ii,
                end_idx=jj,
                start_time_s=float(t[ii]),
                end_time_s=float(t[jj]),
                peak_p=peak,
            )
        )
    return active, events


def _detect_events_confirm_no_active_from_pers(
    ps: np.ndarray,
    t: np.ndarray,
    *,
    tau_low: float,
    cooldown_s: float,
    pers: np.ndarray,
    confirm_s: float,
    require_low: bool,
    has_confirm_scores: bool,
    gate_ok: Optional[np.ndarray],
) -> List[AlertEvent]:
    """Confirm-path detector optimized for return_active=False."""
    events: List[AlertEvent] = []
    cooldown_until = -1e9
    in_event = False
    start_i = 0
    peak = 0.0

    pending = False
    pending_start_i = 0
    pending_deadline_s = -1e9
    pending_peak = 0.0
    armed = True

    for i in range(pers.size):
        ti = float(t[i])
        ps_i = float(ps[i])

        if require_low and ps_i < tau_low:
            armed = True

        if in_event:
            if ps_i > peak:
                peak = ps_i
            if ps_i < tau_low:
                in_event = False
                end_i = i
                events.append(
                    AlertEvent(
                        start_idx=int(start_i),
                        end_idx=int(end_i),
                        start_time_s=float(t[start_i]),
                        end_time_s=float(t[end_i]),
                        peak_p=float(peak),
                    )
                )
                cooldown_until = float(t[end_i]) + cooldown_s
            continue

        if pending:
            if ps_i > pending_peak:
                pending_peak = ps_i
            if ps_i < tau_low:
                pending = False
                if require_low:
                    armed = True
                continue
            if ti > pending_deadline_s:
                pending = False
                continue
            if ps_i >= tau_low and (gate_ok[i] if gate_ok is not None else True):
                in_event = True
                pending = False
                start_i = int(pending_start_i)
                peak = float(pending_peak)
            continue

        if ti < cooldown_until:
            continue
        if not pers[i]:
            continue

        if require_low and (not armed):
            continue

        if has_confirm_scores:
            armed = False
            pending = True
            pending_start_i = int(i)
            pending_deadline_s = float(t[i]) + confirm_s
            pending_peak = ps_i
            continue

        in_event = True
        armed = False
        start_i = int(i)
        peak = ps_i

    if in_event:
        end_i = ps.size - 1
        events.append(
            AlertEvent(
                start_idx=int(start_i),
                end_idx=int(end_i),
                start_time_s=float(t[start_i]),
                end_time_s=float(t[end_i]),
                peak_p=float(peak),
            )
        )
    return events


def _detect_confirm_segments_no_active_from_pers(
    ps: np.ndarray,
    t: np.ndarray,
    *,
    tau_low: float,
    cooldown_s: float,
    pers: np.ndarray,
    confirm_s: float,
    require_low: bool,
    has_confirm_scores: bool,
    gate_ok: Optional[np.ndarray],
    gate_idx: Optional[np.ndarray] = None,
    confirm_deadline_idx: Optional[np.ndarray] = None,
    cooldown_after_idx: Optional[np.ndarray] = None,
    low_idx: Optional[np.ndarray] = None,
    pers_idx: Optional[np.ndarray] = None,
    assume_idx_prepared: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Confirm-path detector optimized for sweep metrics (returns start/end indices)."""
    n = int(ps.size)
    if n == 0 or pers.size != n:
        return np.asarray([], dtype=np.int32), np.asarray([], dtype=np.int32)
    if not bool(has_confirm_scores):
        return _detect_non_confirm_segments_from_pers(
            ps,
            t,
            tau_low=tau_low,
            cooldown_s=cooldown_s,
            pers=pers,
            low_idx=low_idx,
            pers_idx=pers_idx,
            assume_idx_prepared=assume_idx_prepared,
        )

    cooldown_idx = 0
    in_event = False
    start_i = 0

    pending = False
    pending_start_i = 0
    pending_deadline_idx = n
    armed = True
    if low_idx is None:
        low_idx = np.flatnonzero(ps < float(tau_low))
    elif not assume_idx_prepared:
        if not (isinstance(low_idx, np.ndarray) and np.issubdtype(low_idx.dtype, np.integer) and low_idx.ndim == 1):
            low_idx = np.asarray(low_idx, dtype=np.int64).reshape(-1)
    if pers_idx is None:
        pers_idx = np.flatnonzero(pers)
    elif not assume_idx_prepared:
        if not (isinstance(pers_idx, np.ndarray) and np.issubdtype(pers_idx.dtype, np.integer) and pers_idx.ndim == 1):
            pers_idx = np.asarray(pers_idx, dtype=np.int64).reshape(-1)
    n_low = int(low_idx.size)
    n_pers = int(pers_idx.size)
    if n_pers == 0:
        return np.asarray([], dtype=np.int32), np.asarray([], dtype=np.int32)

    # Number of confirmed segments cannot exceed number of persistence candidates.
    starts = np.empty((n_pers,), dtype=np.int32)
    ends = np.empty((n_pers,), dtype=np.int32)
    k_out = 0

    gate_needed = gate_ok is not None
    gate_ok_arr = gate_ok
    gate_idx_arr = gate_idx
    n_gate = int(gate_idx_arr.size) if isinstance(gate_idx_arr, np.ndarray) else 0
    if gate_needed:
        if isinstance(gate_idx_arr, np.ndarray):
            if n_gate == 0:
                return np.asarray([], dtype=np.int32), np.asarray([], dtype=np.int32)
        elif not np.any(gate_ok_arr):
            return np.asarray([], dtype=np.int32), np.asarray([], dtype=np.int32)
    t_search = t.searchsorted
    pers_search = pers_idx.searchsorted
    low_search = low_idx.searchsorted
    low_ptr = 0
    pers_ptr = 0
    gate_ptr = 0
    deadline_idx_arr = confirm_deadline_idx
    cooldown_idx_arr = cooldown_after_idx
    if not assume_idx_prepared:
        if deadline_idx_arr is not None:
            if not (
                isinstance(deadline_idx_arr, np.ndarray)
                and np.issubdtype(deadline_idx_arr.dtype, np.integer)
                and deadline_idx_arr.ndim == 1
                and int(deadline_idx_arr.size) == n
            ):
                deadline_idx_arr = None
        if cooldown_idx_arr is not None:
            if not (
                isinstance(cooldown_idx_arr, np.ndarray)
                and np.issubdtype(cooldown_idx_arr.dtype, np.integer)
                and cooldown_idx_arr.ndim == 1
                and int(cooldown_idx_arr.size) == n
            ):
                cooldown_idx_arr = None

    i = 0
    while i < n:
        ps_i = ps[i]

        if require_low and ps_i < tau_low:
            armed = True

        if in_event:
            if ps_i < tau_low:
                in_event = False
                starts[k_out] = start_i
                ends[k_out] = i
                k_out += 1
                if cooldown_idx_arr is not None:
                    cooldown_idx = int(cooldown_idx_arr[i])
                else:
                    cooldown_idx = int(t_search(t[i] + cooldown_s, side="left"))
                i += 1
                continue

            # Event is active and still above tau_low: jump directly to next
            # low-threshold crossing rather than scanning frame-by-frame.
            low_ptr = low_search(i, side="right")
            if low_ptr >= n_low:
                break
            end_i = low_idx[low_ptr]
            if require_low:
                armed = True
            in_event = False
            starts[k_out] = start_i
            ends[k_out] = end_i
            k_out += 1
            if cooldown_idx_arr is not None:
                cooldown_idx = int(cooldown_idx_arr[end_i])
            else:
                cooldown_idx = int(t_search(t[end_i] + cooldown_s, side="left"))
            i = end_i + 1
            continue

        if pending:
            if ps_i < tau_low:
                pending = False
                if require_low:
                    armed = True
                i += 1
                continue
            if i >= pending_deadline_idx:
                pending = False
                i += 1
                continue
            if ps_i >= tau_low:
                if gate_needed:
                    if gate_ok_arr[i]:
                        in_event = True
                        pending = False
                        start_i = pending_start_i
                        i += 1
                        continue
                    if n_gate > 0:
                        while gate_ptr < n_gate and gate_idx_arr[gate_ptr] <= i:
                            gate_ptr += 1
                        next_gate = int(gate_idx_arr[gate_ptr]) if gate_ptr < n_gate else n
                        lp = int(low_search(i + 1, side="left"))
                        next_low = int(low_idx[lp]) if lp < n_low else n
                        next_deadline = int(pending_deadline_idx)
                        j = min(next_gate, next_low, next_deadline)
                        i = (j if j > i else i + 1)
                        continue
                else:
                    in_event = True
                    pending = False
                    start_i = pending_start_i
                    i += 1
                    continue
            i += 1
            continue

        if i < cooldown_idx:
            i = cooldown_idx
            continue

        # Jump to the next persistence candidate; skip non-trigger frames.
        pers_ptr = pers_search(i, side="left")
        if pers_ptr >= n_pers:
            break
        i = pers_idx[pers_ptr]
        ps_i = ps[i]
        if require_low and ps_i < tau_low:
            armed = True

        if i < cooldown_idx:
            i = cooldown_idx
            continue

        if require_low and (not armed):
            low_ptr = low_search(i, side="right")
            if low_ptr >= n_low:
                break
            i = low_idx[low_ptr]
            continue
        # has_confirm_scores is guaranteed True here (False path returns early).
        armed = False
        pending = True
        pending_start_i = i
        if deadline_idx_arr is not None:
            pending_deadline_idx = int(deadline_idx_arr[i])
        else:
            pending_deadline_idx = int(t_search(t[i] + confirm_s, side="right"))
        i += 1

    if in_event:
        starts[k_out] = start_i
        ends[k_out] = n - 1
        k_out += 1

    if k_out < 1:
        return np.asarray([], dtype=np.int32), np.asarray([], dtype=np.int32)
    return starts[:k_out], ends[:k_out]


def _count_confirm_segments_no_active_from_pers(
    ps: np.ndarray,
    t: np.ndarray,
    *,
    tau_low: float,
    cooldown_s: float,
    pers: np.ndarray,
    confirm_s: float,
    require_low: bool,
    has_confirm_scores: bool,
    gate_ok: Optional[np.ndarray],
    gate_idx: Optional[np.ndarray] = None,
    confirm_deadline_idx: Optional[np.ndarray] = None,
    cooldown_after_idx: Optional[np.ndarray] = None,
    low_idx: Optional[np.ndarray] = None,
    pers_idx: Optional[np.ndarray] = None,
    assume_idx_prepared: bool = False,
) -> int:
    """Count confirm-path alert segments without materializing start/end arrays."""
    n = int(ps.size)
    if n == 0 or pers.size != n:
        return 0
    if not bool(has_confirm_scores):
        return int(
            _count_non_confirm_segments_from_pers(
                ps,
                t,
                tau_low=tau_low,
                cooldown_s=cooldown_s,
                pers=pers,
                low_idx=low_idx,
                pers_idx=pers_idx,
                assume_idx_prepared=assume_idx_prepared,
            )
        )

    cooldown_idx = 0
    in_event = False

    pending = False
    pending_deadline_idx = n
    armed = True
    if low_idx is None:
        low_idx = np.flatnonzero(ps < float(tau_low))
    elif not assume_idx_prepared:
        if not (isinstance(low_idx, np.ndarray) and np.issubdtype(low_idx.dtype, np.integer) and low_idx.ndim == 1):
            low_idx = np.asarray(low_idx, dtype=np.int64).reshape(-1)
    if pers_idx is None:
        pers_idx = np.flatnonzero(pers)
    elif not assume_idx_prepared:
        if not (isinstance(pers_idx, np.ndarray) and np.issubdtype(pers_idx.dtype, np.integer) and pers_idx.ndim == 1):
            pers_idx = np.asarray(pers_idx, dtype=np.int64).reshape(-1)
    n_low = int(low_idx.size)
    n_pers = int(pers_idx.size)
    if n_pers == 0:
        return 0

    gate_needed = gate_ok is not None
    gate_ok_arr = gate_ok
    gate_idx_arr = gate_idx
    n_gate = int(gate_idx_arr.size) if isinstance(gate_idx_arr, np.ndarray) else 0
    if gate_needed:
        if isinstance(gate_idx_arr, np.ndarray):
            if n_gate == 0:
                return 0
        elif not np.any(gate_ok_arr):
            return 0
    t_search = t.searchsorted
    pers_search = pers_idx.searchsorted
    low_search = low_idx.searchsorted
    low_ptr = 0
    pers_ptr = 0
    gate_ptr = 0
    deadline_idx_arr = confirm_deadline_idx
    cooldown_idx_arr = cooldown_after_idx
    if not assume_idx_prepared:
        if deadline_idx_arr is not None:
            if not (
                isinstance(deadline_idx_arr, np.ndarray)
                and np.issubdtype(deadline_idx_arr.dtype, np.integer)
                and deadline_idx_arr.ndim == 1
                and int(deadline_idx_arr.size) == n
            ):
                deadline_idx_arr = None
        if cooldown_idx_arr is not None:
            if not (
                isinstance(cooldown_idx_arr, np.ndarray)
                and np.issubdtype(cooldown_idx_arr.dtype, np.integer)
                and cooldown_idx_arr.ndim == 1
                and int(cooldown_idx_arr.size) == n
            ):
                cooldown_idx_arr = None

    count = 0
    i = 0
    while i < n:
        ps_i = ps[i]

        if require_low and ps_i < tau_low:
            armed = True

        if in_event:
            if ps_i < tau_low:
                in_event = False
                count += 1
                if cooldown_idx_arr is not None:
                    cooldown_idx = int(cooldown_idx_arr[i])
                else:
                    cooldown_idx = int(t_search(t[i] + cooldown_s, side="left"))
                i += 1
                continue

            low_ptr = low_search(i, side="right")
            if low_ptr >= n_low:
                break
            end_i = low_idx[low_ptr]
            if require_low:
                armed = True
            in_event = False
            count += 1
            if cooldown_idx_arr is not None:
                cooldown_idx = int(cooldown_idx_arr[end_i])
            else:
                cooldown_idx = int(t_search(t[end_i] + cooldown_s, side="left"))
            i = end_i + 1
            continue

        if pending:
            if ps_i < tau_low:
                pending = False
                if require_low:
                    armed = True
                i += 1
                continue
            if i >= pending_deadline_idx:
                pending = False
                i += 1
                continue
            if ps_i >= tau_low:
                if gate_needed:
                    if gate_ok_arr[i]:
                        in_event = True
                        pending = False
                        i += 1
                        continue
                    if n_gate > 0:
                        while gate_ptr < n_gate and gate_idx_arr[gate_ptr] <= i:
                            gate_ptr += 1
                        next_gate = int(gate_idx_arr[gate_ptr]) if gate_ptr < n_gate else n
                        lp = int(low_search(i + 1, side="left"))
                        next_low = int(low_idx[lp]) if lp < n_low else n
                        next_deadline = int(pending_deadline_idx)
                        j = min(next_gate, next_low, next_deadline)
                        i = (j if j > i else i + 1)
                        continue
                else:
                    in_event = True
                    pending = False
                    i += 1
                    continue
            i += 1
            continue

        if i < cooldown_idx:
            i = cooldown_idx
            continue

        pers_ptr = pers_search(i, side="left")
        if pers_ptr >= n_pers:
            break
        i = pers_idx[pers_ptr]
        ps_i = ps[i]
        if require_low and ps_i < tau_low:
            armed = True

        if i < cooldown_idx:
            i = cooldown_idx
            continue

        if require_low and (not armed):
            low_ptr = low_search(i, side="right")
            if low_ptr >= n_low:
                break
            i = low_idx[low_ptr]
            continue
        # has_confirm_scores is guaranteed True here (False path returns early).
        armed = False
        pending = True
        if deadline_idx_arr is not None:
            pending_deadline_idx = int(deadline_idx_arr[i])
        else:
            pending_deadline_idx = int(t_search(t[i] + confirm_s, side="right"))
        i += 1

    if in_event:
        count += 1
    return int(count)


def _detect_non_confirm_segments_from_pers(
    ps: np.ndarray,
    t: np.ndarray,
    *,
    tau_low: float,
    cooldown_s: float,
    pers: np.ndarray,
    low_idx: Optional[np.ndarray] = None,
    pers_idx: Optional[np.ndarray] = None,
    pers_t: Optional[np.ndarray] = None,
    end_for_pers: Optional[np.ndarray] = None,
    has_low_after: Optional[np.ndarray] = None,
    assume_idx_prepared: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return alert segments as (start_idx, end_idx) arrays for confirm=False policy."""
    if pers_idx is None:
        pers_idx = np.flatnonzero(pers)
    elif not assume_idx_prepared:
        if not (isinstance(pers_idx, np.ndarray) and np.issubdtype(pers_idx.dtype, np.integer) and pers_idx.ndim == 1):
            pers_idx = np.asarray(pers_idx, dtype=np.int64).reshape(-1)
    if pers_idx.size == 0:
        return np.asarray([], dtype=np.int32), np.asarray([], dtype=np.int32)

    n = int(ps.size)
    if low_idx is None:
        low_idx = np.flatnonzero(ps < float(tau_low))
    elif not assume_idx_prepared:
        if not (isinstance(low_idx, np.ndarray) and np.issubdtype(low_idx.dtype, np.integer) and low_idx.ndim == 1):
            low_idx = np.asarray(low_idx, dtype=np.int64).reshape(-1)
    n_low = int(low_idx.size)
    if n_low == 0:
        return np.asarray([pers_idx[0]], dtype=np.int32), np.asarray([n - 1], dtype=np.int32)
    if pers_idx.size == 1:
        i0 = pers_idx[0]
        lp = int(low_idx.searchsorted(i0, side="left"))
        if lp < n_low:
            return (
                np.asarray([i0], dtype=np.int32),
                np.asarray([low_idx[lp]], dtype=np.int32),
            )
        return np.asarray([i0], dtype=np.int32), np.asarray([n - 1], dtype=np.int32)

    # Precompute, for each persistence candidate, the first low-threshold index
    # at/after it. This removes inner-pointer walks in the hot loop.
    if end_for_pers is None or has_low_after is None:
        low_ptr_for_pers = low_idx.searchsorted(pers_idx, side="left")
        has_low_after = low_ptr_for_pers < n_low
        safe_ptr = np.minimum(low_ptr_for_pers, max(0, n_low - 1))
        end_for_pers = np.where(has_low_after, low_idx[safe_ptr], (n - 1)).astype(np.int32, copy=False)
    elif not assume_idx_prepared:
        end_for_pers = np.asarray(end_for_pers, dtype=np.int32).reshape(-1)
        has_low_after = np.asarray(has_low_after, dtype=bool).reshape(-1)

    starts = np.empty((pers_idx.size,), dtype=np.int32)
    ends = np.empty((pers_idx.size,), dtype=np.int32)
    k_out = 0
    cooldown_until = -1e9
    last_end = -1
    t_arr = t
    if pers_t is None:
        pers_t = t[pers_idx]
    elif not assume_idx_prepared:
        pers_t = np.asarray(pers_t, dtype=np.float32).reshape(-1)
    pers_search = pers_idx.searchsorted
    pers_t_search = pers_t.searchsorted
    p = 0
    n_p = int(pers_idx.size)
    has_low_arr = has_low_after
    end_arr = end_for_pers
    while p < n_p:
        i = pers_idx[p]
        if i <= last_end:
            p = pers_search(last_end + 1, side="left")
            continue
        if t_arr[i] < cooldown_until:
            p = pers_t_search(cooldown_until, side="left")
            continue

        end_i = end_arr[p]
        has_low = has_low_arr[p]
        if has_low:
            cooldown_until = t_arr[end_i] + cooldown_s
        else:
            end_i = n - 1
            cooldown_until = float("inf")

        starts[k_out] = i
        ends[k_out] = end_i
        k_out += 1
        last_end = end_i
        if not has_low:
            break
        p = pers_search(last_end + 1, side="left")

    if k_out < 1:
        return np.asarray([], dtype=np.int32), np.asarray([], dtype=np.int32)
    return starts[:k_out], ends[:k_out]


def _count_non_confirm_segments_from_pers(
    ps: np.ndarray,
    t: np.ndarray,
    *,
    tau_low: float,
    cooldown_s: float,
    pers: np.ndarray,
    low_idx: Optional[np.ndarray] = None,
    pers_idx: Optional[np.ndarray] = None,
    pers_t: Optional[np.ndarray] = None,
    end_for_pers: Optional[np.ndarray] = None,
    has_low_after: Optional[np.ndarray] = None,
    assume_idx_prepared: bool = False,
) -> int:
    """Count alert segments for confirm=False policy without allocating segment arrays."""
    if pers_idx is None:
        pers_idx = np.flatnonzero(pers)
    elif not assume_idx_prepared:
        if not (isinstance(pers_idx, np.ndarray) and np.issubdtype(pers_idx.dtype, np.integer) and pers_idx.ndim == 1):
            pers_idx = np.asarray(pers_idx, dtype=np.int64).reshape(-1)
    if pers_idx.size == 0:
        return 0

    n = int(ps.size)
    if low_idx is None:
        low_idx = np.flatnonzero(ps < float(tau_low))
    elif not assume_idx_prepared:
        if not (isinstance(low_idx, np.ndarray) and np.issubdtype(low_idx.dtype, np.integer) and low_idx.ndim == 1):
            low_idx = np.asarray(low_idx, dtype=np.int64).reshape(-1)
    n_low = int(low_idx.size)
    if n_low == 0:
        return 1
    if pers_idx.size == 1:
        return 1
    if end_for_pers is None or has_low_after is None:
        low_ptr_for_pers = low_idx.searchsorted(pers_idx, side="left")
        has_low_after = low_ptr_for_pers < n_low
        safe_ptr = np.minimum(low_ptr_for_pers, max(0, n_low - 1))
        end_for_pers = np.where(has_low_after, low_idx[safe_ptr], (n - 1)).astype(np.int32, copy=False)
    elif not assume_idx_prepared:
        end_for_pers = np.asarray(end_for_pers, dtype=np.int32).reshape(-1)
        has_low_after = np.asarray(has_low_after, dtype=bool).reshape(-1)

    cooldown_until = -1e9
    last_end = -1
    count = 0
    t_arr = t
    if pers_t is None:
        pers_t = t[pers_idx]
    elif not assume_idx_prepared:
        pers_t = np.asarray(pers_t, dtype=np.float32).reshape(-1)
    pers_search = pers_idx.searchsorted
    pers_t_search = pers_t.searchsorted
    p = 0
    n_p = int(pers_idx.size)
    has_low_arr = has_low_after
    end_arr = end_for_pers
    while p < n_p:
        i = pers_idx[p]
        if i <= last_end:
            p = pers_search(last_end + 1, side="left")
            continue
        if t_arr[i] < cooldown_until:
            p = pers_t_search(cooldown_until, side="left")
            continue

        end_i = end_arr[p]
        has_low = has_low_arr[p]
        if has_low:
            cooldown_until = t_arr[end_i] + cooldown_s
        else:
            end_i = n - 1
            cooldown_until = float("inf")

        count += 1
        last_end = end_i
        if not has_low:
            break
        p = pers_search(last_end + 1, side="left")
    return int(count)


def _event_match_delay_and_alert_stats_times(
    gt_events: List[Tuple[float, float]],
    alert_start_s: np.ndarray,
    alert_end_s: np.ndarray,
    *,
    slack_s: float,
) -> Tuple[int, float, int, int, int]:
    """Like _event_match_delay_and_alert_stats, but consumes alert start/end arrays."""
    if gt_events:
        gt_arr = np.asarray(gt_events, dtype=np.float32).reshape(-1, 2)
        gt_start_s = gt_arr[:, 0]
        gt_end_s = gt_arr[:, 1]
    else:
        gt_start_s = np.asarray([], dtype=np.float32)
        gt_end_s = np.asarray([], dtype=np.float32)
    return _event_match_delay_and_alert_stats_arrays(
        gt_start_s,
        gt_end_s,
        alert_start_s,
        alert_end_s,
        slack_s=slack_s,
    )


def _event_match_delay_and_alert_stats_arrays(
    gt_start_s: np.ndarray,
    gt_end_s: np.ndarray,
    alert_start_s: np.ndarray,
    alert_end_s: np.ndarray,
    *,
    slack_s: float,
) -> Tuple[int, float, int, int, int]:
    """Array-native delay + true/false alert stats on sorted event intervals."""
    gt_start_s = _as_1d_f32(gt_start_s)
    gt_end_s = _as_1d_f32(gt_end_s)
    alert_start_s = _as_1d_f32(alert_start_s)
    alert_end_s = _as_1d_f32(alert_end_s)
    n_alert = int(alert_start_s.size)
    if n_alert == 0:
        return 0, 0.0, 0, 0, 0
    n_gt = int(gt_start_s.size)
    if n_gt == 0:
        return 0, 0.0, 0, 0, n_alert

    slack = float(slack_s)

    matched_gt = 0
    delay_sum = 0.0
    n_delays = 0
    j = 0
    for g_i in range(n_gt):
        gs = float(gt_start_s[g_i])
        ge = float(gt_end_s[g_i])
        while j < n_alert and (alert_end_s[j] + slack) < gs:
            j += 1

        k = j
        first_alert = None
        ge_slack = ge + slack
        while k < n_alert and alert_start_s[k] <= ge_slack:
            as_ = float(alert_start_s[k])
            ae = float(alert_end_s[k])
            if (ae + slack) >= gs and ge_slack >= as_:
                first_alert = as_
                break
            k += 1

        if first_alert is not None:
            matched_gt += 1
            delay_sum += max(0.0, first_alert - gs)
            n_delays += 1

    true_alerts = 0
    false_alerts = 0
    g = 0
    for i in range(n_alert):
        as_ = float(alert_start_s[i])
        ae = float(alert_end_s[i])
        while g < n_gt and (gt_end_s[g] + slack) < as_:
            g += 1
        if g < n_gt:
            gs = float(gt_start_s[g])
            ge = float(gt_end_s[g])
            if (ae + slack) >= gs and (ge + slack) >= as_:
                true_alerts += 1
            else:
                false_alerts += 1
        else:
            false_alerts += 1

    return matched_gt, delay_sum, n_delays, true_alerts, false_alerts


def _event_match_delay_and_alert_stats_indices(
    gt_start_s: np.ndarray,
    gt_end_s: np.ndarray,
    t: np.ndarray,
    start_idx: np.ndarray,
    end_idx: np.ndarray,
    *,
    slack_s: float,
) -> Tuple[int, float, int, int, int]:
    """Array-native stats using alert start/end indices into a shared time vector."""
    gt_start_s = _as_1d_f32(gt_start_s)
    gt_end_s = _as_1d_f32(gt_end_s)
    t = _as_1d_f32(t)
    start_idx = _as_1d_i32(start_idx)
    end_idx = _as_1d_i32(end_idx)
    return _event_match_delay_and_alert_stats_indices_prepared(
        gt_start_s,
        gt_end_s,
        t,
        start_idx,
        end_idx,
        slack_s=float(slack_s),
    )


def _event_match_delay_and_alert_stats_indices_prepared(
    gt_start_s: np.ndarray,
    gt_end_s: np.ndarray,
    t: np.ndarray,
    start_idx: np.ndarray,
    end_idx: np.ndarray,
    *,
    slack_s: float,
    gt_end_slack_pre: Optional[np.ndarray] = None,
) -> Tuple[int, float, int, int, int]:
    """Like _event_match_delay_and_alert_stats_indices but expects normalized 1D arrays."""
    n_alert = int(start_idx.size)
    if n_alert == 0:
        return 0, 0.0, 0, 0, 0
    n_gt = int(gt_start_s.size)
    if n_gt == 0:
        return 0, 0.0, 0, 0, n_alert

    slack = float(slack_s)
    # Fast disjoint-range guard: if alert and GT timelines cannot overlap at all,
    # every alert is false and delay/match stats are zero.
    alert_first_s = t[start_idx[0]]
    alert_last_e = t[end_idx[-1]]
    gt_first_s = gt_start_s[0]
    gt_last_e = gt_end_s[-1]
    if (alert_last_e + slack) < gt_first_s or alert_first_s > (gt_last_e + slack):
        return 0, 0.0, 0, 0, n_alert

    slack32 = np.float32(slack)
    if (
        isinstance(gt_end_slack_pre, np.ndarray)
        and gt_end_slack_pre.ndim == 1
        and int(gt_end_slack_pre.size) == n_gt
    ):
        gt_end_slack = gt_end_slack_pre
    else:
        gt_end_slack = gt_end_s + slack32

    if n_alert == 1:
        a_s = t[start_idx[0]]
        a_e_slack = t[end_idx[0]] + slack32

        g = int(gt_end_slack.searchsorted(a_s, side="left"))
        matched_gt = 0
        delay_sum = 0.0
        n_delays = 0
        if g < n_gt and a_e_slack >= gt_start_s[g] and gt_end_slack[g] >= a_s:
            gs = gt_start_s[g]
            d = a_s - gs
            matched_gt = 1
            delay_sum = d if d > 0.0 else 0.0
            n_delays = 1
            return matched_gt, delay_sum, n_delays, 1, 0
        return matched_gt, delay_sum, n_delays, 0, 1

    alert_start_s = t[start_idx]
    alert_end_s = t[end_idx]
    alert_end_slack = alert_end_s + slack32
    gt_start = gt_start_s
    gt_end_sl = gt_end_slack

    # Small-array fast path: avoid vectorized temporary allocations when this
    # function is called many times on tiny per-video/per-threshold segments.
    if (n_alert * n_gt) <= 512 and n_alert <= 32 and n_gt <= 32:
        as_arr = alert_start_s
        ae_sl_arr = alert_end_slack
        matched_gt = 0
        delay_sum = 0.0
        n_delays = 0
        j = 0
        for g_i in range(n_gt):
            gs = gt_start[g_i]
            ge_sl = gt_end_sl[g_i]
            while j < n_alert and ae_sl_arr[j] < gs:
                j += 1
            k = j
            first_alert = None
            while k < n_alert and as_arr[k] <= ge_sl:
                if ae_sl_arr[k] >= gs:
                    first_alert = as_arr[k]
                    break
                k += 1
            if first_alert is not None:
                matched_gt += 1
                d = first_alert - gs
                delay_sum += float(d) if d > 0.0 else 0.0
                n_delays += 1

        true_alerts = 0
        false_alerts = 0
        g = 0
        for i in range(n_alert):
            as_i = as_arr[i]
            ae_i = ae_sl_arr[i]
            while g < n_gt and gt_end_sl[g] < as_i:
                g += 1
            if g < n_gt and ae_i >= gt_start[g] and gt_end_sl[g] >= as_i:
                true_alerts += 1
            else:
                false_alerts += 1
        return matched_gt, delay_sum, n_delays, true_alerts, false_alerts

    if n_gt == 1:
        gs = gt_start_s[0]
        ge_slack = gt_end_slack[0]

        # GT matched/delay: first alert with end+slack >= gs and start <= ge+slack.
        j = int(alert_end_slack.searchsorted(gs, side="left"))
        if j < n_alert and alert_start_s[j] <= ge_slack:
            matched_gt = 1
            d = alert_start_s[j] - gs
            delay_sum = d if d > 0.0 else 0.0
            n_delays = 1
        else:
            matched_gt = 0
            delay_sum = 0.0
            n_delays = 0

        # True/false alerts against single GT interval.
        true_alerts = int(np.count_nonzero((alert_end_slack >= gs) & (ge_slack >= alert_start_s)))
        false_alerts = int(n_alert - true_alerts)
        return matched_gt, delay_sum, n_delays, true_alerts, false_alerts

    as_arr = alert_start_s
    ae_sl_arr = alert_end_slack
    # First alert candidate per GT: earliest alert with end+slack >= gt_start.
    cand_idx = ae_sl_arr.searchsorted(gt_start, side="left")
    cand_valid = cand_idx < n_alert
    if np.any(cand_valid):
        cand_idx_v = cand_idx[cand_valid]
        gt_start_v = gt_start[cand_valid]
        gt_end_sl_v = gt_end_sl[cand_valid]
        cand_as = as_arr[cand_idx_v]
        overlap_v = cand_as <= gt_end_sl_v
        if np.any(overlap_v):
            matched_gt = int(np.count_nonzero(overlap_v))
            d = cand_as[overlap_v] - gt_start_v[overlap_v]
            np.maximum(d, 0.0, out=d)
            delay_sum = float(np.sum(d, dtype=np.float64)) if d.size > 0 else 0.0
            n_delays = matched_gt
        else:
            matched_gt = 0
            delay_sum = 0.0
            n_delays = 0
    else:
        matched_gt = 0
        delay_sum = 0.0
        n_delays = 0

    # Vectorized true/false alert accounting.
    g_idx = gt_end_sl.searchsorted(as_arr, side="left")
    valid = g_idx < n_gt
    if np.any(valid):
        g_hit = g_idx[valid]
        true_mask = (ae_sl_arr[valid] >= gt_start[g_hit]) & (gt_end_sl[g_hit] >= as_arr[valid])
        true_alerts = int(np.count_nonzero(true_mask))
    else:
        true_alerts = 0

    false_alerts = int(n_alert - true_alerts)
    return matched_gt, delay_sum, n_delays, true_alerts, false_alerts


def classify_states(
    probs: Sequence[float],
    times_s: Sequence[float],
    cfg: AlertCfg,
    *,
    lying_score: Optional[Sequence[float]] = None,
    motion_score: Optional[Sequence[float]] = None,
) -> Dict[str, Any]:
    """Classify each time step into {clear, suspect, alert}.

    We define:
      - alert: the alert policy is active (see detect_alert_events)
      - suspect: not alert, but smoothed prob is in [tau_low, tau_high)
      - clear: smoothed prob < tau_low

    Returns a dict with masks and the smoothed probabilities.
    """
    p = np.asarray(probs, dtype=np.float32).reshape(-1)
    t = np.asarray(times_s, dtype=np.float32).reshape(-1)
    if p.size == 0:
        return {
            "ps": np.asarray([], dtype=np.float32),
            "clear": np.asarray([], dtype=bool),
            "suspect": np.asarray([], dtype=bool),
            "alert": np.asarray([], dtype=bool),
        }

    ps = ema_smooth(p, cfg.ema_alpha)
    alert_mask, _events = _detect_alert_events_from_smoothed(
        ps,
        t,
        cfg,
        lying_score=lying_score,
        motion_score=motion_score,
    )
    suspect = (~alert_mask) & (ps >= float(cfg.tau_low)) & (ps < float(cfg.tau_high))
    clear = (~alert_mask) & (ps < float(cfg.tau_low))
    return {"ps": ps, "clear": clear, "suspect": suspect, "alert": alert_mask}


# ---------------- event metrics ----------------

def _events_from_positive_windows_arrays(
    times_s: np.ndarray,
    y_true: np.ndarray,
    merge_gap_s: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build GT events from y_true=1 windows as parallel start/end arrays."""
    if isinstance(times_s, np.ndarray) and times_s.ndim == 1 and times_s.dtype == np.float32:
        t = times_s
    else:
        t = np.asarray(times_s, dtype=np.float32).reshape(-1)
    if isinstance(y_true, np.ndarray) and y_true.ndim == 1 and y_true.dtype == np.bool_:
        y = y_true
    else:
        y = np.asarray(y_true).reshape(-1) > 0
    if t.size == 0 or y.size == 0:
        empty = np.asarray([], dtype=np.float32)
        return empty, empty
    if t.size != y.size:
        raise ValueError("times_s and y_true must have same length")

    pos = np.flatnonzero(y)
    if pos.size == 0:
        empty = np.asarray([], dtype=np.float32)
        return empty, empty

    tp = t[pos]  # sorted positive timestamps
    if tp.size == 1:
        v = np.asarray([tp[0]], dtype=np.float32)
        return v, v.copy()

    gap = float(merge_gap_s)
    cuts = np.flatnonzero(np.diff(tp) > gap)
    n_seg = int(cuts.size) + 1
    starts = np.empty((n_seg,), dtype=np.int64)
    ends = np.empty((n_seg,), dtype=np.int64)
    starts[0] = 0
    if cuts.size > 0:
        starts[1:] = cuts + 1
        ends[:-1] = cuts
    ends[-1] = tp.size - 1
    return tp[starts], tp[ends]


def _events_from_positive_windows(times_s: np.ndarray, y_true: np.ndarray, merge_gap_s: float) -> List[Tuple[float, float]]:
    """Build GT events from y_true=1 windows. Each GT event is (start_s, end_s)."""
    s_arr, e_arr = _events_from_positive_windows_arrays(times_s, y_true, merge_gap_s)
    if s_arr.size == 0:
        return []
    return [(float(s), float(e)) for s, e in zip(s_arr, e_arr)]


def gt_events_from_windows(
    times_s: Sequence[float],
    y_true: Sequence[int],
    *,
    merge_gap_s: float = 2.0,
) -> List[Tuple[float, float]]:
    """Public helper: derive GT event intervals from window labels/timestamps."""
    if isinstance(times_s, np.ndarray) and times_s.ndim == 1 and times_s.dtype == np.float32:
        t = times_s
    else:
        t = np.asarray(times_s, dtype=np.float32).reshape(-1)
    if isinstance(y_true, np.ndarray) and y_true.ndim == 1 and y_true.dtype == np.int32:
        y = (y_true == 1)
    else:
        y = np.asarray(y_true).reshape(-1) == 1
    if t.size != y.size:
        raise ValueError("times_s and y_true must have same length")
    return _events_from_positive_windows(t, y, merge_gap_s=float(merge_gap_s))


def _overlap(a: Tuple[float, float], b: Tuple[float, float], *, slack_s: float = 0.0) -> bool:
    (as_, ae) = a
    (bs, be) = b
    return (ae + slack_s) >= bs and (be + slack_s) >= as_


def _match_gt_delays(
    gt_events: List[Tuple[float, float]],
    alert_events: List[AlertEvent],
    *,
    slack_s: float,
) -> Tuple[int, List[float]]:
    """Match each GT to earliest overlapping alert in linear time."""
    if not gt_events or not alert_events:
        return 0, []

    matched_gt = 0
    delays: List[float] = []
    j = 0
    n_alert = len(alert_events)

    for gs, ge in gt_events:
        while j < n_alert and ((alert_events[j].end_time_s + slack_s) < gs):
            j += 1

        k = j
        first_alert = None
        while k < n_alert and alert_events[k].start_time_s <= (ge + slack_s):
            ev = alert_events[k]
            if _overlap((gs, ge), (ev.start_time_s, ev.end_time_s), slack_s=slack_s):
                first_alert = ev.start_time_s
                break
            k += 1

        if first_alert is not None:
            matched_gt += 1
            delays.append(max(0.0, first_alert - gs))

    return matched_gt, delays


def _match_gt_delay_stats(
    gt_events: List[Tuple[float, float]],
    alert_events: Sequence[AlertEvent],
    *,
    slack_s: float,
) -> Tuple[int, float, int]:
    """Match each GT to earliest overlapping alert and return delay summary.

    Returns:
      matched_gt, delay_sum_s, n_delays
    """
    if not gt_events or not alert_events:
        return 0, 0.0, 0

    matched_gt = 0
    delay_sum = 0.0
    n_delays = 0
    j = 0
    n_alert = len(alert_events)

    for gs, ge in gt_events:
        while j < n_alert and ((alert_events[j].end_time_s + slack_s) < gs):
            j += 1

        k = j
        first_alert = None
        while k < n_alert and alert_events[k].start_time_s <= (ge + slack_s):
            ev = alert_events[k]
            if _overlap((gs, ge), (ev.start_time_s, ev.end_time_s), slack_s=slack_s):
                first_alert = ev.start_time_s
                break
            k += 1

        if first_alert is not None:
            matched_gt += 1
            delay_sum += max(0.0, first_alert - gs)
            n_delays += 1

    return matched_gt, delay_sum, n_delays


def _event_match_delay_and_alert_stats(
    gt_events: List[Tuple[float, float]],
    alert_events: Sequence[AlertEvent],
    *,
    slack_s: float,
) -> Tuple[int, float, int, int, int]:
    """Compute delay and alert true/false stats.

    Returns:
      matched_gt, delay_sum_s, n_delays, true_alerts, false_alerts
    """
    n_alert = len(alert_events)
    if n_alert == 0:
        return 0, 0.0, 0, 0, 0
    n_gt = len(gt_events)
    if n_gt == 0:
        return 0, 0.0, 0, 0, n_alert

    slack = float(slack_s)

    # Pass 1: GT match + delay stats.
    matched_gt = 0
    delay_sum = 0.0
    n_delays = 0
    j = 0
    for gs, ge in gt_events:
        while j < n_alert and (alert_events[j].end_time_s + slack) < gs:
            j += 1

        k = j
        first_alert = None
        ge_slack = ge + slack
        while k < n_alert and alert_events[k].start_time_s <= ge_slack:
            ev = alert_events[k]
            as_ = ev.start_time_s
            ae = ev.end_time_s
            if (ae + slack) >= gs and (ge_slack >= as_):
                first_alert = as_
                break
            k += 1

        if first_alert is not None:
            matched_gt += 1
            delay_sum += max(0.0, first_alert - gs)
            n_delays += 1

    # Pass 2: true/false alert counts.
    true_alerts = 0
    false_alerts = 0
    g = 0
    for ev in alert_events:
        as_ = ev.start_time_s
        ae = ev.end_time_s
        while g < n_gt and (gt_events[g][1] + slack) < as_:
            g += 1
        if g < n_gt:
            gs, ge = gt_events[g]
            if (ae + slack) >= gs and (ge + slack) >= as_:
                true_alerts += 1
            else:
                false_alerts += 1
        else:
            false_alerts += 1

    return matched_gt, delay_sum, n_delays, true_alerts, false_alerts


def _count_true_false_alerts(
    alert_intervals: List[Tuple[float, float]],
    gt_events: List[Tuple[float, float]],
    *,
    slack_s: float,
) -> Tuple[int, int]:
    """Count true/false alerts via linear scan on sorted intervals."""
    if not alert_intervals:
        return 0, 0
    if not gt_events:
        return 0, len(alert_intervals)

    true_alerts = 0
    false_alerts = 0
    j = 0
    n_gt = len(gt_events)

    for interval in alert_intervals:
        as_, ae = interval
        while j < n_gt and (gt_events[j][1] + slack_s) < as_:
            j += 1

        if j < n_gt and _overlap(interval, gt_events[j], slack_s=slack_s):
            true_alerts += 1
        else:
            false_alerts += 1

    return true_alerts, false_alerts


def _count_true_false_alert_events(
    alert_events: Sequence[AlertEvent],
    gt_events: List[Tuple[float, float]],
    *,
    slack_s: float,
) -> Tuple[int, int]:
    """Count true/false alerts directly from AlertEvent objects."""
    if not alert_events:
        return 0, 0
    if not gt_events:
        return 0, len(alert_events)

    true_alerts = 0
    false_alerts = 0
    j = 0
    n_gt = len(gt_events)

    for ev in alert_events:
        as_ = ev.start_time_s
        ae = ev.end_time_s
        while j < n_gt and (gt_events[j][1] + slack_s) < as_:
            j += 1

        if j < n_gt and _overlap((as_, ae), gt_events[j], slack_s=slack_s):
            true_alerts += 1
        else:
            false_alerts += 1

    return true_alerts, false_alerts


def event_metrics_from_windows(
    probs: Sequence[float],
    y_true: Sequence[int],
    times_s: Sequence[float],
    alert_cfg: AlertCfg,
    *,
    lying_score: Optional[Sequence[float]] = None,
    motion_score: Optional[Sequence[float]] = None,
    alert_events: Optional[Sequence[AlertEvent]] = None,
    gt_events: Optional[Sequence[Tuple[float, float]]] = None,
    duration_s: Optional[float] = None,
    merge_gap_s: float = 2.0,
    overlap_slack_s: float = 0.0,
) -> Tuple[EventMetrics, Dict[str, Any]]:
    """Compute event-level metrics for a single video.

    - GT events are built from contiguous positive windows (y_true==1), merged by merge_gap_s.
    - Alert events come from detect_alert_events().

    false_alerts_per_{hour,day} counts ONLY alerts that do NOT overlap any GT event.
    (For unlabeled streams where y_true is all -1/0, this equals total alerts.)
    """
    y = np.asarray(y_true, dtype=np.int32).reshape(-1)
    t = np.asarray(times_s, dtype=np.float32).reshape(-1)

    if t.size == 0:
        em = EventMetrics(
            event_recall=float("nan"),
            false_alerts_per_hour=float("nan"),
            false_alerts_per_day=float("nan"),
            mean_delay_s=float("nan"),
            median_delay_s=float("nan"),
            n_gt_events=0,
            n_alert_events=0,
            n_matched_gt=0,
            n_true_alerts=0,
            n_false_alerts=0,
            event_precision=float("nan"),
            event_f1=float("nan"),
        )
        return em, {"gt_events": [], "alert_events": []}

    if duration_s is None:
        duration_s = float(t.max() - t.min()) if t.size else 0.0
    duration_h = float(duration_s) / 3600.0 if duration_s > 0 else float("nan")
    duration_d = float(duration_s) / 86400.0 if duration_s > 0 else float("nan")

    # GT events (ignore y=-1 for unlabeled, treat as negatives)
    if gt_events is None:
        y01 = (y == 1)
        gt_events = _events_from_positive_windows(t, y01, merge_gap_s=float(merge_gap_s))
    else:
        gt_events = list(gt_events)

    # Alert events (optionally reuse a precomputed policy pass).
    if alert_events is None:
        p = np.asarray(probs, dtype=np.float32).reshape(-1)
        if p.size != t.size:
            raise ValueError("probs and times_s must have same length")
        _, alert_events = detect_alert_events(
            p,
            t,
            alert_cfg,
            lying_score=lying_score,
            motion_score=motion_score,
        )
    else:
        alert_events = list(alert_events)
    # Match GT -> earliest overlapping alert for delay
    matched_gt, delays = _match_gt_delays(
        gt_events,
        alert_events,
        slack_s=float(overlap_slack_s),
    )

    recall = float(matched_gt / len(gt_events)) if gt_events else float("nan")

    # Alert precision / false alerts
    true_alerts, false_alerts = _count_true_false_alert_events(
        alert_events,
        gt_events,
        slack_s=float(overlap_slack_s),
    )

    n_alert = len(alert_events)
    precision = float(true_alerts / n_alert) if n_alert > 0 else float("nan")
    if np.isfinite(precision) and np.isfinite(recall) and (precision + recall) > 0:
        f1 = float(2.0 * precision * recall / (precision + recall))
    else:
        f1 = float("nan")

    fa_per_hour = float(false_alerts / duration_h) if np.isfinite(duration_h) and duration_h > 0 else float("nan")
    fa_per_day = float(false_alerts / duration_d) if np.isfinite(duration_d) and duration_d > 0 else float("nan")

    if delays:
        mean_delay = float(np.mean(delays))
        median_delay = float(np.median(delays))
    else:
        mean_delay = float("nan")
        median_delay = float("nan")

    em = EventMetrics(
        event_recall=float(recall),
        false_alerts_per_hour=float(fa_per_hour),
        false_alerts_per_day=float(fa_per_day),
        mean_delay_s=float(mean_delay),
        median_delay_s=float(median_delay),
        n_gt_events=int(len(gt_events)),
        n_alert_events=int(n_alert),
        n_matched_gt=int(matched_gt),
        n_true_alerts=int(true_alerts),
        n_false_alerts=int(false_alerts),
        event_precision=float(precision),
        event_f1=float(f1),
    )
    detail = {
        "gt_events": [{"start_s": float(s), "end_s": float(e)} for (s, e) in gt_events],
        "alert_events": [ev.to_dict() for ev in alert_events],
    }
    return em, detail


def sweep_alert_policy_from_windows(
    probs: Sequence[float],
    y_true: Sequence[int],
    video_ids: Sequence[str],
    w_start: Sequence[int],
    w_end: Sequence[int],
    fps: Sequence[float],
    *,
    alert_base: AlertCfg,
    thr_min: float = 0.05,
    thr_max: float = 0.95,
    thr_step: float = 0.01,
    tau_low_ratio: float = 0.80,
    merge_gap_s: Optional[float] = None,
    overlap_slack_s: float = 0.0,
    time_mode: str = "center",
    fps_default: float = 30.0,
    # Optional confirm-stage heuristic signals (per window).
    lying_score: Optional[Sequence[float]] = None,
    motion_score: Optional[Sequence[float]] = None,
    # Optional: separate negative/unlabeled stream windows for FA/24h estimation.
    # If provided, fa24h/fa_per_hour will be computed from these windows instead
    # of the (often short) labeled validation videos.
    fa_probs: Optional[Sequence[float]] = None,
    fa_lying_score: Optional[Sequence[float]] = None,
    fa_motion_score: Optional[Sequence[float]] = None,
    fa_video_ids: Optional[Sequence[str]] = None,
    fa_w_start: Optional[Sequence[int]] = None,
    fa_w_end: Optional[Sequence[int]] = None,
    fa_fps: Optional[Sequence[float]] = None,
) -> Tuple[Dict[str, List[float]], Dict[str, Any]]:
    """Sweep tau_high (called 'thr' for backwards-compat) under the REAL alert policy.

    Returns:
      sweep: dict of lists (thr, tau_low, precision, recall, f1, fa24h, mean_delay_s, ...)
      meta:  durations, n_videos, etc.

    Notes:
      - FA is FALSE alerts only (alerts not overlapping GT).
      - GT events are derived from y_true positive windows.
    """
    probs = np.asarray(probs, dtype=np.float32).reshape(-1)
    y_true = np.asarray(y_true, dtype=np.int32).reshape(-1)
    vids = np.asarray(video_ids).reshape(-1)
    ws = np.asarray(w_start, dtype=np.int32).reshape(-1)
    we = np.asarray(w_end, dtype=np.int32).reshape(-1)
    fps_arr = np.asarray(fps, dtype=np.float32).reshape(-1)

    use_confirm = bool(alert_base.confirm)
    cooldown_cfg = float(alert_base.cooldown_s)
    ls = None
    ms = None
    if use_confirm:
        if lying_score is not None:
            ls = _as_1d_f32(lying_score)
        if motion_score is not None:
            ms = _as_1d_f32(motion_score)

    if probs.size == 0:
        return {"thr": [], "recall": [], "fa24h": []}, {"n_videos": 0}

    ema_alpha = float(alert_base.ema_alpha)

    # group by video (cached sorted indices)
    groups = _build_video_groups(vids, ws)
    smoothed_main = _ema_precompute_by_groups(probs, groups, ema_alpha)
    per_video = {}
    total_duration_s = 0.0
    for v, idx_full in groups:
        if idx_full.size < 1:
            continue
        y_v = y_true[idx_full]
        ws_v = ws[idx_full]
        we_v = we[idx_full]
        fps_v = _robust_video_fps(fps_arr[idx_full], float(fps_default))
        t_v = times_from_windows(ws_v, we_v, fps_v, mode=time_mode)
        duration_s = float((we_v.max() - ws_v.min() + 1) / max(1e-6, fps_v))
        total_duration_s += max(0.0, duration_s)
        ls_v = ls[idx_full] if ls is not None else None
        ms_v = ms[idx_full] if ms is not None else None
        ps_v = smoothed_main.get(v)
        if ps_v is None:
            ps_v = ema_smooth(probs[idx_full], ema_alpha)
        per_video[v] = (
            ps_v,
            t_v,
            y_v,
            ls_v,
            ms_v,
        )

    # Optional FA-only set (unlabeled/negative stream). All alerts are false by definition.
    per_video_fa = {}
    total_duration_s_fa = 0.0
    if fa_probs is not None and fa_video_ids is not None and fa_w_start is not None and fa_w_end is not None and fa_fps is not None:
        fa_p = np.asarray(fa_probs, dtype=np.float32).reshape(-1)
        fa_ls = None
        fa_ms = None
        if use_confirm:
            if fa_lying_score is not None:
                fa_ls = _as_1d_f32(fa_lying_score)
            if fa_motion_score is not None:
                fa_ms = _as_1d_f32(fa_motion_score)
        fa_vids_arr = np.asarray(fa_video_ids).reshape(-1)
        fa_ws_arr = np.asarray(fa_w_start, dtype=np.int32).reshape(-1)
        fa_we_arr = np.asarray(fa_w_end, dtype=np.int32).reshape(-1)
        fa_fps_arr = np.asarray(fa_fps, dtype=np.float32).reshape(-1)
        ok_sizes = (fa_p.size == fa_vids_arr.size == fa_ws_arr.size == fa_we_arr.size == fa_fps_arr.size)
        if fa_ls is not None:
            ok_sizes = ok_sizes and (fa_ls.size == fa_p.size)
        if fa_ms is not None:
            ok_sizes = ok_sizes and (fa_ms.size == fa_p.size)
        if ok_sizes and fa_p.size > 0:
            fa_groups = _build_video_groups(fa_vids_arr, fa_ws_arr)
            smoothed_fa = _ema_precompute_by_groups(fa_p, fa_groups, ema_alpha)
            for v, idx_full in fa_groups:
                if idx_full.size < 1:
                    continue
                ws_v = fa_ws_arr[idx_full]
                we_v = fa_we_arr[idx_full]
                fps_v = _robust_video_fps(fa_fps_arr[idx_full], float(fps_default))
                t_v = times_from_windows(ws_v, we_v, fps_v, mode=time_mode)
                duration_s = float((we_v.max() - ws_v.min() + 1) / max(1e-6, fps_v))
                total_duration_s_fa += max(0.0, duration_s)
                ls_v = fa_ls[idx_full] if fa_ls is not None else None
                ms_v = fa_ms[idx_full] if fa_ms is not None else None
                ps_v = smoothed_fa.get(v)
                if ps_v is None:
                    ps_v = ema_smooth(fa_p[idx_full], ema_alpha)
                per_video_fa[v] = (
                    ps_v,
                    t_v,
                    ls_v,
                    ms_v,
                )

    if merge_gap_s is None:
        # default: 2x median step between windows (seconds)
        gaps = []
        for _v, item in per_video.items():
            t_v = item[1]
            step_v = _robust_time_step_s(t_v)
            if step_v is not None:
                gaps.append(step_v)
        med_gap = float(np.median(gaps)) if gaps else 0.5
        merge_gap_s = max(0.25, 2.0 * med_gap)
    merge_gap_s = float(merge_gap_s)

    # Precompute GT events once with finalized merge_gap_s.
    per_video_eval: List[
        Tuple[
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            float,
            float,
            int,
            Optional[np.ndarray],
            Optional[np.ndarray],
        ]
    ] = []
    gt_total_const = 0
    for v, item in per_video.items():
        ps_v, t_v, y_v, ls_v, ms_v = item
        gt_start_v, gt_end_v = _events_from_positive_windows_arrays(t_v, (y_v == 1), merge_gap_s=merge_gap_s)
        n_gt_v = int(gt_start_v.size)
        if n_gt_v > 0:
            gt_end_slack_v = gt_end_v + np.float32(overlap_slack_s)
            gt_min_start_v = float(gt_start_v[0])
            gt_max_end_v = float(gt_end_v[-1])
        else:
            gt_start_v = np.asarray([], dtype=np.float32)
            gt_end_v = np.asarray([], dtype=np.float32)
            gt_end_slack_v = np.asarray([], dtype=np.float32)
            gt_min_start_v = float("inf")
            gt_max_end_v = float("-inf")
        gt_total_const += n_gt_v
        per_video_eval.append(
            (
            ps_v,
            t_v,
            gt_start_v,
            gt_end_v,
            gt_end_slack_v,
            gt_min_start_v,
                gt_max_end_v,
                n_gt_v,
                ls_v,
                ms_v,
            )
        )
    thr_values = np.arange(float(thr_min), float(thr_max) + 1e-12, float(thr_step), dtype=np.float32)
    tau_low_values = np.clip(thr_values * float(tau_low_ratio), 0.0, thr_values).astype(np.float32, copy=False)
    n_thr = int(thr_values.size)
    # Adaptive threshold-mask precompute guard to avoid pathological memory use
    # on very long sequences while keeping the common sweep case vectorized.
    _LOW_MASK_MAX_ELEMS = 1_000_000
    per_video_eval_items = []
    EVAL_ITEM_ACTIVE_THR_IDX = 9
    any_confirm_scores_eval = False
    confirm_min_lying_sweep = float(alert_base.confirm_min_lying)
    confirm_max_motion_sweep = float(alert_base.confirm_max_motion)
    for (ps_v, t_v, gt_start_v, gt_end_v, gt_end_slack_v, gt_min_start_v, gt_max_end_v, n_gt_v, ls_v, ms_v) in per_video_eval:
        has_confirm_scores_v = bool(ls_v is not None or ms_v is not None)
        any_confirm_scores_eval = any_confirm_scores_eval or has_confirm_scores_v
        use_confirm_item_v = bool(use_confirm and has_confirm_scores_v)
        (
            pers_template,
            active_thr,
            pers_idx_by_thr,
            pers_t_by_thr,
            low_idx_by_thr,
            end_for_pers_by_thr,
            has_low_after_by_thr,
            no_gt_counts_by_thr,
        ) = _precompute_video_threshold_items(
            ps_v,
            t_v,
            thr_values,
            tau_low_values,
            k=alert_base.k,
            n=alert_base.n,
            use_confirm=use_confirm_item_v,
            cooldown_s=cooldown_cfg,
            low_mask_max_elems=_LOW_MASK_MAX_ELEMS,
            compute_non_confirm_counts=((not use_confirm_item_v) and (n_gt_v < 1)),
        )
        gate_ok_v = None
        gate_idx_v = None
        gate_any_v = True
        has_scores_v = False
        require_low_v = False
        confirm_deadline_idx_v = None
        cooldown_after_idx_v = None
        if use_confirm and (ls_v is not None or ms_v is not None):
            gate_ok_v = _build_confirm_gate_mask(
                ps_v.size,
                ls_v,
                ms_v,
                confirm_min_lying=confirm_min_lying_sweep,
                confirm_max_motion=confirm_max_motion_sweep,
            )
            gate_idx_v = np.flatnonzero(gate_ok_v).astype(np.int32, copy=False)
            gate_any_v = bool(gate_idx_v.size > 0)
            has_scores_v = True
            require_low_v = bool(alert_base.confirm_require_low)
            confirm_deadline_idx_v = np.searchsorted(
                t_v,
                t_v + np.float32(alert_base.confirm_s),
                side="right",
            ).astype(np.int32, copy=False)
            cooldown_after_idx_v = np.searchsorted(
                t_v,
                t_v + np.float32(cooldown_cfg),
                side="left",
            ).astype(np.int32, copy=False)
        per_video_eval_items.append(
            (
                ps_v,
                t_v,
                gt_start_v,
                gt_end_v,
                gt_end_slack_v,
                gt_min_start_v,
                gt_max_end_v,
                n_gt_v,
                pers_template,
                active_thr,
                pers_idx_by_thr,
                pers_t_by_thr,
                low_idx_by_thr,
                end_for_pers_by_thr,
                has_low_after_by_thr,
                gate_ok_v,
                gate_idx_v,
                gate_any_v,
                has_scores_v,
                require_low_v,
                confirm_deadline_idx_v,
                cooldown_after_idx_v,
                no_gt_counts_by_thr,
            )
        )
    thr_eval_video_indices: List[List[int]] = []
    _thr_eval_work: List[List[int]] = [[] for _ in range(n_thr)]
    for i_v, item in enumerate(per_video_eval_items):
        active_thr = item[EVAL_ITEM_ACTIVE_THR_IDX]
        if active_thr.size < 1:
            continue
        iv = i_v
        for i_thr in active_thr:
            _thr_eval_work[i_thr].append(iv)
    thr_eval_video_indices = _thr_eval_work

    per_video_fa_items = []
    FA_ITEM_ACTIVE_THR_IDX = 3
    any_confirm_scores_fa = False
    for _v, (ps_v, t_v, ls_v, ms_v) in per_video_fa.items():
        has_confirm_scores_v = bool(ls_v is not None or ms_v is not None)
        any_confirm_scores_fa = any_confirm_scores_fa or has_confirm_scores_v
        use_confirm_item_v = bool(use_confirm and has_confirm_scores_v)
        (
            pers_template,
            active_thr,
            pers_idx_by_thr,
            pers_t_by_thr,
            low_idx_by_thr,
            end_for_pers_by_thr,
            has_low_after_by_thr,
            fa_counts_by_thr,
        ) = _precompute_video_threshold_items(
            ps_v,
            t_v,
            thr_values,
            tau_low_values,
            k=alert_base.k,
            n=alert_base.n,
            use_confirm=use_confirm_item_v,
            cooldown_s=cooldown_cfg,
            low_mask_max_elems=_LOW_MASK_MAX_ELEMS,
            compute_non_confirm_counts=(not use_confirm_item_v),
        )
        gate_ok_v = None
        gate_idx_v = None
        gate_any_v = True
        has_scores_v = False
        require_low_v = False
        confirm_deadline_idx_v = None
        cooldown_after_idx_v = None
        if use_confirm and (ls_v is not None or ms_v is not None):
            gate_ok_v = _build_confirm_gate_mask(
                ps_v.size,
                ls_v,
                ms_v,
                confirm_min_lying=confirm_min_lying_sweep,
                confirm_max_motion=confirm_max_motion_sweep,
            )
            gate_idx_v = np.flatnonzero(gate_ok_v).astype(np.int32, copy=False)
            gate_any_v = bool(gate_idx_v.size > 0)
            has_scores_v = True
            require_low_v = bool(alert_base.confirm_require_low)
            confirm_deadline_idx_v = np.searchsorted(
                t_v,
                t_v + np.float32(alert_base.confirm_s),
                side="right",
            ).astype(np.int32, copy=False)
            cooldown_after_idx_v = np.searchsorted(
                t_v,
                t_v + np.float32(cooldown_cfg),
                side="left",
            ).astype(np.int32, copy=False)
        per_video_fa_items.append(
            (
                ps_v,
                t_v,
                pers_template,
                active_thr,
                pers_idx_by_thr,
                pers_t_by_thr,
                low_idx_by_thr,
                end_for_pers_by_thr,
                has_low_after_by_thr,
                gate_ok_v,
                gate_idx_v,
                gate_any_v,
                has_scores_v,
                require_low_v,
                confirm_deadline_idx_v,
                cooldown_after_idx_v,
                fa_counts_by_thr,
            )
        )
    thr_fa_video_indices: List[List[int]] = []
    _thr_fa_work: List[List[int]] = [[] for _ in range(n_thr)]
    for i_v, item in enumerate(per_video_fa_items):
        active_thr = item[FA_ITEM_ACTIVE_THR_IDX]
        if active_thr.size < 1:
            continue
        iv = i_v
        for i_thr in active_thr:
            _thr_fa_work[i_thr].append(iv)
    thr_fa_video_indices = _thr_fa_work

    out_thr = np.empty((n_thr,), dtype=np.float32)
    out_tau_low = np.empty((n_thr,), dtype=np.float32)
    out_precision = np.empty((n_thr,), dtype=np.float32)
    out_recall = np.empty((n_thr,), dtype=np.float32)
    out_f1 = np.empty((n_thr,), dtype=np.float32)
    out_fa24h = np.empty((n_thr,), dtype=np.float32)
    out_fa_per_hour = np.empty((n_thr,), dtype=np.float32)
    out_mean_delay = np.empty((n_thr,), dtype=np.float32)
    out_median_delay = np.empty((n_thr,), dtype=np.float32)
    out_n_gt = np.empty((n_thr,), dtype=np.int32)
    out_n_alert = np.empty((n_thr,), dtype=np.int32)
    out_n_true = np.empty((n_thr,), dtype=np.int32)
    out_n_false = np.empty((n_thr,), dtype=np.int32)
    out_n_false_fa = np.empty((n_thr,), dtype=np.int32) if per_video_fa else None

    overlap_slack = float(overlap_slack_s)
    # If confirm is enabled but no confirm signals exist anywhere, behavior is
    # equivalent to non-confirm policy; skip confirm branch globally.
    use_confirm_sweep = bool(alert_base.confirm and (any_confirm_scores_eval or any_confirm_scores_fa))
    cooldown_sweep = float(alert_base.cooldown_s)
    confirm_sweep = float(alert_base.confirm_s)
    _fn_detect_confirm = _detect_confirm_segments_no_active_from_pers
    _fn_count_confirm = _count_confirm_segments_no_active_from_pers
    _fn_detect_non_confirm = _detect_non_confirm_segments_from_pers
    _fn_count_non_confirm = _count_non_confirm_segments_from_pers
    _fn_event_stats = _event_match_delay_and_alert_stats_indices_prepared
    ref_dur_s = float(total_duration_s_fa) if per_video_fa else float(total_duration_s)
    dur_h = (ref_dur_s / 3600.0) if ref_dur_s > 0 else float("nan")
    dur_d = (ref_dur_s / 86400.0) if ref_dur_s > 0 else float("nan")
    collect_delay_stats = (gt_total_const > 0)

    for thr_idx, thr in enumerate(thr_values):
        tau_low = float(tau_low_values[thr_idx])

        gt_total = gt_total_const
        matched_gt_total = 0
        alert_total = 0
        true_alert_total = 0
        false_alert_total = 0
        false_alert_total_fa = 0
        thr_eval_idx = thr_eval_video_indices[thr_idx]
        n_eval_videos_thr = len(thr_eval_idx)
        delays_video_means = (
            np.empty((n_eval_videos_thr,), dtype=np.float32)
            if (collect_delay_stats and n_eval_videos_thr > 0)
            else None
        )
        n_delay_means = 0

        for i_v in thr_eval_idx:
            (
                ps_v,
                t_v,
                gt_start_v,
                gt_end_v,
                gt_end_slack_v,
                gt_min_start_v,
                gt_max_end_v,
                _n_gt_v,
                pers_template,
                _active_thr,
                pers_idx_by_thr,
                pers_t_by_thr,
                low_idx_by_thr,
                end_for_pers_by_thr,
                has_low_after_by_thr,
                gate_ok_v,
                gate_idx_v,
                gate_any_v,
                has_scores_v,
                require_low_v,
                confirm_deadline_idx_v,
                cooldown_after_idx_v,
                no_gt_counts_by_thr,
            ) = per_video_eval_items[i_v]
            matched_gt_v = 0
            delay_sum_v = 0.0
            n_delay_v = 0
            true_v = 0
            false_v = 0
            n_alert_v = 0
            if use_confirm_sweep:
                if has_scores_v and (not gate_any_v):
                    n_alert_v = 0
                else:
                    pers_idx_i = pers_idx_by_thr[thr_idx]
                    low_idx_i = low_idx_by_thr[thr_idx]
                    if _n_gt_v < 1:
                        if (not has_scores_v) and (no_gt_counts_by_thr is not None):
                            n_alert_v = no_gt_counts_by_thr[thr_idx]
                        else:
                            n_alert_v = int(
                                _fn_count_confirm(
                                    ps_v,
                                    t_v,
                                    tau_low=tau_low,
                                    cooldown_s=cooldown_sweep,
                                    pers=pers_template,
                                    confirm_s=confirm_sweep,
                                    require_low=require_low_v,
                                    has_confirm_scores=has_scores_v,
                                    gate_ok=gate_ok_v,
                                    gate_idx=gate_idx_v,
                                    confirm_deadline_idx=confirm_deadline_idx_v,
                                    cooldown_after_idx=cooldown_after_idx_v,
                                    low_idx=low_idx_i,
                                    pers_idx=pers_idx_i,
                                    assume_idx_prepared=True,
                                )
                            )
                        matched_gt_v = 0
                        delay_sum_v = 0.0
                        n_delay_v = 0
                        true_v = 0
                        false_v = n_alert_v
                        matched_gt_total += matched_gt_v
                        alert_total += n_alert_v
                        true_alert_total += true_v
                        false_alert_total += false_v
                        continue
                    starts_v, ends_v = _fn_detect_confirm(
                        ps_v,
                        t_v,
                        tau_low=tau_low,
                        cooldown_s=cooldown_sweep,
                        pers=pers_template,
                        confirm_s=confirm_sweep,
                        require_low=require_low_v,
                        has_confirm_scores=has_scores_v,
                        gate_ok=gate_ok_v,
                        gate_idx=gate_idx_v,
                        confirm_deadline_idx=confirm_deadline_idx_v,
                        cooldown_after_idx=cooldown_after_idx_v,
                        low_idx=low_idx_i,
                        pers_idx=pers_idx_i,
                        assume_idx_prepared=True,
                    )
                    n_alert_v = starts_v.shape[0]
                    if n_alert_v < 1:
                        pass
                    elif _n_gt_v < 1:
                        matched_gt_v = 0
                        delay_sum_v = 0.0
                        n_delay_v = 0
                        true_v = 0
                        false_v = n_alert_v
                    elif (t_v[ends_v[-1]] + overlap_slack) < gt_min_start_v or t_v[starts_v[0]] > (gt_max_end_v + overlap_slack):
                        matched_gt_v = 0
                        delay_sum_v = 0.0
                        n_delay_v = 0
                        true_v = 0
                        false_v = n_alert_v
                    else:
                        matched_gt_v, delay_sum_v, n_delay_v, true_v, false_v = _fn_event_stats(
                            gt_start_v,
                            gt_end_v,
                            t_v,
                            starts_v,
                            ends_v,
                            slack_s=overlap_slack,
                            gt_end_slack_pre=gt_end_slack_v,
                        )
            else:
                if _n_gt_v < 1:
                    # For non-confirm path with no GT, counts are precomputed.
                    assert no_gt_counts_by_thr is not None
                    n_alert_v = no_gt_counts_by_thr[thr_idx]
                    matched_gt_v = 0
                    delay_sum_v = 0.0
                    n_delay_v = 0
                    true_v = 0
                    false_v = n_alert_v
                else:
                    pers_idx_i = pers_idx_by_thr[thr_idx]
                    pers_t_i = pers_t_by_thr[thr_idx]
                    low_idx_i = low_idx_by_thr[thr_idx]
                    end_for_pers_i = end_for_pers_by_thr[thr_idx]
                    has_low_after_i = has_low_after_by_thr[thr_idx]
                    n_pers_i = pers_idx_i.shape[0]
                    n_low_i = low_idx_i.shape[0]
                    if n_pers_i == 1:
                        s0 = int(pers_idx_i[0])
                        if n_low_i > 0:
                            lp = int(low_idx_i.searchsorted(s0, side="left"))
                            e0 = int(low_idx_i[lp]) if lp < n_low_i else (int(ps_v.size) - 1)
                        else:
                            e0 = int(ps_v.size) - 1
                        starts_v = np.asarray([s0], dtype=np.int32)
                        ends_v = np.asarray([e0], dtype=np.int32)
                    else:
                        starts_v, ends_v = _fn_detect_non_confirm(
                            ps_v,
                            t_v,
                            tau_low=tau_low,
                            cooldown_s=cooldown_sweep,
                            pers=pers_template,
                            low_idx=low_idx_i,
                            pers_idx=pers_idx_i,
                            pers_t=pers_t_i,
                            end_for_pers=end_for_pers_i,
                            has_low_after=has_low_after_i,
                            assume_idx_prepared=True,
                        )
                    n_alert_v = starts_v.shape[0]
                    if n_alert_v < 1:
                        matched_gt_v = 0
                        delay_sum_v = 0.0
                        n_delay_v = 0
                        true_v = 0
                        false_v = 0
                    elif (t_v[ends_v[-1]] + overlap_slack) < gt_min_start_v or t_v[starts_v[0]] > (gt_max_end_v + overlap_slack):
                        matched_gt_v = 0
                        delay_sum_v = 0.0
                        n_delay_v = 0
                        true_v = 0
                        false_v = n_alert_v
                    else:
                        matched_gt_v, delay_sum_v, n_delay_v, true_v, false_v = _fn_event_stats(
                            gt_start_v,
                            gt_end_v,
                            t_v,
                            starts_v,
                            ends_v,
                            slack_s=overlap_slack,
                            gt_end_slack_pre=gt_end_slack_v,
                        )

            matched_gt_total += matched_gt_v
            alert_total += n_alert_v
            true_alert_total += true_v
            false_alert_total += false_v
            if collect_delay_stats and n_delay_v > 0:
                # Keep prior behavior: aggregate per-video mean delay, not per-event delay.
                if delays_video_means is not None:
                    delays_video_means[n_delay_means] = float(delay_sum_v / n_delay_v)
                    n_delay_means += 1

        # If FA-only stream is provided, compute FA rate from it (more realistic for deployment)
        if per_video_fa:
            for i_v in thr_fa_video_indices[thr_idx]:
                (
                    ps_v,
                    t_v,
                    pers_template,
                    _active_thr,
                    pers_idx_by_thr,
                    pers_t_by_thr,
                    low_idx_by_thr,
                    end_for_pers_by_thr,
                    has_low_after_by_thr,
                    gate_ok_v,
                    gate_idx_v,
                    gate_any_v,
                    has_scores_v,
                    require_low_v,
                    confirm_deadline_idx_v,
                    cooldown_after_idx_v,
                    fa_counts_by_thr,
                ) = per_video_fa_items[i_v]
                if use_confirm_sweep:
                    if has_scores_v and (not gate_any_v):
                        pass
                    else:
                        if (not has_scores_v) and (fa_counts_by_thr is not None):
                            false_alert_total_fa += fa_counts_by_thr[thr_idx]
                        else:
                            pers_idx_i = pers_idx_by_thr[thr_idx]
                            low_idx_i = low_idx_by_thr[thr_idx]
                            false_alert_total_fa += (
                                _fn_count_confirm(
                                    ps_v,
                                    t_v,
                                    tau_low=tau_low,
                                    cooldown_s=cooldown_sweep,
                                    pers=pers_template,
                                    confirm_s=confirm_sweep,
                                    require_low=require_low_v,
                                    has_confirm_scores=has_scores_v,
                                    gate_ok=gate_ok_v,
                                    gate_idx=gate_idx_v,
                                    confirm_deadline_idx=confirm_deadline_idx_v,
                                    cooldown_after_idx=cooldown_after_idx_v,
                                    low_idx=low_idx_i,
                                    pers_idx=pers_idx_i,
                                    assume_idx_prepared=True,
                                )
                            )
                else:
                    # For non-confirm FA path, counts are precomputed.
                    assert fa_counts_by_thr is not None
                    false_alert_total_fa += fa_counts_by_thr[thr_idx]

        recall = float(matched_gt_total / gt_total) if gt_total > 0 else 0.0
        precision = float(true_alert_total / alert_total) if alert_total > 0 else float("nan")
        if np.isfinite(precision) and np.isfinite(recall) and (precision + recall) > 0:
            f1 = float(2.0 * precision * recall / (precision + recall))
        else:
            f1 = float("nan")

        # FA/24h reference: prefer FA-only stream if provided; otherwise use validation videos.
        ref_false = false_alert_total_fa if per_video_fa else false_alert_total
        fa_h = float(ref_false / dur_h) if np.isfinite(dur_h) and dur_h > 0 else float("nan")
        fa_d = float(ref_false / dur_d) if np.isfinite(dur_d) and dur_d > 0 else float("nan")

        out_thr[thr_idx] = thr
        out_tau_low[thr_idx] = tau_low
        out_precision[thr_idx] = precision
        out_recall[thr_idx] = recall
        out_f1[thr_idx] = f1
        out_fa24h[thr_idx] = fa_d
        out_fa_per_hour[thr_idx] = fa_h
        if delays_video_means is not None and n_delay_means > 0:
            dvm = delays_video_means[:n_delay_means]
            out_mean_delay[thr_idx] = float(np.mean(dvm))
            out_median_delay[thr_idx] = float(np.median(dvm))
        else:
            out_mean_delay[thr_idx] = float("nan")
            out_median_delay[thr_idx] = float("nan")
        out_n_gt[thr_idx] = gt_total
        out_n_alert[thr_idx] = alert_total
        out_n_true[thr_idx] = true_alert_total
        out_n_false[thr_idx] = false_alert_total
        if out_n_false_fa is not None:
            out_n_false_fa[thr_idx] = false_alert_total_fa

    out = {
        "thr": out_thr.tolist(),
        "tau_low": out_tau_low.tolist(),
        "precision": out_precision.tolist(),
        "recall": out_recall.tolist(),
        "f1": out_f1.tolist(),
        "fa24h": out_fa24h.tolist(),
        "fa_per_hour": out_fa_per_hour.tolist(),
        "mean_delay_s": out_mean_delay.tolist(),
        "median_delay_s": out_median_delay.tolist(),
        "n_gt_events": out_n_gt.tolist(),
        "n_alert_events": out_n_alert.tolist(),
        "n_true_alerts": out_n_true.tolist(),
        "n_false_alerts": out_n_false.tolist(),
    }
    if out_n_false_fa is not None:
        out["n_false_alerts_fa"] = out_n_false_fa.tolist()

    meta = {
        "n_videos": int(len(per_video)),
        "total_duration_s": float(total_duration_s),
        "n_fa_videos": int(len(per_video_fa)),
        "total_duration_s_fa": float(total_duration_s_fa),
        "merge_gap_s": float(merge_gap_s),
        "time_mode": str(time_mode),
        "overlap_slack_s": float(overlap_slack_s),
    }
    return out, meta


# =========================
# 3-state triage + mode SMs
# =========================

TRIAGE_FALL = "fall"
TRIAGE_NOT_FALL = "not_fall"
TRIAGE_UNCERTAIN = "uncertain"

EVENT_POSSIBLE = "possible_fall"
EVENT_CONFIRMED = "fall_detected"
EVENT_RESOLVED = "resolved"


@dataclass
class TriageCfg:
    """Two-threshold triage (Option 1) + optional uncertainty gate (Option 2)."""
    tau_low: float = 0.05
    tau_high: float = 0.90
    ema_alpha: float = 0.20  # EMA smoothing on probability/mean
    sigma_max: Optional[float] = None  # if provided and sigma > sigma_max => UNCERTAIN


@dataclass
class SingleModeCfg:
    """Temporal logic for a single model (Mode 1 or Mode 2)."""
    possible_k: int = 3
    possible_T_s: float = 2.0
    confirm_T_s: float = 3.6  # keep worst-case confirmed latency ~6s (W=1.6 + possible<=2.4 + confirm<=3.6)
    confirm_k_fall: int = 2
    cooldown_possible_s: float = 15.0
    cooldown_confirmed_s: float = 60.0

    # Optional skeleton-derived confirmation gate (applied at CONFIRMED time).
    confirm_use_scores: bool = False
    confirm_min_lying: float = 0.65
    confirm_max_motion: float = 0.08


@dataclass
class DualModeCfg:
    """Temporal logic for dual-model fusion (Mode 3)."""
    possible_k: int = 3
    possible_T_s: float = 2.0
    confirm_T_s: float = 3.6
    confirm_k_tcn: int = 1
    confirm_k_gcn: int = 1
    require_both: bool = True
    cooldown_possible_s: float = 15.0
    cooldown_confirmed_s: float = 60.0

    # Optional skeleton-derived confirmation gate (applied at CONFIRMED time).
    confirm_use_scores: bool = False
    confirm_min_lying: float = 0.65
    confirm_max_motion: float = 0.08


@dataclass
class TriageEvent:
    kind: str
    t_sec: float
    info: Dict[str, Any]


def triage_state(mu: float, tau_low: float, tau_high: float, sigma: Optional[float] = None, sigma_max: Optional[float] = None) -> str:
    """Return fall / uncertain / not_fall using Option 1 (+ optional Option 2 gate)."""
    if sigma is not None and sigma_max is not None and float(sigma) > float(sigma_max):
        return TRIAGE_UNCERTAIN
    if float(mu) >= float(tau_high):
        return TRIAGE_FALL
    if float(mu) <= float(tau_low):
        return TRIAGE_NOT_FALL
    return TRIAGE_UNCERTAIN


class SingleTriageStateMachine:
    """State machine for a single model.

    - Normal streaming: triage on EMA-smoothed score.
    - 'possible_fall' triggers after K suspicious states in a trailing T seconds.
    - 'fall_detected' triggers if >=confirm_k_fall FALL states within confirm_T_s.
    """

    def __init__(self, triage_cfg: TriageCfg, mode_cfg: SingleModeCfg):
        self.triage_cfg = triage_cfg
        self.mode_cfg = mode_cfg
        self.reset()

    def reset(self) -> None:
        self._ema: Optional[float] = None
        self._state: str = "idle"  # idle|confirm|cooldown_possible|cooldown_confirmed
        self._t_state0: float = 0.0
        self._sus_times: List[float] = []
        self._fall_times: List[float] = []
        self._cooldown_until: float = -1.0

    def _ema_update(self, x: float) -> float:
        a = float(self.triage_cfg.ema_alpha)
        if self._ema is None:
            self._ema = float(x)
        else:
            self._ema = a * float(x) + (1.0 - a) * float(self._ema)
        return float(self._ema)

    def step(self, t_sec: float, p_or_mu: float, sigma: Optional[float] = None, lying: Optional[float] = None, motion: Optional[float] = None) -> List[TriageEvent]:
        evs: List[TriageEvent] = []
        t = float(t_sec)

        mu = self._ema_update(float(p_or_mu))
        s = triage_state(mu, self.triage_cfg.tau_low, self.triage_cfg.tau_high, sigma=sigma, sigma_max=self.triage_cfg.sigma_max)

        # cooldown handling
        if t < self._cooldown_until:
            return evs

        if self._state == "idle":
            if s != TRIAGE_NOT_FALL:
                self._sus_times.append(t)
            # trim
            T = float(self.mode_cfg.possible_T_s)
            self._sus_times = [x for x in self._sus_times if (t - x) <= T]
            if len(self._sus_times) >= int(self.mode_cfg.possible_k):
                # possible fall
                evs.append(TriageEvent(EVENT_POSSIBLE, t, {"mu": mu, "sigma": sigma}))
                self._state = "confirm"
                self._t_state0 = t
                self._fall_times = []
                self._sus_times = []
            return evs

        if self._state == "confirm":
            if s == TRIAGE_FALL:
                self._fall_times.append(t)
            # trim to confirm window
            Tc = float(self.mode_cfg.confirm_T_s)
            self._fall_times = [x for x in self._fall_times if (t - x) <= Tc]
            if len(self._fall_times) >= int(self.mode_cfg.confirm_k_fall):
                gate = bool(getattr(self.mode_cfg, "confirm_use_scores", False)) and (lying is not None or motion is not None)
                if gate:
                    ok_lying = True if lying is None else (float(lying) >= float(getattr(self.mode_cfg, "confirm_min_lying", 0.65)))
                    ok_motion = True if motion is None else (float(motion) <= float(getattr(self.mode_cfg, "confirm_max_motion", 0.08)))
                    if not (ok_lying and ok_motion):
                        # Do not confirm yet; wait for more evidence or timeout.
                        pass
                    else:
                        evs.append(TriageEvent(EVENT_CONFIRMED, t, {"mu": mu, "sigma": sigma, "lying": lying, "motion": motion}))
                        self._state = "idle"
                        self._cooldown_until = t + float(self.mode_cfg.cooldown_confirmed_s)
                        self._fall_times = []
                        return evs
                else:
                    evs.append(TriageEvent(EVENT_CONFIRMED, t, {"mu": mu, "sigma": sigma, "lying": lying, "motion": motion}))
                    self._state = "idle"
                    self._cooldown_until = t + float(self.mode_cfg.cooldown_confirmed_s)
                    self._fall_times = []
                    return evs

            # time out -> resolved
            if (t - self._t_state0) > Tc:
                evs.append(TriageEvent(EVENT_RESOLVED, t, {"mu": mu, "sigma": sigma}))
                self._state = "idle"
                self._cooldown_until = t + float(self.mode_cfg.cooldown_possible_s)
                self._fall_times = []
            return evs

        return evs


class DualTriageStateMachine:
    """Fusion state machine for Mode 3 (TCN + GCN).

    Possible fall triggers if *either* model is suspicious (UNCERTAIN or FALL)
    for K times within possible_T_s.

    Confirmed fall triggers if:
      - require_both=True: each model reaches its own confirm_k in the confirm window.
      - else: either model reaches (confirm_k_tcn) within the confirm window.
    """

    def __init__(self, triage_tcn: TriageCfg, triage_gcn: TriageCfg, mode_cfg: DualModeCfg):
        self.triage_tcn = triage_tcn
        self.triage_gcn = triage_gcn
        self.mode_cfg = mode_cfg
        self.reset()

    def reset(self) -> None:
        self._ema_tcn: Optional[float] = None
        self._ema_gcn: Optional[float] = None
        self._state: str = "idle"
        self._t_state0: float = 0.0
        self._sus_times: List[float] = []
        self._fall_tcn: List[float] = []
        self._fall_gcn: List[float] = []
        self._cooldown_until: float = -1.0

    def _ema_update(self, which: str, x: float) -> float:
        a_t = float(self.triage_tcn.ema_alpha)
        a_g = float(self.triage_gcn.ema_alpha)
        if which == "tcn":
            if self._ema_tcn is None:
                self._ema_tcn = float(x)
            else:
                self._ema_tcn = a_t * float(x) + (1.0 - a_t) * float(self._ema_tcn)
            return float(self._ema_tcn)
        else:
            if self._ema_gcn is None:
                self._ema_gcn = float(x)
            else:
                self._ema_gcn = a_g * float(x) + (1.0 - a_g) * float(self._ema_gcn)
            return float(self._ema_gcn)

    def step(
        self,
        t_sec: float,
        p_tcn: float,
        p_gcn: float,
        sigma_tcn: Optional[float] = None,
        sigma_gcn: Optional[float] = None,
        lying: Optional[float] = None,
        motion: Optional[float] = None,
    ) -> List[TriageEvent]:
        evs: List[TriageEvent] = []
        t = float(t_sec)

        if t < self._cooldown_until:
            return evs

        mu_t = self._ema_update("tcn", float(p_tcn))
        mu_g = self._ema_update("gcn", float(p_gcn))

        s_t = triage_state(mu_t, self.triage_tcn.tau_low, self.triage_tcn.tau_high, sigma=sigma_tcn, sigma_max=self.triage_tcn.sigma_max)
        s_g = triage_state(mu_g, self.triage_gcn.tau_low, self.triage_gcn.tau_high, sigma=sigma_gcn, sigma_max=self.triage_gcn.sigma_max)

        suspicious = (s_t != TRIAGE_NOT_FALL) or (s_g != TRIAGE_NOT_FALL)

        if self._state == "idle":
            if suspicious:
                self._sus_times.append(t)
            Tp = float(self.mode_cfg.possible_T_s)
            self._sus_times = [x for x in self._sus_times if (t - x) <= Tp]
            if len(self._sus_times) >= int(self.mode_cfg.possible_k):
                evs.append(TriageEvent(EVENT_POSSIBLE, t, {"mu_tcn": mu_t, "mu_gcn": mu_g, "sigma_tcn": sigma_tcn, "sigma_gcn": sigma_gcn}))
                self._state = "confirm"
                self._t_state0 = t
                self._fall_tcn = []
                self._fall_gcn = []
                self._sus_times = []
            return evs

        if self._state == "confirm":
            if s_t == TRIAGE_FALL:
                self._fall_tcn.append(t)
            if s_g == TRIAGE_FALL:
                self._fall_gcn.append(t)

            Tc = float(self.mode_cfg.confirm_T_s)
            self._fall_tcn = [x for x in self._fall_tcn if (t - x) <= Tc]
            self._fall_gcn = [x for x in self._fall_gcn if (t - x) <= Tc]

            ok_t = len(self._fall_tcn) >= int(self.mode_cfg.confirm_k_tcn)
            ok_g = len(self._fall_gcn) >= int(self.mode_cfg.confirm_k_gcn)

            confirmed = (ok_t and ok_g) if bool(self.mode_cfg.require_both) else (ok_t or ok_g)
            if confirmed:
                gate = bool(getattr(self.mode_cfg, "confirm_use_scores", False)) and (lying is not None or motion is not None)
                if gate:
                    ok_lying = True if lying is None else (float(lying) >= float(getattr(self.mode_cfg, "confirm_min_lying", 0.65)))
                    ok_motion = True if motion is None else (float(motion) <= float(getattr(self.mode_cfg, "confirm_max_motion", 0.08)))
                    if not (ok_lying and ok_motion):
                        # Do not confirm yet; wait for more evidence or timeout.
                        pass
                    else:
                        evs.append(TriageEvent(EVENT_CONFIRMED, t, {"mu_tcn": mu_t, "mu_gcn": mu_g, "sigma_tcn": sigma_tcn, "sigma_gcn": sigma_gcn, "lying": lying, "motion": motion}))
                        self._state = "idle"
                        self._cooldown_until = t + float(self.mode_cfg.cooldown_confirmed_s)
                        self._fall_tcn = []
                        self._fall_gcn = []
                        return evs
                else:
                    evs.append(TriageEvent(EVENT_CONFIRMED, t, {"mu_tcn": mu_t, "mu_gcn": mu_g, "sigma_tcn": sigma_tcn, "sigma_gcn": sigma_gcn, "lying": lying, "motion": motion}))
                    self._state = "idle"
                    self._cooldown_until = t + float(self.mode_cfg.cooldown_confirmed_s)
                    self._fall_tcn = []
                    self._fall_gcn = []
                    return evs

            if (t - self._t_state0) > Tc:
                evs.append(TriageEvent(EVENT_RESOLVED, t, {"mu_tcn": mu_t, "mu_gcn": mu_g, "sigma_tcn": sigma_tcn, "sigma_gcn": sigma_gcn}))
                self._state = "idle"
                self._cooldown_until = t + float(self.mode_cfg.cooldown_possible_s)
                self._fall_tcn = []
                self._fall_gcn = []
            return evs

        return evs


# ---------------- operating point selection ----------------

def pick_ops_from_sweep(
    sweep: Dict[str, List[float]],
    *,
    op1_recall: float = 0.95,
    op3_fa24h: float = 2.0,
) -> Dict[str, Dict[str, float]]:
    """Pick deployment operating points from a sweep.

    Inputs
    - sweep: dict of lists as returned by sweep_alert_policy_from_windows()

    Outputs
    - dict with keys OP1/OP2/OP3, each containing tau_high/tau_low and the
      corresponding sweep metrics at that index.

    Selection rules (practical + stable):
    - OP2: best F1.
    - OP1: among points with recall >= op1_recall, choose the smallest FA/day;
           tie-break by higher F1 then precision.
    - OP3: among points with FA/day <= op3_fa24h, choose the highest recall;
           tie-break by higher F1 then precision.

    If a constraint has no feasible points, we fall back gracefully.
    """
    def _arr(key: str, fill: float = float('nan')) -> np.ndarray:
        v = sweep.get(key, None)
        if v is None:
            return np.asarray([], dtype=np.float32)
        a = np.asarray(v, dtype=np.float32)
        if a.ndim != 1:
            a = a.reshape(-1)
        return a

    thr = _arr('thr')
    if thr.size == 0:
        return {}

    tau_low = _arr('tau_low')
    if tau_low.size != thr.size:
        tau_low = np.full_like(thr, np.nan, dtype=np.float32)

    prec = _arr('precision')
    rec = _arr('recall')
    f1 = _arr('f1')
    fa = _arr('fa24h')
    if prec.size != thr.size:
        prec = np.full_like(thr, np.nan, dtype=np.float32)
    if rec.size != thr.size:
        rec = np.full_like(thr, np.nan, dtype=np.float32)
    if f1.size != thr.size:
        f1 = np.full_like(thr, np.nan, dtype=np.float32)
    if fa.size != thr.size:
        fa = np.full_like(thr, np.nan, dtype=np.float32)

    def _nanargmax(a: np.ndarray) -> int:
        if a.size == 0:
            return 0
        ok = np.isfinite(a)
        if ok.any():
            return int(np.argmax(np.where(ok, a, -np.inf)))
        return 0

    def _nanargmin(a: np.ndarray) -> int:
        if a.size == 0:
            return 0
        ok = np.isfinite(a)
        if ok.any():
            return int(np.argmin(np.where(ok, a, np.inf)))
        return 0

    # OP2: best F1
    i2 = _nanargmax(f1)

    # OP1: recall constraint, then minimal FA/day
    mask1 = np.isfinite(rec) & (rec >= float(op1_recall))
    if mask1.any():
        idxs = np.where(mask1)[0]
        # sort by (fa asc, f1 desc, precision desc)
        keys = np.lexsort((
            -np.nan_to_num(prec[idxs], nan=-np.inf),
            -np.nan_to_num(f1[idxs], nan=-np.inf),
            np.nan_to_num(fa[idxs], nan=np.inf),
        ))
        i1 = int(idxs[keys[0]])
    else:
        # fallback: closest to target recall, then best F1
        if np.isfinite(rec).any():
            dist = np.abs(rec - float(op1_recall))
            i1 = int(np.argmin(np.where(np.isfinite(dist), dist, np.inf)))
            # if multiple tie, prefer higher F1
            tied = np.where(np.isfinite(dist) & (dist == dist[i1]))[0]
            if tied.size > 1:
                i1 = int(tied[_nanargmax(f1[tied])])
        else:
            i1 = i2

    # OP3: FA/day constraint, then maximize recall
    mask3 = np.isfinite(fa) & (fa <= float(op3_fa24h))
    if mask3.any():
        idxs = np.where(mask3)[0]
        # sort by (rec desc, f1 desc, precision desc)
        keys = np.lexsort((
            -np.nan_to_num(prec[idxs], nan=-np.inf),
            -np.nan_to_num(f1[idxs], nan=-np.inf),
            -np.nan_to_num(rec[idxs], nan=-np.inf),
        ))
        i3 = int(idxs[keys[0]])
    else:
        # fallback: minimal FA/day overall
        i3 = _nanargmin(fa)

    def _pack(i: int) -> Dict[str, float]:
        return {
            'tau_high': float(thr[i]),
            'tau_low': float(tau_low[i]) if np.isfinite(tau_low[i]) else float(thr[i]) * 0.8,
            'precision': float(prec[i]) if np.isfinite(prec[i]) else float('nan'),
            'recall': float(rec[i]) if np.isfinite(rec[i]) else float('nan'),
            'f1': float(f1[i]) if np.isfinite(f1[i]) else float('nan'),
            'fa24h': float(fa[i]) if np.isfinite(fa[i]) else float('nan'),
        }

    return {
        'OP1': _pack(i1),
        'OP2': _pack(i2),
        'OP3': _pack(i3),
    }
    ema_alpha = float(alert_base.ema_alpha)
    overlap_slack = float(overlap_slack_s)
