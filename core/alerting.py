#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
core/alerting.py

Real-time alert policy + event-level metrics + triage state machines.

This file contains three main layers:

A) Alert Policy (deployment-like)
   - Smooth probabilities with EMA
   - Start alert with persistence (k-of-n) above tau_high
   - End alert with hysteresis below tau_low
   - Cooldown blocks frequent re-triggering
   - Optional: quality-adaptive thresholds (increase thresholds if pose quality is low)
   - Optional: confirmation stage (pending -> confirmed) using extra heuristics

B) Event-level Metrics (offline evaluation)
   - Convert window-level labels (y_true) into GT events
   - Run alert policy to obtain alert events
   - Compute recall, precision, F1, false alerts per hour/day, detection delay

C) Triage State Machines (your 3-state “fall / uncertain / not_fall” logic)
   - Single model and dual model “possible -> confirmed -> resolved” sequences

Important conventions
--------------------
- probs: probability sequence in [0, 1] (caller applies sigmoid already)
- times_s: seconds timestamps per probability sample (same length as probs)
- indices in AlertEvent are indices into the probs/times arrays (NOT video frame ids)

Span/window definitions:
- Many parts of the repo use window center time for evaluation,
  so GT events from windows are approximate (good enough for OP fitting).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


# ============================================================
# 1) Config + data structures
# ============================================================

@dataclass(frozen=True)
class AlertCfg:
    """
    Parameters controlling the real-time alert policy.

    ema_alpha:
      EMA smoothing factor for probabilities.
      alpha=0 -> no smoothing
      alpha close to 1 -> very reactive (less smoothing)
      alpha small -> smoother, slower

    k, n:
      Persistence trigger: require >=k "high" samples within last n samples
      before starting an alert.
      Example: k=2, n=3 means: in last 3 windows, at least 2 must be above tau_high.

    tau_high:
      Start threshold (after k-of-n)

    tau_low:
      End threshold (hysteresis). Once alert is active, it stays active while
      p >= tau_low and ends when p < tau_low.

    cooldown_s:
      After an alert ends, block new alert starts for this many seconds.

    quality_adapt + quality_*:
      Optional robustness layer. If enabled, thresholds increase when quality is low:
        tau_high_eff = tau_high + quality_boost * (1 - q)
        tau_low_eff  = tau_low  + quality_boost_low * (1 - q)
      where q in [0,1] is "pose quality".

      quality_min:
        If >0, block new alert starts when q < quality_min.

    confirm + confirm_*:
      Optional confirmation stage.
      If enabled, a persistence trigger creates a "pending" start.
      Pending is promoted to real alert if a confirm condition is met within confirm_s.

      confirm_require_low:
        If True, require an "armed" state: the stream must go below tau_low at least once
        before a new alert can start. This prevents immediate re-triggering from
        lingering in the uncertain band.
    """

    # Base thresholds + persistence
    ema_alpha: float = 0.20
    k: int = 2
    n: int = 3
    tau_high: float = 0.90
    tau_low: float = 0.70
    cooldown_s: float = 30.0

    # Optional pose-quality adaptation
    quality_adapt: bool = False
    quality_min: float = 0.0
    quality_boost: float = 0.15
    quality_boost_low: float = 0.05

    # Optional confirmation stage
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

            quality_adapt=bool(d.get("quality_adapt", False)),
            quality_min=float(d.get("quality_min", 0.0)),
            quality_boost=float(d.get("quality_boost", 0.15)),
            quality_boost_low=float(d.get("quality_boost_low", 0.05)),

            confirm=bool(d.get("confirm", False)),
            confirm_s=float(d.get("confirm_s", 2.0)),
            confirm_min_lying=float(d.get("confirm_min_lying", 0.65)),
            confirm_max_motion=float(d.get("confirm_max_motion", 0.08)),
            confirm_require_low=bool(d.get("confirm_require_low", True)),
        )


@dataclass
class AlertEvent:
    """
    One alert interval.

    start_idx/end_idx:
      indices into the probs/times arrays.

    start_time_s/end_time_s:
      actual seconds timestamps.

    peak_p:
      max smoothed probability reached during the event.
    """
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
    """
    Event-level evaluation metrics for one video.

    event_recall:
      fraction of GT events that overlap at least one alert event.

    false_alerts_per_hour/day:
      number of alert events NOT overlapping any GT event, normalized by duration.

    mean/median_delay_s:
      delay from GT event start to first overlapping alert start.

    counts:
      n_gt_events, n_alert_events, matched, true/false alert events

    event_precision/F1:
      precision and F1 at event level (not window level).
    """
    event_recall: float
    false_alerts_per_hour: float
    false_alerts_per_day: float
    mean_delay_s: float
    median_delay_s: float

    n_gt_events: int
    n_alert_events: int
    n_matched_gt: int
    n_true_alerts: int
    n_false_alerts: int

    event_precision: float
    event_f1: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ============================================================
# 2) Helpers: EMA smoothing, persistence, times
# ============================================================

def ema_smooth(x: np.ndarray, alpha: float) -> np.ndarray:
    """
    Exponential moving average smoothing.

    y[0] = x[0]
    y[i] = alpha*x[i] + (1-alpha)*y[i-1]

    alpha=0 -> returns x
    alpha close to 1 -> minimal smoothing
    """
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    if x.size == 0:
        return x
    a = float(alpha)
    if not (0.0 <= a <= 1.0):
        raise ValueError("ema_alpha must be in [0,1]")
    if a == 0.0:
        return x.copy()

    y = np.empty_like(x)
    y[0] = x[0]
    for i in range(1, x.size):
        y[i] = a * x[i] + (1.0 - a) * y[i - 1]
    return y


def _k_of_n(high01: np.ndarray, k: int, n: int) -> np.ndarray:
    """
    Persistence trigger:
      output[i] = True if sum(high01[i-n+1 : i+1]) >= k

    high01 is int array (0/1).
    """
    high01 = np.asarray(high01, dtype=np.int32).reshape(-1)
    k = int(k)
    n = int(n)

    if high01.size == 0:
        return high01.astype(bool)
    if n <= 1:
        return (high01 >= 1).astype(bool)

    # Sliding window sum via cumulative sum
    csum = np.cumsum(high01, dtype=np.int32)
    out = np.zeros_like(high01, dtype=bool)
    for i in range(high01.size):
        j0 = max(0, i - n + 1)
        s = csum[i] - (csum[j0 - 1] if j0 > 0 else 0)
        out[i] = (s >= k)
    return out


def times_from_windows(ws: Sequence[int], we: Sequence[int], fps: float, *, mode: str = "center") -> np.ndarray:
    """
    Convert window frame indices to timestamps.

    ws/we are window start/end frame indices (end is inclusive in your repo).

    mode:
      - "start" : t = w_start / fps
      - "end"   : t = w_end / fps
      - "center": t = (w_start + w_end)/2 / fps
    """
    ws = np.asarray(ws, dtype=np.float32).reshape(-1)
    we = np.asarray(we, dtype=np.float32).reshape(-1)
    f = float(fps) if float(fps) > 0 else 30.0

    if mode == "start":
        return ws / f
    if mode == "end":
        return we / f
    return (ws + we) * 0.5 / f


# ============================================================
# 3) Alert detection policy (the deploy-like logic)
# ============================================================

def _adapt_threshold(tau: float, q: Optional[float], boost: float, enabled: bool) -> float:
    """
    Quality-adaptive threshold:
      tau_eff = tau + boost*(1-q)

    If not enabled or q is None -> tau_eff = tau
    """
    if not enabled or q is None:
        return float(tau)
    qq = float(np.clip(q, 0.0, 1.0))
    return float(np.clip(float(tau) + float(boost) * (1.0 - qq), 0.0, 1.0))


def detect_alert_events(
    probs: Sequence[float],
    times_s: Sequence[float],
    cfg: AlertCfg,
    *,
    lying_score: Optional[Sequence[float]] = None,
    motion_score: Optional[Sequence[float]] = None,
    quality_score: Optional[Sequence[float]] = None,
) -> Tuple[np.ndarray, List[AlertEvent]]:
    """
    Run the alert policy on a probability sequence.

    Returns:
      active_mask: bool array True when "alert is active"
      events: list of AlertEvent intervals

    Pipeline:
      1) smooth probs with EMA
      2) compute per-step effective thresholds (optional quality adaptation)
      3) persistence trigger using tau_high and k-of-n
      4) start alert when trigger fires (and cooldown/armed allows it)
      5) end alert when prob drops below tau_low
      6) cooldown blocks new starts for cfg.cooldown_s seconds

    Confirmation stage (optional):
      - If cfg.confirm is True AND (lying_score or motion_score is provided),
        we use "pending" state:
          pending starts when persistence triggers,
          then within confirm_s seconds we check heuristics:
            lying_score >= confirm_min_lying
            motion_score <= confirm_max_motion
        If confirmed, we start alert at the pending start.
    """
    p = np.asarray(probs, dtype=np.float32).reshape(-1)
    t = np.asarray(times_s, dtype=np.float32).reshape(-1)

    if p.size == 0:
        return np.asarray([], dtype=bool), []
    if t.size != p.size:
        raise ValueError("detect_alert_events: probs and times_s must have the same length")

    # 1) Smooth probabilities (EMA)
    ps = ema_smooth(p, cfg.ema_alpha)

    # 2) Optional aligned auxiliary sequences
    qs = None
    if quality_score is not None:
        qs = np.asarray(quality_score, dtype=np.float32).reshape(-1)
        if qs.size != ps.size:
            qs = None

    ls = None
    ms = None
    if lying_score is not None:
        ls = np.asarray(lying_score, dtype=np.float32).reshape(-1)
        if ls.size != ps.size:
            ls = None
    if motion_score is not None:
        ms = np.asarray(motion_score, dtype=np.float32).reshape(-1)
        if ms.size != ps.size:
            ms = None

    use_quality = bool(cfg.quality_adapt) and (qs is not None)

    # 3) Build "high" indicator per step for persistence trigger
    #    high[i] = 1 if ps[i] >= tau_high_eff(i) and (quality not too low)
    high = np.zeros_like(ps, dtype=np.int32)
    for i in range(ps.size):
        q_i = float(qs[i]) if qs is not None else None
        tau_hi_i = _adapt_threshold(cfg.tau_high, q_i, cfg.quality_boost, use_quality)

        # Optional: block *new starts* if pose quality is below threshold
        if qs is not None and float(cfg.quality_min) > 0 and float(qs[i]) < float(cfg.quality_min):
            high[i] = 0
        else:
            high[i] = int(float(ps[i]) >= float(tau_hi_i))

    pers = _k_of_n(high, cfg.k, cfg.n)

    active = np.zeros_like(pers, dtype=bool)
    events: List[AlertEvent] = []

    cooldown_until_s = -1e9

    # Event state
    in_event = False
    start_i = 0
    peak = 0.0

    # Armed logic:
    # If confirm_require_low is enabled (commonly used), require at least one dip below tau_low
    # since the last event ended before allowing a new start.
    armed = True

    # Pending confirmation state
    pending = False
    pending_start_i = 0
    pending_deadline_s = -1e9
    pending_peak = 0.0

    for i in range(ps.size):
        ti = float(t[i])
        q_i = float(qs[i]) if qs is not None else None

        # Effective tau_low at this time (if quality-adapt is enabled)
        tau_lo_i = _adapt_threshold(cfg.tau_low, q_i, cfg.quality_boost_low, use_quality)

        # Re-arm only when idle AND stream is clearly low.
        if (not in_event) and (not pending) and float(ps[i]) < float(tau_lo_i):
            armed = True

        # -------------------------
        # If currently in an event
        # -------------------------
        if in_event:
            active[i] = True
            peak = max(peak, float(ps[i]))

            # End condition: drop below tau_low (quality-adapted)
            if float(ps[i]) < float(tau_lo_i):
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

                cooldown_until_s = float(t[end_i]) + float(cfg.cooldown_s)

                # Require a fresh dip below tau_low before starting again.
                armed = False

            continue

        # -------------------------
        # Pending confirmation state
        # -------------------------
        if pending:
            pending_peak = max(pending_peak, float(ps[i]))

            # If we drop below tau_low, abandon pending
            if float(ps[i]) < float(tau_lo_i):
                pending = False
                continue

            # Timeout: if not confirmed in time -> abandon pending
            if float(ti) > float(pending_deadline_s):
                pending = False
                continue

            # Confirmation logic:
            #  - If heuristics exist, apply them
            #  - Otherwise (rare), fall back to probability-only check
            ok_prob = float(ps[i]) >= float(tau_lo_i)

            ok_lying = True
            ok_motion = True
            if ls is not None:
                ok_lying = float(ls[i]) >= float(cfg.confirm_min_lying)
            if ms is not None:
                ok_motion = float(ms[i]) <= float(cfg.confirm_max_motion)

            if ok_prob and ok_lying and ok_motion:
                # Promote pending -> real event
                in_event = True
                pending = False

                start_i = int(pending_start_i)
                peak = float(pending_peak)

                active[i] = True
                armed = False

            continue

        # -------------------------
        # Idle state (not in_event, not pending)
        # -------------------------
        if ti < float(cooldown_until_s):
            continue

        if pers[i]:
            # Optional "require low" (armed) — prevents immediate re-triggering
            if bool(cfg.confirm_require_low) and (not bool(armed)):
                continue

            # If confirm stage enabled AND heuristics exist, start pending instead of immediate event
            if bool(cfg.confirm) and (ls is not None or ms is not None):
                pending = True
                pending_start_i = i
                pending_deadline_s = float(t[i]) + float(cfg.confirm_s)
                pending_peak = float(ps[i])
                armed = False
                continue

            # Otherwise start event immediately
            in_event = True
            start_i = i
            peak = float(ps[i])
            active[i] = True
            armed = False

    # If stream ended while still in event, close it
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


def classify_states(
    probs: Sequence[float],
    times_s: Sequence[float],
    cfg: AlertCfg,
    *,
    lying_score: Optional[Sequence[float]] = None,
    motion_score: Optional[Sequence[float]] = None,
    quality_score: Optional[Sequence[float]] = None,
) -> Dict[str, Any]:
    """
    Classify each time step into:
      - clear   : ps < tau_low_eff
      - suspect : tau_low_eff <= ps < tau_high_eff and not in alert
      - alert   : alert policy is active
      - pending : ps >= tau_high_eff but confirm/gates prevent alert (not in alert)

    Returns:
      {
        "ps": smoothed probabilities,
        "clear": bool mask,
        "suspect": bool mask,
        "alert": bool mask,
        "pending": bool mask,
      }

    NOTE:
    - We run detect_alert_events() using the same cfg (including quality adaptation),
      but set ema_alpha=0 because ps is already smoothed here.
    """
    p = np.asarray(probs, dtype=np.float32).reshape(-1)
    t = np.asarray(times_s, dtype=np.float32).reshape(-1)
    if p.size == 0:
        return {
            "ps": np.asarray([], dtype=np.float32),
            "clear": np.asarray([], dtype=bool),
            "suspect": np.asarray([], dtype=bool),
            "alert": np.asarray([], dtype=bool),
            "pending": np.asarray([], dtype=bool),
        }
    if t.size != p.size:
        raise ValueError("classify_states: probs and times_s must have the same length")

    ps = ema_smooth(p, cfg.ema_alpha)

    # Run alert detection on already-smoothed probabilities
    cfg2 = AlertCfg(
        ema_alpha=0.0,
        k=int(cfg.k),
        n=int(cfg.n),
        tau_high=float(cfg.tau_high),
        tau_low=float(cfg.tau_low),
        cooldown_s=float(cfg.cooldown_s),

        quality_adapt=bool(cfg.quality_adapt),
        quality_min=float(cfg.quality_min),
        quality_boost=float(cfg.quality_boost),
        quality_boost_low=float(cfg.quality_boost_low),

        confirm=bool(cfg.confirm),
        confirm_s=float(cfg.confirm_s),
        confirm_min_lying=float(cfg.confirm_min_lying),
        confirm_max_motion=float(cfg.confirm_max_motion),
        confirm_require_low=bool(cfg.confirm_require_low),
    )

    alert_mask, _ = detect_alert_events(
        ps, t, cfg2,
        lying_score=lying_score,
        motion_score=motion_score,
        quality_score=quality_score,
    )

    # Build effective threshold bands (for suspect/clear)
    qs = None
    if quality_score is not None:
        qs = np.asarray(quality_score, dtype=np.float32).reshape(-1)
        if qs.size != ps.size:
            qs = None

    use_quality = bool(cfg.quality_adapt) and (qs is not None)

    tau_hi = np.full_like(ps, float(cfg.tau_high), dtype=np.float32)
    tau_lo = np.full_like(ps, float(cfg.tau_low), dtype=np.float32)

    if use_quality:
        q = np.clip(qs, 0.0, 1.0)
        tau_hi = np.clip(tau_hi + float(cfg.quality_boost) * (1.0 - q), 0.0, 1.0)
        tau_lo = np.clip(tau_lo + float(cfg.quality_boost_low) * (1.0 - q), 0.0, 1.0)

    suspect = (~alert_mask) & (ps >= tau_lo) & (ps < tau_hi)
    clear = (~alert_mask) & (ps < tau_lo)
    pending = (~alert_mask) & (ps >= tau_hi)

    return {"ps": ps, "clear": clear, "suspect": suspect, "pending": pending, "alert": alert_mask}


# ============================================================
# 4) Event metrics (video-level)
# ============================================================

def _events_from_positive_windows(times_s: np.ndarray, y_true: np.ndarray, merge_gap_s: float) -> List[Tuple[float, float]]:
    """
    Build GT events from contiguous y_true==1 windows.

    times_s are timestamps (often window center times).
    merge_gap_s merges nearby windows into one GT event.
    """
    t = np.asarray(times_s, dtype=np.float32).reshape(-1)
    y = (np.asarray(y_true, dtype=np.int32).reshape(-1) > 0).astype(np.int32)
    if t.size == 0 or y.size == 0:
        return []
    if t.size != y.size:
        raise ValueError("times_s and y_true must have same length")

    pos = np.where(y == 1)[0]
    if pos.size == 0:
        return []

    events: List[Tuple[float, float]] = []
    cur_s = float(t[pos[0]])
    cur_e = float(t[pos[0]])
    gap = float(merge_gap_s)

    for idx in pos[1:]:
        ti = float(t[idx])
        if ti <= cur_e + gap:
            cur_e = max(cur_e, ti)
        else:
            events.append((cur_s, cur_e))
            cur_s = ti
            cur_e = ti

    events.append((cur_s, cur_e))
    return events


def _overlap(a: Tuple[float, float], b: Tuple[float, float], *, slack_s: float = 0.0) -> bool:
    """Return True if time intervals a and b overlap (with optional slack)."""
    (as_, ae) = a
    (bs, be) = b
    return (ae + float(slack_s)) >= bs and (be + float(slack_s)) >= as_


def event_metrics_from_windows(
    probs: Sequence[float],
    y_true: Sequence[int],
    times_s: Sequence[float],
    alert_cfg: AlertCfg,
    *,
    duration_s: Optional[float] = None,
    merge_gap_s: float = 2.0,
    overlap_slack_s: float = 0.0,
    lying_score: Optional[Sequence[float]] = None,
    motion_score: Optional[Sequence[float]] = None,
    quality_score: Optional[Sequence[float]] = None,
) -> Tuple[EventMetrics, Dict[str, Any]]:
    """
    Compute event-level metrics for ONE video.

    GT events:
      derived from y_true==1 windows, merged by merge_gap_s.

    Alert events:
      detect_alert_events(...) output.

    false_alerts_per_{hour,day}:
      count only alerts that do NOT overlap any GT event.

    Note on unlabeled streams:
      If y_true contains only -1/0, then GT events is empty and all alerts become "false".
    """
    p = np.asarray(probs, dtype=np.float32).reshape(-1)
    y = np.asarray(y_true, dtype=np.int32).reshape(-1)
    t = np.asarray(times_s, dtype=np.float32).reshape(-1)

    if p.size == 0:
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

    if t.size != p.size or y.size != p.size:
        raise ValueError("event_metrics_from_windows: probs, y_true, times_s must have same length")

    if duration_s is None:
        duration_s = float(t.max() - t.min()) if t.size else 0.0

    duration_h = float(duration_s) / 3600.0 if duration_s > 0 else float("nan")
    duration_d = float(duration_s) / 86400.0 if duration_s > 0 else float("nan")

    # Build GT events from positive windows only
    y01 = (y == 1).astype(np.int32)
    gt_events = _events_from_positive_windows(t, y01, merge_gap_s=float(merge_gap_s))

    # Alert events from policy
    _, alert_events = detect_alert_events(
        p, t, alert_cfg,
        lying_score=lying_score,
        motion_score=motion_score,
        quality_score=quality_score,
    )
    alert_intervals = [(ev.start_time_s, ev.end_time_s) for ev in alert_events]

    # Match GT -> earliest overlapping alert for delay stats
    matched_gt = 0
    delays: List[float] = []
    for (gs, ge) in gt_events:
        first_alert_start = None
        for ev in alert_events:
            if _overlap((gs, ge), (ev.start_time_s, ev.end_time_s), slack_s=float(overlap_slack_s)):
                first_alert_start = ev.start_time_s
                break
        if first_alert_start is not None:
            matched_gt += 1
            delays.append(max(0.0, float(first_alert_start - gs)))

    recall = float(matched_gt / len(gt_events)) if gt_events else float("nan")

    # Event precision / false alerts
    true_alerts = 0
    false_alerts = 0
    for interval in alert_intervals:
        if any(_overlap(interval, ge, slack_s=float(overlap_slack_s)) for ge in gt_events):
            true_alerts += 1
        else:
            false_alerts += 1

    n_alert = len(alert_events)
    precision = float(true_alerts / n_alert) if n_alert > 0 else float("nan")
    f1 = float("nan")
    if np.isfinite(precision) and np.isfinite(recall) and (precision + recall) > 0:
        f1 = float(2.0 * precision * recall / (precision + recall))

    fa_per_hour = float(false_alerts / duration_h) if np.isfinite(duration_h) and duration_h > 0 else float("nan")
    fa_per_day = float(false_alerts / duration_d) if np.isfinite(duration_d) and duration_d > 0 else float("nan")

    mean_delay = float(np.mean(delays)) if delays else float("nan")
    median_delay = float(np.median(delays)) if delays else float("nan")

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


# ============================================================
# 5) Alert-policy sweep (tau_high sweep under real policy)
# ============================================================

def sweep_alert_policy_from_windows(
    probs: Sequence[float],
    y_true: Sequence[int],
    video_ids: Sequence[str],
    w_start: Sequence[int],
    w_end: Sequence[int],
    fps: Sequence[float],
    *,
    alert_base: AlertCfg,
    lying_score: Optional[Sequence[float]] = None,
    motion_score: Optional[Sequence[float]] = None,
    quality_score: Optional[Sequence[float]] = None,
    thr_min: float = 0.05,
    thr_max: float = 0.95,
    thr_step: float = 0.01,
    tau_low_ratio: float = 0.80,
    merge_gap_s: Optional[float] = None,
    overlap_slack_s: float = 0.0,
    time_mode: str = "center",
    fps_default: float = 30.0,
    # Optional: separate negative/unlabeled stream windows for FA/24h estimation
    fa_probs: Optional[Sequence[float]] = None,
    fa_video_ids: Optional[Sequence[str]] = None,
    fa_w_start: Optional[Sequence[int]] = None,
    fa_w_end: Optional[Sequence[int]] = None,
    fa_fps: Optional[Sequence[float]] = None,
    fa_lying_score: Optional[Sequence[float]] = None,
    fa_motion_score: Optional[Sequence[float]] = None,
    fa_quality_score: Optional[Sequence[float]] = None,
) -> Tuple[Dict[str, List[float]], Dict[str, Any]]:
    """
    Sweep tau_high under the REAL alert policy.

    Returns:
      sweep dict of lists:
        thr (=tau_high), tau_low, precision, recall, f1, fa24h, mean_delay_s, ...
      meta:
        durations, n_videos, merge_gap_s, etc.

    Notes:
      - GT events come from y_true positive windows (span-aware windowing helps).
      - If fa_* stream provided, FA/24h is computed from it (more deployment-realistic).
    """
    probs = np.asarray(probs, dtype=np.float32).reshape(-1)
    y_true = np.asarray(y_true, dtype=np.int32).reshape(-1)
    vids = np.asarray(video_ids).astype(str).reshape(-1)
    ws = np.asarray(w_start, dtype=np.int32).reshape(-1)
    we = np.asarray(w_end, dtype=np.int32).reshape(-1)
    fps_arr = np.asarray(fps, dtype=np.float32).reshape(-1)

    if probs.size == 0:
        return {"thr": [], "recall": [], "fa24h": []}, {"n_videos": 0}

    # Align optional aux arrays
    def _aligned_float(arr: Optional[Sequence[float]]) -> Optional[np.ndarray]:
        if arr is None:
            return None
        a = np.asarray(arr, dtype=np.float32).reshape(-1)
        return a if a.size == probs.size else None

    ls_all = _aligned_float(lying_score)
    ms_all = _aligned_float(motion_score)
    qs_all = _aligned_float(quality_score)

    # Group by video (important: alerting is per video)
    unique_vids = list(dict.fromkeys(list(vids)))

    per_video: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, float, float, np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]] = {}
    total_duration_s = 0.0

    for v in unique_vids:
        mv = vids == v
        if not mv.any():
            continue

        idx = np.argsort(ws[mv])
        p_v = probs[mv][idx]
        y_v = y_true[mv][idx]
        ws_v = ws[mv][idx]
        we_v = we[mv][idx]

        fps_v = float(np.median(fps_arr[mv])) if np.isfinite(fps_arr[mv]).any() else float(fps_default)
        if fps_v <= 0:
            fps_v = float(fps_default)

        t_v = times_from_windows(ws_v, we_v, fps_v, mode=time_mode)
        duration_s = float((we_v.max() - ws_v.min() + 1) / max(1e-6, fps_v))
        total_duration_s += max(0.0, duration_s)

        ls_v = ls_all[mv][idx] if ls_all is not None else None
        ms_v = ms_all[mv][idx] if ms_all is not None else None
        qs_v = qs_all[mv][idx] if qs_all is not None else None

        per_video[v] = (p_v, y_v, t_v, duration_s, fps_v, ws_v, we_v, ls_v, ms_v, qs_v)

    # Optional FA-only stream (all alerts are false alerts by definition)
    per_video_fa = {}
    total_duration_s_fa = 0.0

    if fa_probs is not None and fa_video_ids is not None and fa_w_start is not None and fa_w_end is not None and fa_fps is not None:
        fa_p = np.asarray(fa_probs, dtype=np.float32).reshape(-1)
        fa_vids_arr = np.asarray(fa_video_ids).astype(str).reshape(-1)
        fa_ws_arr = np.asarray(fa_w_start, dtype=np.int32).reshape(-1)
        fa_we_arr = np.asarray(fa_w_end, dtype=np.int32).reshape(-1)
        fa_fps_arr = np.asarray(fa_fps, dtype=np.float32).reshape(-1)

        if fa_p.size == fa_vids_arr.size == fa_ws_arr.size == fa_we_arr.size == fa_fps_arr.size and fa_p.size > 0:
            fa_unique_vids = list(dict.fromkeys(list(fa_vids_arr)))

            fa_ls_all = np.asarray(fa_lying_score, dtype=np.float32).reshape(-1) if fa_lying_score is not None else None
            fa_ms_all = np.asarray(fa_motion_score, dtype=np.float32).reshape(-1) if fa_motion_score is not None else None
            fa_qs_all = np.asarray(fa_quality_score, dtype=np.float32).reshape(-1) if fa_quality_score is not None else None

            for v in fa_unique_vids:
                mv = fa_vids_arr == v
                if not mv.any():
                    continue

                idx = np.argsort(fa_ws_arr[mv])
                p_v = fa_p[mv][idx]
                y_v = np.zeros_like(p_v, dtype=np.int32)  # no GT events on FA stream

                ws_v = fa_ws_arr[mv][idx]
                we_v = fa_we_arr[mv][idx]

                fps_v = float(np.median(fa_fps_arr[mv])) if np.isfinite(fa_fps_arr[mv]).any() else float(fps_default)
                if fps_v <= 0:
                    fps_v = float(fps_default)

                t_v = times_from_windows(ws_v, we_v, fps_v, mode=time_mode)
                duration_s = float((we_v.max() - ws_v.min() + 1) / max(1e-6, fps_v))
                total_duration_s_fa += max(0.0, duration_s)

                ls_v = fa_ls_all[mv][idx] if (fa_ls_all is not None and fa_ls_all.size == fa_p.size) else None
                ms_v = fa_ms_all[mv][idx] if (fa_ms_all is not None and fa_ms_all.size == fa_p.size) else None
                qs_v = fa_qs_all[mv][idx] if (fa_qs_all is not None and fa_qs_all.size == fa_p.size) else None

                per_video_fa[v] = (p_v, y_v, t_v, duration_s, fps_v, ws_v, we_v, ls_v, ms_v, qs_v)

    # Choose merge gap if not provided
    if merge_gap_s is None:
        gaps = []
        for _, pack in per_video.items():
            t_v = pack[2]
            if t_v.size >= 2:
                gaps.append(float(np.median(np.diff(t_v))))
        med_gap = float(np.median(gaps)) if gaps else 0.5
        merge_gap_s = max(0.25, 2.0 * med_gap)

    thr_values = np.arange(float(thr_min), float(thr_max) + 1e-12, float(thr_step), dtype=np.float32)

    out: Dict[str, List[float]] = {
        "thr": [],
        "tau_low": [],
        "precision": [],
        "recall": [],
        "f1": [],
        "fa24h": [],
        "fa_per_hour": [],
        "mean_delay_s": [],
        "median_delay_s": [],
        "n_gt_events": [],
        "n_alert_events": [],
        "n_true_alerts": [],
        "n_false_alerts": [],
    }

    for thr in thr_values:
        tau_high = float(thr)
        tau_low = float(max(0.0, min(tau_high, tau_high * float(tau_low_ratio))))

        cfg = AlertCfg(
            ema_alpha=float(alert_base.ema_alpha),
            k=int(alert_base.k),
            n=int(alert_base.n),
            tau_high=float(tau_high),
            tau_low=float(tau_low),
            cooldown_s=float(alert_base.cooldown_s),

            quality_adapt=bool(alert_base.quality_adapt),
            quality_min=float(alert_base.quality_min),
            quality_boost=float(alert_base.quality_boost),
            quality_boost_low=float(alert_base.quality_boost_low),

            confirm=bool(alert_base.confirm),
            confirm_s=float(alert_base.confirm_s),
            confirm_min_lying=float(alert_base.confirm_min_lying),
            confirm_max_motion=float(alert_base.confirm_max_motion),
            confirm_require_low=bool(alert_base.confirm_require_low),
        )

        gt_total = 0
        matched_gt_total = 0
        alert_total = 0
        true_alert_total = 0
        false_alert_total = 0
        false_alert_total_fa = 0

        delays_all: List[float] = []

        # Evaluate over labeled videos
        for _, (p_v, y_v, t_v, dur_s, _fps_v, _ws_v, _we_v, ls_v, ms_v, qs_v) in per_video.items():
            em, _ = event_metrics_from_windows(
                p_v, y_v, t_v, cfg,
                duration_s=float(dur_s),
                merge_gap_s=float(merge_gap_s),
                overlap_slack_s=float(overlap_slack_s),
                lying_score=ls_v,
                motion_score=ms_v,
                quality_score=qs_v,
            )
            gt_total += int(em.n_gt_events)
            matched_gt_total += int(em.n_matched_gt)
            alert_total += int(em.n_alert_events)
            true_alert_total += int(em.n_true_alerts)
            false_alert_total += int(em.n_false_alerts)

            if np.isfinite(em.mean_delay_s):
                delays_all.append(float(em.mean_delay_s))

        # Optional FA-only stream: compute false alerts from it
        if per_video_fa:
            for _, (p_v, y_v, t_v, dur_s, _fps_v, _ws_v, _we_v, _ls, _ms, _qs) in per_video_fa.items():
                em, _ = event_metrics_from_windows(
                    p_v, y_v, t_v, cfg,
                    duration_s=float(dur_s),
                    merge_gap_s=float(merge_gap_s),
                    overlap_slack_s=float(overlap_slack_s),
                )
                false_alert_total_fa += int(em.n_false_alerts)

        recall = float(matched_gt_total / gt_total) if gt_total > 0 else 0.0
        precision = float(true_alert_total / alert_total) if alert_total > 0 else float("nan")
        f1 = float("nan")
        if np.isfinite(precision) and np.isfinite(recall) and (precision + recall) > 0:
            f1 = float(2.0 * precision * recall / (precision + recall))

        # FA/24h reference: prefer FA-only stream if provided (deployment realism)
        ref_dur_s = float(total_duration_s_fa) if per_video_fa else float(total_duration_s)
        ref_false = int(false_alert_total_fa) if per_video_fa else int(false_alert_total)

        dur_h = float(ref_dur_s) / 3600.0 if ref_dur_s > 0 else float("nan")
        dur_d = float(ref_dur_s) / 86400.0 if ref_dur_s > 0 else float("nan")

        fa_h = float(ref_false / dur_h) if np.isfinite(dur_h) and dur_h > 0 else float("nan")
        fa_d = float(ref_false / dur_d) if np.isfinite(dur_d) and dur_d > 0 else float("nan")

        out["thr"].append(float(tau_high))
        out["tau_low"].append(float(tau_low))
        out["precision"].append(float(precision))
        out["recall"].append(float(recall))
        out["f1"].append(float(f1))
        out["fa24h"].append(float(fa_d))
        out["fa_per_hour"].append(float(fa_h))
        out["mean_delay_s"].append(float(np.mean(delays_all)) if delays_all else float("nan"))
        out["median_delay_s"].append(float(np.median(delays_all)) if delays_all else float("nan"))
        out["n_gt_events"].append(int(gt_total))
        out["n_alert_events"].append(int(alert_total))
        out["n_true_alerts"].append(int(true_alert_total))
        out["n_false_alerts"].append(int(false_alert_total))
        if per_video_fa:
            out.setdefault("n_false_alerts_fa", []).append(int(false_alert_total_fa))

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


# ============================================================
# 6) 3-state triage + state machines (kept compatible)
# ============================================================

TRIAGE_FALL = "fall"
TRIAGE_NOT_FALL = "not_fall"
TRIAGE_UNCERTAIN = "uncertain"

EVENT_POSSIBLE = "possible_fall"
EVENT_CONFIRMED = "fall_detected"
EVENT_RESOLVED = "resolved"


@dataclass
class TriageCfg:
    """
    Two-threshold triage + optional uncertainty gate (sigma).

    tau_low/tau_high:
      same idea as alerting thresholds, but used to classify instantaneous state.

    ema_alpha:
      EMA smoothing on the raw score (probability or mean score)

    sigma_max (optional):
      If provided and sigma > sigma_max -> UNCERTAIN regardless of mu
      (useful for MC-dropout uncertainty gating)
    """
    tau_low: float = 0.05
    tau_high: float = 0.90
    ema_alpha: float = 0.20
    sigma_max: Optional[float] = None


@dataclass
class SingleModeCfg:
    """
    Temporal logic for a single model triage pipeline.

    possible_k within possible_T_s:
      how many 'not clear' states needed to emit EVENT_POSSIBLE

    confirm_k_fall within confirm_T_s:
      how many FALL states needed to emit EVENT_CONFIRMED

    cooldown_*:
      after possible/confirmed, ignore new events for these durations
    """
    possible_k: int = 3
    possible_T_s: float = 2.0
    confirm_T_s: float = 3.6
    confirm_k_fall: int = 2
    cooldown_possible_s: float = 15.0
    cooldown_confirmed_s: float = 60.0


@dataclass
class DualModeCfg:
    """
    Temporal logic for dual-model fusion (TCN + GCN).

    confirm_k_tcn / confirm_k_gcn:
      how many FALL states needed in each model within confirm_T_s.

    require_both:
      True -> both models must satisfy (AND)
      False -> either can satisfy (OR)
    """
    possible_k: int = 3
    possible_T_s: float = 2.0
    confirm_T_s: float = 3.6
    confirm_k_tcn: int = 1
    confirm_k_gcn: int = 1
    require_both: bool = True
    cooldown_possible_s: float = 15.0
    cooldown_confirmed_s: float = 60.0


@dataclass
class TriageEvent:
    kind: str
    t_sec: float
    info: Dict[str, Any]


def triage_state(
    mu: float,
    tau_low: float,
    tau_high: float,
    sigma: Optional[float] = None,
    sigma_max: Optional[float] = None,
) -> str:
    """
    Return fall / uncertain / not_fall.

    If sigma gate is active and sigma is too high -> UNCERTAIN.
    Else:
      mu >= tau_high -> FALL
      mu <= tau_low  -> NOT_FALL
      otherwise      -> UNCERTAIN
    """
    if sigma is not None and sigma_max is not None and float(sigma) > float(sigma_max):
        return TRIAGE_UNCERTAIN
    if float(mu) >= float(tau_high):
        return TRIAGE_FALL
    if float(mu) <= float(tau_low):
        return TRIAGE_NOT_FALL
    return TRIAGE_UNCERTAIN


class SingleTriageStateMachine:
    """
    State machine for a single model triage stream.

    Emits events:
      - EVENT_POSSIBLE
      - EVENT_CONFIRMED
      - EVENT_RESOLVED

    Internal states:
      idle -> confirm -> (confirmed or resolved) -> idle (with cooldown)
    """

    def __init__(self, triage_cfg: TriageCfg, mode_cfg: SingleModeCfg):
        self.triage_cfg = triage_cfg
        self.mode_cfg = mode_cfg
        self.reset()

    def reset(self) -> None:
        self._ema: Optional[float] = None
        self._state: str = "idle"
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

    def step(self, t_sec: float, p_or_mu: float, sigma: Optional[float] = None) -> List[TriageEvent]:
        evs: List[TriageEvent] = []
        t = float(t_sec)

        mu = self._ema_update(float(p_or_mu))
        s = triage_state(mu, self.triage_cfg.tau_low, self.triage_cfg.tau_high, sigma=sigma, sigma_max=self.triage_cfg.sigma_max)

        if t < self._cooldown_until:
            return evs

        if self._state == "idle":
            if s != TRIAGE_NOT_FALL:
                self._sus_times.append(t)

            Tp = float(self.mode_cfg.possible_T_s)
            self._sus_times = [x for x in self._sus_times if (t - x) <= Tp]

            if len(self._sus_times) >= int(self.mode_cfg.possible_k):
                evs.append(TriageEvent(EVENT_POSSIBLE, t, {"mu": mu, "sigma": sigma}))
                self._state = "confirm"
                self._t_state0 = t
                self._fall_times = []
                self._sus_times = []
            return evs

        if self._state == "confirm":
            if s == TRIAGE_FALL:
                self._fall_times.append(t)

            Tc = float(self.mode_cfg.confirm_T_s)
            self._fall_times = [x for x in self._fall_times if (t - x) <= Tc]

            if len(self._fall_times) >= int(self.mode_cfg.confirm_k_fall):
                evs.append(TriageEvent(EVENT_CONFIRMED, t, {"mu": mu, "sigma": sigma}))
                self._state = "idle"
                self._cooldown_until = t + float(self.mode_cfg.cooldown_confirmed_s)
                self._fall_times = []
                return evs

            # Time out
            if (t - self._t_state0) > Tc:
                evs.append(TriageEvent(EVENT_RESOLVED, t, {"mu": mu, "sigma": sigma}))
                self._state = "idle"
                self._cooldown_until = t + float(self.mode_cfg.cooldown_possible_s)
                self._fall_times = []
            return evs

        return evs


class DualTriageStateMachine:
    """
    Dual-model triage (TCN + GCN).

    Emits:
      - EVENT_POSSIBLE when either model is suspicious enough over possible_T_s
      - EVENT_CONFIRMED when confirm conditions satisfy (AND/OR based on require_both)
      - EVENT_RESOLVED if confirmation times out

    This is for your Hybrid fusion workflows.
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
        if which == "tcn":
            a = float(self.triage_tcn.ema_alpha)
            if self._ema_tcn is None:
                self._ema_tcn = float(x)
            else:
                self._ema_tcn = a * float(x) + (1.0 - a) * float(self._ema_tcn)
            return float(self._ema_tcn)

        a = float(self.triage_gcn.ema_alpha)
        if self._ema_gcn is None:
            self._ema_gcn = float(x)
        else:
            self._ema_gcn = a * float(x) + (1.0 - a) * float(self._ema_gcn)
        return float(self._ema_gcn)

    def step(
        self,
        t_sec: float,
        p_tcn: float,
        p_gcn: float,
        sigma_tcn: Optional[float] = None,
        sigma_gcn: Optional[float] = None,
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
                evs.append(TriageEvent(EVENT_CONFIRMED, t, {"mu_tcn": mu_t, "mu_gcn": mu_g, "sigma_tcn": sigma_tcn, "sigma_gcn": sigma_gcn}))
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
