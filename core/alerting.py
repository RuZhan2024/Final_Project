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

def ema_smooth(x: np.ndarray, alpha: float) -> np.ndarray:
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
    """Return a boolean array where index i is True if >=k of the last n samples are 1."""
    high01 = np.asarray(high01, dtype=np.int32).reshape(-1)
    k = int(k); n = int(n)
    if high01.size == 0:
        return high01.astype(bool)
    if n <= 1:
        return (high01 >= 1).astype(bool)
    if k <= 1:
        # any of last n is enough
        out = np.zeros_like(high01, dtype=bool)
        run = 0
        for i in range(high01.size):
            run = min(n, run + 1)
            if high01[max(0, i - n + 1): i + 1].sum() >= 1:
                out[i] = True
        return out
    # sliding window sum
    csum = np.cumsum(high01, dtype=np.int32)
    out = np.zeros_like(high01, dtype=bool)
    for i in range(high01.size):
        j0 = max(0, i - n + 1)
        s = csum[i] - (csum[j0 - 1] if j0 > 0 else 0)
        out[i] = (s >= k)
    return out


def times_from_windows(ws: Sequence[int], we: Sequence[int], fps: float, *, mode: str = "center") -> np.ndarray:
    """Convert window frame indices to timestamps.

    mode:
      - "start": w_start / fps
      - "end":   w_end / fps
      - "center": (w_start + w_end)/2 / fps
    """
    ws = np.asarray(ws, dtype=np.float32).reshape(-1)
    we = np.asarray(we, dtype=np.float32).reshape(-1)
    f = float(fps) if float(fps) > 0 else 30.0
    if mode == "start":
        return ws / f
    if mode == "end":
        return we / f
    return (ws + we) * 0.5 / f


# ---------------- alert detection ----------------

def detect_alert_events(
    probs: Sequence[float],
    times_s: Sequence[float],
    cfg: AlertCfg,
) -> Tuple[np.ndarray, List[AlertEvent]]:
    """Run alert policy on a probability sequence. Return (active_mask, events)."""
    p = np.asarray(probs, dtype=np.float32).reshape(-1)
    t = np.asarray(times_s, dtype=np.float32).reshape(-1)
    if p.size == 0:
        return np.asarray([], dtype=bool), []

    # smooth
    ps = ema_smooth(p, cfg.ema_alpha)

    # persistence trigger uses tau_high
    high = (ps >= float(cfg.tau_high)).astype(np.int32)
    pers = _k_of_n(high, cfg.k, cfg.n)

    active = np.zeros_like(pers, dtype=bool)
    events: List[AlertEvent] = []

    cooldown_until = -1e9  # time (seconds)
    in_event = False
    start_i = 0
    peak = 0.0

    for i in range(pers.size):
        ti = float(t[i])

        if in_event:
            # stay active while ps >= tau_low
            active[i] = True
            peak = max(peak, float(ps[i]))
            if float(ps[i]) < float(cfg.tau_low):
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
                cooldown_until = float(t[end_i]) + float(cfg.cooldown_s)
            continue

        # not in event: can we start?
        if ti < cooldown_until:
            continue

        if pers[i]:
            in_event = True
            start_i = i
            peak = float(ps[i])
            active[i] = True

    if in_event:
        end_i = p.size - 1
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


# ---------------- event metrics ----------------

def _events_from_positive_windows(times_s: np.ndarray, y_true: np.ndarray, merge_gap_s: float) -> List[Tuple[float, float]]:
    """Build GT events from y_true=1 windows. Each GT event is (start_s, end_s)."""
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
    (as_, ae) = a
    (bs, be) = b
    return (ae + slack_s) >= bs and (be + slack_s) >= as_


def event_metrics_from_windows(
    probs: Sequence[float],
    y_true: Sequence[int],
    times_s: Sequence[float],
    alert_cfg: AlertCfg,
    *,
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

    if duration_s is None:
        duration_s = float(t.max() - t.min()) if t.size else 0.0
    duration_h = float(duration_s) / 3600.0 if duration_s > 0 else float("nan")
    duration_d = float(duration_s) / 86400.0 if duration_s > 0 else float("nan")

    # GT events (ignore y=-1 for unlabeled, treat as negatives)
    y01 = (y == 1).astype(np.int32)
    gt_events = _events_from_positive_windows(t, y01, merge_gap_s=float(merge_gap_s))

    # Alert events
    _, alert_events = detect_alert_events(p, t, alert_cfg)
    alert_intervals = [(ev.start_time_s, ev.end_time_s) for ev in alert_events]

    # Match GT -> earliest overlapping alert for delay
    matched_gt = 0
    delays: List[float] = []
    for (gs, ge) in gt_events:
        first_alert = None
        for ev in alert_events:
            if _overlap((gs, ge), (ev.start_time_s, ev.end_time_s), slack_s=float(overlap_slack_s)):
                first_alert = ev.start_time_s
                break
        if first_alert is not None:
            matched_gt += 1
            delays.append(max(0.0, float(first_alert - gs)))

    recall = float(matched_gt / len(gt_events)) if gt_events else float("nan")

    # Alert precision / false alerts
    true_alerts = 0
    false_alerts = 0
    for interval in alert_intervals:
        if any(_overlap(interval, ge, slack_s=float(overlap_slack_s)) for ge in gt_events):
            true_alerts += 1
        else:
            false_alerts += 1

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
    vids = np.asarray(video_ids).astype(str).reshape(-1)
    ws = np.asarray(w_start, dtype=np.int32).reshape(-1)
    we = np.asarray(w_end, dtype=np.int32).reshape(-1)
    fps_arr = np.asarray(fps, dtype=np.float32).reshape(-1)

    if probs.size == 0:
        return {"thr": [], "recall": [], "fa24h": []}, {"n_videos": 0}

    # group by video
    unique_vids = list(dict.fromkeys(list(vids)))
    per_video = {}
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
        per_video[v] = (p_v, y_v, t_v, duration_s, fps_v, ws_v, we_v)

    if merge_gap_s is None:
        # default: 2x median step between windows (seconds)
        gaps = []
        for v, (_, _, t_v, _, _, _, _) in per_video.items():
            if t_v.size >= 2:
                gaps.append(float(np.median(np.diff(t_v))))
        med_gap = float(np.median(gaps)) if gaps else 0.5
        merge_gap_s = max(0.25, 2.0 * med_gap)

    thr_values = np.arange(float(thr_min), float(thr_max) + 1e-12, float(thr_step), dtype=np.float32)

    out = {
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
        )

        gt_total = 0
        matched_gt_total = 0
        alert_total = 0
        true_alert_total = 0
        false_alert_total = 0
        delays_all: List[float] = []

        for v, (p_v, y_v, t_v, dur_s, *_rest) in per_video.items():
            em, _detail = event_metrics_from_windows(
                p_v, y_v, t_v, cfg,
                duration_s=float(dur_s),
                merge_gap_s=float(merge_gap_s),
                overlap_slack_s=float(overlap_slack_s),
            )
            gt_total += int(em.n_gt_events)
            matched_gt_total += int(em.n_matched_gt)
            alert_total += int(em.n_alert_events)
            true_alert_total += int(em.n_true_alerts)
            false_alert_total += int(em.n_false_alerts)
            if np.isfinite(em.mean_delay_s):
                delays_all.append(float(em.mean_delay_s))

        recall = float(matched_gt_total / gt_total) if gt_total > 0 else 0.0
        precision = float(true_alert_total / alert_total) if alert_total > 0 else float("nan")
        if np.isfinite(precision) and np.isfinite(recall) and (precision + recall) > 0:
            f1 = float(2.0 * precision * recall / (precision + recall))
        else:
            f1 = float("nan")

        dur_h = float(total_duration_s) / 3600.0 if total_duration_s > 0 else float("nan")
        dur_d = float(total_duration_s) / 86400.0 if total_duration_s > 0 else float("nan")
        fa_h = float(false_alert_total / dur_h) if np.isfinite(dur_h) and dur_h > 0 else float("nan")
        fa_d = float(false_alert_total / dur_d) if np.isfinite(dur_d) and dur_d > 0 else float("nan")

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

    meta = {
        "n_videos": int(len(per_video)),
        "total_duration_s": float(total_duration_s),
        "merge_gap_s": float(merge_gap_s),
        "time_mode": str(time_mode),
        "overlap_slack_s": float(overlap_slack_s),
    }
    return out, meta
