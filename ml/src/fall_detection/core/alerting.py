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

import ast
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


# ---------------- config + structs ----------------

@dataclass(frozen=True)
class AlertCfg:
    """Deployment-time alert policy thresholds and optional confirmation rules.

    These values sit above raw model probabilities: they define when a score
    sequence becomes a user-visible alert and how quickly another alert may be
    emitted afterwards. The same structure is reused in evaluation so offline
    sweeps match live runtime behavior.
    """
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
    # Optional start guard: block alert starts when lying_score exceeds this.
    # Set to None to disable.
    start_guard_max_lying: Optional[float] = None
    # Optional scene/video scope for start guard. If provided, start guard applies
    # only when video_id starts with one of these prefixes.
    start_guard_prefixes: Optional[List[str]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "AlertCfg":
        d = dict(d or {})
        raw_prefixes = d.get("start_guard_prefixes", None)
        prefixes: Optional[List[str]]
        if raw_prefixes is None:
            prefixes = None
        elif isinstance(raw_prefixes, str):
            raw_s = raw_prefixes.strip()
            vals: List[str] = []
            # Accept stringified list forms emitted by simple YAML writer,
            # e.g. "['SceneA__', 'SceneB__']".
            if raw_s.startswith("[") and raw_s.endswith("]"):
                try:
                    parsed = ast.literal_eval(raw_s)
                except Exception:
                    parsed = None
                if isinstance(parsed, (list, tuple)):
                    vals = [str(x).strip() for x in parsed if str(x).strip()]
                elif isinstance(parsed, str) and parsed.strip():
                    vals = [parsed.strip()]
            if not vals:
                vals = [x.strip() for x in raw_s.split(",") if x.strip()]
            prefixes = vals or None
        elif isinstance(raw_prefixes, (list, tuple)):
            vals = [str(x).strip() for x in raw_prefixes if str(x).strip()]
            prefixes = vals or None
        else:
            prefixes = None
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
            start_guard_max_lying=(
                None if d.get("start_guard_max_lying", None) is None else float(d.get("start_guard_max_lying"))
            ),
            start_guard_prefixes=prefixes,
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
    """Aggregated alert quality for a set of videos under one alert policy.

    The count fields keep the raw matching outcome, while recall/precision/F1
    expose the derived rates used for operating-point selection and reporting.
    """
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
    """Apply causal EMA smoothing to one probability trace."""
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

    ``w_end`` remains inclusive here; the function only chooses which point of
    that closed interval becomes the alert-policy timestamp.

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


def window_span_seconds(w_start: float, w_end: float, fps: float) -> float:
    """Compute clip duration in seconds using inclusive frame indices."""
    f = float(fps) if float(fps) > 0 else 30.0
    return float((float(w_end) - float(w_start) + 1.0) / max(1e-6, f))


# ---------------- alert detection ----------------

def detect_alert_events(
    probs: Sequence[float],
    times_s: Sequence[float],
    cfg: AlertCfg,
    *,
    lying_score: Optional[Sequence[float]] = None,
    motion_score: Optional[Sequence[float]] = None,
    video_id: Optional[str] = None,
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

    Resolution order:
      1) Smooth raw probabilities with EMA.
      2) Open a candidate alert only after the k-of-n `tau_high` rule passes.
      3) Apply optional start guards / confirmation checks.
      4) Keep the alert active until the lower hysteresis threshold fails.
      5) Enforce cooldown before another alert may start.
    """
    p = np.asarray(probs, dtype=np.float32).reshape(-1)
    t = np.asarray(times_s, dtype=np.float32).reshape(-1)
    if p.size == 0:
        return np.asarray([], dtype=bool), []

    # Smoothing happens before persistence so a single noisy spike cannot satisfy
    # the start rule on its own.
    # Reuse detect_alert_events on already-smoothed probabilities so the
    # suspect/alert split matches the real policy without double-smoothing.
    ps = ema_smooth(p, cfg.ema_alpha)

    # The high threshold is reserved for starts; endings use the lower
    # hysteresis threshold so brief post-fall score dips do not fragment events.
    high = (ps >= float(cfg.tau_high)).astype(np.int32)
    pers = _k_of_n(high, cfg.k, cfg.n)

    active = np.zeros_like(pers, dtype=bool)
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

    ls = None
    ms = None
    if lying_score is not None:
        ls = np.asarray(lying_score, dtype=np.float32).reshape(-1)
        if ls.size != p.size:
            if ls.size < p.size:
                pad = np.full((p.size - ls.size,), np.nan, dtype=np.float32)
                ls = np.concatenate([ls, pad], axis=0)
            else:
                ls = ls[: p.size]
    if motion_score is not None:
        ms = np.asarray(motion_score, dtype=np.float32).reshape(-1)
        if ms.size != p.size:
            if ms.size < p.size:
                pad = np.full((p.size - ms.size,), np.nan, dtype=np.float32)
                ms = np.concatenate([ms, pad], axis=0)
            else:
                ms = ms[: p.size]

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

        # confirmation pending (candidate alert start)
        if pending:
            # We are in the uncertain band: no hard alert yet.
            # Promote to a real alert if confirm condition satisfied.
            pending_peak = max(pending_peak, float(ps[i]))

            # If we drop below tau_low, abandon the pending start.
            if float(ps[i]) < float(cfg.tau_low):
                pending = False
                continue

            # Timeout: if we didn't confirm in time, keep it as uncertain (no alert).
            if float(ti) > float(pending_deadline_s):
                pending = False
                continue

            ok_prob = float(ps[i]) >= float(cfg.tau_low)
            ok_lying = True
            ok_motion = True

            # If heuristic signals are finite, apply them.
            # Non-finite values are treated as unavailable (no veto).
            if ls is not None and np.isfinite(ls[i]):
                ok_lying = float(ls[i]) >= float(cfg.confirm_min_lying)
            if ms is not None and np.isfinite(ms[i]):
                ok_motion = float(ms[i]) <= float(cfg.confirm_max_motion)

            if ok_prob and ok_lying and ok_motion:
                # Promote pending -> event
                in_event = True
                pending = False
                start_i = int(pending_start_i)
                peak = float(pending_peak)
                active[i] = True
            continue

        # not in event: can we start?
        if ti < cooldown_until:
            continue

        if pers[i]:
            apply_start_guard = bool(ls is not None and cfg.start_guard_max_lying is not None)
            if apply_start_guard and cfg.start_guard_prefixes:
                if video_id is None:
                    apply_start_guard = False
                else:
                    vid = str(video_id)
                    apply_start_guard = any(vid.startswith(str(pref)) for pref in cfg.start_guard_prefixes)
            if apply_start_guard:
                ls_i = float(ls[i]) if ls is not None else float("nan")
                if np.isfinite(ls_i) and ls_i > float(cfg.start_guard_max_lying):
                    continue
            # Optional "require low": only start after coming from below tau_low.
            # We only apply this when the confirm stage is actually using extra signals.
            if bool(cfg.confirm) and (ls is not None or ms is not None) and bool(cfg.confirm_require_low) and i > 0:
                if float(ps[i - 1]) >= float(cfg.tau_low):
                    # Stay uncertain; do not start.
                    continue

            if bool(cfg.confirm) and (ls is not None or ms is not None):
                pending = True
                pending_start_i = i
                pending_deadline_s = float(t[i]) + float(cfg.confirm_s)
                pending_peak = float(ps[i])
                continue

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


def classify_states(
    probs: Sequence[float],
    times_s: Sequence[float],
    cfg: AlertCfg,
    *,
    lying_score: Optional[Sequence[float]] = None,
    motion_score: Optional[Sequence[float]] = None,
    video_id: Optional[str] = None,
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
    alert_mask, _events = detect_alert_events(
        ps,
        t,
        AlertCfg(
            ema_alpha=0.0,  # already smoothed
            k=int(cfg.k),
            n=int(cfg.n),
            tau_high=float(cfg.tau_high),
            tau_low=float(cfg.tau_low),
            cooldown_s=float(cfg.cooldown_s),
            confirm=bool(cfg.confirm),
            confirm_s=float(cfg.confirm_s),
            confirm_min_lying=float(cfg.confirm_min_lying),
            confirm_max_motion=float(cfg.confirm_max_motion),
            confirm_require_low=bool(cfg.confirm_require_low),
            start_guard_max_lying=cfg.start_guard_max_lying,
            start_guard_prefixes=cfg.start_guard_prefixes,
        ),
        lying_score=lying_score,
        motion_score=motion_score,
        video_id=video_id,
    )
    suspect = (~alert_mask) & (ps >= float(cfg.tau_low)) & (ps < float(cfg.tau_high))
    clear = (~alert_mask) & (ps < float(cfg.tau_low))
    return {"ps": ps, "clear": clear, "suspect": suspect, "alert": alert_mask}


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
    lying_score: Optional[Sequence[float]] = None,
    motion_score: Optional[Sequence[float]] = None,
    video_id: Optional[str] = None,
    duration_s: Optional[float] = None,
    merge_gap_s: float = 2.0,
    overlap_slack_s: float = 0.0,
) -> Tuple[EventMetrics, Dict[str, Any]]:
    """Compute event-level metrics for a single video.

    - GT events are built from contiguous positive windows (y_true==1), merged by merge_gap_s.
    - Alert events come from detect_alert_events().

    false_alerts_per_{hour,day} counts ONLY alerts that do NOT overlap any GT event.
    (For unlabeled streams where y_true is all -1/0, this equals total alerts.)

    Returns:
      EventMetrics:
        Aggregated rates and counts for this one video under `alert_cfg`.
      detail:
        Explicit GT and alert event intervals for debugging/reporting.

    Contract:
      `duration_s` should represent the monitored span for FA/hour and FA/day.
      When it is omitted, the function falls back to the timestamp range, which
      is acceptable for ordered windows but less precise than a caller-supplied
      clip duration.

    Ground truth comes from labeled-positive windows merged into GT events,
    while predictions come from the full alert policy. That distinction is what
    lets delay and false-alert metrics match runtime semantics.
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
    _, alert_events = detect_alert_events(
        p,
        t,
        alert_cfg,
        lying_score=lying_score,
        motion_score=motion_score,
        video_id=video_id,
    )
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
      - If FA-only windows are supplied, FA/day is measured on that stream
        instead of the usually shorter labeled validation clips.
    """
    probs = np.asarray(probs, dtype=np.float32).reshape(-1)
    y_true = np.asarray(y_true, dtype=np.int32).reshape(-1)
    vids = np.asarray(video_ids).astype(str).reshape(-1)
    ws = np.asarray(w_start, dtype=np.int32).reshape(-1)
    we = np.asarray(w_end, dtype=np.int32).reshape(-1)
    fps_arr = np.asarray(fps, dtype=np.float32).reshape(-1)

    ls = None
    ms = None
    if lying_score is not None:
        ls = np.asarray(lying_score, dtype=np.float32).reshape(-1)
    if motion_score is not None:
        ms = np.asarray(motion_score, dtype=np.float32).reshape(-1)

    if probs.size == 0:
        return {"thr": [], "recall": [], "fa24h": []}, {"n_videos": 0}

    # Per-video evaluation keeps event merging and duration accounting aligned
    # with deployment, where alerts are scoped to one stream at a time.
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
        ls_v = ls[mv][idx] if ls is not None else None
        ms_v = ms[mv][idx] if ms is not None else None
        per_video[v] = (p_v, y_v, t_v, duration_s, fps_v, ws_v, we_v, ls_v, ms_v)

    # The optional FA-only stream avoids optimistic FA/day estimates that can
    # happen when validation positives dominate a short labeled subset.
    per_video_fa = {}
    total_duration_s_fa = 0.0
    if fa_probs is not None and fa_video_ids is not None and fa_w_start is not None and fa_w_end is not None and fa_fps is not None:
        fa_p = np.asarray(fa_probs, dtype=np.float32).reshape(-1)
        fa_ls = None
        fa_ms = None
        if fa_lying_score is not None:
            fa_ls = np.asarray(fa_lying_score, dtype=np.float32).reshape(-1)
        if fa_motion_score is not None:
            fa_ms = np.asarray(fa_motion_score, dtype=np.float32).reshape(-1)
        fa_vids_arr = np.asarray(fa_video_ids).astype(str).reshape(-1)
        fa_ws_arr = np.asarray(fa_w_start, dtype=np.int32).reshape(-1)
        fa_we_arr = np.asarray(fa_w_end, dtype=np.int32).reshape(-1)
        fa_fps_arr = np.asarray(fa_fps, dtype=np.float32).reshape(-1)
        ok_sizes = (fa_p.size == fa_vids_arr.size == fa_ws_arr.size == fa_we_arr.size == fa_fps_arr.size)
        if fa_ls is not None:
            ok_sizes = ok_sizes and (fa_ls.size == fa_p.size)
        if fa_ms is not None:
            ok_sizes = ok_sizes and (fa_ms.size == fa_p.size)
        if ok_sizes and fa_p.size > 0:
            fa_unique_vids = list(dict.fromkeys(list(fa_vids_arr)))
            for v in fa_unique_vids:
                mv = fa_vids_arr == v
                if not mv.any():
                    continue
                idx = np.argsort(fa_ws_arr[mv])
                p_v = fa_p[mv][idx]
                # No GT events on FA stream
                y_v = np.zeros_like(p_v, dtype=np.int32)
                ws_v = fa_ws_arr[mv][idx]
                we_v = fa_we_arr[mv][idx]
                fps_v = float(np.median(fa_fps_arr[mv])) if np.isfinite(fa_fps_arr[mv]).any() else float(fps_default)
                if fps_v <= 0:
                    fps_v = float(fps_default)
                t_v = times_from_windows(ws_v, we_v, fps_v, mode=time_mode)
                duration_s = float((we_v.max() - ws_v.min() + 1) / max(1e-6, fps_v))
                total_duration_s_fa += max(0.0, duration_s)
                ls_v = fa_ls[mv][idx] if fa_ls is not None else None
                ms_v = fa_ms[mv][idx] if fa_ms is not None else None
                per_video_fa[v] = (p_v, y_v, t_v, duration_s, fps_v, ws_v, we_v, ls_v, ms_v)

    if merge_gap_s is None:
        # default: 2x median step between windows (seconds)
        gaps = []
        for _v, item in per_video.items():
            t_v = item[2]
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

            confirm=bool(alert_base.confirm),
            confirm_s=float(alert_base.confirm_s),
            confirm_min_lying=float(alert_base.confirm_min_lying),
            confirm_max_motion=float(alert_base.confirm_max_motion),
            confirm_require_low=bool(alert_base.confirm_require_low),
            start_guard_max_lying=alert_base.start_guard_max_lying,
            start_guard_prefixes=alert_base.start_guard_prefixes,
        )

        gt_total = 0
        matched_gt_total = 0
        alert_total = 0
        true_alert_total = 0
        false_alert_total = 0
        false_alert_total_fa = 0
        delays_all: List[float] = []

        for _v, item in per_video.items():
            p_v, y_v, t_v, dur_s = item[0], item[1], item[2], item[3]
            ls_v, ms_v = item[7], item[8]
            em, _detail = event_metrics_from_windows(
                p_v,
                y_v,
                t_v,
                cfg,
                lying_score=ls_v,
                motion_score=ms_v,
                duration_s=float(dur_s),
                merge_gap_s=float(merge_gap_s),
                overlap_slack_s=float(overlap_slack_s),
                video_id=str(_v),
            )
            gt_total += int(em.n_gt_events)
            matched_gt_total += int(em.n_matched_gt)
            alert_total += int(em.n_alert_events)
            true_alert_total += int(em.n_true_alerts)
            false_alert_total += int(em.n_false_alerts)
            if np.isfinite(em.mean_delay_s):
                delays_all.append(float(em.mean_delay_s))

        # If FA-only stream is provided, compute FA rate from it (more realistic for deployment)
        if per_video_fa:
            for _v, item in per_video_fa.items():
                p_v, y_v, t_v, dur_s = item[0], item[1], item[2], item[3]
                ls_v, ms_v = item[7], item[8]
                em, _detail = event_metrics_from_windows(
                    p_v,
                    y_v,
                    t_v,
                    cfg,
                    lying_score=ls_v,
                    motion_score=ms_v,
                    duration_s=float(dur_s),
                    merge_gap_s=float(merge_gap_s),
                    overlap_slack_s=float(overlap_slack_s),
                    video_id=str(_v),
                )
                false_alert_total_fa += int(em.n_false_alerts)

        recall = float(matched_gt_total / gt_total) if gt_total > 0 else 0.0
        precision = float(true_alert_total / alert_total) if alert_total > 0 else float("nan")
        if np.isfinite(precision) and np.isfinite(recall) and (precision + recall) > 0:
            f1 = float(2.0 * precision * recall / (precision + recall))
        else:
            f1 = float("nan")

        # FA/24h reference: prefer FA-only stream if provided; otherwise use validation videos.
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
        # Optional: expose FA-only stream counts (debug)
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

    # Optional lightweight confirmation using skeleton-derived heuristics.
    # (Useful for rejecting fast-sit / kneel / stumble false positives.)
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
