#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""server/online_alert.py

Small online alerting helper that applies the same core idea as
core.alerting.detect_alert_events, but incrementally.

We use the deployment parameters from configs/ops/*.yaml (via
server.deploy_runtime.get_alert_cfg).

This is deliberately lightweight and dependency-free (no torch).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, Optional


def _sf(x: Any, d: float) -> float:
    try:
        v = float(x)
        if v != v:  # NaN
            return d
        return v
    except Exception:
        return d


@dataclass
class OnlineAlertResult:
    triage_state: str
    ps: float
    p_in: float
    high: bool
    pers: bool
    in_event: bool
    started_event: bool
    ended_event: bool
    cooldown_remaining_s: float


class OnlineAlertTracker:
    """Incremental EMA + k-of-n persistence + hysteresis + cooldown.

    Mapping to UI states:
      - fall      : event is active
      - uncertain : not active, but ps >= tau_low (includes brief high spikes)
      - not_fall  : ps < tau_low

    Note: confirmation using lying/motion scores is not applied here because
    the live monitor endpoint does not currently compute those heuristics.
    """

    def __init__(self, alert_cfg: Dict[str, Any]):
        self._cfg: Dict[str, Any] = {}
        self._high_q: Deque[bool] = deque(maxlen=3)
        self._ps_prev: Optional[float] = None
        self._in_event: bool = False
        self._cooldown_until_s: float = -1.0
        self._last_triage: str = "not_fall"
        self.update_cfg(alert_cfg, reset=True)

    @property
    def cfg(self) -> Dict[str, Any]:
        return self._cfg

    def update_cfg(self, alert_cfg: Dict[str, Any], *, reset: bool = False) -> None:
        # Normalise and copy relevant fields.
        cfg = dict(alert_cfg or {})
        # Defaults are conservative.
        n = int(_sf(cfg.get("n"), 3))
        k = int(_sf(cfg.get("k"), 2))
        n = max(1, min(16, n))
        k = max(1, min(n, k))

        self._cfg = {
            "ema_alpha": _sf(cfg.get("ema_alpha"), 0.0),
            "k": k,
            "n": n,
            "tau_high": _sf(cfg.get("tau_high"), 0.85),
            "tau_low": _sf(cfg.get("tau_low"), 0.5),
            "cooldown_s": _sf(cfg.get("cooldown_s"), 3.0),
        }

        if reset:
            self.reset()
            return

        # If shape changes, rebuild queue.
        if self._high_q.maxlen != n:
            self._high_q = deque(list(self._high_q), maxlen=n)

    def reset(self) -> None:
        self._ps_prev = None
        self._in_event = False
        self._cooldown_until_s = -1.0
        self._last_triage = "not_fall"
        self._high_q = deque(maxlen=int(self._cfg.get("n", 3)))

    def step(self, p: float, t_s: float) -> OnlineAlertResult:
        cfg = self._cfg
        alpha = float(cfg["ema_alpha"])
        tau_high = float(cfg["tau_high"])
        tau_low = float(cfg["tau_low"])
        cooldown_s = float(cfg["cooldown_s"])
        k = int(cfg["k"])

        p_in = float(p)
        if self._ps_prev is None or alpha <= 0.0:
            ps = p_in
        else:
            ps = alpha * p_in + (1.0 - alpha) * float(self._ps_prev)
        self._ps_prev = ps

        high = bool(ps >= tau_high)
        self._high_q.append(high)
        pers = bool(sum(1 for v in self._high_q if v) >= k)

        started = False
        ended = False

        # End condition first.
        if self._in_event and ps < tau_low:
            self._in_event = False
            ended = True
            self._cooldown_until_s = float(t_s) + cooldown_s

        # Start condition.
        if (not self._in_event) and (float(t_s) >= float(self._cooldown_until_s)) and pers:
            self._in_event = True
            started = True

        if self._in_event:
            tri = "fall"
        elif ps >= tau_low:
            tri = "uncertain"
        else:
            tri = "not_fall"

        cooldown_remaining = max(0.0, float(self._cooldown_until_s) - float(t_s))
        self._last_triage = tri

        return OnlineAlertResult(
            triage_state=tri,
            ps=float(ps),
            p_in=float(p_in),
            high=high,
            pers=pers,
            in_event=self._in_event,
            started_event=started,
            ended_event=ended,
            cooldown_remaining_s=float(cooldown_remaining),
        )
