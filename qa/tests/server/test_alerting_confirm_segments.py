from __future__ import annotations

import numpy as np

from core.alerting import (
    _count_confirm_segments_no_active_from_pers,
    _detect_confirm_segments_no_active_from_pers,
    _detect_events_confirm_no_active_from_pers,
    _k_of_n,
)


def _events_to_idx(events):
    s = [int(e.start_idx) for e in events]
    e = [int(ev.end_idx) for ev in events]
    return s, e


def test_confirm_segments_match_event_reference_with_scores():
    rng = np.random.default_rng(7)
    n = 120
    t = np.arange(n, dtype=np.float32) * 0.2
    ps = rng.uniform(0.0, 1.0, size=(n,)).astype(np.float32)
    tau_low = 0.55
    tau_high = 0.75
    pers = _k_of_n(ps >= tau_high, 2, 3)
    gate = rng.uniform(0.0, 1.0, size=(n,)) > 0.35

    starts, ends = _detect_confirm_segments_no_active_from_pers(
        ps,
        t,
        tau_low=tau_low,
        cooldown_s=1.0,
        pers=pers,
        confirm_s=0.8,
        require_low=True,
        has_confirm_scores=True,
        gate_ok=gate,
    )
    ev = _detect_events_confirm_no_active_from_pers(
        ps,
        t,
        tau_low=tau_low,
        cooldown_s=1.0,
        pers=pers,
        confirm_s=0.8,
        require_low=True,
        has_confirm_scores=True,
        gate_ok=gate,
    )
    s_ref, e_ref = _events_to_idx(ev)
    assert starts.tolist() == s_ref
    assert ends.tolist() == e_ref


def test_confirm_segments_match_event_reference_without_scores():
    rng = np.random.default_rng(11)
    n = 100
    t = np.arange(n, dtype=np.float32) * 0.1
    ps = rng.uniform(0.0, 1.0, size=(n,)).astype(np.float32)
    tau_low = 0.45
    tau_high = 0.7
    pers = _k_of_n(ps >= tau_high, 2, 3)

    starts, ends = _detect_confirm_segments_no_active_from_pers(
        ps,
        t,
        tau_low=tau_low,
        cooldown_s=0.8,
        pers=pers,
        confirm_s=0.5,
        require_low=False,
        has_confirm_scores=False,
        gate_ok=None,
    )
    ev = _detect_events_confirm_no_active_from_pers(
        ps,
        t,
        tau_low=tau_low,
        cooldown_s=0.8,
        pers=pers,
        confirm_s=0.5,
        require_low=False,
        has_confirm_scores=False,
        gate_ok=None,
    )
    s_ref, e_ref = _events_to_idx(ev)
    assert starts.tolist() == s_ref
    assert ends.tolist() == e_ref


def test_confirm_segments_low_idx_override_matches_default():
    rng = np.random.default_rng(21)
    n = 140
    t = np.arange(n, dtype=np.float32) * 0.15
    ps = rng.uniform(0.0, 1.0, size=(n,)).astype(np.float32)
    tau_low = 0.5
    tau_high = 0.72
    pers = _k_of_n(ps >= tau_high, 2, 3)
    gate = rng.uniform(0.0, 1.0, size=(n,)) > 0.4
    low_idx = np.flatnonzero(ps < tau_low)

    s0, e0 = _detect_confirm_segments_no_active_from_pers(
        ps,
        t,
        tau_low=tau_low,
        cooldown_s=1.2,
        pers=pers,
        confirm_s=0.9,
        require_low=True,
        has_confirm_scores=True,
        gate_ok=gate,
    )
    s1, e1 = _detect_confirm_segments_no_active_from_pers(
        ps,
        t,
        tau_low=tau_low,
        cooldown_s=1.2,
        pers=pers,
        confirm_s=0.9,
        require_low=True,
        has_confirm_scores=True,
        gate_ok=gate,
        low_idx=low_idx,
    )

    assert s0.tolist() == s1.tolist()
    assert e0.tolist() == e1.tolist()


def test_confirm_segments_randomized_match_event_reference():
    rng = np.random.default_rng(314)
    for _ in range(20):
        n = int(rng.integers(60, 180))
        t = (np.arange(n, dtype=np.float32) * rng.uniform(0.08, 0.25)).astype(np.float32)
        ps = rng.uniform(0.0, 1.0, size=(n,)).astype(np.float32)
        tau_low = float(rng.uniform(0.35, 0.65))
        tau_high = float(rng.uniform(max(tau_low, 0.55), 0.9))
        pers = _k_of_n(ps >= tau_high, int(rng.integers(1, 3)), int(rng.integers(2, 5)))
        gate = (rng.uniform(0.0, 1.0, size=(n,)) > 0.45)
        require_low = bool(rng.integers(0, 2))
        cooldown_s = float(rng.uniform(0.3, 2.0))
        confirm_s = float(rng.uniform(0.2, 1.5))

        starts, ends = _detect_confirm_segments_no_active_from_pers(
            ps,
            t,
            tau_low=tau_low,
            cooldown_s=cooldown_s,
            pers=pers,
            confirm_s=confirm_s,
            require_low=require_low,
            has_confirm_scores=True,
            gate_ok=gate,
        )
        ev = _detect_events_confirm_no_active_from_pers(
            ps,
            t,
            tau_low=tau_low,
            cooldown_s=cooldown_s,
            pers=pers,
            confirm_s=confirm_s,
            require_low=require_low,
            has_confirm_scores=True,
            gate_ok=gate,
        )
        s_ref, e_ref = _events_to_idx(ev)
        assert starts.tolist() == s_ref
        assert ends.tolist() == e_ref


def test_confirm_segments_pers_idx_override_matches_default():
    rng = np.random.default_rng(202)
    n = 150
    t = np.arange(n, dtype=np.float32) * 0.12
    ps = rng.uniform(0.0, 1.0, size=(n,)).astype(np.float32)
    tau_low = 0.52
    pers = _k_of_n(ps >= 0.75, 2, 3)
    gate = rng.uniform(0.0, 1.0, size=(n,)) > 0.5
    low_idx = np.flatnonzero(ps < tau_low)
    pers_idx = np.flatnonzero(pers)

    s0, e0 = _detect_confirm_segments_no_active_from_pers(
        ps,
        t,
        tau_low=tau_low,
        cooldown_s=1.0,
        pers=pers,
        confirm_s=0.8,
        require_low=True,
        has_confirm_scores=True,
        gate_ok=gate,
        low_idx=low_idx,
    )
    s1, e1 = _detect_confirm_segments_no_active_from_pers(
        ps,
        t,
        tau_low=tau_low,
        cooldown_s=1.0,
        pers=pers,
        confirm_s=0.8,
        require_low=True,
        has_confirm_scores=True,
        gate_ok=gate,
        low_idx=low_idx,
        pers_idx=pers_idx,
    )
    assert s0.tolist() == s1.tolist()
    assert e0.tolist() == e1.tolist()


def test_confirm_segments_gate_idx_override_matches_default():
    rng = np.random.default_rng(909)
    n = 160
    t = np.arange(n, dtype=np.float32) * 0.11
    ps = rng.uniform(0.0, 1.0, size=(n,)).astype(np.float32)
    tau_low = 0.5
    pers = _k_of_n(ps >= 0.73, 2, 3)
    gate = rng.uniform(0.0, 1.0, size=(n,)) > 0.55
    low_idx = np.flatnonzero(ps < tau_low)
    pers_idx = np.flatnonzero(pers)
    gate_idx = np.flatnonzero(gate)

    s0, e0 = _detect_confirm_segments_no_active_from_pers(
        ps,
        t,
        tau_low=tau_low,
        cooldown_s=0.9,
        pers=pers,
        confirm_s=0.7,
        require_low=True,
        has_confirm_scores=True,
        gate_ok=gate,
        low_idx=low_idx,
        pers_idx=pers_idx,
    )
    s1, e1 = _detect_confirm_segments_no_active_from_pers(
        ps,
        t,
        tau_low=tau_low,
        cooldown_s=0.9,
        pers=pers,
        confirm_s=0.7,
        require_low=True,
        has_confirm_scores=True,
        gate_ok=gate,
        gate_idx=gate_idx,
        low_idx=low_idx,
        pers_idx=pers_idx,
    )
    assert s0.tolist() == s1.tolist()
    assert e0.tolist() == e1.tolist()


def test_count_confirm_segments_matches_detect_with_scores():
    rng = np.random.default_rng(1234)
    n = 180
    t = np.arange(n, dtype=np.float32) * 0.09
    ps = rng.uniform(0.0, 1.0, size=(n,)).astype(np.float32)
    tau_low = 0.48
    pers = _k_of_n(ps >= 0.74, 2, 3)
    gate = rng.uniform(0.0, 1.0, size=(n,)) > 0.5
    gate_idx = np.flatnonzero(gate)
    low_idx = np.flatnonzero(ps < tau_low)
    pers_idx = np.flatnonzero(pers)

    starts, ends = _detect_confirm_segments_no_active_from_pers(
        ps,
        t,
        tau_low=tau_low,
        cooldown_s=1.1,
        pers=pers,
        confirm_s=0.8,
        require_low=True,
        has_confirm_scores=True,
        gate_ok=gate,
        gate_idx=gate_idx,
        low_idx=low_idx,
        pers_idx=pers_idx,
    )
    n_count = _count_confirm_segments_no_active_from_pers(
        ps,
        t,
        tau_low=tau_low,
        cooldown_s=1.1,
        pers=pers,
        confirm_s=0.8,
        require_low=True,
        has_confirm_scores=True,
        gate_ok=gate,
        gate_idx=gate_idx,
        low_idx=low_idx,
        pers_idx=pers_idx,
    )
    assert int(starts.size) == int(ends.size)
    assert int(n_count) == int(starts.size)


def test_count_confirm_segments_matches_detect_without_scores():
    rng = np.random.default_rng(777)
    n = 140
    t = np.arange(n, dtype=np.float32) * 0.13
    ps = rng.uniform(0.0, 1.0, size=(n,)).astype(np.float32)
    tau_low = 0.5
    pers = _k_of_n(ps >= 0.72, 2, 3)
    low_idx = np.flatnonzero(ps < tau_low)
    pers_idx = np.flatnonzero(pers)

    starts, ends = _detect_confirm_segments_no_active_from_pers(
        ps,
        t,
        tau_low=tau_low,
        cooldown_s=0.9,
        pers=pers,
        confirm_s=0.6,
        require_low=False,
        has_confirm_scores=False,
        gate_ok=None,
        low_idx=low_idx,
        pers_idx=pers_idx,
    )
    n_count = _count_confirm_segments_no_active_from_pers(
        ps,
        t,
        tau_low=tau_low,
        cooldown_s=0.9,
        pers=pers,
        confirm_s=0.6,
        require_low=False,
        has_confirm_scores=False,
        gate_ok=None,
        low_idx=low_idx,
        pers_idx=pers_idx,
    )
    assert int(starts.size) == int(ends.size)
    assert int(n_count) == int(starts.size)
