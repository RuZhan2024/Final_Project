from applications.backend import core


def test_prune_session_state_removes_stale_and_caps():
    original_ttl = core.SESSION_TTL_S
    original_cap = core.SESSION_MAX_STATES
    original_state = dict(core._SESSION_STATE)
    try:
        core.SESSION_TTL_S = 120
        core.SESSION_MAX_STATES = 2
        core._SESSION_STATE.clear()

        now_s = 1000.0
        core.touch_session_state("old", now_s=now_s - 200)
        core.touch_session_state("mid", now_s=now_s - 50)
        core.touch_session_state("new", now_s=now_s - 10)

        removed = core.prune_session_state(now_s=now_s)
        assert removed >= 1
        assert "old" not in core._SESSION_STATE
        assert len(core._SESSION_STATE) <= 2
        assert set(core._SESSION_STATE.keys()) == {"mid", "new"}
    finally:
        core._SESSION_STATE.clear()
        core._SESSION_STATE.update(original_state)
        core.SESSION_TTL_S = original_ttl
        core.SESSION_MAX_STATES = original_cap


def test_touch_session_state_updates_last_seen():
    original_state = dict(core._SESSION_STATE)
    try:
        core._SESSION_STATE.clear()
        core.touch_session_state("abc", now_s=10.0)
        first = core._SESSION_STATE["abc"]["last_seen_s"]
        core.touch_session_state("abc", now_s=20.0)
        second = core._SESSION_STATE["abc"]["last_seen_s"]
        assert first == 10.0
        assert second == 20.0
    finally:
        core._SESSION_STATE.clear()
        core._SESSION_STATE.update(original_state)


def test_prune_session_state_clamps_minimums():
    original_ttl = core.SESSION_TTL_S
    original_cap = core.SESSION_MAX_STATES
    original_state = dict(core._SESSION_STATE)
    try:
        # These values are intentionally too small; prune should clamp to sane minimums.
        core.SESSION_TTL_S = 1
        core.SESSION_MAX_STATES = 1
        core._SESSION_STATE.clear()
        now_s = 500.0
        core.touch_session_state("recent_a", now_s=now_s - 30.0)
        core.touch_session_state("recent_b", now_s=now_s - 10.0)
        core.touch_session_state("stale", now_s=now_s - 120.0)

        removed = core.prune_session_state(now_s=now_s)
        assert removed >= 1
        # stale must be dropped due effective min TTL=60s
        assert "stale" not in core._SESSION_STATE
        # max states should clamp to >=10; this set should not be force-trimmed to 1
        assert "recent_a" in core._SESSION_STATE
        assert "recent_b" in core._SESSION_STATE
    finally:
        core._SESSION_STATE.clear()
        core._SESSION_STATE.update(original_state)
        core.SESSION_TTL_S = original_ttl
        core.SESSION_MAX_STATES = original_cap
