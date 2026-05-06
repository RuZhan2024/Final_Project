from applications.backend import runtime_state


def test_prune_session_state_removes_stale_and_caps():
    original_state = dict(runtime_state.SESSION_STATE)
    try:
        runtime_state.SESSION_STATE.clear()

        now_s = 1000.0
        runtime_state.touch_session_state("old", now_s=now_s - 200)
        runtime_state.touch_session_state("mid", now_s=now_s - 50)
        runtime_state.touch_session_state("new", now_s=now_s - 10)

        removed = runtime_state.prune_session_state(now_s=now_s, ttl_s=120, max_states=2)
        assert removed >= 1
        assert "old" not in runtime_state.SESSION_STATE
        assert len(runtime_state.SESSION_STATE) <= 2
        assert set(runtime_state.SESSION_STATE.keys()) == {"mid", "new"}
    finally:
        runtime_state.SESSION_STATE.clear()
        runtime_state.SESSION_STATE.update(original_state)


def test_touch_session_state_updates_last_seen():
    original_state = dict(runtime_state.SESSION_STATE)
    try:
        runtime_state.SESSION_STATE.clear()
        runtime_state.touch_session_state("abc", now_s=10.0)
        first = runtime_state.SESSION_STATE["abc"]["last_seen_s"]
        runtime_state.touch_session_state("abc", now_s=20.0)
        second = runtime_state.SESSION_STATE["abc"]["last_seen_s"]
        assert first == 10.0
        assert second == 20.0
    finally:
        runtime_state.SESSION_STATE.clear()
        runtime_state.SESSION_STATE.update(original_state)


def test_prune_session_state_clamps_minimums():
    original_state = dict(runtime_state.SESSION_STATE)
    try:
        # These values are intentionally too small; prune should clamp to sane minimums.
        runtime_state.SESSION_STATE.clear()
        now_s = 500.0
        runtime_state.touch_session_state("recent_a", now_s=now_s - 30.0)
        runtime_state.touch_session_state("recent_b", now_s=now_s - 10.0)
        runtime_state.touch_session_state("stale", now_s=now_s - 120.0)

        removed = runtime_state.prune_session_state(now_s=now_s, ttl_s=1, max_states=1)
        assert removed >= 1
        # stale must be dropped due effective min TTL=60s
        assert "stale" not in runtime_state.SESSION_STATE
        # max states should clamp to >=10; this set should not be force-trimmed to 1
        assert "recent_a" in runtime_state.SESSION_STATE
        assert "recent_b" in runtime_state.SESSION_STATE
    finally:
        runtime_state.SESSION_STATE.clear()
        runtime_state.SESSION_STATE.update(original_state)
