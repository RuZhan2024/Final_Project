import { useCallback, useMemo, useRef, useState } from "react";

import {
  extractPredictionState,
} from "../../../features/monitor/prediction";
import { CLIP_POST_S, CLIP_PRE_S } from "../constants";
import { labelForTriage } from "../utils";
import type { TimelineEntry, UseMonitorSessionStateOptions } from "../types";

const FALL_HISTORY_DEDUP_MS_DEFAULT = 30_000;

/**
 * Owns monitor-page prediction state derived from backend responses.
 *
 * The hook keeps UI-friendly state separate from transport details: stable
 * triage labels, timeline markers, clip upload queueing, and compact display
 * values for the current monitor session.
 */
export function useMonitorSessionState({
  mode,
  settingsPayload,
  clipFlags,
  pendingClipRef,
  uploadedClipIdsRef,
  maybeFinalizeClipUpload,
  queueClipForEvent,
}: UseMonitorSessionStateOptions) {
  const [triageState, setTriageState] = useState("not_fall");
  const [pFall, setPFall] = useState<number | null>(null);
  const [sigma, setSigma] = useState<number | null>(null);
  const [safeAlert, setSafeAlert] = useState<boolean | null>(null);
  const [recallAlert, setRecallAlert] = useState<boolean | null>(null);
  const [safeState, setSafeState] = useState<string | null>(null);
  const [recallState, setRecallState] = useState<string | null>(null);
  const [timeline, setTimeline] = useState<TimelineEntry[]>([]);

  const lastUiRef = useRef({
    triageState: "not_fall",
    safeAlert: null as boolean | null,
    recallAlert: null as boolean | null,
    safeState: null as string | null,
    recallState: null as string | null,
    pFall: null as number | null,
    sigma: null as number | null,
  });
  const triageStableRef = useRef({ fall: 0, uncertain: 0, safe: 0, last: "not_fall" });
  const timelineSeqRef = useRef(0);
  const timelineSeenEventIdsRef = useRef(new Set<string>());
  const lastFallHistoryTsRef = useRef(0);

  const resetSessionUiState = useCallback(() => {
    /** Clear all frontend session state when a new monitor run starts. */
    setPFall(null);
    setSigma(null);
    setTriageState("not_fall");
    setSafeAlert(null);
    setRecallAlert(null);
    setSafeState(null);
    setRecallState(null);
    setTimeline([]);
    timelineSeqRef.current = 0;
    timelineSeenEventIdsRef.current = new Set();
    lastFallHistoryTsRef.current = 0;
    triageStableRef.current = { fall: 0, uncertain: 0, safe: 0, last: "not_fall" };
    lastUiRef.current = {
      triageState: "not_fall",
      safeAlert: null,
      recallAlert: null,
      safeState: null,
      recallState: null,
      pFall: null,
      sigma: null,
    };
  }, []);

  const addTimelineMarker = useCallback((kind: string, options: { force?: boolean; eventId?: string | number | null; dedupMs?: number } = {}) => {
    /**
     * Append a marker to the recent prediction timeline with fall deduping.
     *
     * Fall markers are deduplicated by event id when available, then by time
     * window. This keeps one persisted event from filling the timeline with
     * repeated markers from overlapping windows.
     */
    const now = Date.now();
    const { force = false, eventId = null, dedupMs = FALL_HISTORY_DEDUP_MS_DEFAULT } = options || {};
    if (!force && kind === "fall") {
      if (eventId != null) {
        const key = String(eventId);
        if (timelineSeenEventIdsRef.current.has(key)) {
          return;
        }
        timelineSeenEventIdsRef.current.add(key);
      } else if (now - Number(lastFallHistoryTsRef.current || 0) < Math.max(1000, Number(dedupMs) || 0)) {
        return;
      }
      lastFallHistoryTsRef.current = now;
    }
    setTimeline((prev) => {
      const next = prev.slice(-49);
      timelineSeqRef.current += 1;
      next.push({ kind, t: now, seq: timelineSeqRef.current });
      return next;
    });
  }, []);

  const applyPredictionResponse = useCallback((data: Record<string, any>) => {
    /**
     * Translate one backend response into stable UI state and side effects.
     *
     * This applies triage smoothing, updates visible prediction labels, queues
     * skeleton clip upload when a new event id appears, and adds timeline
     * markers without letting overlapping fall windows spam the history.
     */
    const endTs = Number(data?.window_end_t_ms || data?.window_end_ts || 0);
    if (Number.isFinite(endTs) && endTs > 0) {
      void maybeFinalizeClipUpload();
    }

    const nextState = extractPredictionState({
      data,
      mode,
      previousStable: triageStableRef.current,
      settingsPayload,
    });
    triageStableRef.current = nextState.stable;

    if (lastUiRef.current.safeAlert !== nextState.safeAlert) {
      lastUiRef.current.safeAlert = nextState.safeAlert;
      setSafeAlert(nextState.safeAlert);
    }
    if (lastUiRef.current.recallAlert !== nextState.recallAlert) {
      lastUiRef.current.recallAlert = nextState.recallAlert;
      setRecallAlert(nextState.recallAlert);
    }
    if (lastUiRef.current.safeState !== nextState.safeState) {
      lastUiRef.current.safeState = nextState.safeState;
      setSafeState(nextState.safeState);
    }
    if (lastUiRef.current.recallState !== nextState.recallState) {
      lastUiRef.current.recallState = nextState.recallState;
      setRecallState(nextState.recallState);
    }
    if (lastUiRef.current.triageState !== nextState.triageState) {
      lastUiRef.current.triageState = nextState.triageState;
      setTriageState(nextState.triageState);
    }

    try {
      const { storeEventClips } = clipFlags;
      const evId = data?.event_id;
      if (storeEventClips && evId != null && Number.isFinite(endTs) && endTs > 0) {
        const sent = uploadedClipIdsRef.current;
        const already = sent && sent.has(String(evId));
        const pending = pendingClipRef.current;
        if (!already && (!pending || String(pending.eventId) !== String(evId))) {
          // Queue at most one pending clip per event id so replay/live overlap
          // does not schedule duplicate uploads for the same backend event.
          queueClipForEvent({
            eventId: evId,
            endTs,
            clipPreS: CLIP_PRE_S,
            clipPostS: CLIP_POST_S,
          });
        }
      }
    } catch {
      // ignore
    }

    if (lastUiRef.current.pFall !== nextState.pFall) {
      lastUiRef.current.pFall = nextState.pFall;
      setPFall(nextState.pFall);
    }
    if (lastUiRef.current.sigma !== nextState.sigma) {
      lastUiRef.current.sigma = nextState.sigma;
      setSigma(nextState.sigma);
    }

    addTimelineMarker(nextState.markerKind, {
      eventId: nextState.markerKind === "fall" ? nextState.eventId : null,
      dedupMs: nextState.dedupMs,
    });
  }, [
    addTimelineMarker,
    clipFlags,
    mode,
    maybeFinalizeClipUpload,
    pendingClipRef,
    queueClipForEvent,
    settingsPayload,
    uploadedClipIdsRef,
  ]);

  const currentPrediction = useMemo(() => labelForTriage(triageState), [triageState]);
  const safePrediction = useMemo(() => {
    if (safeState) return labelForTriage(safeState);
    if (safeAlert == null) return "—";
    return safeAlert ? "FALL DETECTED" : "SAFE";
  }, [safeState, safeAlert]);
  const recallPrediction = useMemo(() => {
    if (recallState) {
      const r = labelForTriage(recallState);
      const s = safeState ? labelForTriage(safeState) : null;
      if (String(r).toLowerCase() === "fall" && String(s || "").toLowerCase() !== "fall") {
        return "Watch";
      }
      return r;
    }
    if (recallAlert == null) return "—";
    if (recallAlert && String(safeState || "").toLowerCase() !== "fall") return "Watch";
    return recallAlert ? "FALL DETECTED" : "SAFE";
  }, [recallState, recallAlert, safeState]);
  const pText = useMemo(() => (pFall == null ? "—" : Number(pFall).toFixed(3)), [pFall]);
  const markers = useMemo(() => {
    const cap = 50;
    const last = timeline.slice(-cap);
    const n = last.length;
    if (n === 0) return [];
    const pad = cap - n;
    return last.map((m, idx) => {
      const slot = pad + idx;
      return {
        key: m.seq ?? m.t ?? idx,
        leftPct: ((slot + 0.5) / cap) * 100,
        kind: m.kind,
      };
    });
  }, [timeline]);
  const timelineStatusText = useMemo(() => {
    if (!timeline.length) return "Waiting for prediction windows…";
    const last = timeline[timeline.length - 1];
    const ts = new Date(Number(last.t) || Date.now()).toLocaleTimeString();
    return `Updated ${ts} · ${timeline.length}/50 windows`;
  }, [timeline]);

  return {
    triageState,
    pFall,
    sigma,
    safeAlert,
    recallAlert,
    safeState,
    recallState,
    timeline,
    currentPrediction,
    safePrediction,
    recallPrediction,
    pText,
    markers,
    timelineStatusText,
    addTimelineMarker,
    applyPredictionResponse,
    resetSessionUiState,
  };
}
