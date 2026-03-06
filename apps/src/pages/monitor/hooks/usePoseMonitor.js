import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import * as mpPose from "@mediapipe/pose";
import * as drawingUtils from "@mediapipe/drawing_utils";

import { apiRequest } from "../../../lib/apiClient";
import {
  CLIP_POST_S,
  CLIP_PRE_S,
  DEGRADED_POSE_MODEL_COMPLEXITY,
  DEGRADED_DRAW_FPS,
  DEGRADED_INFER_FPS,
  LIVE_DRAW_FPS,
  LIVE_POSE_MODEL_COMPLEXITY,
  LOW_FPS_ENTER,
  LOW_FPS_EXIT,
  LOW_FPS_HOLD_MS,
  MAX_PROC_FPS,
  MONITOR_PAYLOAD_Q_SCALE,
  NUM_JOINTS,
  REPLAY_UI_UPDATE_MS,
} from "../constants";
import { clamp01, labelForTriage, prettyModelTag } from "../utils";

const { drawConnectors, drawLandmarks } = drawingUtils;
const TRIAGE_FALL_CONFIRM_N = 2;
const TRIAGE_SAFE_CONFIRM_N = 2;
const TRIAGE_UNCERTAIN_CONFIRM_N = 3;
const FALL_HISTORY_DEDUP_MS_DEFAULT = 30_000;
const CAPTURE_RESOLUTIONS = {
  "480p": { w: 640, h: 480 },
  "540p": { w: 960, h: 540 },
  "720p": { w: 1280, h: 720 },
  "1080p": { w: 1920, h: 1080 },
};

function getClipFlags(settingsPayload) {
  const sys = settingsPayload?.system || {};
  return {
    storeEventClips: Boolean(sys?.store_event_clips),
    anonymize: Boolean(sys?.anonymize_skeleton_data ?? true),
  };
}

function getVideoLoadErrorMessage(videoEl, file) {
  const code = Number(videoEl?.error?.code || 0);
  const byCode = {
    1: "Video loading aborted.",
    2: "Network/media fetch error while loading video.",
    3: "Video decode error (codec may be unsupported).",
    4: "Video format not supported by this browser.",
  };
  const base = byCode[code] || "Unknown video load error.";
  const t = String(file?.type || "").trim();
  const hint = t ? ` File type: ${t}.` : "";
  return `${base}${hint} Try an H.264 MP4 (AAC) or WebM file.`;
}

/**
 * Owns:
 * - camera + MediaPipe Pose
 * - raw pose ring buffer
 * - sending windows to backend
 * - event clip upload (skeleton-only)
 */
export function usePoseMonitor({
  apiBase,
  isActive,
  monitoringOn,
  registerController,
  settingsPayload,
  deployW,
  deployS,
  targetFps,
  mode,
  chosen,
  opCode,
  mcEnabled,
  mcCfg,
  activeDatasetCode,
  chosenSpec,
  onAutoStop,
}) {
  // Refs for camera + mediapipe
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const isActiveRef = useRef(Boolean(isActive));
  const poseRef = useRef(null);
  const streamRef = useRef(null);
  const videoFileRef = useRef(null);
  const videoObjectUrlRef = useRef(null);
  const inputSourceRef = useRef("camera");
  const rafRef = useRef(null);
  const liveFlagRef = useRef(false);

  // Throttle MediaPipe inference so navigation stays responsive when /Monitor isn't visible.
  const lastInferTsRef = useRef(0);
  const inferFpsRef = useRef(15);
  const poseSendBusyRef = useRef(false);
  const lastDrawMsRef = useRef(0);
  const adaptiveDrawFpsRef = useRef(LIVE_DRAW_FPS);
  const adaptiveInferFpsRef = useRef(0);
  const degradedModeRef = useRef(false);
  const lowFpsSinceMsRef = useRef(0);
  const lastReplayUiMsRef = useRef(0);

  // Cap pose processing rate to reduce CPU.
  const lastProcMsRef = useRef(0);

  // Live state
  const [liveRunning, setLiveRunning] = useState(false);
  const [streamFps, setStreamFps] = useState(null);
  const [inputSource, setInputSource] = useState("camera"); // camera | video
  const [captureResolutionPreset, setCaptureResolutionPreset] = useState("720p");
  const [selectedVideoName, setSelectedVideoName] = useState("");
  const [replayFile, setReplayFile] = useState(null);
  const [startError, setStartError] = useState("");
  const [predictError, setPredictError] = useState("");
  const [replayCurrentS, setReplayCurrentS] = useState(0);
  const [replayDurationS, setReplayDurationS] = useState(0);

  // Prediction UI
  const [triageState, setTriageState] = useState("not_fall");
  const [pFall, setPFall] = useState(null);
  const [sigma, setSigma] = useState(null);
  const [safeAlert, setSafeAlert] = useState(null);
  const [recallAlert, setRecallAlert] = useState(null);
  const [safeState, setSafeState] = useState(null);
  const [recallState, setRecallState] = useState(null);
  const lastUiRef = useRef({
    triageState: "not_fall",
    safeAlert: null,
    recallAlert: null,
    safeState: null,
    recallState: null,
    pFall: null,
    sigma: null,
  });

  // Timeline markers: store last 50 prediction windows (safe/uncertain/fall)
  const [timeline, setTimeline] = useState([]); // [{kind:'safe'|'fall'|'uncertain', t:number, seq:number}]

  // Frame buffer for windowing
  const rawFramesRef = useRef([]); // [{t,xy,conf}]
  const lastPoseTsRef = useRef(null);
  const fpsDeltasRef = useRef([]);
  const fpsEstimateRef = useRef(null);
  const lastSentRef = useRef(0);
  const predictInFlightRef = useRef(false);
  const payloadTRef = useRef([]);
  const payloadXYRef = useRef([]);
  const payloadConfRef = useRef([]);

  // Skeleton clip saving
  const pendingClipRef = useRef(null);
  const uploadedClipIdsRef = useRef(new Set());

  // Session id for server-side state machine
  const sessionIdRef = useRef(`monitor-${Math.random().toString(16).slice(2)}`);
  const windowSeqRef = useRef(0);
  const triageStableRef = useRef({ fall: 0, uncertain: 0, safe: 0, last: "not_fall" });
  const timelineSeqRef = useRef(0);
  const timelineSeenEventIdsRef = useRef(new Set());
  const lastFallHistoryTsRef = useRef(0);
  const wsRef = useRef(null);
  const wsReadyRef = useRef(false);
  const wsPendingRef = useRef(null);

  // Keep isActive in a ref (MediaPipe callback shouldn't rebind each render)
  useEffect(() => {
    isActiveRef.current = Boolean(isActive);
    if (!isActiveRef.current) {
      const canvasEl = canvasRef.current;
      if (canvasEl) {
        const ctx = canvasEl.getContext("2d");
        if (ctx) {
          ctx.clearRect(0, 0, canvasEl.width, canvasEl.height);
        }
      }
    }
  }, [isActive]);

  const autoStopMonitoring = useCallback(() => {
    try {
      if (typeof onAutoStop === "function") {
        void Promise.resolve(onAutoStop(false));
      }
    } catch {
      // ignore
    }
  }, [onAutoStop]);

  // Update inference throttle and MediaPipe complexity based on whether this page is active.
  useEffect(() => {
    // Keep live inference close to dataset target FPS for better train/runtime parity.
    const fgFps = Math.min(30, Math.max(8, Number(targetFps) || 23));
    const bgFps = Math.min(5, fgFps);
    inferFpsRef.current = isActiveRef.current ? fgFps : bgFps;

    try {
      if (poseRef.current && poseRef.current.setOptions) {
        poseRef.current.setOptions({
          modelComplexity: LIVE_POSE_MODEL_COMPLEXITY,
          smoothLandmarks: true,
        });
      }
    } catch {
      // ignore
    }
  }, [targetFps, isActive]);

  // Update stream FPS display at a low rate (avoid re-render on every frame)
  useEffect(() => {
    if (!liveRunning) {
      setStreamFps(null);
      adaptiveDrawFpsRef.current = LIVE_DRAW_FPS;
      adaptiveInferFpsRef.current = 0;
      degradedModeRef.current = false;
      lowFpsSinceMsRef.current = 0;
      return;
    }

    const id = setInterval(() => {
      const v = fpsEstimateRef.current;
      const next = v == null ? null : Number(v);
      setStreamFps((prev) => {
        if (prev == null && next == null) return prev;
        if (prev == null || next == null) return next;
        return Math.abs(prev - next) < 0.2 ? prev : next;
      });
    }, 500);

    return () => clearInterval(id);
  }, [liveRunning]);

  const ensureCanvasMatchesVideo = useCallback(() => {
    const videoEl = videoRef.current;
    const canvasEl = canvasRef.current;
    if (!videoEl || !canvasEl) return;

    const vw = videoEl.videoWidth || 1280;
    const vh = videoEl.videoHeight || 720;

    if (canvasEl.width !== vw) canvasEl.width = vw;
    if (canvasEl.height !== vh) canvasEl.height = vh;
  }, []);

  const stopLive = useCallback(() => {
    liveFlagRef.current = false;
    setLiveRunning(false);

    if (rafRef.current) cancelAnimationFrame(rafRef.current);
    rafRef.current = null;

    try {
      poseRef.current?.close?.();
    } catch {
      // ignore
    }
    poseRef.current = null;

    if (streamRef.current) {
      streamRef.current.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
    }

    const videoEl = videoRef.current;
    if (videoEl) {
      try {
        videoEl.pause();
      } catch {
        // ignore
      }
      videoEl.srcObject = null;
      if (videoObjectUrlRef.current) {
        videoEl.removeAttribute("src");
        videoEl.load?.();
      }
    }
    if (videoObjectUrlRef.current) {
      try {
        URL.revokeObjectURL(videoObjectUrlRef.current);
      } catch {
        // ignore
      }
      videoObjectUrlRef.current = null;
    }
    if (wsPendingRef.current) {
      try {
        wsPendingRef.current.reject?.(new Error("ws closed"));
      } catch {
        // ignore
      }
      wsPendingRef.current = null;
    }
    if (wsRef.current) {
      try {
        wsRef.current.close();
      } catch {
        // ignore
      }
    }
    wsRef.current = null;
    wsReadyRef.current = false;

    // Clear buffer
    rawFramesRef.current = [];
    lastPoseTsRef.current = null;
    fpsDeltasRef.current = [];
    lastSentRef.current = 0;

    // Clip upload state
    pendingClipRef.current = null;
    uploadedClipIdsRef.current = new Set();

    // Reset UI
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
    windowSeqRef.current = 0;
    triageStableRef.current = { fall: 0, uncertain: 0, safe: 0, last: "not_fall" };
    setReplayCurrentS(0);
    setReplayDurationS(0);
    lastReplayUiMsRef.current = 0;
    setStartError("");
    setPredictError("");
  }, []);

  const setVideoFile = useCallback((file) => {
    stopLive();
    autoStopMonitoring();
    if (file) {
      videoFileRef.current = file;
      setReplayFile(file);
      setSelectedVideoName(String(file.name || "video"));
      inputSourceRef.current = "video";
      setInputSource("video");
      setStartError("");
    } else {
      videoFileRef.current = null;
      setReplayFile(null);
      setSelectedVideoName("");
      inputSourceRef.current = "camera";
      setInputSource("camera");
      setStartError("");
    }
  }, [autoStopMonitoring, stopLive]);

  const setInputMode = useCallback((mode) => {
    const m = mode === "video" ? "video" : "camera";
    inputSourceRef.current = m;
    setInputSource(m);
    if (m === "camera") {
      videoFileRef.current = null;
      setReplayFile(null);
      setSelectedVideoName("");
    }
    setStartError("");
  }, []);

  const setCaptureResolution = useCallback((preset) => {
    if (!Object.prototype.hasOwnProperty.call(CAPTURE_RESOLUTIONS, preset)) return;
    setCaptureResolutionPreset(String(preset));
  }, []);

  const addTimelineMarker = useCallback((kind, options = {}) => {
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

  const maybeFinalizeClipUpload = useCallback(async () => {
    const { storeEventClips } = getClipFlags(settingsPayload);
    if (!storeEventClips) {
      pendingClipRef.current = null;
      return;
    }

    const pending = pendingClipRef.current;
    if (!pending) return;

    const raw = rawFramesRef.current;
    if (!raw || raw.length < 2) return;

    const lastT = raw[raw.length - 1]?.t;
    if (typeof lastT !== "number") return;
    if (lastT < pending.deadlineTs) return;

    const sent = uploadedClipIdsRef.current;
    if (sent && sent.has(String(pending.eventId))) {
      pendingClipRef.current = null;
      return;
    }

    const startTs = pending.triggerEndTs - pending.preMs;
    const endTs = pending.triggerEndTs + pending.postMs;
    const frames = raw.filter((fr) => fr && typeof fr.t === "number" && fr.t >= startTs && fr.t <= endTs);
    if (!frames || frames.length < 2) {
      pendingClipRef.current = null;
      return;
    }

    try {
      const nFrames = frames.length;
      const tMs = new Array(nFrames);
      const xy = new Array(nFrames);
      const conf = new Array(nFrames);
      for (let i = 0; i < nFrames; i++) {
        const fr = frames[i];
        tMs[i] = fr.t;
        xy[i] = fr.xy;
        conf[i] = fr.conf;
      }

      const clipPayload = {
        resident_id: 1,
        dataset_code: pending.ctx?.dataset_code,
        mode: pending.ctx?.mode,
        op_code: pending.ctx?.op_code,
        use_mc: pending.ctx?.use_mc,
        mc_M: pending.ctx?.mc_M,
        pre_s: pending.preMs / 1000,
        post_s: pending.postMs / 1000,
        t_ms: tMs,
        xy,
        conf,
      };

      const data = await apiRequest(
        apiBase,
        `/api/events/${encodeURIComponent(pending.eventId)}/skeleton_clip`,
        { method: "POST", body: clipPayload }
      );

      if (data?.ok) {
        sent.add(String(pending.eventId));
      }
    } catch (err) {
      console.error("Failed to upload skeleton clip", err);
    } finally {
      pendingClipRef.current = null;
    }
  }, [apiBase, settingsPayload]);

  const ensurePredictWs = useCallback(async () => {
    if (wsRef.current && wsReadyRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      return wsRef.current;
    }

    if (!apiBase) {
      throw new Error("Missing apiBase for WebSocket connection");
    }

    let base = String(apiBase || "").trim();
    if (!base) throw new Error("Empty apiBase");
    if (base.endsWith("/")) base = base.slice(0, -1);
    let wsBase = base.replace(/^http:/i, "ws:").replace(/^https:/i, "wss:");
    let wsPath = "/api/monitor/ws";
    try {
      const u = new URL(base);
      wsBase = u.origin.replace(/^http:/i, "ws:").replace(/^https:/i, "wss:");
      wsPath = (u.pathname || "").replace(/\/+$/, "") + "/api/monitor/ws";
      wsPath = wsPath.replace(/\/{2,}/g, "/");
    } catch {
      // ignore and use fallback from raw string replacement
    }
    const wsUrl = `${wsBase}${wsPath}`;

    const ws = new WebSocket(wsUrl);
    wsRef.current = ws;
    wsReadyRef.current = false;

    return await new Promise((resolve, reject) => {
      let settled = false;
      const t = window.setTimeout(() => {
        if (settled) return;
        settled = true;
        try {
          ws.close();
        } catch {
          // ignore
        }
        reject(new Error("WebSocket connect timeout"));
      }, 3000);

      ws.onopen = () => {
        if (settled) return;
        settled = true;
        window.clearTimeout(t);
        wsReadyRef.current = true;
        resolve(ws);
      };
      ws.onerror = () => {
        if (settled) return;
        settled = true;
        window.clearTimeout(t);
        reject(new Error("WebSocket connect failed"));
      };
      ws.onclose = () => {
        wsReadyRef.current = false;
        if (wsRef.current === ws) wsRef.current = null;
        if (wsPendingRef.current) {
          try {
            wsPendingRef.current.reject?.(new Error("WebSocket closed"));
          } catch {
            // ignore
          }
          wsPendingRef.current = null;
        }
      };
      ws.onmessage = (evt) => {
        const pending = wsPendingRef.current;
        if (!pending) return;
        try {
          const msg = JSON.parse(String(evt?.data || "{}"));
          if (msg?.error) {
            pending.reject(new Error(String(msg?.detail || "predict_window ws error")));
          } else {
            pending.resolve(msg);
          }
        } catch (err) {
          pending.reject(err);
        } finally {
          wsPendingRef.current = null;
        }
      };
    });
  }, [apiBase]);

  const predictViaWs = useCallback(async (payload) => {
    const ws = await ensurePredictWs();
    return await new Promise((resolve, reject) => {
      const timeoutId = window.setTimeout(() => {
        if (wsPendingRef.current) {
          wsPendingRef.current = null;
        }
        reject(new Error("WebSocket predict timeout"));
      }, 5000);

      wsPendingRef.current = {
        resolve: (msg) => {
          window.clearTimeout(timeoutId);
          resolve(msg);
        },
        reject: (err) => {
          window.clearTimeout(timeoutId);
          reject(err);
        },
      };

      try {
        ws.send(JSON.stringify(payload));
      } catch (err) {
        wsPendingRef.current = null;
        window.clearTimeout(timeoutId);
        reject(err);
      }
    });
  }, [ensurePredictWs]);

  const maybeSendWindow = useCallback(async () => {
    if (predictInFlightRef.current) return;
    const now = performance.now();
    const minGapMs = Math.max(250, (deployS / Math.max(1, targetFps)) * 1000);
    if (now - lastSentRef.current < minGapMs) return;

    const raw = rawFramesRef.current;
    if (!raw || raw.length < 2) return;

    const dtMs = 1000 / Math.max(1, Number(targetFps) || 30);
    const needMs = (deployW - 1) * dtMs;
    const endTs = raw[raw.length - 1].t;
    const startNeed = endTs - needMs;

    // Wait until we have enough history to cover the full window duration.
    if (raw[0].t > startNeed) return;

    // Include one frame before startNeed for interpolation on the server.
    let i0 = 0;
    while (i0 < raw.length && raw[i0].t < startNeed) i0++;
    const startIdx = Math.max(0, i0 - 1);
    const slice = raw.slice(startIdx);
    const nSlice = slice.length;

    lastSentRef.current = now;

    const { storeEventClips } = getClipFlags(settingsPayload);

    // Reuse arrays to reduce per-request allocations (guarded by in-flight mutex).
    const tArr = payloadTRef.current;
    const xyArr = payloadXYRef.current;
    const confArr = payloadConfRef.current;
    tArr.length = nSlice;
    xyArr.length = nSlice;
    confArr.length = nSlice;
    for (let i = 0; i < nSlice; i++) {
      const fr = slice[i];
      tArr[i] = fr.t;
      xyArr[i] = fr.xy;
      confArr[i] = fr.conf;
    }

    const xyQ = new Array(nSlice * NUM_JOINTS * 2);
    const confQ = new Array(nSlice * NUM_JOINTS);
    let q = 0;
    let c = 0;
    for (let i = 0; i < nSlice; i++) {
      const fr = slice[i];
      const pts = fr.xy || [];
      const cf = fr.conf || [];
      for (let j = 0; j < NUM_JOINTS; j++) {
        const p = pts[j] || [0, 0];
        const v = cf[j] == null ? 1 : cf[j];
        xyQ[q++] = Math.round(clamp01(Number(p[0]) || 0) * MONITOR_PAYLOAD_Q_SCALE);
        xyQ[q++] = Math.round(clamp01(Number(p[1]) || 0) * MONITOR_PAYLOAD_Q_SCALE);
        confQ[c++] = Math.round(clamp01(Number(v) || 0) * MONITOR_PAYLOAD_Q_SCALE);
      }
    }

    const payload = {
      window_seq: ++windowSeqRef.current,
      session_id: sessionIdRef.current,
      input_source: inputSourceRef.current || "camera",
      resident_id: 1,
      mode,
      dataset_code: activeDatasetCode,
      op_code: opCode || settingsPayload?.system?.active_op_code || "OP-2",

      model_tcn: mode !== "gcn" ? chosen.tcn : null,
      model_gcn: mode !== "tcn" ? chosen.gcn : null,
      model_id: mode === "tcn" ? chosen.tcn : mode === "gcn" ? chosen.gcn : null,

      // The model expects this FPS after resampling.
      fps: targetFps,
      target_fps: targetFps,
      target_T: deployW,

      // Useful for debugging / dashboards (does NOT affect inference)
      capture_fps: streamFps || fpsEstimateRef.current || null,

      timestamp_ms: Date.now(),
      use_mc: Boolean(mcEnabled),
      mc_M: mcCfg?.M,

      // Persist events when monitoring is on OR when we need an event_id for clip saving.
      persist: Boolean(monitoringOn || storeEventClips),

      // Raw pose samples (variable FPS)
      raw_t_ms: tArr,
      raw_shape: [nSlice, NUM_JOINTS],
      raw_xy_q: xyQ,
      raw_conf_q: confQ,

      window_end_t_ms: endTs,
    };

    try {
      predictInFlightRef.current = true;
      const data = await predictViaWs(payload);
      setPredictError("");

      const safeObj = data?.policy_alerts?.safe;
      const recallObj = data?.policy_alerts?.recall;
      const triRaw = String(data?.triage_state || data?.triageState || "not_fall").toLowerCase();
      const safeStateRaw = String(data?.safe_state || safeObj?.state || "").toLowerCase();
      const safeBool =
        typeof data?.safe_alert === "boolean"
          ? data.safe_alert
          : typeof safeObj?.alert === "boolean"
          ? safeObj.alert
          : null;
      const recallBool =
        typeof data?.recall_alert === "boolean"
          ? data.recall_alert
          : typeof recallObj?.alert === "boolean"
          ? recallObj.alert
          : null;
      const nextSafeState = (data?.safe_state || safeObj?.state || null) ?? null;
      const nextRecallState = (data?.recall_state || recallObj?.state || null) ?? null;
      if (lastUiRef.current.safeAlert !== safeBool) {
        lastUiRef.current.safeAlert = safeBool;
        setSafeAlert(safeBool);
      }
      if (lastUiRef.current.recallAlert !== recallBool) {
        lastUiRef.current.recallAlert = recallBool;
        setRecallAlert(recallBool);
      }
      if (lastUiRef.current.safeState !== nextSafeState) {
        lastUiRef.current.safeState = nextSafeState;
        setSafeState(nextSafeState);
      }
      if (lastUiRef.current.recallState !== nextRecallState) {
        lastUiRef.current.recallState = nextRecallState;
        setRecallState(nextRecallState);
      }

      // In high-performance mode, current prediction should follow backend safe channel directly.
      let triCandidate = safeStateRaw || triRaw || "not_fall";
      if (triCandidate !== "fall" && triCandidate !== "uncertain") triCandidate = "not_fall";
      const st = triageStableRef.current;
      if (triCandidate === "fall" && safeBool !== false) {
        st.fall += 1;
        st.uncertain = 0;
        st.safe = 0;
      } else if (triCandidate === "uncertain") {
        st.uncertain += 1;
        st.fall = 0;
        st.safe = 0;
      } else {
        st.safe += 1;
        st.fall = 0;
        st.uncertain = 0;
      }
      let triStable = st.last || "not_fall";
      if (st.fall >= TRIAGE_FALL_CONFIRM_N) triStable = "fall";
      else if (st.safe >= TRIAGE_SAFE_CONFIRM_N) triStable = "not_fall";
      else if (st.uncertain >= TRIAGE_UNCERTAIN_CONFIRM_N && st.fall === 0) triStable = "uncertain";
      st.last = triStable;
      if (lastUiRef.current.triageState !== triStable) {
        lastUiRef.current.triageState = triStable;
        setTriageState(triStable);
      }

      // Schedule a skeleton clip upload (pre + post seconds around this window end)
      try {
        const evId = data?.event_id;
        if (storeEventClips && evId != null) {
          const sent = uploadedClipIdsRef.current;
          const already = sent && sent.has(String(evId));
          const pending = pendingClipRef.current;

          if (!already && (!pending || String(pending.eventId) !== String(evId))) {
            pendingClipRef.current = {
              eventId: evId,
              triggerEndTs: endTs,
              deadlineTs: endTs + CLIP_POST_S * 1000,
              preMs: CLIP_PRE_S * 1000,
              postMs: CLIP_POST_S * 1000,
              ctx: {
                dataset_code: activeDatasetCode,
                mode,
                op_code: opCode || settingsPayload?.system?.active_op_code || "OP-2",
                use_mc: Boolean(mcEnabled),
                mc_M: mcCfg?.M,
              },
            };
          }
        }
      } catch {
        // ignore
      }

      // pFall display from the active model output.
      let mu = null;
      let sig = null;

      const mOut = mode === "hybrid" ? data?.models?.tcn : data?.models?.[mode];
      if (mOut) {
        // Use tracker-smoothed score first so P(fall) matches current triage semantics.
        mu = mOut?.triage?.ps != null
          ? Number(mOut.triage.ps)
          : mOut?.p_alert_in != null
            ? Number(mOut.p_alert_in)
            : mOut?.mu != null
              ? Number(mOut.mu)
              : mOut?.p_det != null
                ? Number(mOut.p_det)
                : null;
        sig = mOut?.sigma != null ? Number(mOut.sigma) : null;
      }

      if (lastUiRef.current.pFall !== mu) {
        lastUiRef.current.pFall = mu;
        setPFall(mu);
      }
      if (lastUiRef.current.sigma !== sig) {
        lastUiRef.current.sigma = sig;
        setSigma(sig);
      }

      // Timeline must match Current Prediction exactly.
      let markerKind = "safe";
      const triNorm = String(triStable || "").toLowerCase();
      if (triNorm === "fall") markerKind = "fall";
      else if (triNorm === "uncertain") markerKind = "uncertain";
      const dedupMs = Math.round(
        Math.max(
          1000,
          1000 * Number(settingsPayload?.system?.alert_cooldown_sec || FALL_HISTORY_DEDUP_MS_DEFAULT / 1000)
        )
      );
      addTimelineMarker(markerKind, {
        eventId: markerKind === "fall" ? data?.event_id ?? null : null,
        dedupMs,
      });
    } catch (err) {
      console.error("Error calling /api/monitor/predict_window", err);
      setPredictError(String(err?.message || err || "predict_window failed"));
    } finally {
      predictInFlightRef.current = false;
    }
  }, [
    activeDatasetCode,
    addTimelineMarker,
    chosen,
    deployS,
    deployW,
    mcCfg,
    mcEnabled,
    mode,
    monitoringOn,
    opCode,
    predictViaWs,
    settingsPayload,
    streamFps,
    targetFps,
  ]);

  const handlePoseResults = useCallback(
    (results) => {
      // Cap processing rate.
      const nowMs = performance.now();
      const procCap = Math.min(MAX_PROC_FPS, Math.max(8, Number(targetFps) + 2));
      if (nowMs - lastProcMsRef.current < 1000 / procCap) return;
      lastProcMsRef.current = nowMs;

      // Adaptive draw FPS: if capture FPS stays low, reduce draw load temporarily.
      const estFps = Number(fpsEstimateRef.current);
      if (Number.isFinite(estFps) && estFps > 0) {
        if (estFps < LOW_FPS_ENTER) {
          if (!lowFpsSinceMsRef.current) lowFpsSinceMsRef.current = nowMs;
          if (nowMs - lowFpsSinceMsRef.current > LOW_FPS_HOLD_MS) {
            adaptiveDrawFpsRef.current = DEGRADED_DRAW_FPS;
            adaptiveInferFpsRef.current = DEGRADED_INFER_FPS;
            if (!degradedModeRef.current && poseRef.current?.setOptions) {
              degradedModeRef.current = true;
              try {
                poseRef.current.setOptions({
                  modelComplexity: DEGRADED_POSE_MODEL_COMPLEXITY,
                  smoothLandmarks: false,
                });
              } catch {
                // ignore
              }
            }
          }
        } else if (estFps > LOW_FPS_EXIT) {
          lowFpsSinceMsRef.current = 0;
          adaptiveDrawFpsRef.current = LIVE_DRAW_FPS;
          adaptiveInferFpsRef.current = 0;
          if (degradedModeRef.current && poseRef.current?.setOptions) {
            degradedModeRef.current = false;
            try {
              poseRef.current.setOptions({
                modelComplexity: LIVE_POSE_MODEL_COMPLEXITY,
                smoothLandmarks: true,
              });
            } catch {
              // ignore
            }
          }
        }
      }

      const canvasEl = canvasRef.current;
      if (!canvasEl) return;
      const ctx = canvasEl.getContext("2d");

      const doDraw = isActiveRef.current;
      const landmarks = results.poseLandmarks;

      // No pose detected.
      if (!landmarks || !landmarks.length) {
        if (doDraw && ctx) {
          const drawNowMs = performance.now();
          if (drawNowMs - lastDrawMsRef.current < 1000 / Math.max(1, adaptiveDrawFpsRef.current)) {
            return;
          }
          lastDrawMsRef.current = drawNowMs;
          ensureCanvasMatchesVideo();
          ctx.fillStyle = "#020617";
          ctx.fillRect(0, 0, canvasEl.width, canvasEl.height);
          ctx.font = "14px system-ui, -apple-system, Segoe UI, Roboto";
          ctx.fillStyle = "#94a3b8";
          ctx.fillText("No pose detected…", 16, 24);
        }
        return;
      }

      // Draw only when this page is active.
      if (doDraw && ctx) {
        const drawNowMs = performance.now();
        if (drawNowMs - lastDrawMsRef.current < 1000 / Math.max(1, adaptiveDrawFpsRef.current)) {
          // Skip frequent repaint to reduce main-thread pressure.
        } else {
          lastDrawMsRef.current = drawNowMs;
        ensureCanvasMatchesVideo();
        const w = canvasEl.width;
        const h = canvasEl.height;

        ctx.fillStyle = "#020617";
        ctx.fillRect(0, 0, w, h);

        drawConnectors(ctx, landmarks, mpPose.POSE_CONNECTIONS, {
          color: "#22c55e",
          lineWidth: 2,
        });
        drawLandmarks(ctx, landmarks, { color: "#fbbf24", lineWidth: 1 });

        ctx.font = "12px system-ui, -apple-system, Segoe UI, Roboto";
        ctx.fillStyle = "#94a3b8";
        ctx.fillText(new Date().toLocaleTimeString(), 16, h - 16);
        }
      }

      // Build raw frame
      const xyFrame = new Array(NUM_JOINTS);
      const confFrame = new Array(NUM_JOINTS);

      for (let i = 0; i < NUM_JOINTS; i++) {
        const lm = landmarks[i];
        if (!lm) {
          xyFrame[i] = [0, 0];
          confFrame[i] = 0;
        } else {
          xyFrame[i] = [clamp01(lm.x), clamp01(lm.y)];
          confFrame[i] = typeof lm.visibility === "number" ? clamp01(lm.visibility) : 1.0;
        }
      }

      const tNow = performance.now();

      // Estimate FPS from callback timing (smoothed)
      const last = lastPoseTsRef.current;
      if (typeof last === "number") {
        const dt = tNow - last;
        if (dt > 0 && dt < 5000) {
          const arr = fpsDeltasRef.current;
          arr.push(dt);
          if (arr.length > 30) arr.shift();

          const meanDt = arr.reduce((a, b) => a + b, 0) / Math.max(1, arr.length);
          const estFps = meanDt > 0 ? 1000 / meanDt : null;
          if (estFps && Number.isFinite(estFps)) {
            const prev = fpsEstimateRef.current;
            fpsEstimateRef.current = prev == null ? estFps : prev * 0.8 + estFps * 0.2;
          }
        }
      }
      lastPoseTsRef.current = tNow;

      // Push raw
      const raw = rawFramesRef.current;
      raw.push({ t: tNow, xy: xyFrame, conf: confFrame });

      // Keep ~12s max
      const maxRaw = Math.max(600, Math.ceil(targetFps * 12));
      if (raw.length > maxRaw) raw.splice(0, raw.length - maxRaw);

      // Clip upload + inference
      void maybeFinalizeClipUpload();
      void maybeSendWindow();
    },
    [ensureCanvasMatchesVideo, maybeFinalizeClipUpload, maybeSendWindow, targetFps]
  );

  const startLive = useCallback(async () => {
    const videoEl = videoRef.current;
    if (!videoEl) return false;

    try {
      setStartError("");
      liveFlagRef.current = true;
      setLiveRunning(true);

      const source = inputSourceRef.current || "camera";
      const file = videoFileRef.current || replayFile;
      const useVideoFile = source === "video";
      if (useVideoFile && !file) {
        throw new Error("Replay mode selected but no video file is loaded.");
      }
      if (useVideoFile) {
        if (videoObjectUrlRef.current) {
          try {
            URL.revokeObjectURL(videoObjectUrlRef.current);
          } catch {
            // ignore
          }
          videoObjectUrlRef.current = null;
        }
        try {
          videoEl.pause();
        } catch {
          // ignore
        }
        videoEl.srcObject = null;
        videoEl.removeAttribute("src");
        try {
          videoEl.load?.();
        } catch {
          // ignore
        }
        const u = URL.createObjectURL(file);
        videoObjectUrlRef.current = u;
        videoEl.src = u;
        videoEl.currentTime = 0;
        videoEl.muted = true;
        videoEl.playsInline = true;
        videoEl.preload = "auto";
      } else {
        const cap = CAPTURE_RESOLUTIONS[captureResolutionPreset] || CAPTURE_RESOLUTIONS["720p"];
        const stream = await navigator.mediaDevices.getUserMedia({
          video: {
            width: { ideal: cap.w, max: cap.w },
            height: { ideal: cap.h, max: cap.h },
            frameRate: { ideal: targetFps, max: targetFps },
          },
          audio: false,
        });
        streamRef.current = stream;
        videoEl.srcObject = stream;
      }

      await new Promise((resolve, reject) => {
        if (videoEl.readyState >= 1) {
          resolve();
          return;
        }
        let done = false;
        const timer = window.setTimeout(() => {
          if (done) return;
          done = true;
          cleanup();
          reject(new Error("Video metadata load timeout"));
        }, 10000);
        const cleanup = () => {
          videoEl.removeEventListener("loadedmetadata", onLoaded);
          videoEl.removeEventListener("canplay", onCanPlay);
          videoEl.removeEventListener("error", onErr);
          window.clearTimeout(timer);
        };
        const onLoaded = () => {
          if (done) return;
          done = true;
          cleanup();
          resolve();
        };
        const onCanPlay = () => {
          if (done) return;
          done = true;
          cleanup();
          resolve();
        };
        const onErr = () => {
          if (done) return;
          done = true;
          cleanup();
          reject(new Error(getVideoLoadErrorMessage(videoEl, file)));
        };
        videoEl.addEventListener("loadedmetadata", onLoaded, { once: true });
        videoEl.addEventListener("canplay", onCanPlay, { once: true });
        videoEl.addEventListener("error", onErr, { once: true });
        try {
          videoEl.load?.();
        } catch {
          // ignore
        }
      });

      try {
        await videoEl.play();
      } catch (e) {
        throw new Error(`Video play failed: ${String(e?.message || e)}`);
      }
      ensureCanvasMatchesVideo();
      if (useVideoFile) {
        setReplayCurrentS(Number(videoEl.currentTime || 0));
        setReplayDurationS(Number(videoEl.duration || 0));
      }

      // Try camera-reported FPS if available
      if (streamRef.current) {
        try {
          const track = streamRef.current.getVideoTracks()[0];
          const settings = track?.getSettings?.();
          if (settings && typeof settings.frameRate === "number") {
            fpsEstimateRef.current = settings.frameRate;
            setStreamFps(settings.frameRate);
          }
        } catch {
          // ignore
        }
      }

      const pose = new mpPose.Pose({
        locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/pose/${file}`,
      });

      pose.setOptions({
        modelComplexity: LIVE_POSE_MODEL_COMPLEXITY,
        smoothLandmarks: true,
        minDetectionConfidence: 0.5,
        minTrackingConfidence: 0.5,
      });

      pose.onResults(handlePoseResults);
      poseRef.current = pose;

      const loop = async () => {
        if (!liveFlagRef.current) return;
        if ((inputSourceRef.current || "camera") === "video" && videoEl.ended) {
          stopLive();
          autoStopMonitoring();
          return;
        }
        if ((inputSourceRef.current || "camera") === "video") {
          const drawNowMs = performance.now();
          if (drawNowMs - lastReplayUiMsRef.current >= REPLAY_UI_UPDATE_MS) {
            lastReplayUiMsRef.current = drawNowMs;
            const ct = Number(videoEl.currentTime || 0);
            const du = Number(videoEl.duration || 0);
            if (Number.isFinite(ct)) {
              setReplayCurrentS((prev) => (Math.abs(Number(prev || 0) - ct) < 0.08 ? prev : ct));
            }
            if (Number.isFinite(du)) {
              setReplayDurationS((prev) => (Math.abs(Number(prev || 0) - du) < 0.08 ? prev : du));
            }
          }
        }

        const now = performance.now();
        const target = Math.max(
          1,
          Number(adaptiveInferFpsRef.current) || Number(inferFpsRef.current) || 15
        );
        const minIntervalMs = 1000 / target;

        if (now - lastInferTsRef.current >= minIntervalMs) {
          lastInferTsRef.current = now;

          if (videoEl.readyState >= 2 && poseRef.current && !poseSendBusyRef.current) {
            poseSendBusyRef.current = true;
            poseRef.current
              .send({ image: videoEl })
              .catch((err) => {
                console.error("pose.send error", err);
              })
              .finally(() => {
                poseSendBusyRef.current = false;
              });
          }
        }

        rafRef.current = requestAnimationFrame(loop);
      };

      void loop();
      return true;
    } catch (err) {
      console.error("Error starting monitor", err);
      setStartError(String(err?.message || err || "Failed to start monitor"));
      stopLive();
      return false;
    }
  }, [autoStopMonitoring, captureResolutionPreset, ensureCanvasMatchesVideo, handlePoseResults, replayFile, stopLive, targetFps]);

  // Register start/stop with the MonitoringContext so other pages can toggle runtime.
  useEffect(() => {
    if (!registerController) return;

    registerController({
      start: startLive,
      stop: stopLive,
    });

    return () => registerController(null);
  }, [registerController, startLive, stopLive]);

  // Ensure <video> is set for iOS-style behaviour.
  useEffect(() => {
    const v = videoRef.current;
    if (v) v.playsInline = true;
  }, []);

  // Cleanup
  useEffect(() => {
    return () => stopLive();
  }, [stopLive]);

  const resetSession = useCallback(async () => {
    try {
      try {
        await apiRequest(
          apiBase,
          `/api/monitor/reset_session?session_id=${encodeURIComponent(sessionIdRef.current)}`,
          { method: "POST" }
        );
      } catch (err) {
        if (Number(err?.status) === 404) {
          await apiRequest(
            apiBase,
            `/api/v1/monitor/reset_session?session_id=${encodeURIComponent(sessionIdRef.current)}`,
            { method: "POST" }
          );
        } else {
          throw err;
        }
      }
    } catch {
      // ignore
    }
  }, [apiBase]);

  const seekReplay = useCallback((ratio) => {
    const videoEl = videoRef.current;
    if (!videoEl) return;
    const du = Number(videoEl.duration || 0);
    if (!Number.isFinite(du) || du <= 0) return;
    const r = Math.max(0, Math.min(1, Number(ratio) || 0));
    videoEl.currentTime = r * du;
    setReplayCurrentS(Number(videoEl.currentTime || 0));
    setReplayDurationS(du);
  }, []);

  const testFall = useCallback(async () => {
    // 1) Try backend helper endpoint (if present)
    try {
      const data = await apiRequest(apiBase, "/api/events/test_fall", {
        method: "POST",
        body: { resident_id: 1, model_code: prettyModelTag(settingsPayload?.system?.active_model_code) },
      });
      if (data || data?.ok) {
        addTimelineMarker("fall", { force: true });
        return;
      }
    } catch {
      // ignore
    }

    // 2) Fallback: record a test notification
    try {
      await apiRequest(apiBase, "/api/notifications/test", {
        method: "POST",
        body: { resident_id: 1, channel: "email", message: "[UI] Test Fall button pressed" },
      });
    } catch {
      // ignore
    }

    // 3) UI-only marker
    addTimelineMarker("fall", { force: true });
  }, [addTimelineMarker, apiBase, settingsPayload]);

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

  const captureFpsText = useMemo(() => {
    const v = streamFps;
    if (v == null) return "—";
    const n = Number(v);
    if (!Number.isFinite(n) || n <= 0) return "—";
    return n.toFixed(1);
  }, [streamFps]);

  const modelFpsText = useMemo(() => {
    const n = Number(targetFps);
    if (!Number.isFinite(n) || n <= 0) return "—";
    return n.toFixed(1);
  }, [targetFps]);

  const pText = useMemo(() => (pFall == null ? "—" : Number(pFall).toFixed(3)), [pFall]);

  const markers = useMemo(() => {
    const cap = 50;
    const last = timeline.slice(-cap);
    const n = last.length;
    if (n === 0) return [];

    // Fixed 50-slot strip:
    // - newest window always appears at the right edge slot
    // - oldest window falls off the left edge when n > 50
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

  // Optional: expose the server's OP-2 tau for debugging.
  const tauHighFromSpec = chosenSpec?.tau_high != null ? Number(chosenSpec.tau_high) : null;

  return {
    videoRef,
    canvasRef,

    liveRunning,
    streamFps,

    triageState,
    currentPrediction,
    pFall,
    pText,
    sigma,
    safeAlert,
    recallAlert,
    safePrediction,
    recallPrediction,

    markers,
    timelineStatusText,
    captureFpsText,
    modelFpsText,

    resetSession,
    testFall,
    inputSource,
    captureResolutionPreset,
    setCaptureResolution,
    selectedVideoName,
    setVideoFile,
    setInputMode,
    startError,
    predictError,
    replayCurrentS,
    replayDurationS,
    seekReplay,

    tauHighFromSpec,
  };
}
