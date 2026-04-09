import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import * as mpPose from "@mediapipe/pose";
import * as drawingUtils from "@mediapipe/drawing_utils";

import { apiRequest } from "../../../lib/apiClient";
import { readBool } from "../../../lib/booleans";
import {
  fetchReplayClipBlob,
  resetMonitorSession,
  triggerTestFall,
  uploadSkeletonClip,
} from "../../../features/monitor/api";
import {
  prepareCameraStream as prepareCameraStreamSource,
  prepareReplayVideo as prepareReplayVideoSource,
  resetVideoSource as resetVideoSourceResources,
  syncReplayPlaybackRate as applyReplayPlaybackRate,
} from "../../../features/monitor/media";
import {
  buildPendingClip,
  buildPredictPayload,
  extractPredictionState,
} from "../../../features/monitor/prediction";
import { createMonitorSocketClient } from "../../../features/monitor/socketClient";
import {
  queueReplayWindowEnds,
  sliceLiveWindowFrames,
  sliceReplayWindowFrames,
} from "../../../features/monitor/windowing";
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
  NUM_JOINTS,
  REPLAY_UI_UPDATE_MS,
} from "../constants";
import { clamp01, labelForTriage, prettyModelTag } from "../utils";

const { drawConnectors, drawLandmarks } = drawingUtils;
const FALL_HISTORY_DEDUP_MS_DEFAULT = 30_000;
const WS_CONNECT_TIMEOUT_MS = 7000;
const WS_PREDICT_TIMEOUT_MS = 12000;
const REPLAY_POSE_MODEL_COMPLEXITY = 1;
const REPLAY_MIN_DETECTION_CONFIDENCE = 0.35;
const REPLAY_MIN_TRACKING_CONFIDENCE = 0.35;
const CAPTURE_RESOLUTIONS = {
  "480p": { w: 640, h: 480 },
  "540p": { w: 960, h: 540 },
  "720p": { w: 1280, h: 720 },
  "1080p": { w: 1920, h: 1080 },
};

function getClipFlags(settingsPayload) {
  const sys = settingsPayload?.system || {};
  return {
    storeEventClips: readBool(sys?.store_event_clips, false),
    anonymize: readBool(sys?.anonymize_skeleton_data, true),
  };
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
  showLivePreview,
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
  const replayClipRef = useRef(null);
  const videoObjectUrlRef = useRef(null);
  const inputSourceRef = useRef("camera");
  const showLivePreviewRef = useRef(Boolean(showLivePreview));
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
  const [replayClip, setReplayClipState] = useState(null);
  const [startError, setStartError] = useState("");
  const [startInfo, setStartInfo] = useState("");
  const [predictError, setPredictError] = useState("");
  const [replayCurrentS, setReplayCurrentS] = useState(0);
  const [replayDurationS, setReplayDurationS] = useState(0);
  // MediaPipe result callbacks can outlive the render that started monitoring.
  // Read the latest persisted-monitoring flag from a ref so replay windows do not
  // keep sending stale `persist: false` after monitoring is toggled on.
  const monitoringOnRef = useRef(Boolean(monitoringOn));

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
  const replayPredictLatencyMsRef = useRef(0);
  const replayNextWindowEndRef = useRef(null);
  const replayWindowQueueRef = useRef([]);

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
  const runTokenRef = useRef(0);
  const predictClientRef = useRef(null);

  const makeSessionId = useCallback(
    () => `monitor-${Date.now().toString(36)}-${Math.random().toString(16).slice(2)}`,
    []
  );

  const syncReplayPlaybackRate = useCallback((videoEl) => {
    applyReplayPlaybackRate(
      videoEl,
      Number(replayPredictLatencyMsRef.current) || 0,
      inputSourceRef.current || "camera"
    );
  }, []);

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

  useEffect(() => {
    monitoringOnRef.current = Boolean(monitoringOn);
  }, [monitoringOn]);

  useEffect(() => {
    showLivePreviewRef.current = Boolean(showLivePreview);
  }, [showLivePreview]);

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

  const resetVideoSource = useCallback(() => {
    resetVideoSourceResources({
      streamRef,
      videoRef,
      videoObjectUrlRef,
    });
  }, []);

  const resetPredictionTransport = useCallback(() => {
    predictClientRef.current?.close?.("WebSocket closed during mode switch");
    predictClientRef.current = null;
  }, []);

  const stopLive = useCallback(() => {
    runTokenRef.current += 1;
    liveFlagRef.current = false;
    setLiveRunning(false);

    if (rafRef.current) cancelAnimationFrame(rafRef.current);
    rafRef.current = null;
    poseSendBusyRef.current = false;

    resetVideoSource();
    resetPredictionTransport();

    // Clear buffer
    rawFramesRef.current = [];
    lastPoseTsRef.current = null;
    fpsDeltasRef.current = [];
    lastSentRef.current = 0;
    replayPredictLatencyMsRef.current = 0;
    replayNextWindowEndRef.current = null;
    replayWindowQueueRef.current = [];

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
    setStartInfo("");
    setPredictError("");
  }, [resetPredictionTransport, resetVideoSource]);

  const resetFrontendSessionState = useCallback(() => {
    rawFramesRef.current = [];
    lastPoseTsRef.current = null;
    fpsDeltasRef.current = [];
    lastSentRef.current = 0;
    replayPredictLatencyMsRef.current = 0;
    replayNextWindowEndRef.current = null;
    replayWindowQueueRef.current = [];
    pendingClipRef.current = null;
    uploadedClipIdsRef.current = new Set();
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

  const setReplayClip = useCallback((clip) => {
    stopLive();
    autoStopMonitoring();
    sessionIdRef.current = makeSessionId();
    resetFrontendSessionState();
    if (clip && typeof clip === "object") {
      replayClipRef.current = clip;
      setReplayClipState(clip);
      setSelectedVideoName(String(clip.name || clip.filename || clip.id || "video"));
      inputSourceRef.current = "video";
      setInputSource("video");
      setStartError("");
    } else {
      replayClipRef.current = null;
      setReplayClipState(null);
      setSelectedVideoName("");
      inputSourceRef.current = "video";
      setInputSource("video");
      setStartError("");
    }
  }, [autoStopMonitoring, makeSessionId, resetFrontendSessionState, stopLive]);

  const setInputMode = useCallback((mode) => {
    const m = mode === "video" ? "video" : "camera";
    stopLive();
    sessionIdRef.current = makeSessionId();
    resetFrontendSessionState();
    inputSourceRef.current = m;
    setInputSource(m);
    if (m === "camera") {
      replayClipRef.current = null;
      setReplayClipState(null);
      setSelectedVideoName("");
    }
    setStartError("");
  }, [makeSessionId, resetFrontendSessionState, stopLive]);

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

      const data = await uploadSkeletonClip(apiBase, pending.eventId, clipPayload);

      if (data?.ok) {
        sent.add(String(pending.eventId));
      }
    } catch (err) {
      console.error("Failed to upload skeleton clip", err);
    } finally {
      pendingClipRef.current = null;
    }
  }, [apiBase, settingsPayload]);

  const ensurePredictClient = useCallback(() => {
    if (predictClientRef.current) return predictClientRef.current;
    predictClientRef.current = createMonitorSocketClient({
      apiBase,
      connectTimeoutMs: WS_CONNECT_TIMEOUT_MS,
      predictTimeoutMs: WS_PREDICT_TIMEOUT_MS,
    });
    return predictClientRef.current;
  }, [apiBase]);

  const predictViaWs = useCallback(
    async (payload) => {
      return await ensurePredictClient().predict(payload);
    },
    [ensurePredictClient]
  );

  const predictViaHttp = useCallback(
    async (payload) =>
      await apiRequest(apiBase, "/api/monitor/predict_window", {
        method: "POST",
        body: payload,
      }),
    [apiBase]
  );

  const applyPredictionResponse = useCallback((data) => {
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
      const { storeEventClips } = getClipFlags(settingsPayload);
      const evId = data?.event_id;
      if (storeEventClips && evId != null && Number.isFinite(endTs) && endTs > 0) {
        const sent = uploadedClipIdsRef.current;
        const already = sent && sent.has(String(evId));
        const pending = pendingClipRef.current;
        if (!already && (!pending || String(pending.eventId) !== String(evId))) {
          pendingClipRef.current = buildPendingClip({
            eventId: evId,
            endTs,
            activeDatasetCode,
            mode,
            opCode,
            settingsPayload,
            mcEnabled,
            mcCfg,
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
    activeDatasetCode,
    addTimelineMarker,
    mcCfg,
    mcEnabled,
    mode,
    opCode,
    settingsPayload,
    maybeFinalizeClipUpload,
  ]);

  const buildPredictionPayload = useCallback(({ sourceMode, frames, windowEndTs, location }) => {
    const nFrames = Array.isArray(frames) ? frames.length : 0;
    if (nFrames < 2) return null;

    const { storeEventClips } = getClipFlags(settingsPayload);
    const payload = buildPredictPayload({
      slice: frames,
      sessionId: sessionIdRef.current,
      windowSeq: ++windowSeqRef.current,
      inputSource: sourceMode,
      location,
      mode,
      datasetCode: activeDatasetCode,
      opCode: opCode || settingsPayload?.system?.active_op_code || "OP-2",
      chosen,
      targetFps,
      deployW,
      streamFps: streamFps || fpsEstimateRef.current || null,
      mcEnabled,
      mcCfg,
      persist: monitoringOnRef.current || storeEventClips,
      endTs: windowEndTs,
    });

    return {
      ...payload,
      compact_response: sourceMode === "video",
    };
  }, [
    activeDatasetCode,
    chosen,
    deployW,
    mcCfg,
    mcEnabled,
    mode,
    opCode,
    settingsPayload,
    streamFps,
    targetFps,
  ]);

  const queueReplayWindows = useCallback(() => {
    return queueReplayWindowEnds({
      rawFrames: rawFramesRef.current,
      targetFps,
      deployW,
      deployS,
      nextWindowEndRef: replayNextWindowEndRef,
      queueRef: replayWindowQueueRef,
    });
  }, [deployS, deployW, targetFps]);

  const buildReplayPayload = useCallback(() => {
    const queue = replayWindowQueueRef.current;
    const raw = rawFramesRef.current;
    if (!queue.length || !raw || raw.length < 2) return null;

    const windowEndTs = Number(queue[0]);
    const frames = sliceReplayWindowFrames({
      rawFrames: raw,
      targetFps,
      deployW,
      windowEndTs,
    });
    if (!frames) return null;

    return buildPredictionPayload({
      sourceMode: "video",
      frames,
      windowEndTs,
      location: String(selectedVideoName || replayClipRef.current?.name || "replay_video"),
    });
  }, [buildPredictionPayload, deployW, selectedVideoName, targetFps]);

  const buildLivePayload = useCallback(() => {
    const liveSlice = sliceLiveWindowFrames({
      rawFrames: rawFramesRef.current,
      targetFps,
      deployW,
    });
    if (!liveSlice) return null;

    return buildPredictionPayload({
      sourceMode: "camera",
      frames: liveSlice.frames,
      windowEndTs: liveSlice.endTs,
      location: "camera_live",
    });
  }, [buildPredictionPayload, deployW, targetFps]);

  const runPredictionRequest = useCallback(async (payload, sourceMode) => {
    const videoEl = videoRef.current;
    try {
      predictInFlightRef.current = true;
      const t0 = performance.now();
      const data =
        sourceMode === "video" ? await predictViaHttp(payload) : await predictViaWs(payload);
      replayPredictLatencyMsRef.current = Math.max(0, performance.now() - t0);
      setPredictError("");
      applyPredictionResponse(data);
      return true;
    } catch (err) {
      if (String(err?.name || "") === "AbortError") {
        setPredictError("");
        return false;
      }
      console.error("Error calling /api/monitor/predict_window", err);
      setPredictError(String(err?.message || err || "predict_window failed"));
      return false;
    } finally {
      predictInFlightRef.current = false;
      if (sourceMode === "video") {
        syncReplayPlaybackRate(videoEl);
      }
    }
  }, [applyPredictionResponse, predictViaHttp, predictViaWs, syncReplayPlaybackRate]);

  const drainReplayQueue = useCallback(async () => {
    if ((inputSourceRef.current || "camera") !== "video") return;
    void queueReplayWindows();
    if (predictInFlightRef.current) return;
    if (!replayWindowQueueRef.current.length) return;

    while (!predictInFlightRef.current && replayWindowQueueRef.current.length) {
      const payload = buildReplayPayload();
      if (!payload) return;
      const nextWindowEndTs = Number(payload.window_end_t_ms);
      const ok = await runPredictionRequest(payload, "video");
      const front = Number(replayWindowQueueRef.current[0]);
      if (Number.isFinite(front) && front === nextWindowEndTs) {
        replayWindowQueueRef.current.shift();
      }
      if (!ok) return;
    }
  }, [buildReplayPayload, queueReplayWindows, runPredictionRequest]);

  const maybeSendWindow = useCallback(async () => {
    const sourceMode = inputSourceRef.current || "camera";
    if (sourceMode === "video") {
      await drainReplayQueue();
      return;
    }

    if (predictInFlightRef.current) return;
    const now = performance.now();
    const minGapMs = Math.max(250, (deployS / Math.max(1, targetFps)) * 1000);
    const gapRemainMs = minGapMs - (now - lastSentRef.current);
    if (gapRemainMs > 0) return;

    const payload = buildLivePayload();
    if (!payload) return;
    lastSentRef.current = now;

    await runPredictionRequest(payload, sourceMode);
  }, [
    buildLivePayload,
    deployS,
    drainReplayQueue,
    runPredictionRequest,
    targetFps,
  ]);

  const handlePoseResults = useCallback(
    (results) => {
      const isReplay = inputSourceRef.current === "video";
      // Cap processing rate.
      const nowMs = performance.now();
      const procCap = Math.min(MAX_PROC_FPS, Math.max(8, Number(targetFps) + 2));
      if (nowMs - lastProcMsRef.current < 1000 / procCap) return;
      lastProcMsRef.current = nowMs;

      // Adaptive draw FPS: if capture FPS stays low, reduce draw load temporarily.
      const estFps = Number(fpsEstimateRef.current);
      if (!isReplay && Number.isFinite(estFps) && estFps > 0) {
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
      const showTransparentPreview =
        (inputSourceRef.current || "camera") === "camera" && showLivePreviewRef.current;

      const hasLandmarks = Boolean(landmarks && landmarks.length);

      // No pose detected.
      if (!hasLandmarks) {
        if (doDraw && ctx) {
          const drawNowMs = performance.now();
          if (drawNowMs - lastDrawMsRef.current < 1000 / Math.max(1, adaptiveDrawFpsRef.current)) {
            // Keep processing so replay/live still advance the raw window buffer.
          } else {
            lastDrawMsRef.current = drawNowMs;
            ensureCanvasMatchesVideo();
            ctx.clearRect(0, 0, canvasEl.width, canvasEl.height);
            if (!showTransparentPreview) {
              ctx.fillStyle = "#020617";
              ctx.fillRect(0, 0, canvasEl.width, canvasEl.height);
            }
            ctx.fillStyle = showTransparentPreview ? "rgba(2, 6, 23, 0.68)" : "#020617";
            ctx.fillRect(12, 8, 156, 28);
            ctx.font = "14px system-ui, -apple-system, Segoe UI, Roboto";
            ctx.fillStyle = "#94a3b8";
            ctx.fillText("No pose detected…", 20, 27);
          }
        }
      }

      // Draw only when this page is active.
      if (hasLandmarks && doDraw && ctx) {
        const drawNowMs = performance.now();
        if (drawNowMs - lastDrawMsRef.current < 1000 / Math.max(1, adaptiveDrawFpsRef.current)) {
          // Skip frequent repaint to reduce main-thread pressure.
        } else {
          lastDrawMsRef.current = drawNowMs;
        ensureCanvasMatchesVideo();
        const w = canvasEl.width;
        const h = canvasEl.height;

        ctx.clearRect(0, 0, w, h);
        if (!showTransparentPreview) {
          ctx.fillStyle = "#020617";
          ctx.fillRect(0, 0, w, h);
        }

        drawConnectors(ctx, landmarks, mpPose.POSE_CONNECTIONS, {
          color: "#22c55e",
          lineWidth: 2,
        });
        drawLandmarks(ctx, landmarks, { color: "#fbbf24", lineWidth: 1 });

        ctx.font = "12px system-ui, -apple-system, Segoe UI, Roboto";
        ctx.fillStyle = "#94a3b8";
        ctx.fillStyle = showTransparentPreview ? "rgba(2, 6, 23, 0.68)" : "#020617";
        ctx.fillRect(12, h - 34, 120, 22);
        ctx.fillStyle = "#94a3b8";
        ctx.fillText(new Date().toLocaleTimeString(), 16, h - 16);
        }
      }

      // Build raw frame
      const xyFrame = new Array(NUM_JOINTS);
      const confFrame = new Array(NUM_JOINTS);

      for (let i = 0; i < NUM_JOINTS; i++) {
        const lm = hasLandmarks ? landmarks[i] : null;
        if (!lm) {
          xyFrame[i] = [0, 0];
          confFrame[i] = 0;
        } else {
          xyFrame[i] = [
            Number.isFinite(Number(lm.x)) ? Number(lm.x) : 0,
            Number.isFinite(Number(lm.y)) ? Number(lm.y) : 0,
          ];
          confFrame[i] = typeof lm.visibility === "number" ? clamp01(lm.visibility) : 1.0;
        }
      }

      const replayVideoTsMs =
        isReplay && videoRef.current ? Number(videoRef.current.currentTime || 0) * 1000 : null;
      let tNow =
        isReplay && Number.isFinite(replayVideoTsMs) ? Number(replayVideoTsMs) : performance.now();
      if (isReplay && typeof lastPoseTsRef.current === "number" && tNow <= lastPoseTsRef.current) {
        tNow = lastPoseTsRef.current + 1;
      }

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
      const maxRaw =
        isReplay ? Math.max(2400, Math.ceil(targetFps * 120)) : Math.max(600, Math.ceil(targetFps * 12));
      if (raw.length > maxRaw) raw.splice(0, raw.length - maxRaw);

      // Clip upload + inference
      void maybeFinalizeClipUpload();
      void maybeSendWindow();
    },
    [ensureCanvasMatchesVideo, maybeFinalizeClipUpload, maybeSendWindow, targetFps]
  );

  const initPosePipeline = useCallback(async (videoEl, { warmup = false } = {}) => {
    const isReplay = inputSourceRef.current === "video";
    if (!poseRef.current) {
      const pose = new mpPose.Pose({
        locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/pose/${file}`,
      });

      pose.setOptions({
        modelComplexity: isReplay ? REPLAY_POSE_MODEL_COMPLEXITY : LIVE_POSE_MODEL_COMPLEXITY,
        smoothLandmarks: true,
        minDetectionConfidence: isReplay ? REPLAY_MIN_DETECTION_CONFIDENCE : 0.5,
        minTrackingConfidence: isReplay ? REPLAY_MIN_TRACKING_CONFIDENCE : 0.5,
      });

      pose.onResults(handlePoseResults);
      poseRef.current = pose;
    } else if (poseRef.current?.setOptions) {
      try {
        poseRef.current.setOptions({
          modelComplexity: isReplay ? REPLAY_POSE_MODEL_COMPLEXITY : LIVE_POSE_MODEL_COMPLEXITY,
          smoothLandmarks: true,
          minDetectionConfidence: isReplay ? REPLAY_MIN_DETECTION_CONFIDENCE : 0.5,
          minTrackingConfidence: isReplay ? REPLAY_MIN_TRACKING_CONFIDENCE : 0.5,
        });
      } catch {
        // ignore
      }
    }

    if (warmup && videoEl?.readyState >= 2 && poseRef.current) {
      await poseRef.current.send({ image: videoEl });
    }
  }, [handlePoseResults]);

  const prepareReplayVideo = useCallback(async (videoEl, clip, clipUrl) => {
    await prepareReplayVideoSource({
      videoEl,
      clip,
      clipUrl,
      videoObjectUrlRef,
      setStartInfo,
      fetchReplayClipBlob,
      resetVideoSource,
    });
  }, [resetVideoSource]);

  const prepareCameraStream = useCallback(async (videoEl) => {
    const captureResolution =
      CAPTURE_RESOLUTIONS[captureResolutionPreset] || CAPTURE_RESOLUTIONS["720p"];
    await prepareCameraStreamSource({
      videoEl,
      captureResolution,
      targetFps,
      streamRef,
    });
  }, [captureResolutionPreset, targetFps]);

  const resetSession = useCallback(async () => {
    try {
      await resetMonitorSession(apiBase, sessionIdRef.current);
    } catch {
      // ignore
    }
  }, [apiBase]);

  const startLive = useCallback(async () => {
    const videoEl = videoRef.current;
    if (!videoEl) return false;

    const runToken = runTokenRef.current + 1;
    runTokenRef.current = runToken;

    try {
      sessionIdRef.current = makeSessionId();
      resetFrontendSessionState();
      await resetSession();
      setStartError("");
      setStartInfo("");
      liveFlagRef.current = true;
      setLiveRunning(true);

      const source = inputSourceRef.current || "camera";
      const useVideoFile = source === "video";
      const clip = replayClipRef.current || replayClip;
      const clipFile = clip?.file instanceof File ? clip.file : null;
      const clipUrl = typeof clip?.url === "string" && clip.url ? clip.url : "";
      if (useVideoFile && !clipFile && !clipUrl) {
        throw new Error("Replay mode selected but no replay clip is loaded.");
      }

      if (useVideoFile) {
        await prepareReplayVideo(videoEl, clip, clipUrl);
        if (runTokenRef.current !== runToken) return false;
        setStartInfo("Preparing replay detector...");
      } else {
        await prepareCameraStream(videoEl);
        if (runTokenRef.current !== runToken) return false;
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

      await initPosePipeline(videoEl, { warmup: true });
      if (runTokenRef.current !== runToken) return false;
      if (useVideoFile) {
        setStartInfo("");
      }

      try {
        await videoEl.play();
      } catch (e) {
        throw new Error(`Video play failed: ${String(e?.message || e)}`);
      }

      const loop = async () => {
        if (!liveFlagRef.current || runTokenRef.current !== runToken) return;
        if ((inputSourceRef.current || "camera") === "video" && videoEl.ended) {
          void buildReplayPayload();
          void drainReplayQueue();
          const replayBusy =
            predictInFlightRef.current || replayWindowQueueRef.current.length > 0;
          if (replayBusy) {
            setStartInfo("Finalizing replay predictions...");
            rafRef.current = requestAnimationFrame(loop);
            return;
          }
          setStartInfo("");
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

          if (
            videoEl.readyState >= 2 &&
            videoEl.videoWidth > 0 &&
            videoEl.videoHeight > 0 &&
            poseRef.current &&
            !poseSendBusyRef.current
          ) {
            poseSendBusyRef.current = true;
            poseRef.current
              .send({ image: videoEl })
              .catch((err) => {
                const msg = String(err?.message || err || "");
                if (!/no video|ROI width and height must be > 0|Aborted\(native code called abort\)/i.test(msg)) {
                  console.error("pose.send error", err);
                }
              })
              .finally(() => {
                if (runTokenRef.current === runToken) {
                  poseSendBusyRef.current = false;
                }
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
  }, [autoStopMonitoring, buildReplayPayload, drainReplayQueue, ensureCanvasMatchesVideo, initPosePipeline, makeSessionId, prepareCameraStream, prepareReplayVideo, replayClip, resetFrontendSessionState, resetSession, stopLive]);

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
    return () => {
      stopLive();
      try {
        poseRef.current?.close?.();
      } catch {
        // ignore
      }
      poseRef.current = null;
    };
  }, [stopLive]);

  const seekReplay = useCallback((ratio) => {
    const videoEl = videoRef.current;
    if (!videoEl) return;
    const du = Number(videoEl.duration || 0);
    if (!Number.isFinite(du) || du <= 0) return;
    const r = Math.max(0, Math.min(1, Number(ratio) || 0));
    sessionIdRef.current = makeSessionId();
    resetFrontendSessionState();
    void resetSession();
    videoEl.currentTime = r * du;
    setReplayCurrentS(Number(videoEl.currentTime || 0));
    setReplayDurationS(du);
  }, [makeSessionId, resetFrontendSessionState, resetSession]);

  const testFall = useCallback(async () => {
    try {
      const data = await triggerTestFall(
        apiBase,
        prettyModelTag(settingsPayload?.system?.active_model_code)
      );
      if (data || data?.ok) {
        addTimelineMarker("fall", { force: true });
        return;
      }
    } catch {
      // ignore
    }
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
    replayClip,
    setReplayClip,
    setInputMode,
    startError,
    startInfo,
    predictError,
    replayCurrentS,
    replayDurationS,
    seekReplay,

    tauHighFromSpec,
  };
}
