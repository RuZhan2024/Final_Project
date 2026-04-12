import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import {
  fetchReplayClipBlob,
  resetMonitorSession,
  triggerTestFall,
} from "../../../features/monitor/api";
import {
  resetVideoSource as resetVideoSourceResources,
  syncReplayPlaybackRate as applyReplayPlaybackRate,
} from "../../../features/monitor/media";
import { useMonitorClipCapture } from "./useMonitorClipCapture";
import { useMonitorMediaRuntime } from "./useMonitorMediaRuntime";
import { useMonitorPredictionLoop } from "./useMonitorPredictionLoop";
import { useMonitorPoseProcessing } from "./useMonitorPoseProcessing";
import { useMonitorRuntimeController } from "./useMonitorRuntimeController";
import { useMonitorSessionState } from "./useMonitorSessionState";
import {
  CAPTURE_RESOLUTIONS,
  LIVE_DRAW_FPS,
  LIVE_POSE_MODEL_COMPLEXITY,
} from "../constants";
import { prettyModelTag } from "../utils";
import type { ReplayClip, SpecModel } from "../../../features/monitor/types";
import type { SettingsResponse } from "../../../features/settings/types";
import type { ChosenSpecs, MonitorControllerHandle, MonitorMode } from "../types";

interface UsePoseMonitorOptions {
  apiBase: string;
  isActive: boolean;
  monitoringOn: boolean;
  showLivePreview: boolean;
  registerController?: ((controller: MonitorControllerHandle | null) => void) | null;
  settingsPayload: SettingsResponse | null;
  deployW: number;
  deployS: number;
  targetFps: number;
  mode: MonitorMode | string;
  chosen: ChosenSpecs;
  opCode: string | null;
  mcEnabled: boolean;
  mcCfg: { M: number | null; M_confirm: number | null };
  activeDatasetCode: string;
  chosenSpec: SpecModel | null;
  replayPersistEvents?: boolean;
  onAutoStop?: ((next: boolean) => void | boolean | Promise<void | boolean>) | null;
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
  replayPersistEvents = false,
  onAutoStop,
}: UsePoseMonitorOptions) {
  // Refs for camera + mediapipe
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const isActiveRef = useRef(Boolean(isActive));
  const poseRef = useRef<any>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const replayClipRef = useRef<ReplayClip | null>(null);
  const videoObjectUrlRef = useRef<string | null>(null);
  const inputSourceRef = useRef("camera");
  const showLivePreviewRef = useRef(Boolean(showLivePreview));
  const rafRef = useRef<number | null>(null);
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
  const [streamFps, setStreamFps] = useState<number | null>(null);
  const [inputSource, setInputSource] = useState<"camera" | "video">("camera");
  const [captureResolutionPreset, setCaptureResolutionPreset] = useState<"480p" | "540p" | "720p" | "1080p">("720p");
  const [selectedVideoName, setSelectedVideoName] = useState("");
  const [replayClip, setReplayClipState] = useState<ReplayClip | null>(null);
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
  // Frame buffer for windowing
  const rawFramesRef = useRef<any[]>([]);
  const lastPoseTsRef = useRef<number | null>(null);
  const fpsDeltasRef = useRef<number[]>([]);
  const fpsEstimateRef = useRef<number | null>(null);
  const lastSentRef = useRef(0);
  const predictInFlightRef = useRef(false);
  const replayPredictLatencyMsRef = useRef(0);
  const replayNextWindowEndRef = useRef<number | null>(null);
  const replayWindowQueueRef = useRef<number[]>([]);

  // Session id for server-side state machine
  const sessionIdRef = useRef(`monitor-${Math.random().toString(16).slice(2)}`);
  const windowSeqRef = useRef(0);
  const runTokenRef = useRef(0);
  const predictClientRef = useRef<any>(null);

  const makeSessionId = useCallback(
    () => `monitor-${Date.now().toString(36)}-${Math.random().toString(16).slice(2)}`,
    []
  );

  const syncReplayPlaybackRate = useCallback((videoEl: HTMLVideoElement | null) => {
    applyReplayPlaybackRate(
      videoEl,
      Number(replayPredictLatencyMsRef.current) || 0,
      inputSourceRef.current || "camera"
    );
  }, []);

  const {
    clipFlags,
    pendingClipRef,
    uploadedClipIdsRef,
    clearPendingClipTimer,
    resetClipCaptureState,
    maybeFinalizeClipUpload,
    queueClipForEvent,
  } = useMonitorClipCapture({
    apiBase,
    settingsPayload,
    rawFramesRef,
    activeDatasetCode,
    mode,
    opCode,
    mcEnabled,
    mcCfg,
  });

  const {
    triageState,
    pFall,
    sigma,
    safeAlert,
    recallAlert,
    currentPrediction,
    safePrediction,
    recallPrediction,
    pText,
    markers,
    timelineStatusText,
    addTimelineMarker,
    applyPredictionResponse,
    resetSessionUiState,
  } = useMonitorSessionState({
    mode,
    settingsPayload,
    clipFlags,
    pendingClipRef,
    uploadedClipIdsRef,
    maybeFinalizeClipUpload,
    queueClipForEvent,
  });

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
    resetClipCaptureState();

    // Reset UI
    windowSeqRef.current = 0;
    resetSessionUiState();
    setReplayCurrentS(0);
    setReplayDurationS(0);
    lastReplayUiMsRef.current = 0;
    setStartError("");
    setStartInfo("");
    setPredictError("");
  }, [resetClipCaptureState, resetPredictionTransport, resetSessionUiState, resetVideoSource]);

  const resetFrontendSessionState = useCallback(() => {
    rawFramesRef.current = [];
    lastPoseTsRef.current = null;
    fpsDeltasRef.current = [];
    lastSentRef.current = 0;
    replayPredictLatencyMsRef.current = 0;
    replayNextWindowEndRef.current = null;
    replayWindowQueueRef.current = [];
    resetClipCaptureState();
    windowSeqRef.current = 0;
    resetSessionUiState();
  }, [resetClipCaptureState, resetSessionUiState]);

  const setReplayClip = useCallback((clip: ReplayClip | null) => {
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

  const setInputMode = useCallback((mode: "camera" | "video" | string) => {
    const m: "camera" | "video" = mode === "video" ? "video" : "camera";
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

  const setCaptureResolution = useCallback((preset: string) => {
    if (!Object.prototype.hasOwnProperty.call(CAPTURE_RESOLUTIONS, preset)) return;
    setCaptureResolutionPreset(String(preset));
  }, []);

  const {
    buildReplayPayload,
    drainReplayQueue,
    maybeSendWindow,
  } = useMonitorPredictionLoop({
    apiBase,
    activeDatasetCode,
    chosen,
    deployS,
    deployW,
    targetFps,
    streamFps: streamFps || fpsEstimateRef.current || null,
    mode,
    opCode,
    settingsPayload,
    mcEnabled,
    mcCfg,
    replayPersistEvents,
    selectedVideoName,
    replayClipRef,
    inputSourceRef,
    monitoringOnRef,
    rawFramesRef,
    sessionIdRef,
    windowSeqRef,
    videoRef,
    predictInFlightRef,
    replayPredictLatencyMsRef,
    replayNextWindowEndRef,
    replayWindowQueueRef,
    lastSentRef,
    predictClientRef,
    applyPredictionResponse,
    syncReplayPlaybackRate,
    maybeFinalizeClipUpload,
    clipFlags,
  });

  const { handlePoseResults } = useMonitorPoseProcessing({
    targetFps,
    canvasRef,
    videoRef,
    poseRef,
    inputSourceRef,
    isActiveRef,
    showLivePreviewRef,
    lastProcMsRef,
    lastDrawMsRef,
    adaptiveDrawFpsRef,
    adaptiveInferFpsRef,
    degradedModeRef,
    lowFpsSinceMsRef,
    rawFramesRef,
    lastPoseTsRef,
    fpsDeltasRef,
    fpsEstimateRef,
    ensureCanvasMatchesVideo,
    maybeFinalizeClipUpload,
    maybeSendWindow,
  });

  const {
    initPosePipeline,
    prepareReplayVideo,
    prepareCameraStream,
    syncReplayTimeState,
    getActiveReplaySource,
  } = useMonitorMediaRuntime({
    targetFps,
    captureResolutionPreset,
    replayClipRef,
    inputSourceRef,
    videoRef,
    poseRef,
    streamRef,
    videoObjectUrlRef,
    setStartInfo,
    setReplayCurrentS,
    setReplayDurationS,
    handlePoseResults,
    resetVideoSource,
    fetchReplayClipBlob,
  });

  useEffect(() => {
    if (!poseRef.current?.onResults) return;
    // MediaPipe keeps the original callback until it is explicitly rebound.
    // Without this, Settings changes (for example store_event_clips) leave the
    // live monitor running against stale closures from the first mount.
    poseRef.current.onResults(handlePoseResults);
  }, [handlePoseResults]);

  useEffect(() => {
    return () => {
      clearPendingClipTimer();
    };
  }, [clearPendingClipTimer]);

  const resetSession = useCallback(async () => {
    try {
      await resetMonitorSession(apiBase, sessionIdRef.current);
    } catch {
      // ignore
    }
  }, [apiBase]);

  const { startLive, seekReplay } = useMonitorRuntimeController({
    targetFps,
    apiBase,
    autoStopMonitoring,
    buildReplayPayload,
    drainReplayQueue,
    ensureCanvasMatchesVideo,
    getActiveReplaySource,
    initPosePipeline,
    makeSessionId,
    prepareCameraStream,
    prepareReplayVideo,
    resetFrontendSessionState,
    resetSession,
    stopLive,
    syncReplayTimeState,
    inputSourceRef,
    videoRef,
    streamRef,
    liveFlagRef,
    runTokenRef,
    rafRef,
    poseRef,
    poseSendBusyRef,
    predictInFlightRef,
    replayWindowQueueRef,
    adaptiveInferFpsRef,
    inferFpsRef,
    lastInferTsRef,
    lastReplayUiMsRef,
    replayPredictLatencyMsRef,
    sessionIdRef,
    fpsEstimateRef,
    setLiveRunning,
    setStartError,
    setStartInfo,
    setReplayCurrentS,
    setReplayDurationS,
  });

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

  const testFall = useCallback(async () => {
    try {
      const data = await triggerTestFall(
        apiBase,
        prettyModelTag(settingsPayload?.system?.active_model_code)
      );
      if (data?.ok) {
        addTimelineMarker("fall", { force: true });
        setPredictError("");
        return true;
      }
      setPredictError("Test fall request was accepted locally but backend did not confirm success.");
      return false;
    } catch (err: any) {
      setPredictError(String(err?.message || err || "Test fall request failed"));
      return false;
    }
  }, [addTimelineMarker, apiBase, settingsPayload, setPredictError]);

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
