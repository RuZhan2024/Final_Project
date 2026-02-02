import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import * as mpPose from "@mediapipe/pose";
import * as drawingUtils from "@mediapipe/drawing_utils";

import { apiRequest } from "../../../lib/apiClient";
import { CLIP_POST_S, CLIP_PRE_S, MAX_PROC_FPS, NUM_JOINTS } from "../constants";
import { clamp01, labelForTriage, prettyModelTag } from "../utils";

const { drawConnectors, drawLandmarks } = drawingUtils;

function safeNumber(x) {
  const n = Number(x);
  return Number.isFinite(n) ? n : null;
}

function getClipFlags(settingsPayload) {
  const sys = settingsPayload?.system || {};
  return {
    storeEventClips: Boolean(sys?.store_event_clips),
    anonymize: Boolean(sys?.anonymize_skeleton_data ?? true),
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
}) {
  // Refs for camera + mediapipe
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const isActiveRef = useRef(Boolean(isActive));
  const poseRef = useRef(null);
  const streamRef = useRef(null);
  const rafRef = useRef(null);
  const liveFlagRef = useRef(false);

  // Throttle MediaPipe inference so navigation stays responsive when /Monitor isn't visible.
  const lastInferTsRef = useRef(0);
  const inferFpsRef = useRef(15);

  // Cap pose processing rate to reduce CPU.
  const lastProcMsRef = useRef(0);

  // Live state
  const [liveRunning, setLiveRunning] = useState(false);
  const [streamFps, setStreamFps] = useState(null);

  // Prediction UI
  const [triageState, setTriageState] = useState("not_fall");
  const [pFall, setPFall] = useState(null);
  const [sigma, setSigma] = useState(null);

  // Timeline markers: store last 50 non-safe windows
  const [timeline, setTimeline] = useState([]); // [{kind:'fall'|'uncertain', t: number}]

  // Frame buffer for windowing
  const rawFramesRef = useRef([]); // [{t,xy,conf}]
  const lastPoseTsRef = useRef(null);
  const fpsDeltasRef = useRef([]);
  const fpsEstimateRef = useRef(null);
  const lastSentRef = useRef(0);

  // Skeleton clip saving
  const pendingClipRef = useRef(null);
  const uploadedClipIdsRef = useRef(new Set());

  // Session id for server-side state machine
  const sessionIdRef = useRef(`monitor-${Math.random().toString(16).slice(2)}`);

  // Keep isActive in a ref (MediaPipe callback shouldn't rebind each render)
  useEffect(() => {
    isActiveRef.current = Boolean(isActive);
  }, [isActive]);

  // Update inference throttle and MediaPipe complexity based on whether this page is active.
  useEffect(() => {
    const fgFps = Math.min(12, Math.max(5, Number(targetFps) || 12));
    const bgFps = Math.min(5, fgFps);
    inferFpsRef.current = isActiveRef.current ? fgFps : bgFps;

    try {
      if (poseRef.current && poseRef.current.setOptions) {
        poseRef.current.setOptions({ modelComplexity: isActiveRef.current ? 1 : 0 });
      }
    } catch {
      // ignore
    }
  }, [targetFps, isActive]);

  // Update stream FPS display at a low rate (avoid re-render on every frame)
  useEffect(() => {
    if (!liveRunning) {
      setStreamFps(null);
      return;
    }

    const id = setInterval(() => {
      const v = fpsEstimateRef.current;
      setStreamFps(v == null ? null : v);
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
    setTimeline([]);
  }, []);

  const addTimelineMarker = useCallback((kind) => {
    setTimeline((prev) => {
      const next = prev.slice(-49);
      next.push({ kind, t: Date.now() });
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
      const clipPayload = {
        resident_id: 1,
        dataset_code: pending.ctx?.dataset_code,
        mode: pending.ctx?.mode,
        op_code: pending.ctx?.op_code,
        use_mc: pending.ctx?.use_mc,
        mc_M: pending.ctx?.mc_M,
        pre_s: pending.preMs / 1000,
        post_s: pending.postMs / 1000,
        t_ms: frames.map((fr) => fr.t),
        xy: frames.map((fr) => fr.xy),
        conf: frames.map((fr) => fr.conf),
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

  const maybeSendWindow = useCallback(async () => {
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

    lastSentRef.current = now;

    const { storeEventClips } = getClipFlags(settingsPayload);

    const payload = {
      session_id: sessionIdRef.current,
      resident_id: 1,
      mode,
      dataset_code: activeDatasetCode,
      op_code: opCode || settingsPayload?.system?.active_op_code || "OP-2",

      model_tcn: mode === "dual" ? chosen.tcn : mode === "tcn" ? chosen.tcn : null,
      model_gcn: mode === "dual" ? chosen.gcn : mode === "gcn" ? chosen.gcn : null,
      model_id: mode === "dual" ? null : mode === "tcn" ? chosen.tcn : chosen.gcn,

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
      raw_t_ms: slice.map((fr) => fr.t),
      raw_xy: slice.map((fr) => fr.xy),
      raw_conf: slice.map((fr) => fr.conf),

      window_end_t_ms: endTs,
    };

    try {
      const data = await apiRequest(apiBase, "/api/monitor/predict_window", { method: "POST", body: payload });

      const tri = data?.triage_state || data?.triageState || "not_fall";
      setTriageState(tri);

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

      // pFall display: use overall mu (single) or conservative fusion for dual
      let mu = null;
      let sig = null;

      if (mode === "dual") {
        const mt = data?.models?.tcn;
        const mg = data?.models?.gcn;
        const mh = data?.models?.hybrid;
        const muT = safeNumber(mt?.mu);
        const muG = safeNumber(mg?.mu);

        if (mh?.mu != null) mu = Number(mh.mu);
        else mu = Math.min(muT ?? 1, muG ?? 1);

        const sT = safeNumber(mt?.sigma);
        const sG = safeNumber(mg?.sigma);
        if (mh?.sigma != null) sig = Number(mh.sigma);
        else sig = Math.max(sT ?? 0, sG ?? 0);
      } else {
        const mOut = data?.models?.[mode];
        if (mOut) {
          mu = mOut?.mu != null ? Number(mOut.mu) : mOut?.p_det != null ? Number(mOut.p_det) : null;
          sig = mOut?.sigma != null ? Number(mOut.sigma) : null;
        }
      }

      setPFall(mu);
      setSigma(sig);

      // timeline markers
      const triNorm = String(tri || "").toLowerCase();
      if (triNorm === "fall") addTimelineMarker("fall");
      else if (triNorm === "uncertain") addTimelineMarker("uncertain");
    } catch (err) {
      console.error("Error calling /api/monitor/predict_window", err);
    }
  }, [
    activeDatasetCode,
    addTimelineMarker,
    apiBase,
    chosen,
    deployS,
    deployW,
    mcCfg,
    mcEnabled,
    mode,
    monitoringOn,
    opCode,
    settingsPayload,
    streamFps,
    targetFps,
  ]);

  const handlePoseResults = useCallback(
    (results) => {
      // Cap processing rate.
      const nowMs = performance.now();
      if (nowMs - lastProcMsRef.current < 1000 / MAX_PROC_FPS) return;
      lastProcMsRef.current = nowMs;

      const canvasEl = canvasRef.current;
      if (!canvasEl) return;
      const ctx = canvasEl.getContext("2d");

      const doDraw = isActiveRef.current;
      const landmarks = results.poseLandmarks;

      // No pose detected.
      if (!landmarks || !landmarks.length) {
        if (doDraw && ctx) {
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
      liveFlagRef.current = true;
      setLiveRunning(true);

      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 640, height: 480, frameRate: { ideal: targetFps } },
        audio: false,
      });

      streamRef.current = stream;
      videoEl.srcObject = stream;

      await new Promise((resolve) => {
        const onLoaded = () => resolve();
        videoEl.onloadedmetadata = onLoaded;
      });

      await videoEl.play();
      ensureCanvasMatchesVideo();

      // Try camera-reported FPS if available
      try {
        const track = stream.getVideoTracks()[0];
        const settings = track?.getSettings?.();
        if (settings && typeof settings.frameRate === "number") {
          fpsEstimateRef.current = settings.frameRate;
          setStreamFps(settings.frameRate);
        }
      } catch {
        // ignore
      }

      const pose = new mpPose.Pose({
        locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/pose/${file}`,
      });

      pose.setOptions({
        modelComplexity: isActiveRef.current ? 1 : 0,
        smoothLandmarks: true,
        minDetectionConfidence: 0.5,
        minTrackingConfidence: 0.5,
      });

      pose.onResults(handlePoseResults);
      poseRef.current = pose;

      const loop = async () => {
        if (!liveFlagRef.current) return;

        const now = performance.now();
        const target = Math.max(1, Number(inferFpsRef.current) || 15);
        const minIntervalMs = 1000 / target;

        if (now - lastInferTsRef.current >= minIntervalMs) {
          lastInferTsRef.current = now;

          if (videoEl.readyState >= 2 && poseRef.current) {
            try {
              await poseRef.current.send({ image: videoEl });
            } catch (err) {
              console.error("pose.send error", err);
            }
          }
        }

        rafRef.current = requestAnimationFrame(loop);
      };

      void loop();
      return true;
    } catch (err) {
      console.error("Error starting monitor", err);
      stopLive();
      return false;
    }
  }, [ensureCanvasMatchesVideo, handlePoseResults, stopLive, targetFps]);

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
      await apiRequest(
        apiBase,
        `/api/monitor/reset_session?session_id=${encodeURIComponent(sessionIdRef.current)}`,
        { method: "POST" }
      );
    } catch {
      // ignore
    }
  }, [apiBase]);

  const testFall = useCallback(async () => {
    // 1) Try backend helper endpoint (if present)
    try {
      const data = await apiRequest(apiBase, "/api/events/test_fall", {
        method: "POST",
        body: { resident_id: 1, model_code: prettyModelTag(settingsPayload?.system?.active_model_code) },
      });
      if (data || data?.ok) {
        addTimelineMarker("fall");
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
    addTimelineMarker("fall");
  }, [addTimelineMarker, apiBase, settingsPayload]);

  const currentPrediction = useMemo(() => labelForTriage(triageState), [triageState]);

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
    const last50 = timeline.slice(-50);
    const n = last50.length;
    if (n === 0) return [];

    return last50.map((m, idx) => ({
      leftPct: ((idx + 1) / (n + 1)) * 100,
      kind: m.kind,
    }));
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

    markers,
    captureFpsText,
    modelFpsText,

    resetSession,
    testFall,

    tauHighFromSpec,
  };
}
