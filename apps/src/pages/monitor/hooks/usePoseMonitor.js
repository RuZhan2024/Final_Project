import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import * as mpPose from "@mediapipe/pose";
import * as drawingUtils from "@mediapipe/drawing_utils";

import { apiRequest } from "../../../lib/apiClient";
import { CLIP_POST_S, CLIP_PRE_S, MAX_PROC_FPS, NUM_JOINTS } from "../constants";
import { clamp01, labelForTriage, prettyModelTag } from "../utils";

const { drawConnectors, drawLandmarks } = drawingUtils;
const XY_DIM = NUM_JOINTS * 2;

function safeNumber(x) {
  const n = Number(x);
  return Number.isFinite(n) ? n : null;
}

function positiveNumberOrUndef(x) {
  const n = Number(x);
  return Number.isFinite(n) && n > 0 ? n : undefined;
}

function getClipFlags(settingsPayload) {
  const sys = settingsPayload?.system || {};
  return {
    storeEventClips: Boolean(sys?.store_event_clips),
    anonymize: Boolean(sys?.anonymize_skeleton_data ?? true),
  };
}

function stripNilFields(obj) {
  const out = {};
  for (const [k, v] of Object.entries(obj || {})) {
    if (v !== null && typeof v !== "undefined") out[k] = v;
  }
  return out;
}

function packFrameSlice(raw, startIdx, endIdx) {
  const n = endIdx - startIdx + 1;
  const t_ms = new Array(n);
  // Build JSON-ready dense arrays directly to avoid an extra Array.from copy pass.
  const xyFlat = new Array(n * XY_DIM).fill(0);
  const confFlat = new Array(n * NUM_JOINTS).fill(0);
  let j = 0;
  for (let i = startIdx; i <= endIdx; i += 1, j += 1) {
    const fr = raw[i];
    const t = Number(fr?.t);
    t_ms[j] = Number.isFinite(t) ? Math.round(t) : 0;
    const xySrc = fr?.xyFlat;
    const confSrc = fr?.confFlat;
    const xyBase = j * XY_DIM;
    const confBase = j * NUM_JOINTS;
    if (xySrc && xySrc.length === XY_DIM) {
      for (let k = 0; k < XY_DIM; k += 1) xyFlat[xyBase + k] = xySrc[k];
    }
    if (confSrc && confSrc.length === NUM_JOINTS) {
      for (let k = 0; k < NUM_JOINTS; k += 1) confFlat[confBase + k] = confSrc[k];
    }
  }
  return {
    t_ms,
    xy_flat: xyFlat,
    conf_flat: confFlat,
  };
}

function lowerBoundFrameTs(raw, targetTs) {
  let lo = 0;
  let hi = raw.length;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    const t = raw[mid]?.t;
    if (typeof t !== "number" || !Number.isFinite(t) || t < targetTs) lo = mid + 1;
    else hi = mid;
  }
  return lo;
}

function upperBoundFrameTs(raw, targetTs) {
  let lo = 0;
  let hi = raw.length;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    const t = raw[mid]?.t;
    if (typeof t !== "number" || !Number.isFinite(t) || t <= targetTs) lo = mid + 1;
    else hi = mid;
  }
  return lo;
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
  const lastClockMsRef = useRef(0);
  const clockTextRef = useRef("");

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
  const rawFramesRef = useRef([]); // [{t,xyFlat,confFlat}]
  const lastPoseTsRef = useRef(null);
  const fpsDeltasRef = useRef([]);
  const fpsDeltaSumRef = useRef(0);
  const fpsEstimateRef = useRef(null);
  const lastSentRef = useRef(0);

  // Skeleton clip saving
  const pendingClipRef = useRef(null);
  const uploadedClipIdsRef = useRef(new Set());
  const clipUploadInFlightRef = useRef(false);
  const predictInFlightRef = useRef(false);
  const bgTickScheduledRef = useRef(false);
  const bgTaskIdRef = useRef(null);
  const bgTaskTypeRef = useRef(null);

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
    fpsDeltaSumRef.current = 0;
    lastSentRef.current = 0;

    // Clip upload state
    pendingClipRef.current = null;
    uploadedClipIdsRef.current = new Set();
    bgTickScheduledRef.current = false;
    if (bgTaskIdRef.current != null) {
      if (bgTaskTypeRef.current === "idle" && typeof window !== "undefined" && window.cancelIdleCallback) {
        window.cancelIdleCallback(bgTaskIdRef.current);
      } else if (bgTaskTypeRef.current === "timeout" && typeof window !== "undefined") {
        window.clearTimeout(bgTaskIdRef.current);
      }
    }
    bgTaskIdRef.current = null;
    bgTaskTypeRef.current = null;

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
    if (clipUploadInFlightRef.current) return;

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
    const startIdx = lowerBoundFrameTs(raw, startTs);
    const endExclusive = upperBoundFrameTs(raw, endTs);
    const endIdx = endExclusive - 1;

    if (startIdx < 0 || endIdx <= startIdx || startIdx >= raw.length) {
      pendingClipRef.current = null;
      return;
    }

    try {
      clipUploadInFlightRef.current = true;
      const { t_ms, xy_flat, conf_flat } = packFrameSlice(raw, startIdx, endIdx);

      const clipPayload = stripNilFields({
        resident_id: 1,
        dataset_code: pending.ctx?.dataset_code,
        mode: pending.ctx?.mode,
        op_code: pending.ctx?.op_code,
        use_mc: pending.ctx?.use_mc,
        mc_M: pending.ctx?.mc_M,
        mc_sigma_tol: pending.ctx?.mc_sigma_tol,
        mc_se_tol: pending.ctx?.mc_se_tol,
        pre_s: pending.preMs / 1000,
        post_s: pending.postMs / 1000,
        t_ms,
        xy_flat,
        conf_flat,
        raw_joints: NUM_JOINTS,
      });

      const data = await apiRequest(
        apiBase,
        `/api/events/${encodeURIComponent(pending.eventId)}/skeleton_clip`,
        { method: "POST", body: clipPayload }
      );

      if (data?.ok) {
        sent.add(String(pending.eventId));
        // Bound memory in very long monitoring sessions.
        while (sent.size > 512) {
          const first = sent.values().next().value;
          if (first == null) break;
          sent.delete(first);
        }
      }
    } catch (err) {
      console.error("Failed to upload skeleton clip", err);
    } finally {
      clipUploadInFlightRef.current = false;
      pendingClipRef.current = null;
    }
  }, [apiBase, settingsPayload]);

  const maybeSendWindow = useCallback(async () => {
    const now = performance.now();
    const minGapMs = Math.max(250, (deployS / Math.max(1, targetFps)) * 1000);
    if (now - lastSentRef.current < minGapMs) return;
    if (predictInFlightRef.current) return;

    const raw = rawFramesRef.current;
    if (!raw || raw.length < 2) return;

    const dtMs = 1000 / Math.max(1, Number(targetFps) || 30);
    const needMs = (deployW - 1) * dtMs;
    const endTs = raw[raw.length - 1].t;
    const startNeed = endTs - needMs;

    // Wait until we have enough history to cover the full window duration.
    if (raw[0].t > startNeed) return;

    // Include one frame before startNeed for interpolation on the server.
    const i0 = lowerBoundFrameTs(raw, startNeed);
    const startIdx = Math.max(0, i0 - 1);
    const { t_ms: raw_t_ms, xy_flat: raw_xy_flat, conf_flat: raw_conf_flat } = packFrameSlice(
      raw,
      startIdx,
      raw.length - 1
    );

    lastSentRef.current = now;

    const { storeEventClips } = getClipFlags(settingsPayload);
    const mcSigmaTol = positiveNumberOrUndef(settingsPayload?.deploy?.mc?.sigma_tol);
    const mcSeTol = positiveNumberOrUndef(settingsPayload?.deploy?.mc?.se_tol);

    const payload = stripNilFields({
      session_id: sessionIdRef.current,
      mode,
      dataset_code: activeDatasetCode,
      op_code: opCode || settingsPayload?.system?.active_op_code || "OP-2",

      model_tcn: mode === "dual" ? chosen.tcn : mode === "tcn" ? chosen.tcn : null,
      model_gcn: mode === "dual" ? chosen.gcn : mode === "gcn" ? chosen.gcn : null,

      // The model expects this FPS after resampling.
      target_fps: targetFps,
      target_T: deployW,
      use_mc: Boolean(mcEnabled),
      mc_M: mcCfg?.M,
      mc_sigma_tol: mcSigmaTol,
      mc_se_tol: mcSeTol,

      // Persist events when monitoring is on OR when we need an event_id for clip saving.
      persist: Boolean(monitoringOn || storeEventClips),

      // Raw pose samples (variable FPS)
      raw_t_ms,
      raw_xy_flat,
      raw_conf_flat,
      raw_joints: NUM_JOINTS,

      window_end_t_ms: endTs,
    });

    try {
      predictInFlightRef.current = true;
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
                mc_sigma_tol: mcSigmaTol,
                mc_se_tol: mcSeTol,
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
    } finally {
      predictInFlightRef.current = false;
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

  const scheduleBackgroundTick = useCallback(() => {
    if (bgTickScheduledRef.current) return;
    bgTickScheduledRef.current = true;
    const run = () => {
      bgTickScheduledRef.current = false;
      bgTaskIdRef.current = null;
      bgTaskTypeRef.current = null;
      void maybeFinalizeClipUpload();
      void maybeSendWindow();
    };
    if (typeof window !== "undefined" && typeof window.requestIdleCallback === "function") {
      bgTaskTypeRef.current = "idle";
      bgTaskIdRef.current = window.requestIdleCallback(run, { timeout: 50 });
      return;
    }
    bgTaskTypeRef.current = "timeout";
    bgTaskIdRef.current = setTimeout(run, 0);
  }, [maybeFinalizeClipUpload, maybeSendWindow]);

  const handlePoseResults = useCallback(
    (results) => {
      // Cap processing rate.
      const nowMs = performance.now();
      if (nowMs - lastProcMsRef.current < 1000 / MAX_PROC_FPS) return;
      lastProcMsRef.current = nowMs;
      const tNow = Math.round(nowMs);

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

        if (tNow - lastClockMsRef.current >= 500 || !clockTextRef.current) {
          clockTextRef.current = new Date().toLocaleTimeString();
          lastClockMsRef.current = tNow;
        }
        ctx.font = "12px system-ui, -apple-system, Segoe UI, Roboto";
        ctx.fillStyle = "#94a3b8";
        ctx.fillText(clockTextRef.current, 16, h - 16);
      }

      // Build raw frame
      const xyFlat = new Float32Array(NUM_JOINTS * 2);
      const confFlat = new Float32Array(NUM_JOINTS);

      for (let i = 0; i < NUM_JOINTS; i++) {
        const lm = landmarks[i];
        const b = i * 2;
        if (!lm) {
          xyFlat[b] = 0;
          xyFlat[b + 1] = 0;
          confFlat[i] = 0;
        } else {
          xyFlat[b] = clamp01(lm.x);
          xyFlat[b + 1] = clamp01(lm.y);
          confFlat[i] = typeof lm.visibility === "number" ? clamp01(lm.visibility) : 1.0;
        }
      }

      // Estimate FPS from callback timing (smoothed)
      const last = lastPoseTsRef.current;
      if (typeof last === "number") {
        const dt = tNow - last;
        if (dt > 0 && dt < 5000) {
          const arr = fpsDeltasRef.current;
          arr.push(dt);
          fpsDeltaSumRef.current += dt;
          if (arr.length > 30) {
            const dropped = arr.shift();
            if (typeof dropped === "number") fpsDeltaSumRef.current -= dropped;
          }

          const meanDt = fpsDeltaSumRef.current / Math.max(1, arr.length);
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
      raw.push({ t: tNow, xyFlat, confFlat });

      // Keep ~12s max. Trim in batches to avoid O(n) shifts on every frame.
      const maxRaw = Math.max(600, Math.ceil(targetFps * 12));
      const trimChunk = Math.max(32, Math.ceil(targetFps));
      if (raw.length > maxRaw + trimChunk) raw.splice(0, raw.length - maxRaw);

      // Clip upload + inference run on a deferred tick to keep pose callback light.
      scheduleBackgroundTick();
    },
    [ensureCanvasMatchesVideo, scheduleBackgroundTick, targetFps]
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
