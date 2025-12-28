import React, { useEffect, useMemo, useRef, useState } from "react";
import * as mpPose from "@mediapipe/pose";
import * as drawingUtils from "@mediapipe/drawing_utils";
import styles from "./Monitor.module.css";

import { useMonitoring } from "../monitoring/MonitoringContext";

const { POSE_CONNECTIONS } = mpPose;
const { drawConnectors, drawLandmarks } = drawingUtils;

// Prefer env var, fallback to localhost
const API_BASE =
  typeof process !== "undefined" && process.env && process.env.REACT_APP_API_BASE
    ? process.env.REACT_APP_API_BASE
    : "http://localhost:8000";

const NUM_JOINTS = 33;

function clamp01(x) {
  if (x < 0) return 0;
  if (x > 1) return 1;
  return x;
}

/**
 * Build a fixed-length [T,33,2] + [T,33] window at a target FPS by resampling
 * the raw pose frames (which arrive at variable FPS).
 */
function buildResampledWindow(rawFrames, targetFps, windowSize) {
  if (!rawFrames || rawFrames.length === 0) return null;

  const dt = 1000 / targetFps;
  const endTs = rawFrames[rawFrames.length - 1].t;
  const startTs = endTs - (windowSize - 1) * dt;

  // Need at least one sample at/before startTs to cover full window
  if (rawFrames[0].t > startTs) return null;

  const xyOut = [];
  const confOut = [];

  let j = 0;
  for (let k = 0; k < windowSize; k++) {
    const tWanted = startTs + k * dt;
    while (j + 1 < rawFrames.length && rawFrames[j + 1].t <= tWanted) j++;
    const fr = rawFrames[j] || rawFrames[0];
    xyOut.push(fr.xy);
    confOut.push(fr.conf);
  }

  return { xy: xyOut, conf: confOut, startTs, endTs };
}

function normModeFromCode(code) {
  const c = (code || "").toUpperCase();
  if (c === "TCN") return "tcn";
  if (c === "GCN") return "gcn";
  if (c === "HYBRID") return "dual";
  return "gcn";
}

function labelForTriage(triageState) {
  const t = (triageState || "").toLowerCase();
  if (t === "fall") return "Fall";
  if (t === "uncertain") return "Uncertain";
  return "No fall";
}

function prettyModelTag(activeModelCode) {
  const c = (activeModelCode || "").toUpperCase();
  if (c === "HYBRID") return "HYBRID";
  if (c === "TCN") return "TCN";
  if (c === "GCN") return "GCN";
  return "GCN";
}

function pickFirstByArch(models, arch) {
  const a = (arch || "").toLowerCase();
  const m = (models || []).find((x) => (x?.arch || "").toLowerCase() === a);
  return m?.id || "";
}

function pickModelPair(models) {
  const tcn = pickFirstByArch(models, "tcn");
  const gcn = pickFirstByArch(models, "gcn");
  return { tcn, gcn };
}

function MonitorDemo({ isActive = true } = {}) {
  // Backend config
  const [deployW, setDeployW] = useState(48);
  const [deployS, setDeployS] = useState(12);
  const [mcCfg, setMcCfg] = useState({ M: 10, M_confirm: 25 });
  const [activeModelCode, setActiveModelCode] = useState("GCN");
  const MAX_PROC_FPS = 30;
  const lastProcMsRef = useRef(0);

  // Global monitoring (persisted via /api/settings)
  const {
    monitoringOn,
    setMonitoringOn,
    registerController,
    error: monitoringErr,
    settings: settingsPayload,
    loaded: settingsLoaded,
    apiBase: apiBaseFromCtx,
  } = useMonitoring();

  const apiBase = apiBaseFromCtx || API_BASE;

  // Loaded models
  const [models, setModels] = useState([]);
  const [modelsErr, setModelsErr] = useState(null);

  // Live state
  const [liveRunning, setLiveRunning] = useState(false);
  const [streamFps, setStreamFps] = useState(null);

  // Prediction UI
  const [triageState, setTriageState] = useState("not_fall");
  const [pFall, setPFall] = useState(null);
  const [sigma, setSigma] = useState(null);

  // Timeline markers (store indices of non-safe windows in last 50)
  const [timeline, setTimeline] = useState([]); // array of {i, kind: 'fall'|'uncertain'}

  // Model info
  const [modelTag, setModelTag] = useState("GCN");
  const [tauHigh, setTauHigh] = useState(null);
  // Thresholds (from DB operating point / settings)
  const [fallThreshold, setFallThreshold] = useState(null);
  const [opThrDetect, setOpThrDetect] = useState(null);
  const [opThrLow, setOpThrLow] = useState(null);
  const [opThrHigh, setOpThrHigh] = useState(null);
  const [opCode, setOpCode] = useState(null);

  // Refs for camera + mediapipe
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const isActiveRef = useRef(isActive);
  useEffect(() => {
    isActiveRef.current = Boolean(isActive);
  }, [isActive]);
  const poseRef = useRef(null);
  const streamRef = useRef(null);
  const rafRef = useRef(null);
  const liveFlagRef = useRef(false);

  // Throttle MediaPipe inference so the UI stays responsive while monitoring runs in background.
  const lastInferTsRef = useRef(0);
  const inferFpsRef = useRef(15);


  // Mode + chosen models
  const mode = useMemo(() => normModeFromCode(activeModelCode), [activeModelCode]);

  const chosen = useMemo(() => {
    if (mode === "dual") return pickModelPair(models);
    if (mode === "tcn") return { tcn: pickFirstByArch(models, "tcn"), gcn: "" };
    return { tcn: "", gcn: pickFirstByArch(models, "gcn") };
  }, [mode, models]);

  const chosenSpec = useMemo(() => {
    if (mode === "tcn") return models.find((m) => m.id === chosen.tcn) || null;
    if (mode === "gcn") return models.find((m) => m.id === chosen.gcn) || null;
    // dual: show GCN as primary tag in UI, but keep both
    return models.find((m) => m.id === chosen.gcn) || models.find((m) => m.id === chosen.tcn) || null;
  }, [mode, models, chosen]);

  const targetFps = useMemo(() => {
    const f = chosenSpec?.fps_default;
    if (typeof f === "number" && Number.isFinite(f) && f > 0) return f;
    return 30;
  }, [chosenSpec]);

  useEffect(() => {
    // When not on the /Monitor page, keep monitoring running but reduce load so navigation stays smooth.
    // Keep this conservative so React routing remains snappy while the pipeline runs.
    const fgFps = Math.min(12, Math.max(5, Number(targetFps) || 12));
    const bgFps = Math.min(5, fgFps);
    inferFpsRef.current = isActiveRef.current ? fgFps : bgFps;

    // Lower MediaPipe complexity when the page is not visible.
    try {
      if (poseRef.current && poseRef.current.setOptions) {
        poseRef.current.setOptions({ modelComplexity: isActiveRef.current ? 1 : 0 });
      }
    } catch {
      // ignore
    }
  }, [targetFps, isActive]);

  // Expose start/stop to the app-level toggle (so Dashboard can turn monitoring on/off)
  const startLiveFnRef = useRef(null);
  const stopLiveFnRef = useRef(null);

  // Frame buffer for windowing
  const rawFramesRef = useRef([]); // [{t,xy,conf}]
  const lastPoseTsRef = useRef(null);
  const fpsDeltasRef = useRef([]);
  const fpsEstimateRef = useRef(null);

  // Throttled UI update for stream FPS (avoid React re-render every frame)
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
  const lastSentRef = useRef(0);

  // Session id for server-side state machine
  const sessionIdRef = useRef(`monitor-demo-${Math.random().toString(16).slice(2)}`);
  // Keep Model Info box in sync
  useEffect(() => {
    setModelTag(prettyModelTag(activeModelCode));
    if (chosenSpec?.tau_high != null) setTauHigh(Number(chosenSpec.tau_high));
  }, [activeModelCode, chosenSpec]);

  // Apply global settings so this page follows changes made on /Settings.
  useEffect(() => {
    if (!settingsLoaded || !settingsPayload) return;
    const s = settingsPayload;

    const W = s?.deploy?.window?.W;
    const S = s?.deploy?.window?.S;
    if (typeof W === "number") setDeployW(W);
    if (typeof S === "number") setDeployS(S);

    const mc = s?.deploy?.mc;
    if (mc && typeof mc === "object") {
      setMcCfg({ M: mc.M, M_confirm: mc.M_confirm });
    }

    const sys = s?.system || {};
    if (sys?.active_model_code) setActiveModelCode(sys.active_model_code);
    if (typeof sys?.fall_threshold === "number") setFallThreshold(Number(sys.fall_threshold));

    // Load thresholds from the active operating point (preferred when DB schema is v2).
    let cancelled = false;
    (async () => {
      try {
        const opId = sys?.active_operating_point;
        const codeRaw = String(sys?.active_model_code || "GCN").toUpperCase();
        const tryCodes = codeRaw === "HYBRID" ? ["HYBRID", "GCN"] : [codeRaw];
        let picked = null;
        for (const code of tryCodes) {
          const rop = await fetch(
            `${apiBase}/api/operating_points?model_code=${encodeURIComponent(code)}`
          );
          if (!rop.ok) continue;
          const opData = await rop.json();
          const ops = Array.isArray(opData?.operating_points)
            ? opData.operating_points
            : Array.isArray(opData)
              ? opData
              : [];
          const byId = opId != null ? ops.find((o) => Number(o.id) === Number(opId)) : null;
          picked =
            byId ||
            ops.find((o) => String(o.code || o.op_code || "").toUpperCase() === "OP-2") ||
            ops[0] ||
            null;
          if (picked) break;
        }
        if (!picked || cancelled) return;

        setOpCode(picked.code || picked.op_code || null);
        const det = picked.thr_detect != null ? Number(picked.thr_detect) : null;
        const low =
          picked.thr_low_conf != null
            ? Number(picked.thr_low_conf)
            : picked.threshold_low != null
              ? Number(picked.threshold_low)
              : null;
        const high =
          picked.thr_high_conf != null
            ? Number(picked.thr_high_conf)
            : picked.threshold_high != null
              ? Number(picked.threshold_high)
              : null;

        setOpThrDetect(det);
        setOpThrLow(low);
        setOpThrHigh(high);
      } catch {
        // ignore
      }
    })();

    return () => {
      cancelled = true;
    };
  }, [settingsLoaded, settingsPayload, apiBase]);

  // Load models list
  useEffect(() => {
    (async () => {
      try {
        setModelsErr(null);
        const rm = await fetch(`${apiBase}/api/models/summary`);
        if (!rm.ok) throw new Error(await rm.text());
        const data = await rm.json();
        const arr = Array.isArray(data?.models) ? data.models : [];
        setModels(arr);
      } catch (e) {
        setModelsErr(String(e?.message || e));
        setModels([]);
      }
    })();
  }, []);

  // Canvas sizing helper
  function ensureCanvasMatchesVideo() {
    const videoEl = videoRef.current;
    const canvasEl = canvasRef.current;
    if (!videoEl || !canvasEl) return;

    const vw = videoEl.videoWidth || 1280;
    const vh = videoEl.videoHeight || 720;

    // Only resize when changed (avoid flicker)
    if (canvasEl.width !== vw) canvasEl.width = vw;
    if (canvasEl.height !== vh) canvasEl.height = vh;
  }

  // Start/Stop camera + pose
  async function startLive() {
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

      // MediaPipe Pose
      const pose = new mpPose.Pose({
        locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/pose/${file}`,
      });

      pose.setOptions({
        // Use the lightest model by default. When /Monitor is the active page,
        // the effect above may bump complexity up if needed.
        modelComplexity: 0,
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

      loop();
      return true;
    } catch (err) {
      console.error("Error starting monitor-demo:", err);
      stopLive();
      return false;
    }
  }

  function stopLive() {
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

    // Reset UI
    setPFall(null);
    setSigma(null);
    setTriageState("not_fall");
    setTimeline([]);
  }

  // Keep controller refs up-to-date, then register with the app-level toggle.
  useEffect(() => {
    startLiveFnRef.current = startLive;
    stopLiveFnRef.current = stopLive;
  });

  useEffect(() => {
    registerController({
      start: () => startLiveFnRef.current && startLiveFnRef.current(),
      stop: () => stopLiveFnRef.current && stopLiveFnRef.current(),
    });
    return () => registerController(null);
  }, [registerController]);

  // Keep video off-screen (we want skeleton-only view)
  useEffect(() => {
    const v = videoRef.current;
    if (v) v.playsInline = true;
  }, []);

  // Cleanup
  useEffect(() => {
    return () => stopLive();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Pose callback
  function handlePoseResults(results) {
    // Option A: cap pose processing rate to MAX_PROC_FPS
    const nowMs = performance.now();
    if (nowMs - lastProcMsRef.current < 1000 / MAX_PROC_FPS) return;
    lastProcMsRef.current = nowMs;
    const canvasEl = canvasRef.current;
    if (!canvasEl) return;

    const ctx = canvasEl.getContext("2d");
      const doDraw = isActiveRef.current;

      const landmarks = results.poseLandmarks;

      // If no pose is detected, keep the pipeline running but don't crash the UI.
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

      // Draw only when this page is active (so other pages stay responsive)
      if (doDraw && ctx) {
        ensureCanvasMatchesVideo();
        const w = canvasEl.width;
        const h = canvasEl.height;

        ctx.fillStyle = "#020617";
        ctx.fillRect(0, 0, w, h);

        // Simple skeleton overlay
        drawConnectors(ctx, landmarks, mpPose.POSE_CONNECTIONS, {
          color: "#22c55e",
          lineWidth: 2,
        });
        drawLandmarks(ctx, landmarks, { color: "#fbbf24", lineWidth: 1 });

        // Timestamp
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
    // Keep ~10s max (enough for windowing)
    const maxRaw = Math.max(600, Math.ceil(targetFps * 12));
    if (raw.length > maxRaw) raw.splice(0, raw.length - maxRaw);

    // Only send once we have a full resampled window
    maybeSendWindow();
  }

  function addTimelineMarker(kind) {
    setTimeline((prev) => {
      const next = prev.slice(-49); // keep last 49, then add 1
      next.push({ i: (prev.length ? prev[prev.length - 1].i + 1 : 0), kind });
      return next;
    });
  }

    async function maybeSendWindow() {
    const now = performance.now();
    // Stride is defined in frames at the *target/model* FPS (e.g., S=12 at 30 FPS => ~0.4s).
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

    // Best-practice: send *raw* frames + timestamps; server resamples to target_fps/target_T.
    const payload = {
      session_id: sessionIdRef.current,
      resident_id: 1,
      mode: mode,
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
      use_mc: true,
      persist: Boolean(monitoringOn),

      // Raw pose samples (variable FPS)
      raw_t_ms: slice.map((fr) => fr.t),
      raw_xy: slice.map((fr) => fr.xy),
      raw_conf: slice.map((fr) => fr.conf),

      // Optional (helps server align the window end exactly)
      window_end_t_ms: endTs,
    };

    try {
      const resp = await fetch(`${apiBase}/api/monitor/predict_window`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      if (!resp.ok) {
        console.error("Backend error", await resp.text());
        return;
      }

      const data = await resp.json();

      const tri = data?.triage_state || data?.triageState || "not_fall";
      setTriageState(tri);

      // pFall display: use overall mu (single) or max of (tcn,gcn) for dual
      let mu = null;
      let sig = null;

      if (mode === "dual") {
        const mt = data?.models?.tcn;
        const mg = data?.models?.gcn;
        const muT = mt?.mu != null ? Number(mt.mu) : null;
        const muG = mg?.mu != null ? Number(mg.mu) : null;
        mu = Math.max(muT ?? 0, muG ?? 0);
        // show sigma if both present (max)
        const sT = mt?.sigma != null ? Number(mt.sigma) : null;
        const sG = mg?.sigma != null ? Number(mg.sigma) : null;
        sig = Math.max(sT ?? 0, sG ?? 0);
        // threshold (use the "primary" spec if available)
        if (chosenSpec?.tau_high != null) setTauHigh(Number(chosenSpec.tau_high));
      } else {
        const mOut = data?.models?.[mode];
        if (mOut) {
          mu = mOut?.mu != null ? Number(mOut.mu) : mOut?.p_det != null ? Number(mOut.p_det) : null;
          sig = mOut?.sigma != null ? Number(mOut.sigma) : null;
          if (mOut?.triage?.tau_high != null) setTauHigh(Number(mOut.triage.tau_high));
        }
      }

      setPFall(mu);
      setSigma(sig);

      // timeline markers: mark uncertain/fall
      const triNorm = (tri || "").toLowerCase();
      if (triNorm === "fall") addTimelineMarker("fall");
      else if (triNorm === "uncertain") addTimelineMarker("uncertain");
    } catch (err) {
      console.error("Error calling /api/monitor/predict_window:", err);
    }
  }

  async function resetSession() {
    try {
      await fetch(`${apiBase}/api/monitor/reset_session?session_id=${encodeURIComponent(sessionIdRef.current)}`, {
        method: "POST",
      });
    } catch {
      // ignore
    }
  }

  async function testFall() {
    // 1) Try backend helper endpoint (if present)
    try {
      const r = await fetch(`${apiBase}/api/events/test_fall`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ resident_id: 1, model_code: prettyModelTag(activeModelCode) }),
      });
      if (r.ok) {
        addTimelineMarker("fall");
        return;
      }
    } catch {
      // ignore
    }

    // 2) Fallback: record a test notification (if DB exists)
    try {
      await fetch(`${apiBase}/api/notifications/test`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          resident_id: 1,
          channel: "email",
          message: "[UI] Test Fall button pressed",
        }),
      });
    } catch {
      // ignore
    }

    // 3) UI-only marker
    addTimelineMarker("fall");
  }

  const currentPrediction = labelForTriage(triageState);
  const pText = pFall == null ? "—" : pFall.toFixed(3);

  // markers: translate last 50 markers to positions (0..100%)
  const markers = useMemo(() => {
    // We don't know exact index within 50; approximate by spreading over the bar.
    const last50 = timeline.slice(-50);
    const n = last50.length;
    if (n === 0) return [];
    return last50.map((m, idx) => ({
      leftPct: ((idx + 1) / (n + 1)) * 100,
      kind: m.kind,
    }));
  }, [timeline]);

  // Display both FPS numbers:
  // - Capture FPS: the actual MediaPipe callback rate (variable)
  // - Model FPS: the FPS the server resamples to before inference (fixed, from model spec)
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

  return (
    <div className={styles.pageContainer}>
      <h2 className={styles.pageTitle}>Live Monitor</h2>

      <div className={styles.content}>
        {/* LEFT COLUMN (2/3) */}
        <div className={styles.leftColumn}>
          <div className={styles.card}>
            <div className={styles.cardHeader}>
              <h3>Live Monitoring</h3>
              <p>Real-time Skeleton tracking and fall detection</p>
            </div>

            {/* Video/Skeleton View (skeleton-only canvas) */}
            <div className={styles.videoPlaceholder}>
              <video
                ref={videoRef}
                muted
                playsInline
                style={{
                  position: "absolute",
                  inset: 0,
                  width: "100%",
                  height: "100%",
                  objectFit: "cover",
                  opacity: 0, // keep hidden (skeleton-only)
                }}
              />
              <canvas
                ref={canvasRef}
                style={{
                  position: "absolute",
                  inset: 0,
                  width: "100%",
                  height: "100%",
                }}
              />
            </div>

            {/* Prediction Result Box */}
            <div className={styles.predictionBox}>
              <div className={styles.predictionItem}>
                <span className={styles.label}>Current Prediction</span>
                <span className={styles.value}>{currentPrediction}</span>
              </div>
              <div className={styles.predictionItem}>
                <span className={styles.label}>P (fall)</span>
                <span className={styles.value}>{pText}</span>
              </div>
            </div>
          </div>
        </div>

        {/* RIGHT COLUMN (1/3) */}
        <div className={styles.rightColumn}>
          {/* Controls Card */}
          <div className={styles.card}>
            <h3>Controls</h3>
            {modelsErr && (
              <p className={styles.subText} style={{ color: "#B45309" }}>
                Backend error: {modelsErr}
              </p>
            )}
            {monitoringErr && (
              <p className={styles.subText} style={{ color: "#B45309" }}>
                Settings error: {monitoringErr}
              </p>
            )}
            <div className={styles.buttonGroup}>
              {!monitoringOn ? (
                <button
                  className={styles.btnGray}
                  onClick={() => {
                    // Don't await here: we want getUserMedia to run under a user gesture.
                    resetSession();
                    setMonitoringOn(true);
                  }}
                  title="Start"
                >
                  Start
                </button>
              ) : (
                <button className={styles.btnGray} onClick={() => setMonitoringOn(false)}>
                  Stop
                </button>
              )}
              <button className={styles.btnRed} onClick={testFall}>
                Test Fall
              </button>
            </div>
          </div>

          {/* Timeline Card */}
          <div className={styles.card}>
            <h3>Event Timeline</h3>
            <p className={styles.subText}>Last 50 prediction windows</p>
            <div className={styles.timelineBar}>
              {markers.map((m, idx) => (
                <div
                  key={idx}
                  className={styles.marker}
                  style={{
                    left: `${m.leftPct}%`,
                    opacity: m.kind === "uncertain" ? 0.5 : 1,
                  }}
                  title={m.kind}
                />
              ))}
            </div>
          </div>

          {/* Model Info Card */}
          <div className={styles.card}>
            <h3>Model Info</h3>
            <div className={styles.infoTable}>
              <div className={styles.infoRow}>
                <span>Model:</span>
                <span className={styles.tag}>{modelTag}</span>
              </div>
              <div className={styles.infoRow}>
                <span>Window Size</span>
                <span>{deployW}</span>
              </div>
              <div className={styles.infoRow}>
                <span>Stride</span>
                <span>{deployS}</span>
              </div>
              <div className={styles.infoRow}>
                <span>Threshold</span>
                <span>
                  {(() => {
                    const v = opThrDetect ?? opThrHigh ?? fallThreshold ?? tauHigh;
                    if (v == null) return "—";
                    const tag = opCode ? ` (${opCode})` : "";
                    return `${Number(v).toFixed(2)}${tag}`;
                  })()}
                </span>
              </div>
            </div>
            {/* Keep extra info out of the design, but useful for debugging */}
            <p className={styles.subText} style={{ marginTop: 12 }}>
              Mode: {mode} • Capture FPS: {captureFpsText} • Model FPS: {modelFpsText} • MC: {mcCfg?.M ?? "—"}/{mcCfg?.M_confirm ?? "—"}
              {sigma != null ? ` • σ=${sigma.toFixed(3)}` : ""}
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}

export default MonitorDemo;