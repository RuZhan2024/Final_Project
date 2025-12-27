// src/pages/Monitor.jsx
import React, { useEffect, useMemo, useRef, useState } from "react";
import * as mpPose from "@mediapipe/pose";
import * as drawingUtils from "@mediapipe/drawing_utils";

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

export default function Monitor() {
  // Server-configured window (fallback to 48/12)
  const [windowSize, setWindowSize] = useState(48);
  const [stride, setStride] = useState(12);

  // Models come from backend (/api/models/summary)
  const [models, setModels] = useState([]);
  const [modelsErr, setModelsErr] = useState(null);

  const [selectedModel, setSelectedModel] = useState("");
  const [liveRunning, setLiveRunning] = useState(false);

  const [backendOk, setBackendOk] = useState(null); // null/true/false
  const [pFall, setPFall] = useState(null);
  const [threshold, setThreshold] = useState(null);
  const [framesCollected, setFramesCollected] = useState(0);
  const [frameIndex, setFrameIndex] = useState(0);
  const [statusText, setStatusText] = useState("SAFE");
  const [triageState, setTriageState] = useState("not_fall");

  const [streamFps, setStreamFps] = useState(null);

  const videoRef = useRef(null);
  const canvasRef = useRef(null);

  const poseRef = useRef(null);
  const streamRef = useRef(null);
  const rafRef = useRef(null);
  const liveFlagRef = useRef(false);

  const rawFramesRef = useRef([]); // [{t,xy,conf}]
  const lastPoseTsRef = useRef(null);
  const fpsDeltasRef = useRef([]);

  const modelIdRef = useRef("");
  const lastSentRef = useRef(0);

  const sessionIdRef = useRef(`monitor-${Math.random().toString(16).slice(2)}`);

  const selectedModelObj = useMemo(
    () => models.find((m) => m.id === selectedModel) || null,
    [models, selectedModel]
  );

  const targetFps = useMemo(() => {
    const f = selectedModelObj?.fps_default;
    if (typeof f === "number" && Number.isFinite(f) && f > 0) return f;
    return 30;
  }, [selectedModelObj]);

  const displayedTargetFps = targetFps;

  // Load settings + model list from backend (so frontend always matches server)
  useEffect(() => {
    (async () => {
      try {
        // Settings (window W/S)
        try {
          const rs = await fetch(`${API_BASE}/api/settings`);
          if (rs.ok) {
            const s = await rs.json();
            const W = s?.deploy?.window?.W;
            const S = s?.deploy?.window?.S;
            if (typeof W === "number") setWindowSize(W);
            if (typeof S === "number") setStride(S);
          }
        } catch {
          // ignore
        }

        setModelsErr(null);
        const r = await fetch(`${API_BASE}/api/models/summary`);
        if (!r.ok) throw new Error(await r.text());
        const data = await r.json();
        const arr = Array.isArray(data?.models) ? data.models : [];
        setModels(arr);

        // Ensure selected model exists
        if (arr.length > 0) {
          const fallback = arr.find((m) => (m?.arch || "").toLowerCase() === "gcn") || arr[0];
          if (!selectedModel || !arr.some((m) => m.id === selectedModel)) {
            setSelectedModel(fallback.id);
            modelIdRef.current = fallback.id;
          }
        }
      } catch (e) {
        setModelsErr(String(e?.message || e));
        setModels([]);
      }
    })();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Update modelIdRef when selected model changes
  useEffect(() => {
    modelIdRef.current = selectedModel;
  }, [selectedModel]);

  // Probe backend
  useEffect(() => {
    (async () => {
      try {
        const r = await fetch(`${API_BASE}/api/health`);
        setBackendOk(r.ok);
      } catch {
        setBackendOk(false);
      }
    })();
  }, []);

  function ensureCanvasMatchesVideo() {
    const videoEl = videoRef.current;
    const canvasEl = canvasRef.current;
    if (!videoEl || !canvasEl) return;

    const vw = videoEl.videoWidth || 640;
    const vh = videoEl.videoHeight || 480;

    if (canvasEl.width !== vw) canvasEl.width = vw;
    if (canvasEl.height !== vh) canvasEl.height = vh;
  }

  async function startLive() {
    if (liveRunning) return;

    const videoEl = videoRef.current;
    if (!videoEl) return;

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
        modelComplexity: 1,
        smoothLandmarks: true,
        minDetectionConfidence: 0.5,
        minTrackingConfidence: 0.5,
      });

      pose.onResults(handlePoseResults);
      poseRef.current = pose;

      const loop = async () => {
        if (!liveFlagRef.current) return;
        if (videoEl.readyState >= 2 && poseRef.current) {
          try {
            await poseRef.current.send({ image: videoEl });
          } catch (err) {
            console.error("pose.send error", err);
          }
        }
        rafRef.current = requestAnimationFrame(loop);
      };

      loop();
    } catch (err) {
      console.error("Error starting live monitoring:", err);
      stopLiveInternal();
    }
  }

  function stopLiveInternal() {
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

    rawFramesRef.current = [];
    lastPoseTsRef.current = null;
    fpsDeltasRef.current = [];
    lastSentRef.current = 0;

    setFramesCollected(0);
    setFrameIndex(0);
    setPFall(null);
    setThreshold(null);
    setStatusText("SAFE");
    setTriageState("not_fall");
  }

  useEffect(() => {
    return () => stopLiveInternal();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  function handlePoseResults(results) {
    const canvasEl = canvasRef.current;
    if (!canvasEl) return;

    const ctx = canvasEl.getContext("2d");
    if (!ctx) return;

    ensureCanvasMatchesVideo();
    const w = canvasEl.width;
    const h = canvasEl.height;

    // Clear + dark bg
    ctx.save();
    ctx.clearRect(0, 0, w, h);
    ctx.fillStyle = "#020617";
    ctx.fillRect(0, 0, w, h);

    const landmarks = results.poseLandmarks;
    if (!landmarks || !landmarks.length) {
      ctx.restore();
      return;
    }

    drawConnectors(ctx, landmarks, POSE_CONNECTIONS, { color: "#22c55e", lineWidth: 2 });
    drawLandmarks(ctx, landmarks, { color: "#fbbf24", lineWidth: 1 });
    ctx.restore();

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

    // Estimate FPS from callback timing
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
          setStreamFps((prev) => (prev == null ? estFps : prev * 0.8 + estFps * 0.2));
        }
      }
    }
    lastPoseTsRef.current = tNow;

    // Push raw
    const raw = rawFramesRef.current;
    raw.push({ t: tNow, xy: xyFrame, conf: confFrame });

    // Keep ~10s max
    const maxRaw = Math.max(600, Math.ceil(targetFps * 12));
    if (raw.length > maxRaw) raw.splice(0, raw.length - maxRaw);

    // Window readiness (approx)
    const dtTarget = 1000 / targetFps;
    const endTs = raw[raw.length - 1].t;
    const startNeed = endTs - (windowSize - 1) * dtTarget;
    const collected =
      raw[0].t <= startNeed
        ? windowSize
        : Math.max(
            0,
            Math.min(windowSize, Math.floor((endTs - raw[0].t) / dtTarget) + 1)
          );

    setFramesCollected(collected);
    setFrameIndex((prev) => prev + 1);

    if (collected === windowSize) maybeSendWindow();
  }

  async function maybeSendWindow() {
    const now = performance.now();
    const minGapMs = Math.max(250, (stride / Math.max(1, targetFps)) * 1000);
    if (now - lastSentRef.current < minGapMs) return;
    lastSentRef.current = now;

    const raw = rawFramesRef.current;
    const win = buildResampledWindow(raw, targetFps, windowSize);
    if (!win) return;

    const modelId = modelIdRef.current;
    const arch = (selectedModelObj?.arch || "gcn").toLowerCase();

    try {
      const resp = await fetch(`${API_BASE}/api/monitor/predict_window`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          session_id: sessionIdRef.current,
          resident_id: 1,
          mode: arch,
          model_id: modelId,
          fps: streamFps || targetFps,
          timestamp_ms: Date.now(),
          use_mc: true,
          persist: false, // demo page does not persist by default
          xy: win.xy,
          conf: win.conf,
        }),
      });

      if (!resp.ok) {
        console.error("Backend error", await resp.text());
        return;
      }

      const data = await resp.json();
      const tri = (data?.triage_state || "not_fall").toLowerCase();
      setTriageState(tri);

      const mOut = data?.models?.[arch];
      const mu =
        mOut?.mu != null
          ? Number(mOut.mu)
          : mOut?.p_det != null
          ? Number(mOut.p_det)
          : null;

      const tau =
        mOut?.triage?.tau_high != null
          ? Number(mOut.triage.tau_high)
          : selectedModelObj?.tau_high != null
          ? Number(selectedModelObj.tau_high)
          : null;

      setPFall(mu);
      setThreshold(tau);

      if (tri === "fall") setStatusText("FALL DETECTED");
      else if (tri === "uncertain") setStatusText("UNCERTAIN");
      else setStatusText("SAFE");
    } catch (err) {
      console.error("Error calling /api/monitor/predict_window:", err);
    }
  }

  const circleColor =
    triageState === "fall"
      ? "#ef4444"
      : triageState === "uncertain"
      ? "#f59e0b"
      : "#10b981";

  return (
    <div style={{ padding: "1.5rem" }}>
      <h1 style={{ fontSize: "1.5rem", marginBottom: "0.25rem" }}>Live Monitor</h1>

      <p style={{ marginBottom: "0.75rem", color: "#4b5563" }}>
        Webcam → MediaPipe Pose → resample to model FPS → window (T={windowSize}) → FastAPI
      </p>

      {/* Backend status */}
      <div style={{ marginBottom: "0.75rem", fontSize: "0.9rem", color: "#4b5563" }}>
        <strong>Backend:</strong>{" "}
        {backendOk == null ? "unknown" : backendOk ? "OK" : "NOT REACHABLE"}{" "}
        <span style={{ color: "#9ca3af" }}>({API_BASE})</span>
      </div>

      {modelsErr && (
        <div style={{ marginBottom: "0.75rem", color: "#b45309" }}>
          Could not load /api/models/summary: {modelsErr}
        </div>
      )}

      {/* Controls */}
      <div
        style={{
          display: "flex",
          alignItems: "center",
          gap: "1rem",
          marginBottom: "1rem",
          flexWrap: "wrap",
        }}
      >
        <div>
          <div style={{ fontSize: "0.8rem", color: "#4b5563" }}>Model:</div>
          <select
            value={selectedModel}
            onChange={(e) => setSelectedModel(e.target.value)}
            disabled={liveRunning}
            style={{
              padding: "0.25rem 0.5rem",
              borderRadius: 9999,
              border: "1px solid #d1d5db",
            }}
          >
            {models.map((m) => (
              <option key={m.id} value={m.id}>
                {m.label || `${m.arch?.toUpperCase?.() || "MODEL"}: ${m.id}`}
              </option>
            ))}
          </select>
        </div>

        <button
          onClick={startLive}
          disabled={liveRunning || !selectedModel}
          style={{
            padding: "0.5rem 1rem",
            borderRadius: 9999,
            border: "none",
            backgroundColor: liveRunning ? "#9ca3af" : "#2563eb",
            color: "white",
            cursor: liveRunning ? "default" : "pointer",
          }}
        >
          {liveRunning ? "Live Running" : "▶ Start live"}
        </button>

        <button
          onClick={stopLiveInternal}
          style={{
            padding: "0.5rem 1rem",
            borderRadius: 9999,
            border: "1px solid #d1d5db",
            backgroundColor: "white",
            cursor: "pointer",
          }}
        >
          ⏹ Stop
        </button>

        <div style={{ fontSize: "0.85rem", color: "#4b5563" }}>
          <div>
            <strong>Target FPS:</strong> {displayedTargetFps}
          </div>
          <div>
            <strong>Stream FPS:</strong> {streamFps != null ? streamFps.toFixed(1) : "-"}
          </div>
        </div>
      </div>

      {/* Status + skeleton */}
      <div style={{ display: "flex", gap: "2rem", alignItems: "flex-start", flexWrap: "wrap" }}>
        <div
          style={{
            width: 200,
            height: 200,
            borderRadius: "9999px",
            border: `6px solid ${circleColor}`,
            display: "flex",
            flexDirection: "column",
            alignItems: "center",
            justifyContent: "center",
          }}
        >
          <div style={{ fontSize: "0.85rem", color: "#6b7280" }}>p(fall)</div>
          <div style={{ fontSize: "1.8rem", fontWeight: 600 }}>
            {pFall != null ? pFall.toFixed(2) : "-"}
          </div>
          <div
            style={{
              marginTop: "0.25rem",
              fontWeight: 600,
              color: circleColor,
              fontSize: "0.9rem",
              textAlign: "center",
            }}
          >
            {statusText}
          </div>
          <div style={{ marginTop: "0.25rem", fontSize: "0.75rem", color: "#6b7280" }}>
            Model: {selectedModelObj?.label || selectedModel || "—"}
          </div>
        </div>

        <div style={{ flex: 1, minWidth: 320 }}>
          <div style={{ marginBottom: "0.25rem", fontSize: "0.9rem", fontWeight: 500 }}>
            Skeleton preview
          </div>

          <canvas
            ref={canvasRef}
            width={640}
            height={480}
            style={{
              width: "100%",
              maxWidth: 640,
              aspectRatio: "4 / 3",
              backgroundColor: "#020617",
              borderRadius: "0.75rem",
              border: "1px solid #111827",
            }}
          />

          {/* IMPORTANT: keep video decoding by NOT using display:none */}
          <video
            ref={videoRef}
            playsInline
            muted
            style={{
              position: "absolute",
              left: "-9999px",
              top: "0",
              width: 640,
              height: 480,
              opacity: 0,
              pointerEvents: "none",
            }}
          />
        </div>
      </div>

      {/* Details */}
      <div
        style={{
          marginTop: "1.5rem",
          padding: "1rem 1.25rem",
          borderRadius: "0.75rem",
          border: "1px solid #e5e7eb",
        }}
      >
        <h2 style={{ fontSize: "1rem", marginBottom: "0.5rem" }}>Monitoring details</h2>
        <p style={{ margin: 0 }}>
          <strong>Collected frames:</strong> {framesCollected}/{windowSize}
        </p>
        <p style={{ margin: 0 }}>
          <strong>Frame index:</strong> {frameIndex}
        </p>
        <p style={{ margin: 0 }}>
          <strong>Decision threshold:</strong> {threshold != null ? threshold.toFixed(3) : "-"}
        </p>
      </div>
    </div>
  );
}
