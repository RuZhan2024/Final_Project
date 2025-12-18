// src/pages/Monitor.jsx
import React, { useEffect, useMemo, useRef, useState } from "react";
import * as mpPose from "@mediapipe/pose";
import * as drawingUtils from "@mediapipe/drawing_utils";

const { POSE_CONNECTIONS } = mpPose;
const { drawConnectors, drawLandmarks } = drawingUtils;

// Prefer env var, fallback to localhost
const API_BASE = (typeof process !== "undefined" && process.env && process.env.REACT_APP_API_BASE)
  ? process.env.REACT_APP_API_BASE
  : "http://localhost:8000";


const WINDOW_SIZE = 48;
const NUM_JOINTS = 33;

// If server doesn't provide fps, use these defaults
const MODEL_TARGET_FPS_FALLBACK = {
  le2i: 25,
  urfd: 30,
  caucafall: 23,
};

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
  // Models come from backend (/api/models/summary)
  const [models, setModels] = useState([]);
  const [modelsErr, setModelsErr] = useState(null);

  const [selectedModel, setSelectedModel] = useState("le2i");
  const [liveRunning, setLiveRunning] = useState(false);

  const [backendOk, setBackendOk] = useState(null); // null/true/false
  const [pFall, setPFall] = useState(null);
  const [threshold, setThreshold] = useState(null);
  const [framesCollected, setFramesCollected] = useState(0);
  const [frameIndex, setFrameIndex] = useState(0);
  const [statusText, setStatusText] = useState("SAFE");

  const [streamFps, setStreamFps] = useState(null);
  const [targetFps, setTargetFps] = useState(MODEL_TARGET_FPS_FALLBACK.le2i);

  const videoRef = useRef(null);
  const canvasRef = useRef(null);

  const poseRef = useRef(null);
  const streamRef = useRef(null);
  const rafRef = useRef(null);
  const liveFlagRef = useRef(false);

  const rawFramesRef = useRef([]); // [{t,xy,conf}]
  const lastPoseTsRef = useRef(null);
  const fpsDeltasRef = useRef([]);

  const modelIdRef = useRef("le2i");
  const lastSentRef = useRef(0);

  const selectedModelObj = useMemo(
    () => models.find((m) => m.id === selectedModel) || null,
    [models, selectedModel]
  );

  // Load model list from backend (so frontend always matches server)
  useEffect(() => {
    (async () => {
      try {
        setModelsErr(null);
        const r = await fetch(`${API_BASE}/api/models/summary`);
        if (!r.ok) throw new Error(await r.text());
        const data = await r.json();
        const arr = Array.isArray(data?.models) ? data.models : [];
        setModels(arr);

        // Ensure selected model exists
        if (arr.length > 0 && !arr.some((m) => m.id === selectedModel)) {
          setSelectedModel(arr[0].id);
        }
      } catch (e) {
        setModelsErr(String(e?.message || e));
        // fallback to a safe set that matches your backend payload type
        setModels([
          { id: "le2i", label: "TCN trained on LE2I" },
          { id: "urfd", label: "TCN trained on URFD" },
          { id: "caucafall", label: "TCN trained on CAUCAFall" },
        ]);
      }
    })();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Sync refs + target fps
  useEffect(() => {
    modelIdRef.current = selectedModel;
    const fallback = MODEL_TARGET_FPS_FALLBACK[selectedModel] ?? 30;
    setTargetFps(fallback);
  }, [selectedModel]);

  // Cleanup on unmount
  useEffect(() => {
    return () => stopLiveInternal();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  function resetState() {
    setPFall(null);
    setThreshold(null);
    setFramesCollected(0);
    setFrameIndex(0);
    setStatusText("SAFE");
    setStreamFps(null);

    rawFramesRef.current = [];
    lastPoseTsRef.current = null;
    fpsDeltasRef.current = [];
  }

  async function pingBackend() {
    try {
      const r = await fetch(`${API_BASE}/api/health`);
      setBackendOk(r.ok);
      return r.ok;
    } catch {
      setBackendOk(false);
      return false;
    }
  }

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

    resetState();
    setLiveRunning(true);
    liveFlagRef.current = true;

    const ok = await pingBackend();
    if (!ok) {
      console.error("Backend not reachable at", API_BASE);
      // Still allow pose drawing even if backend is down
    }

    try {
      const videoEl = videoRef.current;
      const canvasEl = canvasRef.current;
      if (!videoEl || !canvasEl) {
        console.error("Video or canvas ref missing");
        stopLiveInternal();
        return;
      }

      // Ask camera for an ideal FPS near target
      const tgtFps = MODEL_TARGET_FPS_FALLBACK[modelIdRef.current] ?? 30;

      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 640, height: 480, frameRate: { ideal: tgtFps } },
        audio: false,
      });

      streamRef.current = stream;
      videoEl.srcObject = stream;

      // Wait for metadata so videoWidth/videoHeight are valid
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
        locateFile: (file) =>
          // If you get 404s here, pin a version:
          // `https://cdn.jsdelivr.net/npm/@mediapipe/pose@0.5.1675469404/${file}`
          `https://cdn.jsdelivr.net/npm/@mediapipe/pose/${file}`,
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

    if (rafRef.current) {
      cancelAnimationFrame(rafRef.current);
      rafRef.current = null;
    }

    if (poseRef.current && poseRef.current.close) {
      try {
        poseRef.current.close();
      } catch {
        // ignore
      }
    }
    poseRef.current = null;

    if (streamRef.current) {
      streamRef.current.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
    }

    resetState();
  }

  function handlePoseResults(results) {
    const canvasEl = canvasRef.current;
    if (!canvasEl) return;

    const ctx = canvasEl.getContext("2d");
    if (!ctx) return;

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
    const maxRaw = 600;
    if (raw.length > maxRaw) raw.splice(0, raw.length - maxRaw);

    // Window readiness
    const modelId = modelIdRef.current || "le2i";
    const tgt = MODEL_TARGET_FPS_FALLBACK[modelId] ?? 30;
    const dtTarget = 1000 / tgt;
    const endTs = raw[raw.length - 1].t;
    const startNeed = endTs - (WINDOW_SIZE - 1) * dtTarget;

    const collected = raw[0].t <= startNeed
      ? WINDOW_SIZE
      : Math.max(0, Math.min(WINDOW_SIZE, Math.floor((endTs - raw[0].t) / dtTarget) + 1));

    setFramesCollected(collected);
    setFrameIndex((prev) => prev + 1);

    if (collected === WINDOW_SIZE) maybeSendWindow();
  }

  async function maybeSendWindow() {
    const now = performance.now();
    if (now - lastSentRef.current < 500) return; // ~2 req/sec
    lastSentRef.current = now;

    const modelId = modelIdRef.current || "le2i";
    const tgtFps = MODEL_TARGET_FPS_FALLBACK[modelId] ?? 30;

    const raw = rawFramesRef.current;
    const win = buildResampledWindow(raw, tgtFps, WINDOW_SIZE);
    if (!win) return;

    try {
      const resp = await fetch(`${API_BASE}/api/monitor/predict_window`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ model_id: modelId, xy: win.xy, conf: win.conf }),
      });

      if (!resp.ok) {
        console.error("Backend error", await resp.text());
        return;
      }

      const data = await resp.json();
      const pf = data.p_fall;
      const thr = data.threshold;

      setPFall(pf);
      setThreshold(thr);
      setStatusText(pf >= thr ? "FALL DETECTED" : "SAFE");
    } catch (err) {
      console.error("Error calling /api/monitor/predict_window:", err);
    }
  }

  const circleColor = statusText === "FALL DETECTED" ? "#dc2626" : "#16a34a";
  const displayedTargetFps = targetFps ?? 30;

  return (
    <div style={{ padding: "1.5rem" }}>
      <h1 style={{ fontSize: "1.5rem", marginBottom: "0.25rem" }}>
        Live Monitor
      </h1>

      <p style={{ marginBottom: "0.75rem", color: "#4b5563" }}>
        Webcam → MediaPipe Pose → resample to model FPS → window (T={WINDOW_SIZE}) → FastAPI
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
                {m.label || m.id}
              </option>
            ))}
          </select>
        </div>

        <button
          onClick={startLive}
          disabled={liveRunning}
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
            <strong>Stream FPS:</strong>{" "}
            {streamFps != null ? streamFps.toFixed(1) : "-"}
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
          <div style={{ marginTop: "0.25rem", fontWeight: 600, color: circleColor, fontSize: "0.9rem", textAlign: "center" }}>
            {statusText}
          </div>
          <div style={{ marginTop: "0.25rem", fontSize: "0.75rem", color: "#6b7280" }}>
            Model: {selectedModelObj?.label || selectedModel}
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
        <h2 style={{ fontSize: "1rem", marginBottom: "0.5rem" }}>
          Monitoring details
        </h2>
        <p style={{ margin: 0 }}>
          <strong>Collected frames:</strong> {framesCollected}/{WINDOW_SIZE}
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
