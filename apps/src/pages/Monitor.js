// src/pages/Monitor.jsx
import React, { useState, useRef, useEffect } from "react";
import * as mpPose from "@mediapipe/pose";
import * as drawingUtils from "@mediapipe/drawing_utils";

const { POSE_CONNECTIONS } = mpPose;
const { drawConnectors, drawLandmarks } = drawingUtils;

const API_BASE = "http://localhost:8000"; // FastAPI backend
const WINDOW_SIZE = 48;
const NUM_JOINTS = 33;

const MODEL_OPTIONS = [
  { id: "le2i", label: "LE2I TCN" },
  { id: "urfd", label: "URFD TCN" },
  { id: "caucafall", label: "CAUCAFall TCN" },
];

function Monitor() {
  const [selectedModel, setSelectedModel] = useState("le2i");
  const [liveRunning, setLiveRunning] = useState(false);

  const [pFall, setPFall] = useState(null);
  const [threshold, setThreshold] = useState(null);
  const [framesCollected, setFramesCollected] = useState(0);
  const [frameIndex, setFrameIndex] = useState("-");
  const [statusText, setStatusText] = useState("SAFE");

  const videoRef = useRef(null);        // hidden video
  const canvasRef = useRef(null);       // visible skeleton canvas
  const poseRef = useRef(null);         // MediaPipe Pose instance
  const streamRef = useRef(null);       // MediaStream
  const rafRef = useRef(null);          // rAF id
  const liveFlagRef = useRef(false);    // so loop can see state
  const windowBufferRef = useRef([]);   // rolling T=48 frames
  const modelIdRef = useRef("le2i");    // latest selected model
  const lastSentRef = useRef(0);        // throttle backend calls

  useEffect(() => {
    modelIdRef.current = selectedModel;
  }, [selectedModel]);

  // cleanup on unmount
  useEffect(() => {
    return () => {
      stopLiveInternal();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  function resetState() {
    setPFall(null);
    setThreshold(null);
    setFramesCollected(0);
    setFrameIndex("-");
    setStatusText("SAFE");
    windowBufferRef.current = [];
  }

  async function startLive() {
    if (liveRunning) return;

    resetState();
    setLiveRunning(true);
    liveFlagRef.current = true;

    try {
      const videoEl = videoRef.current;
      const canvasEl = canvasRef.current;
      if (!videoEl || !canvasEl) {
        console.error("Video or canvas ref missing");
        setLiveRunning(false);
        liveFlagRef.current = false;
        return;
      }

      // 1) getUserMedia
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 640, height: 480 },
        audio: false,
      });
      streamRef.current = stream;
      videoEl.srcObject = stream;
      await videoEl.play();

      // 2) MediaPipe Pose (note: use namespace import)
      const pose = new mpPose.Pose({
        locateFile: (file) =>
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

      // 3) pose loop
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
      poseRef.current.close();
    }
    poseRef.current = null;

    if (streamRef.current) {
      streamRef.current.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
    }

    resetState();
  }

  function handleStopClick() {
    stopLiveInternal();
  }

  function handlePoseResults(results) {
    const canvasEl = canvasRef.current;
    if (!canvasEl) return;

    const ctx = canvasEl.getContext("2d");
    const w = canvasEl.width;
    const h = canvasEl.height;

    // Clear + dark background
    ctx.save();
    ctx.clearRect(0, 0, w, h);
    ctx.fillStyle = "#020617";
    ctx.fillRect(0, 0, w, h);

    const landmarks = results.poseLandmarks;
    if (!landmarks || !landmarks.length) {
      ctx.restore();
      return;
    }

    // ---- draw skeleton (no extra scaling, drawing_utils handles it) ----
    drawConnectors(ctx, landmarks, POSE_CONNECTIONS, {
      color: "#22c55e",
      lineWidth: 2,
    });
    drawLandmarks(ctx, landmarks, { color: "#fbbf24", lineWidth: 1 });
    ctx.restore();

    // ---- build one frame for the window ----
    const xyFrame = [];
    const confFrame = [];
    for (let i = 0; i < NUM_JOINTS; i++) {
      const lm = landmarks[i];
      if (!lm) {
        xyFrame.push([0, 0]);
        confFrame.push(0);
      } else {
        xyFrame.push([lm.x, lm.y]); // normalized
        confFrame.push(
          typeof lm.visibility === "number" ? lm.visibility : 1.0
        );
      }
    }

    const buf = windowBufferRef.current;
    buf.push({ xy: xyFrame, conf: confFrame });
    if (buf.length > WINDOW_SIZE) buf.shift();

    setFramesCollected(buf.length);
    setFrameIndex((prev) =>
      typeof prev === "number" ? prev + 1 : 0
    );

    // send to backend once we have a full window (throttled)
    if (buf.length === WINDOW_SIZE) {
      maybeSendWindow();
    }
  }

  async function maybeSendWindow() {
    const now = performance.now();
    if (now - lastSentRef.current < 500) return; // at most ~2 req/sec
    lastSentRef.current = now;

    const modelId = modelIdRef.current || "le2i";
    const buf = windowBufferRef.current;
    if (buf.length < WINDOW_SIZE) return;

    const xy = buf.map((f) => f.xy);     // [T,33,2]
    const conf = buf.map((f) => f.conf); // [T,33]

    try {
      const resp = await fetch(`${API_BASE}/api/monitor/predict_window`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ model_id: modelId, xy, conf }),
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

  const circleColor =
    statusText === "FALL DETECTED" ? "#dc2626" : "#16a34a";

  return (
    <div style={{ padding: "1.5rem" }}>
      <h1 style={{ fontSize: "1.5rem", marginBottom: "0.25rem" }}>
        Live Monitor
      </h1>
      <p style={{ marginBottom: "1rem", color: "#4b5563" }}>
        Webcam → MediaPipe Pose → skeleton window (T=48) → TCN model on
        FastAPI backend. Only the skeleton (no RGB video) is shown for privacy.
      </p>

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
            {MODEL_OPTIONS.map((m) => (
              <option key={m.id} value={m.id}>
                {m.label}
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
          onClick={handleStopClick}
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
      </div>

      {/* Status + skeleton preview */}
      <div
        style={{
          display: "flex",
          gap: "2rem",
          alignItems: "flex-start",
          flexWrap: "wrap",
        }}
      >
        {/* Status circle */}
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
          <div
            style={{
              marginTop: "0.25rem",
              fontSize: "0.75rem",
              color: "#6b7280",
            }}
          >
            Live skeleton window
          </div>
        </div>

        {/* Skeleton canvas */}
        <div style={{ flex: 1, minWidth: 320 }}>
          <div
            style={{
              marginBottom: "0.25rem",
              fontSize: "0.9rem",
              fontWeight: 500,
            }}
          >
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
          {/* hidden video */}
          <video
            ref={videoRef}
            playsInline
            muted
            style={{ display: "none" }}
          />
        </div>
      </div>

      {/* Monitoring details */}
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
          <strong>Selected model:</strong>{" "}
          {
            MODEL_OPTIONS.find((m) => m.id === selectedModel)?.label ??
            selectedModel
          }
        </p>
        <p style={{ margin: 0 }}>
          <strong>Mode:</strong> Live (webcam + MediaPipe Pose)
        </p>
        <p style={{ margin: 0 }}>
          <strong>Frame index:</strong> {String(frameIndex)}
        </p>
        <p style={{ margin: 0 }}>
          <strong>Collected frames (live window):</strong>{" "}
          {framesCollected}/{WINDOW_SIZE}
        </p>
        <p style={{ margin: 0 }}>
          <strong>Decision threshold:</strong>{" "}
          {threshold != null ? threshold.toFixed(2) : "-"}
        </p>
        <p
          style={{
            marginTop: "0.5rem",
            fontSize: "0.85rem",
            color: "#4b5563",
          }}
        >
          Every {WINDOW_SIZE} frames, the page builds a pose window (33 joints ×
          2D) and periodically sends it to the FastAPI backend. When the fall
          probability crosses the threshold, the status flips from SAFE to{" "}
          <span style={{ color: "#dc2626", fontWeight: 600 }}>
            FALL DETECTED
          </span>
          .
        </p>
      </div>
    </div>
  );
}

export default Monitor;
