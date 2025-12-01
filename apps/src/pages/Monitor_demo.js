// src/pages/Monitor.jsx
import React, { useState, useEffect, useRef } from "react";

const API_BASE = "http://localhost:8000"; // adjust if your backend is on a different port

const MODEL_OPTIONS = [
  { id: "le2i", label: "LE2I model", endpoint: "/api/demo/le2i_fall" },
  { id: "urfd", label: "URFD model", endpoint: "/api/demo/urfd_fall" },
  {
    id: "caucafall",
    label: "CAUCAFall model",
    endpoint: "/api/demo/caucafall_fall",
  },
];

function Monitor() {
  const [modelId, setModelId] = useState("le2i");

  const [points, setPoints] = useState([]);
  const [threshold, setThreshold] = useState(null);
  const [currentIndex, setCurrentIndex] = useState(0);
  const [playing, setPlaying] = useState(false);

  const timerRef = useRef(null);

  const current = points[currentIndex] || null;
  const isFall = current ? current.fall : false;
  const prob = current ? current.p_fall : 0;

  const statusText = isFall ? "FALL DETECTED" : "SAFE";
  const statusColor = isFall ? "#dc2626" : "#16a34a"; // red / green

  async function handleStartDemo() {
    // stop previous playback
    if (timerRef.current) {
      clearInterval(timerRef.current);
      timerRef.current = null;
    }

    setPlaying(false);
    setCurrentIndex(0);
    setPoints([]);
    setThreshold(null);

    const selectedModel = MODEL_OPTIONS.find((m) => m.id === modelId);
    if (!selectedModel) {
      alert(`Unknown model: ${modelId}`);
      return;
    }

    try {
      const resp = await fetch(`${API_BASE}${selectedModel.endpoint}`, {
        method: "POST",
      });

      if (!resp.ok) {
        const text = await resp.text();
        throw new Error(text || `HTTP ${resp.status}`);
      }

      const data = await resp.json();

      const pts = data.points || [];
      const thr = data.threshold;

      setPoints(pts);
      setThreshold(thr);

      if (pts.length === 0) {
        alert(`No points returned from ${selectedModel.endpoint}`);
        return;
      }

      setPlaying(true);

      const fps = data.fps || 5;
      const intervalMs = 1000 / fps;

      timerRef.current = setInterval(() => {
        setCurrentIndex((prev) => {
          const next = prev + 1;
          if (next >= pts.length) {
            // stop at end
            if (timerRef.current) {
              clearInterval(timerRef.current);
              timerRef.current = null;
            }
            setPlaying(false);
            return prev;
          }
          return next;
        });
      }, intervalMs);
    } catch (err) {
      console.error("Demo error:", err);
      alert("Error starting demo. Check console / backend.");
    }
  }

  // cleanup on unmount
  useEffect(() => {
    return () => {
      if (timerRef.current) {
        clearInterval(timerRef.current);
      }
    };
  }, []);

  const selectedLabel =
    MODEL_OPTIONS.find((m) => m.id === modelId)?.label || modelId;

  return (
    <div style={{ padding: "1.5rem" }}>
      <h1 style={{ fontSize: "1.5rem", marginBottom: "0.5rem" }}>
        Live Monitor
      </h1>

      <p style={{ marginBottom: "1rem", color: "#4b5563" }}>
        Replays a pre-recorded sequence from the selected dataset through the
        trained TCN model and shows the fall probability over time.
      </p>

      {/* Model selector + button */}
      <div
        style={{
          display: "flex",
          gap: "1rem",
          alignItems: "center",
          marginBottom: "1rem",
          flexWrap: "wrap",
        }}
      >
        <label>
          Model:&nbsp;
          <select
            value={modelId}
            onChange={(e) => {
              // stop current playback when switching model
              if (timerRef.current) {
                clearInterval(timerRef.current);
                timerRef.current = null;
              }
              setPlaying(false);
              setCurrentIndex(0);
              setPoints([]);
              setThreshold(null);
              setModelId(e.target.value);
            }}
          >
            {MODEL_OPTIONS.map((m) => (
              <option key={m.id} value={m.id}>
                {m.label}
              </option>
            ))}
          </select>
        </label>

        <button
          onClick={handleStartDemo}
          disabled={playing}
          style={{
            padding: "0.5rem 1rem",
            borderRadius: 9999,
            border: "none",
            backgroundColor: playing ? "#9ca3af" : "#2563eb",
            color: "white",
            cursor: playing ? "default" : "pointer",
          }}
        >
          {playing ? "Playing..." : `▶ Play ${selectedLabel} Demo`}
        </button>
      </div>

      <div
        style={{
          display: "flex",
          gap: "1rem",
          alignItems: "center",
          marginBottom: "1rem",
          flexWrap: "wrap",
        }}
      >
        {/* Status circle */}
        <div
          style={{
            width: 140,
            height: 140,
            borderRadius: "9999px",
            border: `4px solid ${statusColor}`,
            display: "flex",
            flexDirection: "column",
            alignItems: "center",
            justifyContent: "center",
          }}
        >
          <div style={{ fontSize: "0.85rem", color: "#6b7280" }}>p(fall)</div>
          <div style={{ fontSize: "1.8rem", fontWeight: 600 }}>
            {points.length ? prob.toFixed(2) : "-"}
          </div>
          <div
            style={{
              marginTop: "0.25rem",
              fontWeight: 600,
              color: statusColor,
              fontSize: "0.9rem",
            }}
          >
            {points.length ? statusText : "READY"}
          </div>
        </div>

        {/* Info panel */}
        <div style={{ flex: 1, minWidth: 220 }}>
          <div style={{ marginBottom: "0.25rem" }}>
            <strong>Dataset model:</strong> {selectedLabel}
          </div>
          <div style={{ marginBottom: "0.25rem" }}>
            <strong>Frame:</strong>{" "}
            {points.length ? current.t : "-"} /{" "}
            {points.length ? points.length - 1 : "-"}
          </div>
          <div style={{ marginBottom: "0.25rem" }}>
            <strong>Threshold:</strong>{" "}
            {threshold != null ? threshold.toFixed(2) : "-"}
          </div>
          <div style={{ fontSize: "0.875rem", color: "#4b5563" }}>
            When p(fall) crosses the selected operating-point threshold, the
            status turns red and an alert would be logged in a full deployment.
          </div>
        </div>
      </div>

      {/* Debug log of all time steps */}
      {points.length > 0 && (
        <div
          style={{
            borderTop: "1px solid #e5e7eb",
            paddingTop: "0.75rem",
            fontSize: "0.8rem",
           
            overflowY: "auto",
          }}
        >
          {points.map((p, idx) => (
            <div
              key={p.t}
              style={{
                display: "flex",
                justifyContent: "space-between",
                color: idx === currentIndex ? "#111827" : "#6b7280",
                fontWeight: idx === currentIndex ? 600 : 400,
              }}
            >
              <span>t={p.t}</span>
              <span>p_fall={p.p_fall.toFixed(3)}</span>
              <span>{p.fall ? "FALL" : "safe"}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

export default Monitor;
