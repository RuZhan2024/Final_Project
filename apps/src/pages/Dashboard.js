import React, { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import styles from "./Dashboard.module.css";

// Prefer env var, fallback to localhost
const API_BASE =
  typeof process !== "undefined" && process.env && process.env.REACT_APP_API_BASE
    ? process.env.REACT_APP_API_BASE
    : "http://localhost:8000";

function Dashboard() {
  const navigate = useNavigate();

  const [loading, setLoading] = useState(true);
  const [err, setErr] = useState(null);

  const [status, setStatus] = useState("normal"); // normal|alert
  const [fallsDetected, setFallsDetected] = useState(0);
  const [falseAlarms, setFalseAlarms] = useState(0);

  const [modelName, setModelName] = useState("—");
  const [monitoringEnabled, setMonitoringEnabled] = useState(true);
  const [latencyMs, setLatencyMs] = useState(null);
  const [apiOnline, setApiOnline] = useState(null); // null/true/false

  async function loadSummary() {
    try {
      setErr(null);
      const r = await fetch(`${API_BASE}/api/dashboard/summary`);
      if (!r.ok) throw new Error(await r.text());
      const data = await r.json();

      setStatus(data?.status || "normal");
      setFallsDetected(Number(data?.today?.falls_detected ?? 0));
      setFalseAlarms(Number(data?.today?.false_alarms ?? 0));

      setModelName(data?.system?.model_name || "—");
      setMonitoringEnabled(Boolean(data?.system?.monitoring_enabled ?? true));
      setLatencyMs(
        data?.system?.last_latency_ms == null ? null : Number(data.system.last_latency_ms)
      );
      setApiOnline(Boolean(data?.system?.api_online ?? true));
      setLoading(false);
    } catch (e) {
      setErr(String(e?.message || e));
      setLoading(false);
    }
  }

  useEffect(() => {
    loadSummary();
    const t = setInterval(loadSummary, 3000);
    return () => clearInterval(t);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  async function toggleMonitoring() {
    const next = !monitoringEnabled;
    setMonitoringEnabled(next); // optimistic
    try {
      const r = await fetch(`${API_BASE}/api/settings`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ monitoring_enabled: next }),
      });
      if (!r.ok) throw new Error(await r.text());
      // refresh to ensure UI matches DB
      loadSummary();
    } catch (e) {
      // revert on error
      setMonitoringEnabled(!next);
      console.error("Failed to update monitoring_enabled:", e);
    }
  }

  const statusLabel = status === "alert" ? "Alert" : "Normal";
  const toggleBg = monitoringEnabled ? "#4F46E5" : "#9CA3AF";
  const knobStyle = monitoringEnabled
    ? { right: 3, left: "auto" }
    : { left: 3, right: "auto" };

  const apiOk = apiOnline === true;
  const dotStyle = apiOnline == null
    ? {}
    : apiOk
      ? {}
      : { backgroundColor: "#EF4444", boxShadow: "0 0 0 4px #FEE2E2" };

  return (
    <div className={styles.container}>
      <h2 className={styles.pageTitle}>Dashboard</h2>

      {/* --- Top Section (Two Columns) --- */}
      <div className={styles.topRow}>
        {/* Card 1: Resident Status */}
        <div className={styles.card}>
          <h3 className={styles.cardTitle}>Resident Status</h3>
          <div className={styles.residentContent}>
            <span className={styles.statusText}>{statusLabel}</span>
            <button
              className={styles.actionButton}
              onClick={() => navigate("/monitor-demo")}
            >
              View Live Feed
            </button>
          </div>
        </div>

        {/* Card 2: Today's Summary */}
        <div className={styles.card}>
          <h3 className={styles.cardTitle}>Today's Summary</h3>
          <div className={styles.summaryGrid}>
            <div className={styles.summaryBox}>
              <span className={styles.bigNumber}>{fallsDetected}</span>
              <span className={styles.label}>Falls Detected</span>
            </div>
            <div className={styles.summaryBox}>
              <span className={styles.bigNumber}>{falseAlarms}</span>
              <span className={styles.label}>Confirmed False Alarms</span>
            </div>
          </div>
        </div>
      </div>

      {/* --- Bottom Section (System Status) --- */}
      <div className={styles.fullWidthCard}>
        <h3 className={styles.cardTitle}>System Status</h3>

        {err && (
          <div style={{ marginBottom: 12, color: "#B45309" }}>
            Backend error: {err}
          </div>
        )}

        <div className={styles.systemGrid}>
          {/* Item 1: Model */}
          <div className={styles.systemBox}>
            <span className={styles.boxLabel}>Model</span>
            <div className={styles.centerContent}>
              <span className={styles.tag}>{modelName}</span>
            </div>
          </div>

          {/* Item 2: Monitoring Toggle */}
          <div className={styles.systemBox}>
            <span className={styles.boxLabel}>Monitoring</span>
            <div className={styles.centerContent}>
              {/* CSS-only Toggle Switch representation */}
              <div
                className={styles.toggleSwitch}
                onClick={toggleMonitoring}
                title="Toggle monitoring"
                style={{ backgroundColor: toggleBg }}
              >
                <div className={styles.toggleKnob} style={knobStyle}></div>
              </div>
            </div>
          </div>

          {/* Item 3: Latency */}
          <div className={styles.systemBox}>
            <span className={styles.boxLabel}>Latency</span>
            <div className={styles.centerContent}>
              <span className={styles.valueText}>
                {latencyMs == null ? "—" : `${Math.round(latencyMs)}ms`}
              </span>
            </div>
          </div>

          {/* Item 4: API Health */}
          <div className={styles.systemBox}>
            <span className={styles.boxLabel}>API Health</span>
            <div className={styles.centerContent}>
              <div className={styles.statusIndicator}>
                <span className={styles.dot} style={dotStyle}></span>
                <span>{apiOnline == null ? "Unknown" : apiOk ? "Online" : "Offline"}</span>
              </div>
            </div>
          </div>
        </div>

        {loading && (
          <div style={{ marginTop: 12, color: "#6B7280", fontSize: "0.9rem" }}>
            Loading…
          </div>
        )}
      </div>
    </div>
  );
}

export default Dashboard;
