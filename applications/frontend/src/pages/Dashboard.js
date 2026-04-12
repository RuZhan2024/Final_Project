import React, { useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import styles from "./Dashboard.module.css";

import { useMonitoring } from "../monitoring/MonitoringContext";
import { modelCodeToLabel } from "../lib/modelCodes";
import { useDashboardSummary } from "./dashboard/hooks/useDashboardSummary";

function Dashboard() {
  const navigate = useNavigate();
  const [statusMsg, setStatusMsg] = useState("");
  const [localErr, setLocalErr] = useState("");

  const { monitoringOn, toggleMonitoringOn, error: monitoringErr, settings, apiBase } = useMonitoring();
  const { data, loading, error } = useDashboardSummary(apiBase);

  const status = data?.status || "normal";
  const fallsDetected = Number(data?.today?.falls_detected ?? 0);
  const falseAlarms = Number(data?.today?.false_alarms ?? 0);
  const latencyMs = data?.system?.last_latency_ms == null ? null : Number(data.system.last_latency_ms);
  const apiOnline = data?.system?.api_online == null ? null : Boolean(data.system.api_online);

  const modelLabel = useMemo(() => {
    const activeModelCode = settings?.system?.active_model_code;
    if (activeModelCode) return modelCodeToLabel(activeModelCode);
    return data?.system?.model_name || "—";
  }, [settings, data]);


  const statusLabel = status === "alert" ? "Alert" : "Normal";
  const toggleBg = monitoringOn ? "#4F46E5" : "#9CA3AF";
  const knobStyle = monitoringOn
    ? { right: 3, left: "auto" }
    : { left: 3, right: "auto" };

  const apiOk = apiOnline === true;
  const dotStyle = apiOnline == null
    ? {}
    : apiOk
      ? {}
      : { backgroundColor: "#EF4444", boxShadow: "0 0 0 4px #FEE2E2" };

  const showToast = (msg) => {
    setStatusMsg(msg);
    window.setTimeout(() => setStatusMsg(""), 2800);
  };

  const onToggleMonitoring = async () => {
    setLocalErr("");
    const prev = Boolean(monitoringOn);
    try {
      const next = await toggleMonitoringOn();
      if (next === prev) {
        showToast(
          next
            ? "Monitoring remains ON. If you expected OFF, check monitor session state."
            : "Monitoring remains OFF. If you expected ON, check camera permission and API status."
        );
        return;
      }
      showToast(
        next
          ? "Monitoring enabled from Dashboard. Live detection is now active."
          : "Monitoring disabled from Dashboard. Live detection is now paused."
      );
    } catch (e) {
      setLocalErr(`Could not change monitoring state. ${String(e?.message || e)}`);
    }
  };

  return (
    <div className={styles.container}>
      <h2 className={styles.pageTitle}>Dashboard</h2>
      {statusMsg && <div className={`${styles.toast} ${styles.toastSuccess}`}>{statusMsg}</div>}
      {localErr && <div className={`${styles.toast} ${styles.toastError}`}>{localErr}</div>}

      {/* --- Top Section (Two Columns) --- */}
      <div className={styles.topRow}>
        {/* Card 1: Resident Status */}
        <div className={styles.card}>
          <h3 className={styles.cardTitle}>Resident Status</h3>
          <div className={styles.residentContent}>
            <span className={styles.statusText}>{statusLabel}</span>
            <button
              className={styles.actionButton}
              onClick={() => navigate("/monitor")}
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

        {error && (
          <div className={styles.statusNotice}>
            Backend error: {error}
          </div>
        )}

        {monitoringErr && (
          <div className={styles.statusNotice}>
            Settings error: {monitoringErr}
          </div>
        )}

        <div className={styles.systemGrid}>
          {/* Item 1: Model */}
          <div className={styles.systemBox}>
            <span className={styles.boxLabel}>Model</span>
            <div className={styles.centerContent}>
              <span className={styles.tag}>{modelLabel}</span>
            </div>
          </div>

          {/* Item 2: Monitoring Toggle */}
          <div className={styles.systemBox}>
            <span className={styles.boxLabel}>Monitoring</span>
            <div className={styles.centerContent}>
              {/* CSS-only Toggle Switch representation */}
              <div
                className={styles.toggleSwitch}
                onClick={onToggleMonitoring}
                title="Enable or disable live monitoring"
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
          <div className={styles.mutedNotice}>
            Loading…
          </div>
        )}
      </div>
    </div>
  );
}

export default Dashboard;
