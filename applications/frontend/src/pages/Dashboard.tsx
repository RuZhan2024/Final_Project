import React, { useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";

import type { DashboardSummary } from "../features/dashboard/types";
import { modelCodeToLabel } from "../lib/modelCodes";
import { useMonitoring } from "../monitoring/MonitoringContext";
import { useDashboardSummary } from "./dashboard/hooks/useDashboardSummary";

import styles from "./Dashboard.module.css";

const dashboardHelp = {
  residentStatus: "Shows whether the resident is currently in a normal state or whether the system has an active fall alert.",
  fallsDetected: "The number of fall events detected by the system today. These may still need human review.",
  confirmedFalls: "The number of today's detected fall events that have been reviewed and confirmed as real falls.",
  model: "The fall-detection model currently selected for monitoring.",
  monitoring: "Shows whether live monitoring is enabled. Turning it off pauses live fall detection.",
  latency: "The most recent backend prediction response time. Lower values mean faster system response.",
  apiHealth: "Shows whether the frontend can currently reach the backend service.",
};

function InfoHint({ text }: { text: string }) {
  return (
    <span className={styles.infoHint} tabIndex={0} title={text} aria-label={text} data-tooltip={text}>
      i
    </span>
  );
}

/**
 * Dashboard page showing status cards and a lightweight monitoring toggle.
 */
function Dashboard() {
  const navigate = useNavigate();
  const [statusMsg, setStatusMsg] = useState("");
  const [localErr, setLocalErr] = useState("");

  const { monitoringOn, toggleMonitoringOn, error: monitoringErr, settings, apiBase } = useMonitoring();
  const { data, loading, error } = useDashboardSummary(apiBase);

  const summary = (data || {}) as DashboardSummary & Record<string, unknown>;
  const status = summary.status || "normal";
  const fallsDetected = Number(summary.today?.falls_detected ?? 0);
  const confirmedFalls = Number(summary.today?.confirmed_falls ?? 0);
  const latencyMs = summary.system?.last_latency_ms == null ? null : Number(summary.system.last_latency_ms);
  const apiOnline = summary.system?.api_online == null ? null : Boolean(summary.system.api_online);

  const modelLabel = useMemo(() => {
    const activeModelCode = settings?.system?.active_model_code;
    // Prefer the persisted settings selection; summary.model_name is only a fallback for older responses.
    if (activeModelCode) return modelCodeToLabel(activeModelCode);
    return summary.system?.model_name || "—";
  }, [settings?.system?.active_model_code, summary.system?.model_name]);

  const statusLabel = status === "alert" ? "Alert" : "Normal";
  const toggleBg = monitoringOn ? "#4F46E5" : "#9CA3AF";
  const knobStyle = monitoringOn ? { right: 3, left: "auto" } : { left: 3, right: "auto" };

  const apiOk = apiOnline === true;
  const dotStyle =
    apiOnline == null
      ? {}
      : apiOk
        ? {}
        : { backgroundColor: "#EF4444", boxShadow: "0 0 0 4px #FEE2E2" };

  const showToast = (message: string) => {
    setStatusMsg(message);
    window.setTimeout(() => setStatusMsg(""), 2800);
  };

  const onToggleMonitoring = async (): Promise<void> => {
    setLocalErr("");
    const previous = Boolean(monitoringOn);
    try {
      const next = await toggleMonitoringOn();
      if (next === previous) {
        // The provider returns the effective runtime state, which may differ if startup/shutdown failed.
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
    } catch (err) {
      setLocalErr(`Could not change monitoring state. ${String((err as Error)?.message || err)}`);
    }
  };

  return (
    <div className={styles.container}>
      <h2 className={styles.pageTitle}>Dashboard</h2>
      {statusMsg && <div className={`${styles.toast} ${styles.toastSuccess}`}>{statusMsg}</div>}
      {localErr && <div className={`${styles.toast} ${styles.toastError}`}>{localErr}</div>}

      <div className={styles.topRow}>
        <div className={styles.card}>
          <h3 className={`${styles.cardTitle} ${styles.titleWithHint}`}>
            <span>Resident Status</span>
            <InfoHint text={dashboardHelp.residentStatus} />
          </h3>
          <div className={styles.residentContent}>
            <span className={styles.statusText}>{statusLabel}</span>
            <button className={styles.actionButton} onClick={() => navigate("/monitor")}>
              View Live Feed
            </button>
          </div>
        </div>

        <div className={styles.card}>
          <h3 className={styles.cardTitle}>Today's Summary</h3>
          <div className={styles.summaryGrid}>
            <div className={styles.summaryBox}>
              <span className={styles.bigNumber}>{fallsDetected}</span>
              <span className={styles.labelWithHint}>
                <span className={styles.label}>Falls Detected</span>
                <InfoHint text={dashboardHelp.fallsDetected} />
              </span>
            </div>
            <div className={styles.summaryBox}>
              <span className={styles.bigNumber}>{confirmedFalls}</span>
              <span className={styles.labelWithHint}>
                <span className={styles.label}>Confirmed Falls</span>
                <InfoHint text={dashboardHelp.confirmedFalls} />
              </span>
            </div>
          </div>
        </div>
      </div>

      <div className={styles.fullWidthCard}>
        <h3 className={styles.cardTitle}>System Status</h3>

        {error && <div className={styles.statusNotice}>Backend error: {error}</div>}
        {monitoringErr && <div className={styles.statusNotice}>Settings error: {monitoringErr}</div>}

        <div className={styles.systemGrid}>
          <div className={styles.systemBox}>
            <span className={styles.boxLabel}>
              <span>Model</span>
              <InfoHint text={dashboardHelp.model} />
            </span>
            <div className={styles.centerContent}>
              <span className={styles.tag}>{modelLabel}</span>
            </div>
          </div>

          <div className={styles.systemBox}>
            <span className={styles.boxLabel}>
              <span>Monitoring</span>
              <InfoHint text={dashboardHelp.monitoring} />
            </span>
            <div className={styles.centerContent}>
              <div
                className={styles.toggleSwitch}
                onClick={() => {
                  // Dashboard uses the same provider-level toggle as Settings/Monitor to avoid split state.
                  void onToggleMonitoring();
                }}
                title="Enable or disable live monitoring"
                style={{ backgroundColor: toggleBg }}
              >
                <div className={styles.toggleKnob} style={knobStyle} />
              </div>
            </div>
          </div>

          <div className={styles.systemBox}>
            <span className={styles.boxLabel}>
              <span>Latency</span>
              <InfoHint text={dashboardHelp.latency} />
            </span>
            <div className={styles.centerContent}>
              <span className={styles.valueText}>
                {latencyMs == null ? "—" : `${Math.round(latencyMs)}ms`}
              </span>
            </div>
          </div>

          <div className={styles.systemBox}>
            <span className={styles.boxLabel}>
              <span>API Health</span>
              <InfoHint text={dashboardHelp.apiHealth} />
            </span>
            <div className={styles.centerContent}>
              <div className={styles.statusIndicator}>
                <span className={styles.dot} style={dotStyle} />
                <span>{apiOnline == null ? "Unknown" : apiOk ? "Online" : "Offline"}</span>
              </div>
            </div>
          </div>
        </div>

        {loading && <div className={styles.mutedNotice}>Loading…</div>}
      </div>

      <div className={styles.infoCard}>
        <h3 className={styles.infoCardTitle}>How to read this dashboard</h3>
        <ul className={styles.infoList}>
          <li>
            <span className={styles.infoKeyword}>Falls Detected</span>
            <span>shows fall events found by the system today.</span>
          </li>
          <li>
            <span className={styles.infoKeyword}>Confirmed Falls</span>
            <span>shows events reviewed and confirmed as real falls.</span>
          </li>
          <li>
            <span className={styles.infoKeyword}>Monitoring</span>
            <span>shows whether live fall detection is currently running.</span>
          </li>
          <li>
            <span className={styles.infoKeyword}>API Health</span>
            <span>shows whether the dashboard is connected to the backend service.</span>
          </li>
        </ul>
      </div>
    </div>
  );
}

export default Dashboard;
