import React from "react";

import styles from "../../Monitor.module.css";

export function ControlsCard({
  monitoringOn,
  setMonitoringOn,
  resetSession,
  testFall,
  modelsErr,
  monitoringErr,
  summaryErr,
  apiSummary,
}) {
  return (
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
      {summaryErr && (
        <p className={styles.subText} style={{ color: "#B45309" }}>
          Summary error: {summaryErr}
        </p>
      )}

      {!summaryErr && apiSummary?.system && (
        <p className={styles.subText}>
          API: {apiSummary.system.api_online ? "Online" : "Offline"} • Last latency: {apiSummary.system.last_latency_ms ?? "—"} ms
        </p>
      )}

      <div className={styles.buttonGroup}>
        {!monitoringOn ? (
          <button
            className={styles.btnGray}
            onClick={() => {
              // Don’t await: we want getUserMedia to run under a user gesture.
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
  );
}
