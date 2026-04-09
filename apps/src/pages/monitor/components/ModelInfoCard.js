import React from "react";

import styles from "../../Monitor.module.css";

export function ModelInfoCard({
  modelTag,
  deployW,
  deployS,
  tauLow,
  tauHigh,
  opCode,
  confirmK,
  confirmN,
  cooldownS,
  mode,
  captureFpsText,
  modelFpsText,
  mcCfg,
  sigma,
}) {
  const tauHighText = (() => {
    const v = tauHigh;
    if (v == null) return "—";
    const tag = opCode ? ` (${opCode})` : "";
    return `${Number(v).toFixed(2)}${tag}`;
  })();

  return (
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
          <span>τ_low</span>
          <span>{tauLow != null ? Number(tauLow).toFixed(2) : "—"}</span>
        </div>
        <div className={styles.infoRow}>
          <span>τ_high</span>
          <span>{tauHighText}</span>
        </div>
        <div className={styles.infoRow}>
          <span>Confirm (k/n)</span>
          <span>{confirmK != null && confirmN != null ? `${confirmK}/${confirmN}` : "—"}</span>
        </div>
        <div className={styles.infoRow}>
          <span>Cooldown (s)</span>
          <span>{cooldownS != null ? cooldownS : "—"}</span>
        </div>
      </div>

      <p className={`${styles.subText} ${styles.infoSummary}`}>
        Mode: {mode} • Capture FPS: {captureFpsText} • Model FPS: {modelFpsText} • Live MC: {mcCfg?.M ?? "—"}/{mcCfg?.M_confirm ?? "—"}
        {sigma != null ? ` • σ=${Number(sigma).toFixed(3)}` : ""}
      </p>
    </div>
  );
}
