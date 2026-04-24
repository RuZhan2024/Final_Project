import React from "react";

import styles from "../../Monitor.module.css";

interface ModelInfoCardProps {
  modelTag: string;
  deployW: number | null;
  deployS: number | null;
  tauLow: number | null;
  tauHigh: number | null;
  opCode: string | null;
  confirmK: number | null;
  confirmN: number | null;
  cooldownS: number | null;
  mode: string;
  captureFpsText: string;
  modelFpsText: string;
  mcCfg: { M: number | null; M_confirm: number | null };
  sigma: number | null;
}

/**
 * Summary card for the active model/runtime parameters shown beside the monitor.
 */
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
}: ModelInfoCardProps) {
  const tauHighText = (() => {
    const v = tauHigh;
    if (v == null) return "—";
    // When available, keep the OP code next to tau_high because settings may override presets.
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

      {/* Keep the compact footer for values that are useful in demos but too noisy for the main table. */}
      <p className={`${styles.subText} ${styles.infoSummary}`}>
        Mode: {mode} • Capture FPS: {captureFpsText} • Model FPS: {modelFpsText} • Live MC: {mcCfg?.M ?? "—"}/{mcCfg?.M_confirm ?? "—"}
        {sigma != null ? ` • σ=${Number(sigma).toFixed(3)}` : ""}
      </p>
    </div>
  );
}
