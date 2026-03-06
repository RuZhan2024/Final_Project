import React from "react";

import styles from "../../Monitor.module.css";

function predictionToneStyle(prediction) {
  const t = String(prediction || "").trim().toLowerCase();
  if (t === "fall" || t === "fall detected") return { color: "#d32f2f" };
  if (t === "uncertain" || t === "watch") return { color: "#f5f508" };
  if (t === "no fall" || t === "safe" || t === "normal") return { color: "#34dc31" };
  return undefined;
}

function predictionDisplayText(prediction) {
  const raw = String(prediction || "").trim();
  if (!raw) return raw;
  const t = raw.toLowerCase();
  if (t === "no fall") return "Normal";
  return prediction;
}

export function LiveMonitorCard({
  videoRef,
  canvasRef,
  currentPrediction,
  pText,
  safePrediction,
  recallPrediction,
  inputSource,
  captureFpsText,
  modelFpsText,
}) {
  return (
    <div className={styles.card}>
      <div className={styles.cardHeader}>
        <h3>Live Monitoring</h3>
        <p>Real-time Skeleton tracking and fall detection</p>
      </div>

      <div className={styles.videoPlaceholder}>
        <video
          ref={videoRef}
          muted
          playsInline
          style={{
            position: "absolute",
            inset: 0,
            width: "100%",
            height: "100%",
            objectFit: "cover",
            opacity: inputSource === "video" ? 0.35 : 0, // show replay video under skeleton
          }}
        />
        <canvas
          ref={canvasRef}
          style={{
            position: "absolute",
            inset: 0,
            width: "100%",
            height: "100%",
          }}
        />
      </div>

      <div className={styles.predictionBox}>
        <div className={styles.predictionItem}>
          <span className={styles.label}>Current Prediction</span>
          <span className={styles.value} style={predictionToneStyle(currentPrediction)}>
            {predictionDisplayText(currentPrediction)}
          </span>
        </div>
        <div className={styles.predictionItem}>
          <span className={styles.label}>P (fall)</span>
          <span className={styles.value}>{pText}</span>
        </div>
        <div className={styles.predictionItem}>
          <span className={styles.label}>Safe Channel</span>
          <span className={styles.value}>{safePrediction ? predictionDisplayText(safePrediction) : "—"}</span>
        </div>
        <div className={styles.predictionItem}>
          <span className={styles.label}>Recall Channel (Aggressive)</span>
          <span className={styles.value}>{recallPrediction ? predictionDisplayText(recallPrediction) : "—"}</span>
        </div>
        <div className={styles.predictionItem}>
          <span className={styles.label}>Capture FPS / Target FPS</span>
          <span className={styles.value}>
            {captureFpsText || "—"} / {modelFpsText || "—"}
          </span>
        </div>
      </div>
    </div>
  );
}
