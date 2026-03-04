import React from "react";

import styles from "../../Monitor.module.css";

export function LiveMonitorCard({
  videoRef,
  canvasRef,
  currentPrediction,
  pText,
  safePrediction,
  recallPrediction,
  inputSource,
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
          <span className={styles.value}>{currentPrediction}</span>
        </div>
        <div className={styles.predictionItem}>
          <span className={styles.label}>P (fall)</span>
          <span className={styles.value}>{pText}</span>
        </div>
        <div className={styles.predictionItem}>
          <span className={styles.label}>Safe Channel</span>
          <span className={styles.value}>{safePrediction || "—"}</span>
        </div>
        <div className={styles.predictionItem}>
          <span className={styles.label}>Recall Channel (Aggressive)</span>
          <span className={styles.value}>{recallPrediction || "—"}</span>
        </div>
      </div>
    </div>
  );
}
