import React from "react";

import styles from "../../Monitor.module.css";

export function LiveMonitorCard({ videoRef, canvasRef, currentPrediction, pText }) {
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
            opacity: 0, // keep hidden (skeleton-only)
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
      </div>
    </div>
  );
}
