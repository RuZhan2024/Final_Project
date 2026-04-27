import React from "react";

import type { InputSource } from "../types";

import styles from "../../Monitor.module.css";

function predictionToneStyle(prediction: string) {
  const t = String(prediction || "").trim().toLowerCase();
  if (t === "fall" || t === "fall detected") return { color: "#d32f2f" };
  if (t === "uncertain" || t === "watch") return { color: "#f5f508" };
  if (t === "no fall" || t === "safe" || t === "normal") return { color: "#34dc31" };
  return undefined;
}

function predictionDisplayText(prediction: string) {
  const raw = String(prediction || "").trim();
  if (!raw) return raw;
  const t = raw.toLowerCase();
  // The UI says "Normal" where the backend/frontend state contract says "no fall".
  if (t === "no fall") return "Normal";
  return prediction;
}

interface LiveMonitorCardProps {
  videoRef: React.RefObject<HTMLVideoElement>;
  canvasRef: React.RefObject<HTMLCanvasElement>;
  currentPrediction: string;
  pText: string;
  inputSource: InputSource;
  showLivePreview: boolean;
  onToggleLivePreview: () => void;
  captureFpsText: string;
  modelFpsText: string;
}

/**
 * Primary monitor card showing the video/canvas stack and current prediction state.
 */
export function LiveMonitorCard({
  videoRef,
  canvasRef,
  currentPrediction,
  pText,
  inputSource,
  showLivePreview,
  onToggleLivePreview,
  captureFpsText,
  modelFpsText,
}: LiveMonitorCardProps) {
  const videoLayerClass =
    inputSource === "video"
      ? styles.videoLayerReplay
      // Live camera preview can be hidden while keeping pose drawing and inference active.
      : showLivePreview
        ? styles.videoLayerLiveVisible
        : styles.videoLayerHidden;

  return (
    <div className={styles.card}>
      <div className={styles.cardHeader}>
        <div className={styles.cardHeaderText}>
          <h3>Live Monitoring</h3>
          <p>Real-time Skeleton tracking and fall detection</p>
        </div>
        {inputSource === "camera" ? (
          <label className={styles.previewToggleRow}>
            <span>Show live preview</span>
            <button
              type="button"
              role="switch"
              aria-checked={showLivePreview}
              aria-label="Toggle live video preview"
              onClick={onToggleLivePreview}
              className={`${styles.previewToggle} ${showLivePreview ? styles.previewToggleOn : ""}`}
            >
              <span className={styles.previewToggleThumb}></span>
            </button>
          </label>
        ) : null}
      </div>

      <div className={styles.videoPlaceholder}>
        <video
          ref={videoRef}
          muted
          playsInline
          className={`${styles.videoLayer} ${videoLayerClass}`}
        />
        <canvas
          ref={canvasRef}
          className={styles.canvasLayer}
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
          <span className={styles.label}>Capture FPS / Target FPS</span>
          <span className={styles.value}>
            {captureFpsText || "—"} / {modelFpsText || "—"}
          </span>
        </div>
      </div>
    </div>
  );
}
