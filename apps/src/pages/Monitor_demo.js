import React from "react";
import styles from "./Monitor.module.css";

function Monitor() {
  return (
    <div className={styles.pageContainer}>
      <h2 className={styles.pageTitle}>Live Monitor</h2>
      
      <div className={styles.content}>
        
        {/* LEFT COLUMN (2/3) */}
        <div className={styles.leftColumn}>
          <div className={styles.card}>
            <div className={styles.cardHeader}>
              <h3>Live Monitoring</h3>
              <p>Real-time Skeleton tracking and fall detection</p>
            </div>
            
            {/* Video Placeholder */}
            <div className={styles.videoPlaceholder}>
              {/* This svg mimics the 'X' in your wireframe */}
              <svg viewBox="0 0 100 100" preserveAspectRatio="none">
                <line x1="0" y1="0" x2="100" y2="100" stroke="#ccc" strokeWidth="0.5" />
                <line x1="100" y1="0" x2="0" y2="100" stroke="#ccc" strokeWidth="0.5" />
                <rect x="0" y="0" width="100" height="100" fill="none" stroke="#ccc" strokeWidth="1" />
              </svg>
            </div>

            {/* Prediction Result Box */}
            <div className={styles.predictionBox}>
              <div className={styles.predictionItem}>
                <span className={styles.label}>Current Prediction</span>
                <span className={styles.value}>No fall</span>
              </div>
              <div className={styles.predictionItem}>
                <span className={styles.label}>P (fall)</span>
                <span className={styles.value}>0.552</span>
              </div>
            </div>
          </div>
        </div>

        {/* RIGHT COLUMN (1/3) */}
        <div className={styles.rightColumn}>
          
          {/* Controls Card */}
          <div className={styles.card}>
            <h3>Controls</h3>
            <div className={styles.buttonGroup}>
              <button className={styles.btnGray}>Pause</button>
              <button className={styles.btnRed}>Test Fall</button>
            </div>
          </div>

          {/* Timeline Card */}
          <div className={styles.card}>
            <h3>Event Timeline</h3>
            <p className={styles.subText}>Last 50 prediction windows</p>
            <div className={styles.timelineBar}>
              {/* Simulating the red markers in the green bar */}
              <div className={styles.marker} style={{ left: '20%' }}></div>
              <div className={styles.marker} style={{ left: '45%' }}></div>
              <div className={styles.marker} style={{ left: '80%' }}></div>
            </div>
          </div>

          {/* Model Info Card */}
          <div className={styles.card}>
            <h3>Model Info</h3>
            <div className={styles.infoTable}>
              <div className={styles.infoRow}>
                <span>Model:</span>
                <span className={styles.tag}>GCN</span>
              </div>
              <div className={styles.infoRow}>
                <span>Window Size</span>
                <span>150</span>
              </div>
              <div className={styles.infoRow}>
                <span>Stride</span>
                <span>30</span>
              </div>
              <div className={styles.infoRow}>
                <span>Threshold</span>
                <span>0.95</span>
              </div>
            </div>
          </div>

        </div>
      </div>
    </div>
  );
}

export default Monitor;