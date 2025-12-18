import React from "react";
import styles from "./Dashboard.module.css";

function Dashboard() {
  return (
    <div className={styles.container}>
      <h2 className={styles.pageTitle}>Dashboard</h2>

      {/* --- Top Section (Two Columns) --- */}
      <div className={styles.topRow}>
        
        {/* Card 1: Resident Status */}
        <div className={styles.card}>
          <h3 className={styles.cardTitle}>Resident Status</h3>
          <div className={styles.residentContent}>
            <span className={styles.statusText}>Normal</span>
            <button className={styles.actionButton}>View Live Feed</button>
          </div>
        </div>

        {/* Card 2: Today's Summary */}
        <div className={styles.card}>
          <h3 className={styles.cardTitle}>Today's Summary</h3>
          <div className={styles.summaryGrid}>
            <div className={styles.summaryBox}>
              <span className={styles.bigNumber}>1</span>
              <span className={styles.label}>Falls Detected</span>
            </div>
            <div className={styles.summaryBox}>
              <span className={styles.bigNumber}>3</span>
              <span className={styles.label}>Confirmed False Alarms</span>
            </div>
          </div>
        </div>
      </div>

      {/* --- Bottom Section (System Status) --- */}
      <div className={styles.fullWidthCard}>
        <h3 className={styles.cardTitle}>System Status</h3>
        <div className={styles.systemGrid}>
          
          {/* Item 1: Model */}
          <div className={styles.systemBox}>
            <span className={styles.boxLabel}>Model</span>
            <div className={styles.centerContent}>
              <span className={styles.tag}>GCN</span>
            </div>
          </div>

          {/* Item 2: Monitoring Toggle */}
          <div className={styles.systemBox}>
            <span className={styles.boxLabel}>Monitoring</span>
            <div className={styles.centerContent}>
              {/* CSS-only Toggle Switch representation */}
              <div className={styles.toggleSwitch}>
                <div className={styles.toggleKnob}></div>
              </div>
            </div>
          </div>

          {/* Item 3: Latency */}
          <div className={styles.systemBox}>
            <span className={styles.boxLabel}>Latency</span>
            <div className={styles.centerContent}>
              <span className={styles.valueText}>120ms</span>
            </div>
          </div>

          {/* Item 4: API Health */}
          <div className={styles.systemBox}>
            <span className={styles.boxLabel}>API Health</span>
            <div className={styles.centerContent}>
              <div className={styles.statusIndicator}>
                <span className={styles.dot}></span>
                <span>Online</span>
              </div>
            </div>
          </div>

        </div>
      </div>

    </div>
  );
}

export default Dashboard;