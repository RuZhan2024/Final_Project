import React from "react";

import styles from "../../Monitor.module.css";

export function TimelineCard({ markers, statusText }) {
  return (
    <div className={styles.card}>
      <h3>Event Timeline</h3>
      <p className={styles.subText}>{statusText || "Last 50 prediction windows"}</p>

      <div className={styles.timelineBar}>
        {markers.map((m, idx) => (
          <div
            key={m.key ?? idx}
            className={styles.marker}
            style={{
              left: `${m.leftPct}%`,
              opacity: m.kind === "safe" ? 0 : m.kind === "uncertain" ? 0.5 : 1,
              backgroundColor: "#d32f2f",
            }}
            title={m.kind}
          />
        ))}
      </div>
    </div>
  );
}
