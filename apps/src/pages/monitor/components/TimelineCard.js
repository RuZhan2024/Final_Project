import React from "react";

import styles from "../../Monitor.module.css";

export function TimelineCard({ markers }) {
  return (
    <div className={styles.card}>
      <h3>Event Timeline</h3>
      <p className={styles.subText}>Last 50 prediction windows</p>

      <div className={styles.timelineBar}>
        {markers.map((m, idx) => (
          <div
            key={idx}
            className={styles.marker}
            style={{
              left: `${m.leftPct}%`,
              opacity: m.kind === "uncertain" ? 0.5 : 1,
            }}
            title={m.kind}
          />
        ))}
      </div>
    </div>
  );
}
