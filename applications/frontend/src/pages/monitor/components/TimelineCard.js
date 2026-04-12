import React from "react";

import styles from "../../Monitor.module.css";

export function TimelineCard({ markers, statusText }) {
  const kindToColor = (kind) => {
    if (kind === "fall") return "#d32f2f";
    if (kind === "uncertain") return "#f5f508";
    if (kind === "safe") return "#34dc31";
    return "#d1fae5";
  };

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
              "--marker-color": kindToColor(m.kind),
              opacity: m.kind === "uncertain" ? 0.85 : 1,
            }}
            title={m.kind}
          />
        ))}
      </div>
    </div>
  );
}
