import React from "react";
import styles from "./Events.module.css";

function Events() {
  // Sample data to mimic the table in your image
  const eventsData = [
    { id: 1, time: "23/11/2025", type: "fall", model: "TCN", prob: "85%", delay: "0.8", status: "Pending Review" },
    { id: 2, time: "23/11/2025", type: "fall", model: "TCN", prob: "85%", delay: "0.8", status: "Pending Review" },
    { id: 3, time: "23/11/2025", type: "fall", model: "TCN", prob: "85%", delay: "0.8", status: "Pending Review" },
    { id: 4, time: "23/11/2025", type: "fall", model: "TCN", prob: "85%", delay: "0.8", status: "Pending Review" },
    { id: 5, time: "23/11/2025", type: "fall", model: "TCN", prob: "85%", delay: "0.8", status: "Pending Review" },
    { id: 6, time: "23/11/2025", type: "fall", model: "TCN", prob: "85%", delay: "0.8", status: "Pending Review" },
    { id: 7, time: "23/11/2025", type: "fall", model: "TCN", prob: "85%", delay: "0.8", status: "Pending Review" },
  ];

  return (
    <div className={styles.container}>
      <h2 className={styles.pageTitle}>Event History & Review</h2>

      {/* --- Section 1: Top Statistics Cards --- */}
      <div className={styles.statsGrid}>
        <div className={styles.statCard}>
          <span className={styles.statNumber}>0</span>
          <span className={styles.statLabel}>Falls Today</span>
        </div>
        <div className={styles.statCard}>
          <span className={styles.statNumber}>0</span>
          <span className={styles.statLabel}>Falls Alarm Today</span>
        </div>
        <div className={styles.statCard}>
          <span className={styles.statNumber}>~0.7</span>
          <span className={styles.statLabel}>FA/24h Estimate</span>
        </div>
        <div className={styles.statCard}>
          <span className={styles.statNumber}>1.2s</span>
          <span className={styles.statLabel}>AVG Detect Delay</span>
        </div>
      </div>

      {/* --- Section 2: Filters --- */}
      <div className={styles.filterCard}>
        <h3 className={styles.sectionTitle}>Filters</h3>
        <div className={styles.filterInputs}>
          {/* Mimicking the Date Picker */}
          <input 
            type="text" 
            placeholder="20/11/2025 - 23/11/2025" 
            className={styles.inputField} 
          />
          {/* Dropdowns */}
          <select className={styles.selectField} defaultValue="">
            <option value="" disabled hidden>All Types</option>
            <option value="fall">Fall</option>
            <option value="manual">Manual</option>
          </select>
          <select className={styles.selectField} defaultValue="">
            <option value="" disabled hidden>All Status</option>
            <option value="pending">Pending Review</option>
            <option value="confirmed">Confirmed</option>
          </select>
          <select className={styles.selectField} defaultValue="">
            <option value="" disabled hidden>All Models</option>
            <option value="tcn">TCN</option>
            <option value="gcn">GCN</option>
          </select>
        </div>
      </div>

      {/* --- Section 3: Data Table --- */}
      <div className={styles.tableCard}>
        <table className={styles.eventsTable}>
          <thead>
            <tr>
              <th>Time</th>
              <th>type</th>
              <th>Model</th>
              <th>Probability</th>
              <th>Delay(s)</th>
              <th>Status</th>
              <th>Actions</th>
            </tr>
          </thead>
          <tbody>
            {eventsData.map((row) => (
              <tr key={row.id}>
                <td>{row.time}</td>
                <td>{row.type}</td>
                <td>{row.model}</td>
                <td>{row.prob}</td>
                <td>{row.delay}</td>
                <td>{row.status}</td>
                <td>
                  <button className={styles.viewBtn}>view</button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

    </div>
  );
}

export default Events;