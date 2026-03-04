import React, { useMemo, useState } from "react";
import styles from "./Events.module.css";
import { useMonitoring } from "../monitoring/MonitoringContext";
import { toISODateInput, parseDateSafe, endOfDay } from "../lib/dates";
import { eventStatusLabel, eventTypeLabel, EVENT_STATUS_OPTIONS } from "../lib/eventLabels";
import { useEventsData } from "./events/hooks/useEventsData";

export default function Events() {
  const { apiBase } = useMonitoring();
  // Filters (default: last 7 days)
  const [startDate, setStartDate] = useState(toISODateInput(new Date(Date.now() - 7 * 864e5)));
  const [endDate, setEndDate] = useState(toISODateInput(new Date()));
  const [eventType, setEventType] = useState("All");
  const [status, setStatus] = useState("All");
  const [model, setModel] = useState("All");

  const { events, todaySummary, loading, error, reload, updateStatus } = useEventsData(apiBase, 1);

  const filteredEvents = useMemo(() => {
    const sD = parseDateSafe(startDate);
    const eD0 = parseDateSafe(endDate);
    const eD = eD0 ? endOfDay(eD0) : null;
    return (events || []).filter((ev) => {
      
      const t = parseDateSafe(ev.event_time);
      if (sD && t && t < sD) return false;
      if (eD && t && t > eD) return false;

      if (eventType !== "All") {
        const want =
          eventType === "Fall" ? "fall" : eventType === "Uncertain" ? "uncertain" : "not_fall";
        if ((ev.type || "").toLowerCase() !== want) return false;
      }

      if (status !== "All") {
        const want =
          status === "Unreviewed"
            ? "pending_review"
            : status === "Confirmed"
            ? "confirmed_fall"
            : status === "False Alarm"
            ? "false_alarm"
            : "dismissed";
        if ((ev.status || "").toLowerCase() !== want) return false;
      }

      if (model !== "All") {
        const want = model.toUpperCase();
        if ((ev.model_code || "").toUpperCase() !== want) return false;
      }

      return true;
    });
  }, [events, startDate, endDate, eventType, status, model]);

  // Stats: total, pending, confirmed, false
  const stats = useMemo(() => {
    const total = events.length;
    const pending = events.filter((e) => (e.status || "").toLowerCase() === "pending_review").length;
    const confirmed = events.filter((e) => (e.status || "").toLowerCase() === "confirmed_fall").length;
    const falseAlarms = events.filter((e) => (e.status || "").toLowerCase() === "false_alarm").length;
    return { total, pending, confirmed, falseAlarms };
  }, [events]);

  async function handleView(ev) {
    const current = (ev.status || "pending_review").toLowerCase();
    const next = window.prompt(
      "Update event status?\nUse one of: pending_review, confirmed_fall, false_alarm, dismissed",
      current
    );
    if (!next) return;
    const ok = EVENT_STATUS_OPTIONS.includes(next.toLowerCase());
    if (!ok) {
      alert("Invalid status.");
      return;
    }
    try {
      await updateStatus(ev.id, next.toLowerCase());
    } catch (e) {
      alert(`Failed to update status: ${String(e?.message || e)}`);
    }
  }

  return (
    <div className={styles.container}>
      <h2 className={styles.pageTitle}>Event History</h2>

      {error && (
        <div style={{ marginTop: -8, color: "#B45309" }}>
          Backend error: {error}
        </div>
      )}

      {/* --- Section 1: Stats Grid --- */}
      <div className={styles.statsGrid}>
        <div className={styles.statCard}>
          <span className={styles.statNumber}>{todaySummary.falls}</span>
          <span className={styles.statLabel}>Falls Today</span>
        </div>

        <div className={styles.statCard}>
          <span className={styles.statNumber}>{todaySummary.pending}</span>
          <span className={styles.statLabel}>Pending Review</span>
        </div>

        <div className={styles.statCard}>
          <span className={styles.statNumber}>{todaySummary.false_alarms}</span>
          <span className={styles.statLabel}>False Alarms Today</span>
        </div>

        <div className={styles.statCard}>
          <span className={styles.statNumber}>{stats.total}</span>
          <span className={styles.statLabel}>Events (Total)</span>
        </div>
      </div>

      {/* --- Section 2: Filters --- */}
      <div className={styles.filterCard}>
        <h3 className={styles.sectionTitle}>Filters</h3>
        <div className={styles.filterInputs}>
          <div className={styles.dateWrapper}>
            <input
              className={styles.inputField}
              type="date"
              value={startDate}
              onChange={(e) => setStartDate(e.target.value)}
            />
          </div>

          <div className={styles.dateWrapper}>
            <input
              className={styles.inputField}
              type="date"
              value={endDate}
              onChange={(e) => setEndDate(e.target.value)}
            />
          </div>

          <select className={styles.selectField} value={eventType} onChange={(e) => setEventType(e.target.value)}>
            <option>All</option>
            <option>Fall</option>
            <option>Uncertain</option>
            <option>Safe</option>
          </select>

          <select className={styles.selectField} value={status} onChange={(e) => setStatus(e.target.value)}>
            <option>All</option>
            <option>Unreviewed</option>
            <option>Confirmed</option>
            <option>False Alarm</option>
            <option>Dismissed</option>
          </select>

          <select className={styles.selectField} value={model} onChange={(e) => setModel(e.target.value)}>
            <option>All</option>
            <option>TCN</option>
            <option>GCN</option>
          </select>

          <button
            className={styles.viewBtn}
            onClick={reload}
            style={{ marginLeft: "auto" }}
            title="Refresh events"
          >
            Refresh
          </button>
        </div>
      </div>

      {/* --- Section 3: Table --- */}
      <div className={styles.tableCard}>
        <table className={styles.eventsTable}>
          <thead>
            <tr>
              <th>Date</th>
              <th>Type</th>
              <th>Model</th>
              <th>Confidence</th>
              <th>Status</th>
              <th>Action</th>
            </tr>
          </thead>
          <tbody>
            {loading ? (
              <tr>
                <td colSpan={6} style={{ padding: 16, color: "#6B7280" }}>
                  Loading…
                </td>
              </tr>
            ) : filteredEvents.length === 0 ? (
              <tr>
                <td colSpan={6} style={{ padding: 16, color: "#6B7280" }}>
                  No events found.
                </td>
              </tr>
            ) : (
              filteredEvents.map((ev) => (
                <tr key={ev.id}>
                  <td>
                    {parseDateSafe(ev.event_time)
                      ? parseDateSafe(ev.event_time).toLocaleString()
                      : String(ev.event_time)}
                  </td>
            
                  <td>{eventTypeLabel(ev.type)}</td>
                  <td>{(ev.model_code || "—").toUpperCase()}</td>
                  <td>{ev.p_fall != null ? Number(ev.p_fall).toFixed(2) : "—"}</td>
                  <td>
                    <span className={styles.statusBadge}>{eventStatusLabel(ev.status)}</span>
                  </td>
                  <td>
                    <button className={styles.viewBtn} onClick={() => handleView(ev)}>
                      View
                    </button>
                  </td>
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}
