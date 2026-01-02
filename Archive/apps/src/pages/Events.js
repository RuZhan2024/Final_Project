import React, { useEffect, useMemo, useState } from "react";
import styles from "./Events.module.css";

// Prefer env var, fallback to localhost
const API_BASE =
  typeof process !== "undefined" && process.env && process.env.REACT_APP_API_BASE
    ? process.env.REACT_APP_API_BASE
    : "http://localhost:8000";

function toISODateInput(d) {
  const pad = (n) => String(n).padStart(2, "0");
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}`;
}

function parseDateSafe(s) {
  const d = new Date(s);
  return Number.isFinite(d.getTime()) ? d : null;
}

function statusLabel(s) {
  const x = (s || "").toLowerCase();
  if (x === "confirmed_fall") return "Confirmed";
  if (x === "false_alarm") return "False Alarm";
  if (x === "pending_review") return "Pending Review";
  if (x === "dismissed") return "Dismissed";
  return s || "—";
}

function typeLabel(t) {
  const x = (t || "").toLowerCase();
  if (x === "fall") return "Fall";
  if (x === "uncertain") return "Uncertain";
  if (x === "not_fall") return "Safe";
  return t || "—";
}

export default function Events() {
  // Filters (default: last 7 days)
  const [startDate, setStartDate] = useState(toISODateInput(new Date(Date.now() - 7 * 864e5)));
  const [endDate, setEndDate] = useState(toISODateInput(new Date()));
  const [eventType, setEventType] = useState("All");
  const [status, setStatus] = useState("All");
  const [model, setModel] = useState("All");

  // Data
  const [events, setEvents] = useState([]);
  const [todaySummary, setTodaySummary] = useState({ falls: 0, pending: 0, false_alarms: 0 });

  const [loading, setLoading] = useState(true);
  const [err, setErr] = useState(null);

  async function load() {
    try {
      setErr(null);
      setLoading(true);

      // summary (today)
      try {
        const rs = await fetch(`${API_BASE}/api/events/summary?resident_id=1`);
        if (rs.ok) {
          const s = await rs.json();
          if (s?.today) setTodaySummary(s.today);
        }
      } catch {
        // ignore
      }

      const r = await fetch(`${API_BASE}/api/events?resident_id=1&limit=500`);
      if (!r.ok) throw new Error(await r.text());
      const data = await r.json();
      setEvents(Array.isArray(data?.events) ? data.events : []);
      setLoading(false);
    } catch (e) {
      setErr(String(e?.message || e));
      setLoading(false);
    }
  }

  useEffect(() => {
    load();
  }, []);

  const filteredEvents = useMemo(() => {
    const sD = parseDateSafe(startDate);
    const eD = parseDateSafe(endDate);
    if (eD) eD.setHours(23, 59, 59, 999);

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
        const want = model.toUpperCase() === "HYBRID" ? "HYBRID" : model.toUpperCase();
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

  async function updateEventStatus(ev, newStatus) {
    try {
      const r = await fetch(`${API_BASE}/api/events/${ev.id}/status`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ status: newStatus }),
      });
      if (!r.ok) throw new Error(await r.text());
      await load();
    } catch (e) {
      alert(`Failed to update status: ${String(e?.message || e)}`);
    }
  }

  async function handleView(ev) {
    const current = (ev.status || "pending_review").toLowerCase();
    const next = window.prompt(
      "Update event status?\nUse one of: pending_review, confirmed_fall, false_alarm, dismissed",
      current
    );
    if (!next) return;
    const ok = ["pending_review", "confirmed_fall", "false_alarm", "dismissed"].includes(
      next.toLowerCase()
    );
    if (!ok) {
      alert("Invalid status.");
      return;
    }
    await updateEventStatus(ev, next.toLowerCase());
  }

  return (
    <div className={styles.container}>
      <h2 className={styles.pageTitle}>Event History</h2>

      {err && (
        <div style={{ marginTop: -8, color: "#B45309" }}>
          Backend error: {err}
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
            <option>Hybrid</option>
          </select>

          <button
            className={styles.viewBtn}
            onClick={load}
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
                  <td>{typeLabel(ev.type)}</td>
                  <td>{(ev.model_code || "—").toUpperCase()}</td>
                  <td>{ev.p_fall != null ? Number(ev.p_fall).toFixed(2) : "—"}</td>
                  <td>
                    <span className={styles.statusBadge}>{statusLabel(ev.status)}</span>
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
