import { useCallback, useEffect, useRef, useState } from "react";

import { apiRequest } from "../../../lib/apiClient";

export function useEventsData(apiBase, residentId = 1) {
  const [events, setEvents] = useState([]);
  const [todaySummary, setTodaySummary] = useState({ falls: 0, pending: 0, false_alarms: 0 });
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const abortRef = useRef(null);

  const reload = useCallback(async () => {
    try {
      setError(null);
      setLoading(true);
      abortRef.current?.abort?.();
      const ac = new AbortController();
      abortRef.current = ac;

      // Summary (best-effort)
      try {
        const s = await apiRequest(apiBase, `/api/events/summary?resident_id=${residentId}`, {
          signal: ac.signal,
        });
        if (s?.today) setTodaySummary(s.today);
      } catch {
        // ignore
      }

      const data = await apiRequest(apiBase, `/api/events?resident_id=${residentId}&limit=500`, {
        signal: ac.signal,
      });
      setEvents(Array.isArray(data?.events) ? data.events : []);
      setLoading(false);
    } catch (e) {
      if (e?.name === "AbortError") return;
      setError(String(e?.message || e));
      setLoading(false);
    }
  }, [apiBase, residentId]);

  useEffect(() => {
    reload();
    return () => abortRef.current?.abort?.();
  }, [reload]);

  const updateStatus = useCallback(
    async (eventId, status) => {
      await apiRequest(apiBase, `/api/events/${eventId}/status`, {
        method: "PUT",
        body: { status },
      });
      await reload();
    },
    [apiBase, reload]
  );

  return { events, todaySummary, loading, error, reload, updateStatus };
}
