import { useCallback, useEffect, useRef, useState } from "react";

import {
  fetchEvents,
  fetchEventsSummary,
  updateEventStatus,
} from "../../../features/events/api";
import type {
  EventRecord,
  EventStatus,
  EventsTodaySummary,
} from "../../../features/events/types";

function isTodayLocal(isoLike: string | null | undefined): boolean {
  const d = new Date(isoLike);
  if (Number.isNaN(d.getTime())) return false;
  const now = new Date();
  return (
    d.getFullYear() === now.getFullYear() &&
    d.getMonth() === now.getMonth() &&
    d.getDate() === now.getDate()
  );
}

interface UseEventsDataResult {
  events: EventRecord[];
  todaySummary: EventsTodaySummary;
  loading: boolean;
  error: string | null;
  reload: (options?: { silent?: boolean }) => Promise<void>;
  updateStatus: (eventId: number, status: EventStatus) => Promise<void>;
}

const EMPTY_SUMMARY: EventsTodaySummary = { falls: 0, pending: 0, false_alarms: 0 };

export function useEventsData(apiBase: string, residentId = 1): UseEventsDataResult {
  const [events, setEvents] = useState<EventRecord[]>([]);
  const [todaySummary, setTodaySummary] = useState<EventsTodaySummary>(EMPTY_SUMMARY);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const abortRef = useRef<AbortController | null>(null);

  const reload = useCallback(async ({ silent = false }: { silent?: boolean } = {}): Promise<void> => {
    try {
      setError(null);
      if (!silent) setLoading(true);
      abortRef.current?.abort();
      const ac = new AbortController();
      abortRef.current = ac;

      // Summary (best-effort)
      try {
        const s = await fetchEventsSummary(apiBase, { residentId, signal: ac.signal });
        if (s?.today) setTodaySummary(s.today);
      } catch {
        // ignore
      }

      const data = await fetchEvents(apiBase, { residentId, limit: 500, signal: ac.signal });
      const nextEvents = Array.isArray(data?.events) ? data.events : [];
      setEvents(nextEvents);
      if (!silent) setLoading(false);
    } catch (e: unknown) {
      if ((e as Error)?.name === "AbortError") return;
      setError(String((e as Error)?.message || e));
      if (!silent) setLoading(false);
    }
  }, [apiBase, residentId]);

  useEffect(() => {
    void reload();
    return () => abortRef.current?.abort();
  }, [reload]);

  useEffect(() => {
    function refreshVisible() {
      if (document.visibilityState !== "visible") return;
      void reload({ silent: true });
    }

    function refreshOnFocus() {
      void reload({ silent: true });
    }

    const pollId = window.setInterval(() => {
      if (document.visibilityState !== "visible") return;
      void reload({ silent: true });
    }, 10000);

    document.addEventListener("visibilitychange", refreshVisible);
    window.addEventListener("focus", refreshOnFocus);
    return () => {
      window.clearInterval(pollId);
      document.removeEventListener("visibilitychange", refreshVisible);
      window.removeEventListener("focus", refreshOnFocus);
    };
  }, [reload]);

  const updateStatus = useCallback(
    async (eventId: number, status: EventStatus): Promise<void> => {
      await updateEventStatus(apiBase, eventId, status);
      let prevStatus: string | null = null;
      let eventTime: string | null | undefined = null;
      setEvents((prev) =>
        prev.map((ev) => {
          if (ev.id !== eventId) return ev;
          prevStatus = String(ev.status || "pending_review").toLowerCase();
          eventTime = ev.event_time;
          return { ...ev, status };
        })
      );

      if (isTodayLocal(eventTime)) {
        setTodaySummary((prev) => {
          const next = {
            falls: Number(prev?.falls || 0),
            pending: Number(prev?.pending || 0),
            false_alarms: Number(prev?.false_alarms || 0),
          };
          if (prevStatus === "pending_review") next.pending = Math.max(0, next.pending - 1);
          if (prevStatus === "false_alarm") next.false_alarms = Math.max(0, next.false_alarms - 1);
          if (status === "pending_review") next.pending += 1;
          if (status === "false_alarm") next.false_alarms += 1;
          return next;
        });
      }

      void reload({ silent: true });
    },
    [apiBase, reload]
  );

  return { events, todaySummary, loading, error, reload, updateStatus };
}
