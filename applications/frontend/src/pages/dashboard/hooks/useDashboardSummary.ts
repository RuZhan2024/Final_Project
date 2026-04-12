import { useEffect, useRef, useState } from "react";

import { fetchDashboardSummary } from "../../../features/dashboard/api";
import type { DashboardSummary } from "../../../features/dashboard/types";

interface UseDashboardSummaryOptions {
  intervalMs?: number;
}

interface UseDashboardSummaryResult {
  data: DashboardSummary | null;
  loading: boolean;
  error: string | null;
}

export function useDashboardSummary(
  apiBase: string,
  { intervalMs = 3000 }: UseDashboardSummaryOptions = {}
): UseDashboardSummaryResult {
  const [data, setData] = useState<DashboardSummary | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const timerRef = useRef<number | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  useEffect(() => {
    let mounted = true;

    async function load(): Promise<void> {
      try {
        setError(null);
        abortRef.current?.abort();
        const controller = new AbortController();
        abortRef.current = controller;
        const payload = await fetchDashboardSummary(apiBase, { signal: controller.signal });
        if (!mounted) return;
        setData(payload);
        setLoading(false);
      } catch (err) {
        if (!mounted) return;
        if ((err as Error)?.name === "AbortError") return;
        setError(String((err as Error)?.message || err));
        setLoading(false);
      }
    }

    void load();
    timerRef.current = window.setInterval(load, intervalMs);

    return () => {
      mounted = false;
      abortRef.current?.abort();
      if (timerRef.current) window.clearInterval(timerRef.current);
    };
  }, [apiBase, intervalMs]);

  return { data, loading, error };
}
