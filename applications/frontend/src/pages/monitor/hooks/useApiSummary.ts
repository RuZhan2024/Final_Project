import { useEffect, useState } from "react";

import { apiRequest } from "../../../lib/apiClient";

/**
 * Polls /api/summary so Monitor can show API health/latency.
 */
export function useApiSummary(apiBase: string, intervalMs = 5000, enabled = true) {
  const [summary, setSummary] = useState<unknown>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!enabled) return undefined;

    let cancelled = false;

    const tick = async () => {
      try {
        setError(null);
        const data = await apiRequest(apiBase, "/api/summary");
        if (!cancelled) setSummary(data);
      } catch (e: unknown) {
        if (!cancelled) setError(String((e as Error)?.message || e));
      }
    };

    tick();
    const id = setInterval(tick, Math.max(1000, Number(intervalMs) || 5000));

    return () => {
      cancelled = true;
      clearInterval(id);
    };
  }, [apiBase, intervalMs, enabled]);

  return { summary, error };
}
