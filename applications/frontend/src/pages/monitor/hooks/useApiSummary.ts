import { useEffect, useState } from "react";

import { apiRequest } from "../../../lib/apiClient";

/**
 * Poll /api/summary so Monitor can show API health and latency.
 */
export function useApiSummary(apiBase: string, intervalMs = 5000, enabled = true) {
  const [summary, setSummary] = useState<unknown>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    // Hidden/inactive monitor tabs can disable polling without clearing the last card.
    if (!enabled) return undefined;

    let cancelled = false;

    const tick = async () => {
      try {
        setError(null);
        const data = await apiRequest(apiBase, "/api/summary");
        if (!cancelled) setSummary(data);
      } catch (e: unknown) {
        // Preserve the last good summary card while surfacing the polling error.
        if (!cancelled) setError(String((e as Error)?.message || e));
      }
    };

    tick();
    // Keep polling coarse; summary cards do not need per-frame freshness.
    const id = setInterval(tick, Math.max(1000, Number(intervalMs) || 5000));

    return () => {
      cancelled = true;
      clearInterval(id);
    };
  }, [apiBase, intervalMs, enabled]);

  return { summary, error };
}
