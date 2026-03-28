import { useEffect, useRef, useState } from "react";

import { fetchDashboardSummary } from "../../../features/dashboard/api";

export function useDashboardSummary(apiBase, { intervalMs = 3000 } = {}) {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const timerRef = useRef(null);
  const abortRef = useRef(null);

  useEffect(() => {
    let mounted = true;

    async function load() {
      try {
        setError(null);
        abortRef.current?.abort?.();
        const ac = new AbortController();
        abortRef.current = ac;
        const payload = await fetchDashboardSummary(apiBase, { signal: ac.signal });
        if (!mounted) return;
        setData(payload);
        setLoading(false);
      } catch (e) {
        if (!mounted) return;
        if (e?.name === "AbortError") return;
        setError(String(e?.message || e));
        setLoading(false);
      }
    }

    load();
    timerRef.current = setInterval(load, intervalMs);
    return () => {
      mounted = false;
      abortRef.current?.abort?.();
      if (timerRef.current) clearInterval(timerRef.current);
    };
  }, [apiBase, intervalMs]);

  return { data, loading, error };
}
