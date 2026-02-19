import { useCallback, useEffect, useRef, useState } from "react";

import { apiRequest } from "../../../lib/apiClient";

export function useCaregivers(apiBase, residentId = 1) {
  const [caregivers, setCaregivers] = useState([]);
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

      const data = await apiRequest(apiBase, `/api/caregivers?resident_id=${residentId}`, {
        signal: ac.signal,
      });
      setCaregivers(Array.isArray(data?.caregivers) ? data.caregivers : []);
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

  const upsert = useCallback(
    async (payload) => {
      await apiRequest(apiBase, "/api/caregivers", {
        method: "PUT",
        body: { resident_id: residentId, ...payload },
      });
      await reload();
    },
    [apiBase, residentId, reload]
  );

  return { caregivers, loading, error, reload, upsert };
}
