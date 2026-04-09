import { useCallback, useEffect, useRef, useState } from "react";

import { apiRequest } from "../../../lib/apiClient";

export function useCaregivers(apiBase) {
  const [caregivers, setCaregivers] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const inFlightRef = useRef(null);

  const reload = useCallback(async () => {
    try {
      setError(null);
      setLoading(true);
      if (!inFlightRef.current) {
        inFlightRef.current = apiRequest(apiBase, "/api/caregivers");
      }
      const data = await inFlightRef.current;
      setCaregivers(Array.isArray(data?.caregivers) ? data.caregivers : []);
      setLoading(false);
    } catch (e) {
      setError(String(e?.message || e));
      setLoading(false);
    } finally {
      inFlightRef.current = null;
    }
  }, [apiBase]);

  useEffect(() => {
    reload();
  }, [reload]);

  const upsert = useCallback(
    async (payload) => {
      const data = await apiRequest(apiBase, "/api/caregivers", {
        method: "PUT",
        body: { ...payload },
      });
      await reload();
      return data;
    },
    [apiBase, reload]
  );

  return { caregivers, loading, error, reload, upsert };
}
