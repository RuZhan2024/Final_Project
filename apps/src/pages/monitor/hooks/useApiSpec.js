import { useEffect, useState } from "react";

import { fetchMonitorSpecModels } from "../../../features/monitor/api";

/**
 * Loads /api/spec and normalises it into the shape used by the Monitor UI.
 */
export function useApiSpec(apiBase, enabled = true) {
  const [models, setModels] = useState([]);
  const [error, setError] = useState(null);

  useEffect(() => {
    if (!enabled) return undefined;

    let cancelled = false;

    (async () => {
      try {
        setError(null);
        const models = await fetchMonitorSpecModels(apiBase);
        if (!cancelled) setModels(models);
      } catch (e) {
        if (!cancelled) {
          setError(String(e?.message || e));
          setModels([]);
        }
      }
    })();

    return () => {
      cancelled = true;
    };
  }, [apiBase, enabled]);

  return { models, error };
}
