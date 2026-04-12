import { useEffect, useState } from "react";

import { fetchMonitorSpecModels } from "../../../features/monitor/api";
import type { SpecModel } from "../../../features/monitor/types";
import type { ApiSpecState } from "../types";

/**
 * Loads /api/spec and normalises it into the shape used by the Monitor UI.
 */
export function useApiSpec(apiBase: string, enabled = true): ApiSpecState {
  const [models, setModels] = useState<SpecModel[]>([]);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!enabled) return undefined;

    let cancelled = false;

    (async () => {
      try {
        setError(null);
        const models = await fetchMonitorSpecModels(apiBase);
        if (!cancelled) setModels(models);
      } catch (e: unknown) {
        if (!cancelled) {
          setError(String((e as Error)?.message || e));
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
