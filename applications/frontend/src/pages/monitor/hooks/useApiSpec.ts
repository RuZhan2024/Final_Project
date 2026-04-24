import { useEffect, useState } from "react";

import { fetchMonitorSpecModels } from "../../../features/monitor/api";
import type { SpecModel } from "../../../features/monitor/types";
import type { ApiSpecState } from "../types";

/**
 * Load /api/spec and normalize it into the shape used by the Monitor UI.
 */
export function useApiSpec(apiBase: string, enabled = true): ApiSpecState {
  const [models, setModels] = useState<SpecModel[]>([]);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    // Hidden monitor pages can skip spec loading until they become active.
    if (!enabled) return undefined;

    let cancelled = false;

    (async () => {
      try {
        setError(null);
        const models = await fetchMonitorSpecModels(apiBase);
        // Replace the whole model list so UI selectors always reflect one backend snapshot.
        if (!cancelled) setModels(models);
      } catch (e: unknown) {
        if (!cancelled) {
          // Clear models on failure so operating-point selection cannot use stale specs.
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
