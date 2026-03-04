import { useEffect, useState } from "react";

import { apiRequest } from "../../../lib/apiClient";

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
        const data = await apiRequest(apiBase, "/api/spec");
        const arr = Array.isArray(data?.models) ? data.models : Array.isArray(data) ? data : [];

        const norm = arr.map((m) => {
          const id = m?.id || m?.key || m?.spec_key || "";
          const op2 = m?.ops?.["OP-2"] || m?.ops?.op2 || null;
          const tau_low = op2?.tau_low != null ? Number(op2.tau_low) : null;
          const tau_high = op2?.tau_high != null ? Number(op2.tau_high) : null;
          return { ...m, id, tau_low, tau_high };
        });

        if (!cancelled) setModels(norm);
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
