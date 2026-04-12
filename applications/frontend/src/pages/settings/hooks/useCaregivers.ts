import { useCallback, useEffect, useRef, useState } from "react";

import { apiRequest } from "../../../lib/apiClient";
import type { CaregiverRecord } from "../../../features/settings/types";

interface CaregiversResponse {
  caregivers?: CaregiverRecord[];
  db_available?: boolean;
  [key: string]: unknown;
}

interface CaregiverUpsertPayload {
  id?: number | null;
  name?: string | null;
  telegram_chat_id?: string | null;
}

interface UseCaregiversResult {
  caregivers: CaregiverRecord[];
  loading: boolean;
  error: string | null;
  reload: () => Promise<void>;
  upsert: (payload: CaregiverUpsertPayload) => Promise<CaregiversResponse>;
}

export function useCaregivers(apiBase: string): UseCaregiversResult {
  const [caregivers, setCaregivers] = useState<CaregiverRecord[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const inFlightRef = useRef<Promise<CaregiversResponse> | null>(null);

  const reload = useCallback(async (): Promise<void> => {
    try {
      setError(null);
      setLoading(true);
      if (!inFlightRef.current) {
        inFlightRef.current = apiRequest<CaregiversResponse>(apiBase, "/api/caregivers");
      }
      const data = await inFlightRef.current;
      setCaregivers(Array.isArray(data?.caregivers) ? data.caregivers : []);
      setLoading(false);
    } catch (e: unknown) {
      setError(String((e as Error)?.message || e));
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
