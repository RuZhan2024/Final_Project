/**
 * Caregiver settings data hook.
 *
 * This hook centralizes caregiver reads/writes for the settings page. It keeps
 * overlapping reload calls from racing and refreshes after writes so backend
 * defaults or schema-shaped fields are reflected immediately.
 */
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
  /**
   * Serializes caregiver fetches so repeated reloads share one request/result.
   *
   * The settings page can trigger refreshes from mount and from save handlers;
   * this guard keeps those paths from racing each other.
   */
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
      // Share one in-flight fetch across repeated reload triggers from mount
      // and save handlers so the page converges on a single result.
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
    async (payload: CaregiverUpsertPayload): Promise<CaregiversResponse> => {
      const data = await apiRequest<CaregiversResponse>(apiBase, "/api/caregivers", {
        method: "PUT",
        body: { ...payload },
      });
      // Re-read after writes so the UI reflects any backend defaults or schema-
      // dependent fields returned by the current deployment.
      await reload();
      return data;
    },
    [apiBase, reload]
  );

  return { caregivers, loading, error, reload, upsert };
}
