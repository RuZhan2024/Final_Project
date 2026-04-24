import { useCallback, useEffect, useState } from "react";

import { fetchReplayClips } from "../../../features/monitor/api";
import type { ReplayClip } from "../../../features/monitor/types";
import type { ReplayClipsState } from "../types";

/**
 * Load replay clips once when the replay picker is active.
 */
export function useReplayClips(apiBase: string, isActive = true): ReplayClipsState {
  const [clips, setClips] = useState<ReplayClip[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [configuredDir, setConfiguredDir] = useState("");
  const [available, setAvailable] = useState(false);

  const refresh = useCallback(async () => {
    setLoading(true);
    setError("");
    try {
      const data = await fetchReplayClips(apiBase);
      setClips(data.clips);
      setConfiguredDir(data.configuredDir);
      setAvailable(data.available);
    } catch (err: unknown) {
      // Keep the picker in a known empty state so stale clips do not survive a failed refresh.
      setClips([]);
      setConfiguredDir("");
      setAvailable(false);
      setError(String((err as Error)?.message || err || "Failed to load replay clips."));
    } finally {
      // Loading state stays local to the picker; selecting a clip should not block the rest of the page.
      setLoading(false);
    }
  }, [apiBase]);

  useEffect(() => {
    // Camera-only sessions do not need replay directory checks on mount.
    if (!isActive) return;
    void refresh();
  }, [isActive, refresh]);

  return {
    clips,
    loading,
    error,
    configuredDir,
    available,
  };
}
