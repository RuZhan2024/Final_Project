import { useCallback, useEffect, useState } from "react";

import { fetchReplayClips } from "../../../features/monitor/api";

export function useReplayClips(apiBase, isActive = true) {
  const [clips, setClips] = useState([]);
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
    } catch (err) {
      setClips([]);
      setConfiguredDir("");
      setAvailable(false);
      setError(String(err?.message || err || "Failed to load replay clips."));
    } finally {
      setLoading(false);
    }
  }, [apiBase]);

  useEffect(() => {
    if (!isActive) return;
    void refresh();
  }, [isActive, refresh]);

  return {
    clips,
    loading,
    error,
    configuredDir,
    available,
    refresh,
  };
}
