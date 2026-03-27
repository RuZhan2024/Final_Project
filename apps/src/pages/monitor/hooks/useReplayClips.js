import { useCallback, useEffect, useState } from "react";

import { apiRequest, joinUrl } from "../../../lib/apiClient";

function classifyClipGroup(clip) {
  if (clip?.category === "fall") return "fall";
  if (clip?.category === "adl") return "adl";
  if (clip?.category === "other") return "other";
  const raw = `${clip?.path || ""} ${clip?.filename || ""} ${clip?.name || ""}`.toLowerCase();
  if (/(^|[/_\-\s])(adl|nonfall|non_fall|non-fall|normal|safe)([/_\-\s]|$)/.test(raw)) {
    return "adl";
  }
  if (/(^|[/_\-\s])(fall|falls|falling)([/_\-\s]|$)/.test(raw)) {
    return "fall";
  }
  return "other";
}

function normalizeClip(apiBase, clip) {
  if (!clip || typeof clip !== "object") return null;
  const relUrl = typeof clip.url === "string" ? clip.url : "";
  const normalized = {
    id: String(clip.id || clip.path || clip.filename || clip.name || ""),
    name: String(clip.name || clip.filename || clip.id || "Replay clip"),
    filename: String(clip.filename || clip.name || clip.id || "Replay clip"),
    path: String(clip.path || clip.id || ""),
    category: String(clip.category || ""),
    sizeBytes: Number(clip.size_bytes || 0),
    url: relUrl ? joinUrl(apiBase, relUrl) : "",
  };
  return {
    ...normalized,
    group: classifyClipGroup(normalized),
  };
}

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
      const data = await apiRequest(apiBase, "api/replay/clips");
      const next = Array.isArray(data?.clips) ? data.clips.map((clip) => normalizeClip(apiBase, clip)).filter(Boolean) : [];
      setClips(next);
      setConfiguredDir(String(data?.configured_dir || ""));
      setAvailable(Boolean(data?.available));
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
