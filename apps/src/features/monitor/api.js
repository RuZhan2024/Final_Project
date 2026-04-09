import { apiRequest, joinUrl } from "../../lib/apiClient";

function normalizeSpecModel(model) {
  const id = model?.id || model?.key || model?.spec_key || "";
  const op2 = model?.ops?.["OP-2"] || model?.ops?.op2 || null;
  const tauLow = op2?.tau_low != null ? Number(op2.tau_low) : null;
  const tauHigh = op2?.tau_high != null ? Number(op2.tau_high) : null;
  return { ...model, id, tau_low: tauLow, tau_high: tauHigh };
}

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

function normalizeReplayClip(apiBase, clip) {
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

export async function fetchMonitorSpecModels(apiBase) {
  const data = await apiRequest(apiBase, "/api/spec");
  const models = Array.isArray(data?.models) ? data.models : Array.isArray(data) ? data : [];
  return models.map(normalizeSpecModel);
}

export async function fetchReplayClips(apiBase) {
  const data = await apiRequest(apiBase, "/api/replay/clips");
  return {
    clips: Array.isArray(data?.clips)
      ? data.clips.map((clip) => normalizeReplayClip(apiBase, clip)).filter(Boolean)
      : [],
    configuredDir: String(data?.configured_dir || ""),
    available: Boolean(data?.available),
  };
}

export async function fetchOperatingPoints(apiBase, modelCode) {
  return await apiRequest(apiBase, `/api/operating_points?model_code=${encodeURIComponent(modelCode)}`);
}

export async function fetchReplayClipBlob(clipUrl, options = {}) {
  const resp = await fetch(String(clipUrl || ""), {
    method: "GET",
    signal: options.signal,
  });
  if (!resp.ok) {
    throw new Error(`Replay clip fetch failed: HTTP ${resp.status}`);
  }
  return await resp.blob();
}

export function buildMonitorWebSocketUrl(apiBase) {
  let base = String(apiBase || "").trim();
  if (!base) throw new Error("Missing apiBase for WebSocket connection");
  if (base.endsWith("/")) base = base.slice(0, -1);

  let wsBase = base.replace(/^http:/i, "ws:").replace(/^https:/i, "wss:");
  let wsPath = "/api/monitor/ws";
  try {
    const parsed = new URL(base);
    wsBase = parsed.origin.replace(/^http:/i, "ws:").replace(/^https:/i, "wss:");
    wsPath = `${(parsed.pathname || "").replace(/\/+$/, "")}/api/monitor/ws`.replace(/\/{2,}/g, "/");
  } catch {
    // Fall back to string replacement for non-standard base values.
  }
  return `${wsBase}${wsPath}`;
}

export async function resetMonitorSession(apiBase, sessionId) {
  const encoded = encodeURIComponent(String(sessionId || ""));
  try {
    return await apiRequest(apiBase, `/api/monitor/reset_session?session_id=${encoded}`, {
      method: "POST",
    });
  } catch (err) {
    if (Number(err?.status) !== 404) throw err;
    return await apiRequest(apiBase, `/api/v1/monitor/reset_session?session_id=${encoded}`, {
      method: "POST",
    });
  }
}

export async function uploadSkeletonClip(apiBase, eventId, clipPayload) {
  return await apiRequest(apiBase, `/api/events/${encodeURIComponent(eventId)}/skeleton_clip`, {
    method: "POST",
    body: clipPayload,
  });
}

export async function triggerTestFall(apiBase, modelCode) {
  return await apiRequest(apiBase, "/api/events/test_fall", {
    method: "POST",
    body: { resident_id: 1, model_code: modelCode },
  });
}
