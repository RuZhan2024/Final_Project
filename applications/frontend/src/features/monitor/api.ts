import { apiRequest, joinUrl, type ApiRequestOptions } from "../../lib/apiClient";
import type {
  ReplayClip,
  ReplayClipsResponse,
  SpecModel,
} from "./types";

/**
 * Frontend API helpers for monitor setup, replay assets, and session control.
 *
 * These functions normalize backend responses into the shapes expected by the
 * monitor UI. Transport details stay here so hooks/components can depend on one
 * stable contract.
 */
function normalizeSpecModel(model: Partial<SpecModel> | null | undefined): SpecModel {
  // Spec ids have appeared under several field names during API evolution.
  // Normalize once here so the rest of the UI can treat `id` as canonical.
  const id = model?.id || model?.key || model?.spec_key || "";
  const op2 = model?.ops?.["OP-2"] || model?.ops?.op2 || null;
  const tauLow = op2?.tau_low != null ? Number(op2.tau_low) : null;
  const tauHigh = op2?.tau_high != null ? Number(op2.tau_high) : null;
  return { ...model, id, tau_low: tauLow, tau_high: tauHigh };
}

function classifyClipGroup(clip: Partial<ReplayClip> | null | undefined): ReplayClip["group"] {
  if (clip?.category === "fall") return "fall";
  if (clip?.category === "adl") return "adl";
  if (clip?.category === "other") return "other";
  const raw = `${clip?.path || ""} ${clip?.filename || ""} ${clip?.name || ""}`.toLowerCase();
  // Replay clip metadata is not fully standardized across datasets, so folder
  // and filename hints are used as a fallback when category is missing.
  if (/(^|[/_\-\s])(adl|nonfall|non_fall|non-fall|normal|safe)([/_\-\s]|$)/.test(raw)) {
    return "adl";
  }
  if (/(^|[/_\-\s])(fall|falls|falling)([/_\-\s]|$)/.test(raw)) {
    return "fall";
  }
  return "other";
}

function normalizeReplayClip(
  apiBase: string,
  clip: Record<string, unknown> | null | undefined
): ReplayClip | null {
  /** Normalize one replay clip row into the UI-facing clip contract. */
  if (!clip || typeof clip !== "object") return null;
  const relUrl = typeof clip.url === "string" ? clip.url : "";
  // The media element needs an absolute URL; keep that rewrite in one place so
  // callers do not have to know whether the backend returned relative paths.
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

export async function fetchMonitorSpecModels(apiBase: string): Promise<SpecModel[]> {
  /** Load deploy spec models and normalize current/legacy response shapes. */
  const data = await apiRequest<{ models?: SpecModel[] } | SpecModel[]>(apiBase, "/api/spec");
  const models = Array.isArray(data)
    ? data
    : data && typeof data === "object" && Array.isArray(data.models)
      ? data.models
      : [];
  return models.map(normalizeSpecModel);
}

export async function fetchReplayClips(apiBase: string): Promise<ReplayClipsResponse> {
  /** Load replay clips plus directory-availability metadata for the UI. */
  const data = await apiRequest<{
    clips?: Record<string, unknown>[];
    configured_dir?: string;
    available?: boolean;
  }>(apiBase, "/api/replay/clips");
  return {
    clips: Array.isArray(data?.clips)
      ? data.clips.map((clip) => normalizeReplayClip(apiBase, clip)).filter(Boolean)
      : [],
    configuredDir: String(data?.configured_dir || ""),
    available: Boolean(data?.available),
  };
}

export async function fetchOperatingPoints(
  apiBase: string,
  modelCode: string,
  datasetCode = "caucafall"
): Promise<unknown> {
  // The backend falls back to CAUCAFall if dataset_code is omitted, so always
  // pass the active dataset when using this legacy compatibility path.
  return await apiRequest<unknown>(
    apiBase,
    `/api/operating_points?model_code=${encodeURIComponent(modelCode)}&dataset_code=${encodeURIComponent(datasetCode)}`
  );
}

export async function fetchReplayClipBlob(
  clipUrl: string,
  options: Pick<ApiRequestOptions, "signal"> = {}
): Promise<Blob> {
  // Binary clip fetches bypass apiRequest because the caller needs the raw Blob
  // rather than JSON parsing and API-shaped error payloads.
  const resp = await fetch(String(clipUrl || ""), {
    method: "GET",
    signal: options.signal,
  });
  if (!resp.ok) {
    throw new Error(`Replay clip fetch failed: HTTP ${resp.status}`);
  }
  return await resp.blob();
}

export function buildMonitorWebSocketUrl(apiBase: string): string {
  /** Build the monitor WebSocket URL from the configured API base. */
  let base = String(apiBase || "").trim();
  if (!base) throw new Error("Missing apiBase for WebSocket connection");
  if (base.endsWith("/")) base = base.slice(0, -1);

  let wsBase = base.replace(/^http:/i, "ws:").replace(/^https:/i, "wss:");
  let wsPath = "/api/monitor/ws";
  try {
    // Preserve any reverse-proxy path prefix so deployments under subpaths do
    // not hard-code the socket endpoint at the origin root.
    const parsed = new URL(base);
    wsBase = parsed.origin.replace(/^http:/i, "ws:").replace(/^https:/i, "wss:");
    wsPath = `${(parsed.pathname || "").replace(/\/+$/, "")}/api/monitor/ws`.replace(/\/{2,}/g, "/");
  } catch {
    // Fall back to string replacement for non-standard base values.
  }
  return `${wsBase}${wsPath}`;
}

export async function resetMonitorSession(
  apiBase: string,
  sessionId: string
): Promise<unknown> {
  /** Reset backend session state, with fallback to the legacy v1 route prefix. */
  const encoded = encodeURIComponent(String(sessionId || ""));
  try {
    return await apiRequest<unknown>(apiBase, `/api/monitor/reset_session?session_id=${encoded}`, {
      method: "POST",
    });
  } catch (err) {
    if (Number((err as { status?: number })?.status) !== 404) throw err;
    return await apiRequest<unknown>(apiBase, `/api/v1/monitor/reset_session?session_id=${encoded}`, {
      method: "POST",
    });
  }
}

export async function uploadSkeletonClip(
  apiBase: string,
  eventId: string | number,
  clipPayload: Record<string, unknown>
): Promise<unknown> {
  /** Upload a skeleton-only clip attachment for a persisted event id. */
  return await apiRequest<unknown>(apiBase, `/api/events/${encodeURIComponent(eventId)}/skeleton_clip`, {
    method: "POST",
    body: clipPayload,
  });
}
