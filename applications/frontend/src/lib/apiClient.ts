/**
 * Shared frontend HTTP transport helpers.
 *
 * Feature-level API modules use this file so URL joining, JSON encoding,
 * raw-response escapes, and backend error propagation behave consistently
 * across pages.
 */
export interface ApiRequestOptions extends Omit<RequestInit, "body" | "headers" | "method"> {
  body?: BodyInit | Record<string, unknown> | unknown[] | null;
  headers?: Record<string, string>;
  method?: string;
  raw?: boolean;
}

export interface ApiRequestError extends Error {
  status?: number;
  url?: string;
}

export function joinUrl(base: string | undefined | null, path: string | undefined | null): string {
  /** Join API base and route path without doubled or missing slashes. */
  const normalizedBase = String(base || "").replace(/\/+$/, "");
  const normalizedPath = String(path || "").replace(/^\/+/, "");
  return `${normalizedBase}/${normalizedPath}`;
}

export async function apiRequest<T = unknown>(
  base: string | undefined | null,
  path: string | undefined | null,
  options: ApiRequestOptions = {}
): Promise<T> {
  /**
   * Shared transport contract for the frontend API helpers.
   *
   * Object bodies default to JSON, while `raw: true` lets feature modules keep
   * access to the original `Response` when they need blob/stream handling.
   */
  const url = joinUrl(base, path);
  const method = options.method || "GET";
  const headers = { ...(options.headers || {}) };
  const hasBody = typeof options.body !== "undefined";

  if (hasBody && !headers["Content-Type"]) {
    // Default object payloads to JSON, while still letting callers override the
    // encoding explicitly for FormData or custom binary transports.
    headers["Content-Type"] = "application/json";
  }

  const response = await fetch(url, {
    ...options,
    method,
    headers,
    body:
      hasBody &&
      headers["Content-Type"] === "application/json" &&
      typeof options.body !== "string" &&
      !(options.body instanceof FormData) &&
      !(options.body instanceof Blob)
        ? JSON.stringify(options.body)
        : (options.body as BodyInit | null | undefined),
  });

  const contentType = response.headers.get("content-type") || "";
  const isJson = contentType.includes("application/json");

  if (!response.ok) {
    // Preserve backend error text when possible so higher-level hooks can show
    // route-specific failures instead of a generic HTTP status.
    const message = isJson
      ? JSON.stringify(await response.json().catch(() => ({})))
      : await response.text().catch(() => "");
    const error = new Error(message || `HTTP ${response.status}`) as ApiRequestError;
    error.status = response.status;
    error.url = url;
    throw error;
  }

  if (options.raw) {
    // Blob/stream callers opt into the untouched Response; everybody else gets
    // parsed JSON/text so feature code can stay transport-agnostic.
    return response as T;
  }

  if (isJson) {
    // Most feature modules consume parsed JSON and should not care about transport details.
    return (await response.json()) as T;
  }

  return (await response.text()) as T;
}
