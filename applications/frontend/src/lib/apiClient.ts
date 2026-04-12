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
  const normalizedBase = String(base || "").replace(/\/+$/, "");
  const normalizedPath = String(path || "").replace(/^\/+/, "");
  return `${normalizedBase}/${normalizedPath}`;
}

export async function apiRequest<T = unknown>(
  base: string | undefined | null,
  path: string | undefined | null,
  options: ApiRequestOptions = {}
): Promise<T> {
  const url = joinUrl(base, path);
  const method = options.method || "GET";
  const headers = { ...(options.headers || {}) };
  const hasBody = typeof options.body !== "undefined";

  if (hasBody && !headers["Content-Type"]) {
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
    const message = isJson
      ? JSON.stringify(await response.json().catch(() => ({})))
      : await response.text().catch(() => "");
    const error = new Error(message || `HTTP ${response.status}`) as ApiRequestError;
    error.status = response.status;
    error.url = url;
    throw error;
  }

  if (options.raw) {
    return response as T;
  }

  if (isJson) {
    return (await response.json()) as T;
  }

  return (await response.text()) as T;
}
