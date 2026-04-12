/**
 * Tiny fetch wrapper so every page handles API errors consistently.
 */

export function joinUrl(base, path) {
  const b = String(base || "").replace(/\/+$/, "");
  const p = String(path || "").replace(/^\/+/, "");
  return `${b}/${p}`;
}

export async function apiRequest(base, path, options = {}) {
  const url = joinUrl(base, path);
  const method = options.method || "GET";
  const headers = { ...(options.headers || {}) };
  const hasBody = typeof options.body !== "undefined";

  if (hasBody && !headers["Content-Type"]) {
    headers["Content-Type"] = "application/json";
  }

  const resp = await fetch(url, {
    method,
    headers,
    body:
      hasBody && headers["Content-Type"] === "application/json" && typeof options.body !== "string"
        ? JSON.stringify(options.body)
        : options.body,
    signal: options.signal,
  });

  const contentType = resp.headers.get("content-type") || "";
  const isJson = contentType.includes("application/json");

  if (!resp.ok) {
    const msg = isJson ? JSON.stringify(await resp.json().catch(() => ({}))) : await resp.text().catch(() => "");
    const err = new Error(msg || `HTTP ${resp.status}`);
    err.status = resp.status;
    err.url = url;
    throw err;
  }

  if (options.raw) return resp;
  if (isJson) return await resp.json();
  return await resp.text();
}
