/** Frontend API-base resolution shared by pages and feature hooks. */
const RENDER_FRONTEND_HOST = "fall-detection-frontend.onrender.com";
const RENDER_BACKEND_URL = "https://fall-detection-backend-e7fg.onrender.com";

function detectHostedFallback(): string {
  // Render frontend deployments should talk to the paired hosted backend by default.
  if (typeof window === "undefined") return "";
  const host = String(window.location?.hostname || "").trim().toLowerCase();
  if (host === RENDER_FRONTEND_HOST) {
    return RENDER_BACKEND_URL;
  }
  return "";
}

export const DEFAULT_API_BASE =
  typeof process !== "undefined" &&
  process.env &&
  process.env.REACT_APP_API_BASE
    ? process.env.REACT_APP_API_BASE
    : detectHostedFallback() || "http://localhost:8000";

export function getApiBase(preferred?: string | null): string {
  // A caller override wins, otherwise fall back to env/host detection.
  return preferred || DEFAULT_API_BASE;
}
