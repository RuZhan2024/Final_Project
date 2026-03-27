// Centralised app config helpers.

const RENDER_FRONTEND_HOST = "fall-detection-frontend.onrender.com";
const RENDER_BACKEND_URL = "https://fall-detection-backend-e7fg.onrender.com";

function detectHostedFallback() {
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

export function getApiBase(preferred) {
  return preferred || DEFAULT_API_BASE;
}
