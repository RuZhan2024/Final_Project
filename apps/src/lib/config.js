// Centralised app config helpers.

export const DEFAULT_API_BASE =
  typeof process !== "undefined" &&
  process.env &&
  process.env.REACT_APP_API_BASE
    ? process.env.REACT_APP_API_BASE
    : "http://localhost:8000";

export function getApiBase(preferred) {
  return preferred || DEFAULT_API_BASE;
}
