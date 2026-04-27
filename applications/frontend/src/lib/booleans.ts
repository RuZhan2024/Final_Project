/** Boolean coercion helper for mixed API/settings payload values. */
export function readBool(value: unknown, fallback = false) {
  if (value == null) return Boolean(fallback);
  if (typeof value === "boolean") return value;
  if (typeof value === "number") return value !== 0;
  if (typeof value === "string") {
    // Settings payloads often arrive as DB/YAML strings, so normalize the common forms here.
    const normalized = value.trim().toLowerCase();
    if (!normalized) return Boolean(fallback);
    if (["1", "true", "yes", "on"].includes(normalized)) return true;
    if (["0", "false", "no", "off"].includes(normalized)) return false;
  }
  // Non-empty unknown strings still follow JS truthiness to match legacy callers.
  return Boolean(value);
}
