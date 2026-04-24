/** Label helpers for event table/status UI. */
export function eventTypeLabel(type: unknown): string {
  const t = String(type || "").toLowerCase();
  if (t === "fall") return "Fall";
  if (t === "uncertain") return "Uncertain";
  if (t === "not_fall") return "Safe";
  const raw = String(type ?? "").trim();
  return raw || "—";
}

export function eventStatusLabel(status: unknown): string {
  const s = String(status || "").toLowerCase();
  if (s === "confirmed_fall") return "Confirmed";
  if (s === "false_alarm") return "Not a Fall";
  if (s === "pending_review") return "Pending Review";
  if (s === "dismissed") return "Dismissed";
  const raw = String(status ?? "").trim();
  return raw || "—";
}

export const EVENT_STATUS_OPTIONS = [
  // Keep these normalized values aligned with backend review-status contracts.
  "pending_review",
  "confirmed_fall",
  "false_alarm",
  "dismissed",
] as const;
