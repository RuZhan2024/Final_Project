// Shared helpers for event type + review status labels.

export function eventTypeLabel(type) {
  const t = String(type || "").toLowerCase();
  if (t === "fall") return "Fall";
  if (t === "uncertain") return "Uncertain";
  if (t === "not_fall") return "Safe";
  return type || "—";
}

export function eventStatusLabel(status) {
  const s = String(status || "").toLowerCase();
  if (s === "confirmed_fall") return "Confirmed";
  if (s === "false_alarm") return "False Alarm";
  if (s === "pending_review") return "Pending Review";
  if (s === "dismissed") return "Dismissed";
  return status || "—";
}

export const EVENT_STATUS_OPTIONS = [
  "pending_review",
  "confirmed_fall",
  "false_alarm",
  "dismissed",
];
