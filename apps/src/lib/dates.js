// Date helpers for the UI.

export function toISODateInput(d) {
  const pad = (n) => String(n).padStart(2, "0");
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}`;
}

export function parseDateSafe(value) {
  const d = new Date(value);
  return Number.isFinite(d.getTime()) ? d : null;
}

export function endOfDay(d) {
  const x = new Date(d.getTime());
  x.setHours(23, 59, 59, 999);
  return x;
}
