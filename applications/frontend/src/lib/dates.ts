export function toISODateInput(d: Date) {
  const pad = (n: number) => String(n).padStart(2, "0");
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}`;
}

export function parseDateSafe(value: string | number | Date | null | undefined) {
  const d = new Date(value as any);
  return Number.isFinite(d.getTime()) ? d : null;
}

export function endOfDay(d: Date) {
  const x = new Date(d.getTime());
  x.setHours(23, 59, 59, 999);
  return x;
}
