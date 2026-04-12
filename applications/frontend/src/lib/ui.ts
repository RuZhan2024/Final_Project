export function clamp(n: unknown, lo: number, hi: number): number {
  const x = Number(n);
  return Math.max(lo, Math.min(hi, Number.isFinite(x) ? x : lo));
}

export function sliderBackground(value: unknown, min: number, max: number): string {
  const v = clamp(value, min, max);
  const pct = ((v - min) / (max - min)) * 100;
  const fill = "#4F46E5";
  const rest = "#E5E7EB";
  return `linear-gradient(to right, ${fill} 0%, ${fill} ${pct}%, ${rest} ${pct}%, ${rest} 100%)`;
}
