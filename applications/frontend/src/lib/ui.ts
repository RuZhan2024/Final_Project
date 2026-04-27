/** Small shared UI helpers for sliders and bounded numeric controls. */
export function clamp(n: unknown, lo: number, hi: number): number {
  // Invalid inputs clamp to the low bound so slider gradients never produce NaN CSS.
  const x = Number(n);
  return Math.max(lo, Math.min(hi, Number.isFinite(x) ? x : lo));
}

export function sliderBackground(value: unknown, min: number, max: number): string {
  // Slider fill is computed here so settings pages can share one visual contract.
  const v = clamp(value, min, max);
  const pct = ((v - min) / (max - min)) * 100;
  const fill = "#4F46E5";
  const rest = "#E5E7EB";
  return `linear-gradient(to right, ${fill} 0%, ${fill} ${pct}%, ${rest} ${pct}%, ${rest} 100%)`;
}
