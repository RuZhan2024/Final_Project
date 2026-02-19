export function clamp(n, lo, hi) {
  const x = Number(n);
  return Math.max(lo, Math.min(hi, Number.isFinite(x) ? x : lo));
}

// Helper for gradient slider background.
// (Uses explicit colors to match the existing theme.)
export function sliderBackground(value, min, max) {
  const v = clamp(value, min, max);
  const pct = ((v - min) / (max - min)) * 100;
  const fill = "#4F46E5";
  const rest = "#E5E7EB";
  return `linear-gradient(to right, ${fill} 0%, ${fill} ${pct}%, ${rest} ${pct}%, ${rest} 100%)`;
}