/** Normalize model-family codes used across monitor, dashboard, and settings UI. */
export type ModelCode = "TCN" | "CTR_GCN";

export function normalizeModelCode(code: unknown): ModelCode {
  const v = String(code || "").trim().toUpperCase().replace("-", "_");
  if (v === "TCN") return "TCN";
  if (v === "CTR_GCN" || v === "CTRGCN") return "CTR_GCN";
  return "TCN";
}

export function modelCodeToLabel(code: unknown): string {
  return normalizeModelCode(code).replace("_", "-");
}

export function modelLabelToCode(label: unknown): ModelCode {
  const v = String(label || "").toLowerCase();
  if (v.includes("ctr")) return "CTR_GCN";
  if (v.includes("tcn")) return "TCN";
  return "TCN";
}
