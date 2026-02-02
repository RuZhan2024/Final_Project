// Shared helpers for mapping model codes/labels.

export function normalizeModelCode(code) {
  const v = String(code || "").trim().toUpperCase();
  if (v === "TCN") return "TCN";
  if (v === "GCN") return "GCN";
  if (v === "HYBRID" || v === "TCN+GCN" || v === "TCN_GCN") return "HYBRID";
  return v || "HYBRID";
}

export function modelCodeToLabel(code) {
  const v = normalizeModelCode(code);
  if (v === "HYBRID") return "Hybrid";
  return v;
}

export function modelLabelToCode(label) {
  const v = String(label || "").toLowerCase();
  if (v.includes("tcn") && v.includes("gcn")) return "HYBRID";
  if (v.includes("tcn")) return "TCN";
  if (v.includes("gcn")) return "GCN";
  if (v.includes("hybrid")) return "HYBRID";
  return "HYBRID";
}
