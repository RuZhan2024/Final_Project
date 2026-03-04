// Shared helpers for mapping model codes/labels.

export function normalizeModelCode(code) {
  const v = String(code || "").trim().toUpperCase();
  if (v === "TCN") return "TCN";
  if (v === "GCN") return "GCN";
  if (v === "HYBRID") return "HYBRID";
  return "TCN";
}

export function modelCodeToLabel(code) {
  const v = normalizeModelCode(code);
  return v;
}

export function modelLabelToCode(label) {
  const v = String(label || "").toLowerCase();
  if (v.includes("hybrid")) return "HYBRID";
  if (v.includes("tcn")) return "TCN";
  if (v.includes("gcn")) return "GCN";
  return "TCN";
}
