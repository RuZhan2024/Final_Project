export function normalizeModelCode(code: unknown) {
  const v = String(code || "").trim().toUpperCase();
  if (v === "TCN") return "TCN";
  if (v === "GCN") return "GCN";
  if (v === "HYBRID") return "HYBRID";
  return "TCN";
}

export function modelCodeToLabel(code: unknown) {
  return normalizeModelCode(code);
}

export function modelLabelToCode(label: unknown) {
  const v = String(label || "").toLowerCase();
  if (v.includes("hybrid")) return "HYBRID";
  if (v.includes("tcn")) return "TCN";
  if (v.includes("gcn")) return "GCN";
  return "TCN";
}
