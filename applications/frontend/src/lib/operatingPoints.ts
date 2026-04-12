export const PRESET_LABELS = ["High Sensitivity", "Balanced", "Low Sensitivity"] as const;

export function normalizeOpCode(opCode: unknown): "OP-1" | "OP-2" | "OP-3" {
  const s = String(opCode || "").trim().toUpperCase().replace("_", "-");
  if (s === "OP1" || s === "OP-1") return "OP-1";
  if (s === "OP2" || s === "OP-2") return "OP-2";
  if (s === "OP3" || s === "OP-3") return "OP-3";
  return "OP-2";
}

export function presetFromOpCode(opCode: unknown): (typeof PRESET_LABELS)[number] {
  const c = String(normalizeOpCode(opCode)).toUpperCase();
  if (c === "OP-1") return "High Sensitivity";
  if (c === "OP-3") return "Low Sensitivity";
  return "Balanced";
}

export function opCodeForPreset(presetLabel: unknown): "OP-1" | "OP-2" | "OP-3" {
  const p = String(presetLabel || "").toLowerCase();
  if (p.includes("high")) return "OP-1";
  if (p.includes("low")) return "OP-3";
  return "OP-2";
}

export function pickOperatingPoint<T extends { id?: number | null; op_code?: string | null; name?: string | null }>(
  operatingPoints: T[] | null | undefined,
  presetLabel: unknown
) {
  const ops = Array.isArray(operatingPoints) ? operatingPoints : [];
  if (!ops.length) return null;

  const want = opCodeForPreset(presetLabel).toLowerCase();

  const byCode = ops.find((o) => String(o.op_code || "").toLowerCase() === want);
  if (byCode) return byCode;

  const byName = ops.find((o) => {
    const n = String(o.name || "").toLowerCase();
    if (want === "op-1") return n.includes("high") || n.includes("recall");
    if (want === "op-3") return n.includes("low") || n.includes("alarm");
    return n.includes("bal");
  });
  if (byName) return byName;

  const sorted = [...ops].sort((a, b) => Number(a.id) - Number(b.id));
  if (want === "op-1") return sorted[0] || null;
  if (want === "op-3") return sorted[sorted.length - 1] || null;
  return sorted[Math.floor(sorted.length / 2)] || null;
}
