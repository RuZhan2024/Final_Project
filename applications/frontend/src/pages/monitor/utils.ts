import type { SpecModel } from "../../features/monitor/types";
import type { MonitorMode } from "./types";

export function clamp01(x: unknown): number {
  const n = Number(x);
  if (!Number.isFinite(n)) return 0;
  if (n < 0) return 0;
  if (n > 1) return 1;
  return n;
}

export function normModeFromCode(code: unknown): MonitorMode {
  const c = String(code || "").toUpperCase();
  if (c === "TCN") return "tcn";
  if (c === "GCN") return "gcn";
  if (c === "HYBRID") return "hybrid";
  return "tcn";
}

export function labelForTriage(triageState: unknown): string {
  const t = String(triageState || "").toLowerCase();
  if (t === "fall") return "Fall";
  if (t === "uncertain") return "Uncertain";
  return "No fall";
}

export function prettyModelTag(activeModelCode: unknown): string {
  const c = String(activeModelCode || "").toUpperCase();
  if (c === "TCN") return "TCN";
  if (c === "GCN") return "GCN";
  if (c === "HYBRID") return "HYBRID";
  return "TCN";
}

export function pickFirstByArch(models: SpecModel[], arch: string, datasetCode: string): string {
  const a = String(arch || "").toLowerCase();
  const d = String(datasetCode || "").toLowerCase();
  const arr = models || [];
  const matched = arr.find((x) => {
    const xa = String(x?.arch || "").toLowerCase();
    const xd = String(x?.dataset_code || x?.dataset || "").toLowerCase();
    return xa === a && (!d || xd === d);
  });
  if (matched?.id) return matched.id;
  const any = arr.find((x) => String(x?.arch || "").toLowerCase() === a);
  if (any?.id) return any.id;
  return matched?.id || "";
}

export function pickModelPair(models: SpecModel[], datasetCode: string): { tcn: string; gcn: string } {
  return {
    tcn: pickFirstByArch(models, "tcn", datasetCode),
    gcn: pickFirstByArch(models, "gcn", datasetCode),
  };
}

export function targetFpsForDataset(datasetCode: unknown): number {
  const ds = String(datasetCode || "").toLowerCase();
  const byDs: Record<string, number> = { le2i: 25, caucafall: 23 };
  const f = byDs[ds];
  return typeof f === "number" && Number.isFinite(f) && f > 0 ? f : 23;
}
