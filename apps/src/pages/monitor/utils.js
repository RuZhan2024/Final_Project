export function clamp01(x) {
  const n = Number(x);
  if (!Number.isFinite(n)) return 0;
  if (n < 0) return 0;
  if (n > 1) return 1;
  return n;
}

export function normModeFromCode(code) {
  const c = String(code || "").toUpperCase();
  if (c === "TCN") return "tcn";
  if (c === "GCN") return "gcn";
  if (c === "HYBRID") return "dual";
  return "gcn";
}

export function labelForTriage(triageState) {
  const t = String(triageState || "").toLowerCase();
  if (t === "fall") return "Fall";
  if (t === "uncertain") return "Uncertain";
  return "No fall";
}

export function prettyModelTag(activeModelCode) {
  const c = String(activeModelCode || "").toUpperCase();
  if (c === "HYBRID") return "HYBRID";
  if (c === "TCN") return "TCN";
  if (c === "GCN") return "GCN";
  return "GCN";
}

export function pickFirstByArch(models, arch, datasetCode) {
  const a = String(arch || "").toLowerCase();
  const d = String(datasetCode || "").toLowerCase();
  const m = (models || []).find((x) => {
    const xa = String(x?.arch || "").toLowerCase();
    const xd = String(x?.dataset_code || x?.dataset || "").toLowerCase();
    return xa === a && (!d || xd === d);
  });
  return m?.id || "";
}

export function pickModelPair(models, datasetCode) {
  return {
    tcn: pickFirstByArch(models, "tcn", datasetCode),
    gcn: pickFirstByArch(models, "gcn", datasetCode),
  };
}

export function targetFpsForDataset(datasetCode) {
  const ds = String(datasetCode || "").toLowerCase();
  const byDs = { le2i: 25, urfd: 30, caucafall: 23, muvim: 30 };
  const f = byDs[ds];
  return typeof f === "number" && Number.isFinite(f) && f > 0 ? f : 30;
}
