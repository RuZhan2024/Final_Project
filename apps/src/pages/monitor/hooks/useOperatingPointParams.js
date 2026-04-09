import { useEffect, useMemo, useState } from "react";

import { fetchOperatingPoints } from "../../../features/monitor/api";

function normaliseUiParams(settingsPayload) {
  const sys = settingsPayload?.system || {};
  const ui = sys?.deploy_params?.ui || sys?.deploy_params || {};

  const op_code = ui?.op_code || sys?.active_op_code || null;

  const tau_high = ui?.tau_high != null ? Number(ui.tau_high) : null;
  const tau_low = ui?.tau_low != null ? Number(ui.tau_low) : null;

  const k = ui?.k ?? ui?.confirm ?? ui?.confirm_k ?? null;
  const n = ui?.n ?? ui?.confirm_n ?? null;
  const cooldown_s = ui?.cooldown_s ?? ui?.cooldownSec ?? ui?.cooldown_sec ?? null;

  return {
    op_code,
    tau_high: tau_high != null && Number.isFinite(tau_high) ? tau_high : null,
    tau_low: tau_low != null && Number.isFinite(tau_low) ? tau_low : null,
    k: k != null && Number.isFinite(Number(k)) ? Number(k) : null,
    n: n != null && Number.isFinite(Number(n)) ? Number(n) : null,
    cooldown_s: cooldown_s != null && Number.isFinite(Number(cooldown_s)) ? Number(cooldown_s) : null,
    hasUi: ui?.tau_high != null || ui?.tau_low != null || ui?.op_code != null || ui?.k != null || ui?.n != null || ui?.cooldown_s != null,
  };
}

async function loadLegacyOperatingPoint(apiBase, modelCode, datasetCode, activeOperatingPointId) {
  const codeRaw = String(modelCode || "TCN").toUpperCase();
  const tryCodes = codeRaw === "GCN" ? ["GCN"] : ["TCN"];

  for (const code of tryCodes) {
    try {
      const data = await fetchOperatingPoints(apiBase, code, datasetCode);
      const ops = Array.isArray(data?.operating_points)
        ? data.operating_points
        : Array.isArray(data)
          ? data
          : [];

      const byId = activeOperatingPointId != null
        ? ops.find((o) => Number(o.id) === Number(activeOperatingPointId))
        : null;

      const picked =
        byId ||
        ops.find((o) => String(o.code || o.op_code || "").toUpperCase() === "OP-2") ||
        ops[0] ||
        null;

      if (!picked) continue;

      const low = picked.thr_low_conf != null
        ? Number(picked.thr_low_conf)
        : picked.threshold_low != null
          ? Number(picked.threshold_low)
          : null;

      const high = picked.thr_high_conf != null
        ? Number(picked.thr_high_conf)
        : picked.threshold_high != null
          ? Number(picked.threshold_high)
          : null;

      return {
        op_code: picked.code || picked.op_code || null,
        tau_low: low != null && Number.isFinite(low) ? low : null,
        tau_high: high != null && Number.isFinite(high) ? high : null,
      };
    } catch {
      // ignore and try next code
    }
  }

  return null;
}

/**
 * Resolve operating point parameters from /api/settings (preferred) with a
 * legacy fallback to /api/operating_points.
 */
export function useOperatingPointParams({ apiBase, settingsPayload, modelCode }) {
  const uiParams = useMemo(() => normaliseUiParams(settingsPayload), [settingsPayload]);

  const [legacy, setLegacy] = useState({ tau_low: null, tau_high: null, op_code: null });

  useEffect(() => {
    let cancelled = false;

    // If YAML-derived deploy params exist, don't fetch legacy DB operating points.
    if (uiParams.hasUi && (uiParams.tau_high != null || uiParams.tau_low != null)) {
      setLegacy({ tau_low: null, tau_high: null, op_code: null });
      return;
    }

    const sys = settingsPayload?.system || {};
    const opId = sys?.active_operating_point;
    const datasetCode = String(sys?.active_dataset_code || "caucafall").toLowerCase();

    (async () => {
      const r = await loadLegacyOperatingPoint(apiBase, modelCode, datasetCode, opId);
      if (!cancelled) {
        setLegacy(r || { tau_low: null, tau_high: null, op_code: null });
      }
    })();

    return () => {
      cancelled = true;
    };
  }, [apiBase, settingsPayload, modelCode, uiParams.hasUi, uiParams.tau_high, uiParams.tau_low]);

  // Final resolved values (prefer UI-derived values, else legacy, else null)
  return {
    opCode: uiParams.op_code || legacy.op_code || null,
    tauLow: uiParams.tau_low != null ? uiParams.tau_low : legacy.tau_low,
    tauHigh: uiParams.tau_high != null ? uiParams.tau_high : legacy.tau_high,
    confirmK: uiParams.k,
    confirmN: uiParams.n,
    cooldownS: uiParams.cooldown_s,
  };
}
