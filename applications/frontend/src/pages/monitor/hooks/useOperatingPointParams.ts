import { useMemo } from "react";

import type { SettingsResponse } from "../../../features/settings/types";
import type { MonitorOperatingPointParams } from "../types";

function normaliseUiParams(settingsPayload: SettingsResponse | null): MonitorOperatingPointParams {
  const sys = settingsPayload?.system || {};
  const ui = sys?.deploy_params?.ui || sys?.deploy_params || {};

  const rawOpCode = ui?.op_code ?? sys?.active_op_code ?? null;
  const tauHigh = ui?.tau_high != null ? Number(ui.tau_high) : null;
  const tauLow = ui?.tau_low != null ? Number(ui.tau_low) : null;
  const k = ui?.k ?? ui?.confirm ?? ui?.confirm_k ?? null;
  const n = ui?.n ?? ui?.confirm_n ?? null;
  const cooldown = ui?.cooldown_s ?? ui?.cooldownSec ?? ui?.cooldown_sec ?? null;

  return {
    opCode: rawOpCode != null ? String(rawOpCode) : null,
    tauLow: tauLow != null && Number.isFinite(tauLow) ? tauLow : null,
    tauHigh: tauHigh != null && Number.isFinite(tauHigh) ? tauHigh : null,
    confirmK: k != null && Number.isFinite(Number(k)) ? Number(k) : null,
    confirmN: n != null && Number.isFinite(Number(n)) ? Number(n) : null,
    cooldownS: cooldown != null && Number.isFinite(Number(cooldown)) ? Number(cooldown) : null,
  };
}

export function useOperatingPointParams({
  settingsPayload,
}: {
  settingsPayload: SettingsResponse | null;
}): MonitorOperatingPointParams {
  return useMemo(() => normaliseUiParams(settingsPayload), [settingsPayload]);
}
