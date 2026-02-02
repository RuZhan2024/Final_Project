import React, { useMemo } from "react";

import styles from "./Monitor.module.css";

import { useMonitoring } from "../monitoring/MonitoringContext";

import { useApiSpec } from "./monitor/hooks/useApiSpec";
import { useApiSummary } from "./monitor/hooks/useApiSummary";
import { useOperatingPointParams } from "./monitor/hooks/useOperatingPointParams";
import { usePoseMonitor } from "./monitor/hooks/usePoseMonitor";

import { ControlsCard } from "./monitor/components/ControlsCard";
import { LiveMonitorCard } from "./monitor/components/LiveMonitorCard";
import { ModelInfoCard } from "./monitor/components/ModelInfoCard";
import { TimelineCard } from "./monitor/components/TimelineCard";

import { normModeFromCode, pickFirstByArch, pickModelPair, prettyModelTag, targetFpsForDataset } from "./monitor/utils";

function safeNumber(x, fallback = null) {
  const n = Number(x);
  return Number.isFinite(n) ? n : fallback;
}

function Monitor({ isActive = true } = {}) {
  const {
    monitoringOn,
    setMonitoringOn,
    registerController,
    error: monitoringErr,
    settings: settingsPayload,
    loaded: settingsLoaded,
    apiBase,
  } = useMonitoring();

  // ---- Settings → derived runtime config ----
  const deployW = useMemo(() => safeNumber(settingsPayload?.deploy?.window?.W, 48), [settingsPayload]);
  const deployS = useMemo(() => safeNumber(settingsPayload?.deploy?.window?.S, 12), [settingsPayload]);

  const mcCfg = useMemo(() => {
    const mc = settingsPayload?.deploy?.mc;
    const M = safeNumber(mc?.M, 10);
    const M_confirm = safeNumber(mc?.M_confirm, 25);
    return { M, M_confirm };
  }, [settingsPayload]);

  const activeModelCode = useMemo(() => {
    const code = settingsPayload?.system?.active_model_code;
    return code ? String(code) : "GCN";
  }, [settingsPayload]);

  const activeDatasetCode = useMemo(() => {
    const code = settingsPayload?.system?.active_dataset_code;
    return code ? String(code) : "muvim";
  }, [settingsPayload]);

  const mcEnabled = useMemo(() => {
    const v = settingsPayload?.system?.mc_enabled;
    return v == null ? true : Boolean(v);
  }, [settingsPayload]);

  const fallThreshold = useMemo(() => {
    const v = settingsPayload?.system?.fall_threshold;
    return typeof v === "number" && Number.isFinite(v) ? Number(v) : null;
  }, [settingsPayload]);

  const mode = useMemo(() => normModeFromCode(activeModelCode), [activeModelCode]);
  const modelTag = useMemo(() => prettyModelTag(activeModelCode), [activeModelCode]);
  const targetFps = useMemo(() => targetFpsForDataset(activeDatasetCode), [activeDatasetCode]);

  // ---- Backend spec + model picking ----
  const { models, error: modelsErr } = useApiSpec(apiBase);

  const chosen = useMemo(() => {
    if (mode === "dual") return pickModelPair(models, activeDatasetCode);
    if (mode === "tcn") return { tcn: pickFirstByArch(models, "tcn", activeDatasetCode), gcn: "" };
    return { tcn: "", gcn: pickFirstByArch(models, "gcn", activeDatasetCode) };
  }, [mode, models, activeDatasetCode]);

  const chosenSpec = useMemo(() => {
    if (mode === "tcn") return models.find((m) => m.id === chosen.tcn) || null;
    if (mode === "gcn") return models.find((m) => m.id === chosen.gcn) || null;
    // dual: show GCN as primary tag in UI, but keep both
    return models.find((m) => m.id === chosen.gcn) || models.find((m) => m.id === chosen.tcn) || null;
  }, [mode, models, chosen]);

  // ---- Operating point params (YAML-derived preferred; legacy DB fallback) ----
  const { opCode, tauLow, tauHigh, confirmK, confirmN, cooldownS } = useOperatingPointParams({
    apiBase,
    settingsPayload,
    modelCode: activeModelCode,
  });

  const resolvedTauHigh = useMemo(() => {
    if (tauHigh != null) return tauHigh;
    if (fallThreshold != null) return fallThreshold;
    if (chosenSpec?.tau_high != null) return Number(chosenSpec.tau_high);
    return null;
  }, [tauHigh, fallThreshold, chosenSpec]);

  // ---- Optional server summary ----
  const { summary: apiSummary, error: summaryErr } = useApiSummary(apiBase);

  // ---- Live pipeline (camera + pose + inference) ----
  const {
    videoRef,
    canvasRef,
    currentPrediction,
    pText,
    sigma,
    markers,
    captureFpsText,
    modelFpsText,
    resetSession,
    testFall,
  } = usePoseMonitor({
    apiBase,
    isActive,
    monitoringOn,
    registerController,
    settingsPayload,
    deployW,
    deployS,
    targetFps,
    mode,
    chosen,
    opCode,
    mcEnabled,
    mcCfg,
    activeDatasetCode,
    chosenSpec,
  });

  // If settings are still loading, keep UI stable but show placeholders.
  const showPlaceholders = !settingsLoaded;

  return (
    <div className={styles.pageContainer}>
      <h2 className={styles.pageTitle}>Live Monitor</h2>

      <div className={styles.content}>
        {/* LEFT COLUMN (2/3) */}
        <div className={styles.leftColumn}>
          <LiveMonitorCard
            videoRef={videoRef}
            canvasRef={canvasRef}
            currentPrediction={showPlaceholders ? "—" : currentPrediction}
            pText={showPlaceholders ? "—" : pText}
          />
        </div>

        {/* RIGHT COLUMN (1/3) */}
        <div className={styles.rightColumn}>
          <ControlsCard
            monitoringOn={monitoringOn}
            setMonitoringOn={setMonitoringOn}
            resetSession={resetSession}
            testFall={testFall}
            modelsErr={modelsErr}
            monitoringErr={monitoringErr}
            summaryErr={summaryErr}
            apiSummary={apiSummary}
          />

          <TimelineCard markers={markers} />

          <ModelInfoCard
            modelTag={modelTag}
            deployW={deployW}
            deployS={deployS}
            tauLow={tauLow}
            tauHigh={resolvedTauHigh}
            opCode={opCode}
            confirmK={confirmK}
            confirmN={confirmN}
            cooldownS={cooldownS}
            mode={mode}
            captureFpsText={captureFpsText}
            modelFpsText={modelFpsText}
            mcCfg={mcCfg}
            sigma={sigma}
          />
        </div>
      </div>
    </div>
  );
}

export default Monitor;
