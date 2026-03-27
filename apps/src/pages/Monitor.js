import React, { useMemo } from "react";

import styles from "./Monitor.module.css";

import { useMonitoring } from "../monitoring/MonitoringContext";

import { useApiSpec } from "./monitor/hooks/useApiSpec";
import { useOperatingPointParams } from "./monitor/hooks/useOperatingPointParams";
import { usePoseMonitor } from "./monitor/hooks/usePoseMonitor";
import { useReplayClips } from "./monitor/hooks/useReplayClips";

import { ControlsCard } from "./monitor/components/ControlsCard";
import { LiveMonitorCard } from "./monitor/components/LiveMonitorCard";
import { ModelInfoCard } from "./monitor/components/ModelInfoCard";
import { TimelineCard } from "./monitor/components/TimelineCard";

import { normModeFromCode, pickFirstByArch, prettyModelTag, targetFpsForDataset } from "./monitor/utils";

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
    return code ? String(code) : "TCN";
  }, [settingsPayload]);

  const activeDatasetCode = useMemo(() => {
    const code = settingsPayload?.system?.active_dataset_code;
    return code ? String(code) : "caucafall";
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
  const targetFps = useMemo(() => targetFpsForDataset(activeDatasetCode), [activeDatasetCode]);

  // ---- Backend spec + model picking ----
  const { models, error: modelsErr } = useApiSpec(apiBase, isActive);
  const {
    clips: replayClips,
    loading: replayClipsLoading,
    error: replayClipsError,
    configuredDir: replayClipsDir,
    available: replayClipsAvailable,
    refresh: refreshReplayClips,
  } = useReplayClips(apiBase, isActive);

  const chosen = useMemo(() => {
    if (mode === "tcn") return { tcn: pickFirstByArch(models, "tcn", activeDatasetCode), gcn: "" };
    if (mode === "gcn") return { tcn: "", gcn: pickFirstByArch(models, "gcn", activeDatasetCode) };
    return {
      tcn: pickFirstByArch(models, "tcn", activeDatasetCode),
      gcn: pickFirstByArch(models, "gcn", activeDatasetCode),
    };
  }, [mode, models, activeDatasetCode]);

  const effectiveMode = useMemo(() => {
    if (mode === "tcn" && chosen.tcn) return "tcn";
    if (mode === "gcn" && chosen.gcn) return "gcn";
    if (mode === "hybrid" && chosen.tcn && chosen.gcn) return "hybrid";
    // Auto-fallback when selected arch is unavailable for the dataset.
    if (chosen.tcn) return "tcn";
    if (chosen.gcn) return "gcn";
    return mode;
  }, [mode, chosen]);

  const modelTag = useMemo(() => prettyModelTag(effectiveMode), [effectiveMode]);

  const chosenSpec = useMemo(() => {
    if (effectiveMode === "tcn") return models.find((m) => m.id === chosen.tcn) || null;
    if (effectiveMode === "hybrid") return null;
    return models.find((m) => m.id === chosen.gcn) || null;
  }, [effectiveMode, models, chosen]);

  const resolvedDatasetCode = useMemo(() => {
    const fromSpec =
      chosenSpec?.dataset_code ||
      chosenSpec?.dataset ||
      (effectiveMode === "tcn" ? (models.find((m) => m.id === chosen.tcn)?.dataset_code || models.find((m) => m.id === chosen.tcn)?.dataset) : null) ||
      (effectiveMode === "gcn" ? (models.find((m) => m.id === chosen.gcn)?.dataset_code || models.find((m) => m.id === chosen.gcn)?.dataset) : null);
    return String(fromSpec || activeDatasetCode || "caucafall");
  }, [chosenSpec, effectiveMode, models, chosen, activeDatasetCode]);

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

  // Monitor page no longer polls /api/summary (dashboard owns summary polling).
  const summaryErr = "";

  // ---- Live pipeline (camera + pose + inference) ----
  const {
    videoRef,
    canvasRef,
    currentPrediction,
    pText,
    safePrediction,
    recallPrediction,
    sigma,
    markers,
    timelineStatusText,
    captureFpsText,
    modelFpsText,
    resetSession,
    testFall,
    inputSource,
    captureResolutionPreset,
    setCaptureResolution,
    selectedVideoName,
    replayClip,
    setReplayClip,
    setInputMode,
    startError,
    startInfo,
    predictError,
    replayCurrentS,
    replayDurationS,
    seekReplay,
  } = usePoseMonitor({
    apiBase,
    isActive,
    monitoringOn,
    registerController,
    settingsPayload,
    deployW,
    deployS,
    targetFps,
    mode: effectiveMode,
    chosen,
    opCode,
    mcEnabled,
    mcCfg,
    activeDatasetCode: resolvedDatasetCode,
    chosenSpec,
    onAutoStop: () => setMonitoringOn(false),
  });

  const hasReplayFile = useMemo(() => Boolean(selectedVideoName), [selectedVideoName]);
  const toastMessages = useMemo(
    () =>
      [
        modelsErr ? { key: `models-${modelsErr}`, tone: "warning", text: `Backend error: ${modelsErr}` } : null,
        monitoringErr ? { key: `settings-${monitoringErr}`, tone: "warning", text: `Settings error: ${monitoringErr}` } : null,
        replayClipsError && inputSource === "video"
          ? { key: `replay-${replayClipsError}`, tone: "warning", text: `Replay clips error: ${replayClipsError}` }
          : null,
        summaryErr ? { key: `summary-${summaryErr}`, tone: "warning", text: `Summary error: ${summaryErr}` } : null,
        startError ? { key: `start-${startError}`, tone: "error", text: `Start error: ${startError}` } : null,
        startInfo && !startError ? { key: `info-${startInfo}`, tone: "info", text: startInfo } : null,
        predictError ? { key: `predict-${predictError}`, tone: "error", text: `Predict error: ${predictError}` } : null,
      ].filter(Boolean),
    [inputSource, modelsErr, monitoringErr, predictError, replayClipsError, startError, startInfo, summaryErr]
  );

  // If settings are still loading, keep UI stable but show placeholders.
  const showPlaceholders = !settingsLoaded;

  return (
    <div className={styles.pageContainer}>
      <div className={styles.toastContainer}>
        {toastMessages.map((toast) => (
          <div
            key={toast.key}
            className={`${styles.toastItem} ${
              toast.tone === "error"
                ? styles.toastError
                : toast.tone === "info"
                  ? styles.toastInfo
                  : styles.toastWarning
            }`}
          >
            {toast.text}
          </div>
        ))}
      </div>

      <h2 className={styles.pageTitle}>Live Monitor</h2>

      <div className={styles.content}>
        {/* LEFT COLUMN (2/3) */}
        <div className={styles.leftColumn}>
          <LiveMonitorCard
            videoRef={videoRef}
            canvasRef={canvasRef}
            currentPrediction={showPlaceholders ? "—" : currentPrediction}
            pText={showPlaceholders ? "—" : pText}
            safePrediction={showPlaceholders ? "—" : safePrediction}
            recallPrediction={showPlaceholders ? "—" : recallPrediction}
            inputSource={inputSource}
            captureFpsText={showPlaceholders ? "—" : captureFpsText}
            modelFpsText={showPlaceholders ? "—" : modelFpsText}
          />
        </div>

        {/* RIGHT COLUMN (1/3) */}
        <div className={styles.rightColumn}>
          <ControlsCard
            monitoringOn={monitoringOn}
            setMonitoringOn={setMonitoringOn}
            resetSession={resetSession}
            testFall={testFall}
            inputSource={inputSource}
            selectedVideoName={selectedVideoName}
            hasReplayFile={hasReplayFile}
            replayClips={replayClips}
            replayClipsLoading={replayClipsLoading}
            replayClipsError={replayClipsError}
            replayClipsDir={replayClipsDir}
            replayClipsAvailable={replayClipsAvailable}
            selectedReplayClipId={replayClip?.id || ""}
            onSwitchRealtime={() => {
              if (monitoringOn) setMonitoringOn(false);
              setInputMode("camera");
            }}
            onSwitchReplay={() => {
              if (monitoringOn) setMonitoringOn(false);
              setInputMode("video");
            }}
            captureResolutionPreset={captureResolutionPreset}
            onChangeCaptureResolution={(preset) => {
              if (monitoringOn) setMonitoringOn(false);
              setCaptureResolution(preset);
            }}
            onSelectReplayClip={(clipId) => {
              const nextClip = replayClips.find((clip) => clip.id === clipId) || null;
              setReplayClip(nextClip);
            }}
            onRefreshReplayClips={refreshReplayClips}
            onClearReplay={() => setReplayClip(null)}
            replayCurrentS={replayCurrentS}
            replayDurationS={replayDurationS}
            onSeekReplay={seekReplay}
          />

          <TimelineCard markers={markers} statusText={timelineStatusText} />

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
            mode={effectiveMode}
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
