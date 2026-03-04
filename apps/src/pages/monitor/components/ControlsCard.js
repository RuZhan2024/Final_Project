import React from "react";

import styles from "../../Monitor.module.css";

export function ControlsCard({
  monitoringOn,
  setMonitoringOn,
  resetSession,
  testFall,
  inputSource,
  selectedVideoName,
  hasReplayFile,
  onSwitchRealtime,
  onSwitchReplay,
  onPickVideo,
  onClearReplay,
  replayCurrentS,
  replayDurationS,
  onSeekReplay,
  startError,
  predictError,
  modelsErr,
  monitoringErr,
  summaryErr,
  apiSummary,
}) {
  const replayPct = replayDurationS > 0 ? Math.max(0, Math.min(100, (replayCurrentS / replayDurationS) * 100)) : 0;
  const fmt = (s) => {
    const v = Math.max(0, Math.floor(Number(s) || 0));
    const mm = String(Math.floor(v / 60)).padStart(2, "0");
    const ss = String(v % 60).padStart(2, "0");
    return `${mm}:${ss}`;
  };

  return (
    <div className={styles.card}>
      <h3>Controls</h3>

      {modelsErr && (
        <p className={styles.subText} style={{ color: "#B45309" }}>
          Backend error: {modelsErr}
        </p>
      )}
      {monitoringErr && (
        <p className={styles.subText} style={{ color: "#B45309" }}>
          Settings error: {monitoringErr}
        </p>
      )}
      {summaryErr && (
        <p className={styles.subText} style={{ color: "#B45309" }}>
          Summary error: {summaryErr}
        </p>
      )}
      {startError && (
        <p className={styles.subText} style={{ color: "#B91C1C" }}>
          Start error: {startError}
        </p>
      )}
      {predictError && (
        <p className={styles.subText} style={{ color: "#B91C1C" }}>
          Predict error: {predictError}
        </p>
      )}

      {!summaryErr && apiSummary?.system && (
        <p className={styles.subText}>
          API: {apiSummary.system.api_online ? "Online" : "Offline"} • Last latency: {apiSummary.system.last_latency_ms ?? "—"} ms
        </p>
      )}

      <div className={styles.buttonGroup}>
        <button
          className={styles.btnGray}
          style={{ opacity: inputSource === "camera" ? 1 : 0.75 }}
          onClick={onSwitchRealtime}
        >
          Realtime Mode
        </button>
        <button
          className={styles.btnGray}
          style={{ opacity: inputSource === "video" ? 1 : 0.75 }}
          onClick={onSwitchReplay}
        >
          Replay Mode
        </button>
      </div>

      <p className={styles.subText}>
        Source: {inputSource === "video" ? `Replay (${selectedVideoName || "no file"})` : "Realtime Camera"}
      </p>

      {inputSource === "video" ? (
        <>
          <div className={styles.buttonGroup}>
            <input
              type="file"
              accept="video/*"
              onChange={(e) => {
                if (monitoringOn) setMonitoringOn(false);
                onPickVideo?.(e?.target?.files?.[0] || null);
              }}
              style={{ maxWidth: 180 }}
            />
            <button
              className={styles.btnGray}
              onClick={() => {
                if (monitoringOn) setMonitoringOn(false);
                onClearReplay?.();
              }}
            >
              Clear File
            </button>
          </div>
          <div style={{ display: "flex", flexDirection: "column", gap: 6, marginBottom: 10 }}>
            <input
              type="range"
              min={0}
              max={100}
              value={replayPct}
              onChange={(e) => onSeekReplay?.(Number(e.target.value) / 100)}
              disabled={!hasReplayFile || replayDurationS <= 0}
            />
            <span className={styles.subText}>
              Replay: {fmt(replayCurrentS)} / {fmt(replayDurationS)}
            </span>
          </div>
          <div className={styles.buttonGroup}>
            {!monitoringOn ? (
              <button
                className={styles.btnGray}
                disabled={!hasReplayFile}
                onClick={() => {
                  resetSession();
                  setMonitoringOn(true);
                }}
                title={hasReplayFile ? "Play Replay" : "Choose a video file first"}
              >
                Play Replay
              </button>
            ) : (
              <button className={styles.btnGray} onClick={() => setMonitoringOn(false)}>
                Stop Replay
              </button>
            )}
            <button className={styles.btnRed} onClick={testFall}>
              Test Fall
            </button>
          </div>
        </>
      ) : (
        <div className={styles.buttonGroup}>
          {!monitoringOn ? (
            <button
              className={styles.btnGray}
              onClick={() => {
                resetSession();
                setMonitoringOn(true);
              }}
              title="Start Realtime"
            >
              Start Realtime
            </button>
          ) : (
            <button className={styles.btnGray} onClick={() => setMonitoringOn(false)}>
              Stop Realtime
            </button>
          )}
          <button className={styles.btnRed} onClick={testFall}>
            Test Fall
          </button>
        </div>
      )}
    </div>
  );
}
