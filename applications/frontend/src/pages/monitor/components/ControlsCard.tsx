import React from "react";

import type { ReplayClip } from "../../../features/monitor/types";
import type { CaptureResolutionPreset, InputSource } from "../types";

import styles from "../../Monitor.module.css";

interface ControlsCardProps {
  monitoringOn: boolean;
  setMonitoringOn: (next: boolean) => void | Promise<boolean>;
  resetSession: () => void;
  testFall: () => void;
  inputSource: InputSource;
  selectedVideoName: string;
  hasReplayFile: boolean;
  replayClips: ReplayClip[];
  replayClipsLoading: boolean;
  replayClipsError: string;
  replayClipsDir: string;
  replayClipsAvailable: boolean;
  selectedReplayClipId: string;
  onSwitchRealtime: () => void;
  onSwitchReplay: () => void;
  captureResolutionPreset: CaptureResolutionPreset | string;
  onChangeCaptureResolution?: (preset: string) => void;
  onSelectReplayClip?: (clipId: string) => void;
  onRefreshReplayClips?: () => void;
  onClearReplay?: () => void;
  replayCurrentS: number;
  replayDurationS: number;
  onSeekReplay?: (positionRatio: number) => void;
  replayPersistEvents: boolean;
  onToggleReplayPersist?: (next: boolean) => void;
}

export function ControlsCard({
  monitoringOn,
  setMonitoringOn,
  resetSession,
  testFall,
  inputSource,
  hasReplayFile,
  replayClips,
  replayClipsLoading,
  replayClipsError,
  replayClipsDir,
  replayClipsAvailable,
  selectedReplayClipId,
  onSwitchRealtime,
  onSwitchReplay,
  captureResolutionPreset,
  onChangeCaptureResolution,
  onSelectReplayClip,
  onRefreshReplayClips,
  onClearReplay,
  replayCurrentS,
  replayDurationS,
  onSeekReplay,
  replayPersistEvents,
  onToggleReplayPersist,
}: ControlsCardProps) {
  const replayPct = replayDurationS > 0 ? Math.max(0, Math.min(100, (replayCurrentS / replayDurationS) * 100)) : 0;
  const groupedReplayClips = (replayClips || []).reduce(
    (acc, clip) => {
      const key = clip?.group === "fall" ? "fall" : clip?.group === "adl" ? "adl" : "other";
      acc[key].push(clip);
      return acc;
    },
    { fall: [], adl: [], other: [] }
  );
  const fmt = (s: number) => {
    const v = Math.max(0, Math.floor(Number(s) || 0));
    const mm = String(Math.floor(v / 60)).padStart(2, "0");
    const ss = String(v % 60).padStart(2, "0");
    return `${mm}:${ss}`;
  };

  return (
    <div className={styles.card}>
      <h3>Controls</h3>

      <div className={styles.buttonGroup}>
        <button
          className={`${styles.btnGray} ${inputSource === "camera" ? styles.btnActive : ""}`}
          onClick={onSwitchRealtime}
        >
          Realtime Mode
        </button>
        <button
          className={`${styles.btnGray} ${inputSource === "video" ? styles.btnActive : ""}`}
          onClick={onSwitchReplay}
        >
          Replay Mode
        </button>
      </div>

      {inputSource === "camera" ? (
        <div className={styles.controlSection}>
          <select
            id="capture-resolution-select"
            value={captureResolutionPreset || "720p"}
            onChange={(e) => onChangeCaptureResolution?.(e.target.value)}
            className={styles.selectInput}
          >
            <option value="480p">480p (640×480)</option>
            <option value="540p">540p (960×540)</option>
            <option value="720p">720p (1280×720)</option>
            <option value="1080p">1080p (1920×1080)</option>
          </select>
        </div>
      ) : null}

      {inputSource === "video" ? (
        <>
          {!replayClipsAvailable && replayClipsDir ? (
            <div className={styles.inlineInfoWarning}>
              Replay clip directory is configured but currently unavailable: {replayClipsDir}
            </div>
          ) : null}
          {replayClipsError ? (
            <div className={styles.inlineInfoWarning}>{replayClipsError}</div>
          ) : null}
          <div className={`${styles.controlSection} ${styles.stackSection}`}>
            <select
              id="replay-clip-select"
              value={selectedReplayClipId || ""}
              disabled={replayClipsLoading || !(replayClips || []).length}
              onChange={(e) => {
                if (monitoringOn) void setMonitoringOn(false);
                onSelectReplayClip?.(e.target.value);
              }}
              className={styles.selectInput}
            >
              <option value="">{replayClipsLoading ? "Loading clips..." : "Select a replay clip"}</option>
              {groupedReplayClips.fall.length ? (
                <optgroup label="Fall Clips">
                  {groupedReplayClips.fall.map((clip) => (
                    <option key={clip.id} value={clip.id}>
                      {clip.name}
                    </option>
                  ))}
                </optgroup>
              ) : null}
              {groupedReplayClips.adl.length ? (
                <optgroup label="ADL Clips">
                  {groupedReplayClips.adl.map((clip) => (
                    <option key={clip.id} value={clip.id}>
                      {clip.name}
                    </option>
                  ))}
                </optgroup>
              ) : null}
              {groupedReplayClips.other.length ? (
                <optgroup label="Fall Clips">
                  {groupedReplayClips.other.map((clip) => (
                    <option key={clip.id} value={clip.id}>
                      {clip.name}
                    </option>
                  ))}
                </optgroup>
              ) : null}
            </select>
            <div className={styles.buttonGroup}>
              <button className={styles.btnGray} onClick={() => onRefreshReplayClips?.()} disabled={replayClipsLoading}>
                {replayClipsLoading ? "Refreshing..." : "Refresh Clips"}
              </button>
            </div>
          </div>
          <div className={styles.toggleRow}>
            <span>Persist Replay Events</span>
            <label className={styles.switch}>
              <input
                type="checkbox"
                checked={Boolean(replayPersistEvents)}
                onChange={(e) => onToggleReplayPersist?.(e.target.checked)}
              />
              <span className={styles.slider}></span>
            </label>
          </div>
          <div className={styles.inlineInfo}>
            <div className={styles.inlineInfoTitle}>Replay Persistence</div>
            <div>
              {replayPersistEvents
                ? "Replay detections will be written to event history and can trigger notifications."
                : "Replay detections stay visual-only unless persistence is explicitly enabled."}
            </div>
          </div>
          <div className={styles.buttonGroup}>
            {!monitoringOn ? (
              <button
                className={styles.btnGray}
                disabled={!hasReplayFile}
                onClick={() => {
                  resetSession();
                  void setMonitoringOn(true);
                }}
                title={hasReplayFile ? "Play Replay" : "Choose a video file first"}
              >
                Play Replay
              </button>
            ) : (
              <button className={styles.btnGray} onClick={() => void setMonitoringOn(false)}>
                Stop Replay
              </button>
            )}
            <button
              className={styles.btnGray}
              onClick={() => {
                if (monitoringOn) void setMonitoringOn(false);
                onClearReplay?.();
              }}
            >
              Clear
            </button>
          </div>
          <div className={styles.rangeSection}>
            <input
              className={styles.rangeInput}
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
        </>
      ) : (
        <div className={styles.buttonGroup}>
          {!monitoringOn ? (
            <button
              className={styles.btnGray}
              onClick={() => {
                resetSession();
                void setMonitoringOn(true);
              }}
              title="Start Realtime"
            >
              Start Realtime
            </button>
          ) : (
            <button className={styles.btnGray} onClick={() => void setMonitoringOn(false)}>
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
