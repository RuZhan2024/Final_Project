import { useCallback } from "react";

import { REPLAY_UI_UPDATE_MS } from "../constants";
import type { UseMonitorRuntimeControllerOptions } from "../types";

/**
 * Orchestrates start/stop and the main monitor runtime loop.
 *
 * This hook owns session resets, source preparation, replay end-of-clip
 * handling, and the requestAnimationFrame loop that feeds MediaPipe. It does
 * not build prediction payloads itself; that contract lives in the prediction
 * hook.
 */
export function useMonitorRuntimeController({
  targetFps,
  autoStopMonitoring,
  buildReplayPayload,
  drainReplayQueue,
  ensureCanvasMatchesVideo,
  getActiveReplaySource,
  initPosePipeline,
  makeSessionId,
  prepareCameraStream,
  prepareReplayVideo,
  resetFrontendSessionState,
  resetSession,
  stopLive,
  syncReplayTimeState,
  inputSourceRef,
  videoRef,
  streamRef,
  liveFlagRef,
  runTokenRef,
  rafRef,
  poseRef,
  poseSendBusyRef,
  predictInFlightRef,
  replayWindowQueueRef,
  adaptiveInferFpsRef,
  inferFpsRef,
  lastInferTsRef,
  lastReplayUiMsRef,
  sessionIdRef,
  fpsEstimateRef,
  setLiveRunning,
  setStartError,
  setStartInfo,
  setReplayCurrentS,
  setReplayDurationS,
}: UseMonitorRuntimeControllerOptions) {
  const startLive = useCallback(async () => {
    /**
     * Start a fresh monitoring run for either camera or replay input.
     *
     * Each invocation increments a run token. Any async step that resolves under
     * an older token must stop quietly so a restarted session cannot revive a
     * stale stream or replay pipeline.
     */
    const videoEl = videoRef.current;
    if (!videoEl) return false;

    const runToken = runTokenRef.current + 1;
    runTokenRef.current = runToken;

    try {
      sessionIdRef.current = makeSessionId();
      resetFrontendSessionState();
      await resetSession();
      setStartError("");
      setStartInfo("");
      liveFlagRef.current = true;
      setLiveRunning(true);

      const source = inputSourceRef.current || "camera";
      const useVideoFile = source === "video";
      const { clip, clipFile, clipUrl } = getActiveReplaySource();
      if (useVideoFile && !clipFile && !clipUrl) {
        throw new Error("Replay mode selected but no replay clip is loaded.");
      }

      if (useVideoFile) {
        await prepareReplayVideo(videoEl, clip, clipUrl);
        if (runTokenRef.current !== runToken) return false;
        setStartInfo("Preparing replay detector...");
      } else {
        await prepareCameraStream(videoEl);
        if (runTokenRef.current !== runToken) return false;
      }
      ensureCanvasMatchesVideo();
      if (useVideoFile) {
        syncReplayTimeState();
      }

      if (streamRef.current) {
        try {
          const track = streamRef.current.getVideoTracks()[0];
          const settings = track?.getSettings?.();
          if (settings && typeof settings.frameRate === "number") {
            fpsEstimateRef.current = settings.frameRate;
          }
        } catch {
          // ignore
        }
      }

      await initPosePipeline(videoEl, { warmup: true });
      if (runTokenRef.current !== runToken) return false;
      if (useVideoFile) {
        setStartInfo("");
      }

      try {
        await videoEl.play();
      } catch (e: any) {
        throw new Error(`Video play failed: ${String(e?.message || e)}`);
      }

      const loop = async () => {
        if (!liveFlagRef.current || runTokenRef.current !== runToken) return;
        if ((inputSourceRef.current || "camera") === "video" && videoEl.ended) {
          void buildReplayPayload();
          void drainReplayQueue();
          // Replay must finish queued prediction windows before stopping;
          // otherwise the final fall segment can disappear from review history.
          const replayBusy =
            predictInFlightRef.current || replayWindowQueueRef.current.length > 0;
          if (replayBusy) {
            setStartInfo("Finalizing replay predictions...");
            rafRef.current = requestAnimationFrame(loop);
            return;
          }
          setStartInfo("");
          stopLive();
          autoStopMonitoring();
          return;
        }
        if ((inputSourceRef.current || "camera") === "video") {
          const drawNowMs = performance.now();
          if (drawNowMs - lastReplayUiMsRef.current >= REPLAY_UI_UPDATE_MS) {
            lastReplayUiMsRef.current = drawNowMs;
            const ct = Number(videoEl.currentTime || 0);
            const du = Number(videoEl.duration || 0);
            if (Number.isFinite(ct)) {
              setReplayCurrentS((prev) => (Math.abs(Number(prev || 0) - ct) < 0.08 ? prev : ct));
            }
            if (Number.isFinite(du)) {
              setReplayDurationS((prev) => (Math.abs(Number(prev || 0) - du) < 0.08 ? prev : du));
            }
          }
        }

        const now = performance.now();
        const target = Math.max(
          1,
          Number(adaptiveInferFpsRef.current) || Number(inferFpsRef.current) || 15
        );
        const minIntervalMs = 1000 / target;

        if (now - lastInferTsRef.current >= minIntervalMs) {
          lastInferTsRef.current = now;

          if (
            videoEl.readyState >= 2 &&
            videoEl.videoWidth > 0 &&
            videoEl.videoHeight > 0 &&
            poseRef.current &&
            !poseSendBusyRef.current
          ) {
            // Keep only one outstanding MediaPipe send call. Concurrent sends can
            // race the latest frame and destabilize the capture cadence.
            poseSendBusyRef.current = true;
            poseRef.current
              .send({ image: videoEl })
              .catch((err: any) => {
                const msg = String(err?.message || err || "");
                if (!/no video|ROI width and height must be > 0|Aborted\(native code called abort\)/i.test(msg)) {
                  console.error("pose.send error", err);
                }
              })
              .finally(() => {
                if (runTokenRef.current === runToken) {
                  poseSendBusyRef.current = false;
                }
              });
          }
        }

        rafRef.current = requestAnimationFrame(loop);
      };

      void loop();
      return true;
    } catch (err: any) {
      console.error("Error starting monitor", err);
      setStartError(String(err?.message || err || "Failed to start monitor"));
      stopLive();
      return false;
    }
  }, [
    adaptiveInferFpsRef,
    autoStopMonitoring,
    buildReplayPayload,
    drainReplayQueue,
    ensureCanvasMatchesVideo,
    fpsEstimateRef,
    getActiveReplaySource,
    inferFpsRef,
    initPosePipeline,
    inputSourceRef,
    lastInferTsRef,
    lastReplayUiMsRef,
    liveFlagRef,
    makeSessionId,
    poseRef,
    poseSendBusyRef,
    predictInFlightRef,
    prepareCameraStream,
    prepareReplayVideo,
    rafRef,
    replayWindowQueueRef,
    resetFrontendSessionState,
    resetSession,
    runTokenRef,
    sessionIdRef,
    setLiveRunning,
    setReplayCurrentS,
    setReplayDurationS,
    setStartError,
    setStartInfo,
    stopLive,
    streamRef,
    syncReplayTimeState,
    videoRef,
  ]);

  const seekReplay = useCallback((ratio: number) => {
    /**
     * Seek replay video and start a fresh logical monitor session.
     *
     * Seeking invalidates previous alert/tracker context, so frontend state and
     * backend session state are both reset before the new position is analysed.
     */
    const videoEl = videoRef.current;
    if (!videoEl) return;
    const du = Number(videoEl.duration || 0);
    if (!Number.isFinite(du) || du <= 0) return;
    const r = Math.max(0, Math.min(1, Number(ratio) || 0));
    sessionIdRef.current = makeSessionId();
    resetFrontendSessionState();
    void resetSession();
    videoEl.currentTime = r * du;
    setReplayCurrentS(Number(videoEl.currentTime || 0));
    setReplayDurationS(du);
  }, [
    makeSessionId,
    resetFrontendSessionState,
    resetSession,
    sessionIdRef,
    setReplayCurrentS,
    setReplayDurationS,
    videoRef,
  ]);

  return { startLive, seekReplay };
}
