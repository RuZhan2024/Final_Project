import { useCallback } from "react";

import { apiRequest } from "../../../lib/apiClient";
import {
  buildPredictPayload,
} from "../../../features/monitor/prediction";
import { createMonitorSocketClient } from "../../../features/monitor/socketClient";
import {
  queueReplayWindowEnds,
  sliceLiveWindowFrames,
  sliceReplayWindowFrames,
} from "../../../features/monitor/windowing";
import type { UseMonitorPredictionLoopOptions } from "../types";

const WS_CONNECT_TIMEOUT_MS = 7000;
const WS_PREDICT_TIMEOUT_MS = 12000;

/**
 * Owns how monitor windows are built and sent for live and replay analysis.
 *
 * Live camera monitoring favours the persistent WebSocket path, while replay
 * keeps deterministic HTTP-style request ordering and latency-aware playback
 * control.
 */
export function useMonitorPredictionLoop({
  apiBase,
  activeDatasetCode,
  chosen,
  deployS,
  deployW,
  targetFps,
  streamFps,
  mode,
  opCode,
  settingsPayload,
  mcEnabled,
  mcCfg,
  selectedVideoName,
  replayClipRef,
  inputSourceRef,
  monitoringOnRef,
  rawFramesRef,
  sessionIdRef,
  windowSeqRef,
  videoRef,
  predictInFlightRef,
  replayPredictLatencyMsRef,
  replayNextWindowEndRef,
  replayWindowQueueRef,
  lastSentRef,
  predictClientRef,
  applyPredictionResponse,
  syncReplayPlaybackRate,
  maybeFinalizeClipUpload,
  clipFlags,
}: UseMonitorPredictionLoopOptions) {
  const ensurePredictClient = useCallback(() => {
    if (predictClientRef.current) return predictClientRef.current;
    // Live mode reuses one socket so window cadence is not dominated by reconnect cost.
    predictClientRef.current = createMonitorSocketClient({
      apiBase,
      connectTimeoutMs: WS_CONNECT_TIMEOUT_MS,
      predictTimeoutMs: WS_PREDICT_TIMEOUT_MS,
    });
    return predictClientRef.current;
  }, [apiBase, predictClientRef]);

  const predictViaWs = useCallback(
    async (payload: Record<string, unknown>) => {
      return await ensurePredictClient().predict(payload);
    },
    [ensurePredictClient]
  );

  const predictViaHttp = useCallback(
    async (payload: Record<string, unknown>) =>
      // Replay keeps HTTP request ordering simple and easier to reason about during seeks.
      await apiRequest(apiBase, "/api/monitor/predict_window", {
        method: "POST",
        body: payload,
      }),
    [apiBase]
  );

  const buildPredictionPayload = useCallback(
    ({
      sourceMode,
      frames,
      windowEndTs,
      location,
    }: {
      sourceMode: "camera" | "video";
      frames: UseMonitorPredictionLoopOptions["rawFramesRef"]["current"];
      windowEndTs: number;
      location: string;
    }) => {
      const nFrames = Array.isArray(frames) ? frames.length : 0;
      if (nFrames < 2) return null;

      // Window sequence numbers are client-owned so the backend can detect stale
      // or out-of-order windows even when transport latency changes.
      const payload = buildPredictPayload({
        slice: frames,
        sessionId: sessionIdRef.current,
        windowSeq: ++windowSeqRef.current,
        inputSource: sourceMode,
        location,
        mode,
        datasetCode: activeDatasetCode,
        opCode: opCode || settingsPayload?.system?.active_op_code || "OP-2",
        chosen,
        targetFps,
        deployW,
        streamFps,
        mcEnabled,
        mcCfg,
        persist: sourceMode === "video" ? false : monitoringOnRef.current,
        endTs: windowEndTs,
      });

      return {
        ...payload,
        compact_response: sourceMode === "video",
      };
    },
    [
      activeDatasetCode,
      chosen,
      deployW,
      mcCfg,
      mcEnabled,
      mode,
      monitoringOnRef,
      opCode,
      sessionIdRef,
      settingsPayload,
      streamFps,
      targetFps,
      windowSeqRef,
    ]
  );

  const queueReplayWindows = useCallback(() => {
    return queueReplayWindowEnds({
      rawFrames: rawFramesRef.current,
      targetFps,
      deployW,
      deployS,
      nextWindowEndRef: replayNextWindowEndRef,
      queueRef: replayWindowQueueRef,
    });
  }, [deployS, deployW, rawFramesRef, replayNextWindowEndRef, replayWindowQueueRef, targetFps]);

  const buildReplayPayload = useCallback(() => {
    const queue = replayWindowQueueRef.current;
    const raw = rawFramesRef.current;
    if (!queue.length || !raw || raw.length < 2) return null;

    const windowEndTs = Number(queue[0]);
    // Replay always builds from the queue head and only removes that head after
    // a matching response, so transient request failures do not skip windows.
    const frames = sliceReplayWindowFrames({
      rawFrames: raw,
      targetFps,
      deployW,
      windowEndTs,
    });
    if (!frames) return null;

    return buildPredictionPayload({
      sourceMode: "video",
      frames,
      windowEndTs,
      location: String(selectedVideoName || replayClipRef.current?.name || "replay_video"),
    });
  }, [
    buildPredictionPayload,
    deployW,
    rawFramesRef,
    replayClipRef,
    replayWindowQueueRef,
    selectedVideoName,
    targetFps,
  ]);

  const buildLivePayload = useCallback(() => {
    const liveSlice = sliceLiveWindowFrames({
      rawFrames: rawFramesRef.current,
      targetFps,
      deployW,
    });
    if (!liveSlice) return null;

    return buildPredictionPayload({
      sourceMode: "camera",
      frames: liveSlice.frames,
      windowEndTs: liveSlice.endTs,
      location: "camera_live",
    });
  }, [buildPredictionPayload, deployW, rawFramesRef, targetFps]);

  const runPredictionRequest = useCallback(
    async (payload: Record<string, any>, sourceMode: "camera" | "video") => {
      const videoEl = videoRef.current;
      try {
        predictInFlightRef.current = true;
        const t0 = performance.now();
        // Replay latency feeds playback throttling; live mode optimizes for
        // lower overhead over the persistent socket connection.
        const data =
          sourceMode === "video" ? await predictViaHttp(payload) : await predictViaWs(payload);
        replayPredictLatencyMsRef.current = Math.max(0, performance.now() - t0);
        applyPredictionResponse(data);
        return true;
      } finally {
        predictInFlightRef.current = false;
        if (sourceMode === "video") {
          // Replay speed follows backend throughput so the UI does not race
          // ahead of queued predictions and hide unprocessed windows.
          syncReplayPlaybackRate(videoEl);
        }
      }
    },
    [
      applyPredictionResponse,
      predictInFlightRef,
      predictViaHttp,
      predictViaWs,
      replayPredictLatencyMsRef,
      syncReplayPlaybackRate,
      videoRef,
    ]
  );

  const drainReplayQueue = useCallback(async () => {
    if ((inputSourceRef.current || "camera") !== "video") return;
    void queueReplayWindows();
    if (predictInFlightRef.current) return;
    if (!replayWindowQueueRef.current.length) return;

    while (!predictInFlightRef.current && replayWindowQueueRef.current.length) {
      const payload = buildReplayPayload();
      if (!payload) return;
      const nextWindowEndTs = Number((payload as { window_end_t_ms?: number }).window_end_t_ms);
      const ok = await runPredictionRequest(payload, "video");
      const front = Number(replayWindowQueueRef.current[0]);
      if (Number.isFinite(front) && front === nextWindowEndTs) {
        // Confirm the head timestamp still matches the processed request because
        // queueReplayWindows may have appended newer items in the meantime.
        replayWindowQueueRef.current.shift();
      }
      if (!ok) return;
    }
  }, [
    buildReplayPayload,
    inputSourceRef,
    predictInFlightRef,
    queueReplayWindows,
    replayWindowQueueRef,
    runPredictionRequest,
  ]);

  const maybeSendWindow = useCallback(async () => {
    const sourceMode = inputSourceRef.current || "camera";
    if (sourceMode === "video") {
      await drainReplayQueue();
      return;
    }

    if (predictInFlightRef.current) return;
    const now = performance.now();
    // Respect the deployment stride rather than sending every available frame;
    // the backend tracker expects overlapping windows at this cadence.
    const minGapMs = Math.max(250, (deployS / Math.max(1, targetFps)) * 1000);
    const gapRemainMs = minGapMs - (now - lastSentRef.current);
    if (gapRemainMs > 0) return;

    const payload = buildLivePayload();
    if (!payload) return;
    lastSentRef.current = now;
    await runPredictionRequest(payload, "camera");
  }, [
    buildLivePayload,
    deployS,
    drainReplayQueue,
    inputSourceRef,
    lastSentRef,
    predictInFlightRef,
    runPredictionRequest,
    targetFps,
  ]);

  return {
    buildReplayPayload,
    drainReplayQueue,
    maybeSendWindow,
    maybeFinalizeClipUpload,
  };
}
