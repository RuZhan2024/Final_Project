import { useCallback } from "react";
import * as mpPose from "@mediapipe/pose";
import * as drawingUtils from "@mediapipe/drawing_utils";

import {
  DEGRADED_DRAW_FPS,
  DEGRADED_INFER_FPS,
  DEGRADED_POSE_MODEL_COMPLEXITY,
  LIVE_DRAW_FPS,
  LIVE_POSE_MODEL_COMPLEXITY,
  LOW_FPS_ENTER,
  LOW_FPS_EXIT,
  LOW_FPS_HOLD_MS,
  MAX_PROC_FPS,
  NUM_JOINTS,
} from "../constants";
import { clamp01 } from "../utils";
import type { UseMonitorPoseProcessingOptions } from "../types";

const { drawConnectors, drawLandmarks } = drawingUtils;

export function useMonitorPoseProcessing({
  targetFps,
  canvasRef,
  videoRef,
  poseRef,
  inputSourceRef,
  isActiveRef,
  showLivePreviewRef,
  lastProcMsRef,
  lastDrawMsRef,
  adaptiveDrawFpsRef,
  adaptiveInferFpsRef,
  degradedModeRef,
  lowFpsSinceMsRef,
  rawFramesRef,
  lastPoseTsRef,
  fpsDeltasRef,
  fpsEstimateRef,
  ensureCanvasMatchesVideo,
  maybeFinalizeClipUpload,
  maybeSendWindow,
}: UseMonitorPoseProcessingOptions) {
  const handlePoseResults = useCallback((results: any) => {
    const isReplay = inputSourceRef.current === "video";
    const nowMs = performance.now();
    const procCap = Math.min(MAX_PROC_FPS, Math.max(8, Number(targetFps) + 2));
    if (nowMs - lastProcMsRef.current < 1000 / procCap) return;
    lastProcMsRef.current = nowMs;

    const estFps = Number(fpsEstimateRef.current);
    if (!isReplay && Number.isFinite(estFps) && estFps > 0) {
      if (estFps < LOW_FPS_ENTER) {
        if (!lowFpsSinceMsRef.current) lowFpsSinceMsRef.current = nowMs;
        if (nowMs - lowFpsSinceMsRef.current > LOW_FPS_HOLD_MS) {
          adaptiveDrawFpsRef.current = DEGRADED_DRAW_FPS;
          adaptiveInferFpsRef.current = DEGRADED_INFER_FPS;
          if (!degradedModeRef.current && poseRef.current?.setOptions) {
            degradedModeRef.current = true;
            try {
              poseRef.current.setOptions({
                modelComplexity: DEGRADED_POSE_MODEL_COMPLEXITY,
                smoothLandmarks: false,
              });
            } catch {
              // ignore
            }
          }
        }
      } else if (estFps > LOW_FPS_EXIT) {
        lowFpsSinceMsRef.current = 0;
        adaptiveDrawFpsRef.current = LIVE_DRAW_FPS;
        adaptiveInferFpsRef.current = 0;
        if (degradedModeRef.current && poseRef.current?.setOptions) {
          degradedModeRef.current = false;
          try {
            poseRef.current.setOptions({
              modelComplexity: LIVE_POSE_MODEL_COMPLEXITY,
              smoothLandmarks: true,
            });
          } catch {
            // ignore
          }
        }
      }
    }

    const canvasEl = canvasRef.current;
    if (!canvasEl) return;
    const ctx = canvasEl.getContext("2d");

    const doDraw = isActiveRef.current;
    const landmarks = results.poseLandmarks;
    const showTransparentPreview =
      (inputSourceRef.current || "camera") === "camera" && showLivePreviewRef.current;
    const hasLandmarks = Boolean(landmarks && landmarks.length);

    if (!hasLandmarks) {
      if (doDraw && ctx) {
        const drawNowMs = performance.now();
        if (drawNowMs - lastDrawMsRef.current >= 1000 / Math.max(1, adaptiveDrawFpsRef.current)) {
          lastDrawMsRef.current = drawNowMs;
          ensureCanvasMatchesVideo();
          ctx.clearRect(0, 0, canvasEl.width, canvasEl.height);
          if (!showTransparentPreview) {
            ctx.fillStyle = "#020617";
            ctx.fillRect(0, 0, canvasEl.width, canvasEl.height);
          }
          ctx.fillStyle = showTransparentPreview ? "rgba(2, 6, 23, 0.68)" : "#020617";
          ctx.fillRect(12, 8, 156, 28);
          ctx.font = "14px system-ui, -apple-system, Segoe UI, Roboto";
          ctx.fillStyle = "#94a3b8";
          ctx.fillText("No pose detected…", 20, 27);
        }
      }
    }

    if (hasLandmarks && doDraw && ctx) {
      const drawNowMs = performance.now();
      if (drawNowMs - lastDrawMsRef.current >= 1000 / Math.max(1, adaptiveDrawFpsRef.current)) {
        lastDrawMsRef.current = drawNowMs;
        ensureCanvasMatchesVideo();
        const w = canvasEl.width;
        const h = canvasEl.height;

        ctx.clearRect(0, 0, w, h);
        if (!showTransparentPreview) {
          ctx.fillStyle = "#020617";
          ctx.fillRect(0, 0, w, h);
        }

        drawConnectors(ctx, landmarks, mpPose.POSE_CONNECTIONS, {
          color: "#22c55e",
          lineWidth: 2,
        });
        drawLandmarks(ctx, landmarks, { color: "#fbbf24", lineWidth: 1 });

        ctx.font = "12px system-ui, -apple-system, Segoe UI, Roboto";
        ctx.fillStyle = showTransparentPreview ? "rgba(2, 6, 23, 0.68)" : "#020617";
        ctx.fillRect(12, h - 34, 120, 22);
        ctx.fillStyle = "#94a3b8";
        ctx.fillText(new Date().toLocaleTimeString(), 16, h - 16);
      }
    }

    const xyFrame = new Array(NUM_JOINTS);
    const confFrame = new Array(NUM_JOINTS);
    for (let i = 0; i < NUM_JOINTS; i += 1) {
      const lm = hasLandmarks ? landmarks[i] : null;
      if (!lm) {
        xyFrame[i] = [0, 0];
        confFrame[i] = 0;
      } else {
        xyFrame[i] = [
          Number.isFinite(Number(lm.x)) ? Number(lm.x) : 0,
          Number.isFinite(Number(lm.y)) ? Number(lm.y) : 0,
        ];
        confFrame[i] = typeof lm.visibility === "number" ? clamp01(lm.visibility) : 1.0;
      }
    }

    const replayVideoTsMs =
      isReplay && videoRef.current ? Number(videoRef.current.currentTime || 0) * 1000 : null;
    let tNow =
      isReplay && Number.isFinite(replayVideoTsMs) ? Number(replayVideoTsMs) : performance.now();
    if (isReplay && typeof lastPoseTsRef.current === "number" && tNow <= lastPoseTsRef.current) {
      tNow = lastPoseTsRef.current + 1;
    }

    const last = lastPoseTsRef.current;
    if (typeof last === "number") {
      const dt = tNow - last;
      if (dt > 0 && dt < 5000) {
        const arr = fpsDeltasRef.current;
        arr.push(dt);
        if (arr.length > 30) arr.shift();

        const meanDt = arr.reduce((a, b) => a + b, 0) / Math.max(1, arr.length);
        const nextFps = meanDt > 0 ? 1000 / meanDt : null;
        if (nextFps && Number.isFinite(nextFps)) {
          const prev = fpsEstimateRef.current;
          fpsEstimateRef.current = prev == null ? nextFps : prev * 0.8 + nextFps * 0.2;
        }
      }
    }
    lastPoseTsRef.current = tNow;

    const raw = rawFramesRef.current;
    raw.push({ t: tNow, xy: xyFrame, conf: confFrame });
    const maxRaw =
      isReplay ? Math.max(2400, Math.ceil(targetFps * 120)) : Math.max(600, Math.ceil(targetFps * 12));
    if (raw.length > maxRaw) raw.splice(0, raw.length - maxRaw);

    void maybeFinalizeClipUpload();
    void maybeSendWindow();
  }, [
    adaptiveDrawFpsRef,
    adaptiveInferFpsRef,
    canvasRef,
    degradedModeRef,
    ensureCanvasMatchesVideo,
    fpsDeltasRef,
    fpsEstimateRef,
    inputSourceRef,
    isActiveRef,
    lastDrawMsRef,
    lastPoseTsRef,
    lastProcMsRef,
    lowFpsSinceMsRef,
    maybeFinalizeClipUpload,
    maybeSendWindow,
    poseRef,
    rawFramesRef,
    showLivePreviewRef,
    targetFps,
    videoRef,
  ]);

  return { handlePoseResults };
}
