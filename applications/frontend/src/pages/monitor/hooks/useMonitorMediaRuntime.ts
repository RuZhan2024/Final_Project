import { useCallback } from "react";
import * as mpPose from "@mediapipe/pose";

import {
  fetchReplayClipBlob,
} from "../../../features/monitor/api";
import {
  prepareCameraStream as prepareCameraStreamSource,
  prepareReplayVideo as prepareReplayVideoSource,
} from "../../../features/monitor/media";
import {
  CAPTURE_RESOLUTIONS,
  LIVE_POSE_MODEL_COMPLEXITY,
} from "../constants";
import type { UseMonitorMediaRuntimeOptions } from "../types";

const REPLAY_POSE_MODEL_COMPLEXITY = 1;
const REPLAY_MIN_DETECTION_CONFIDENCE = 0.35;
const REPLAY_MIN_TRACKING_CONFIDENCE = 0.35;
const MEDIAPIPE_POSE_ASSET_ROOT = `${process.env.PUBLIC_URL || ""}/mediapipe/pose`;

/**
 * Prepares camera/replay media sources and configures the MediaPipe pose runtime.
 *
 * This hook owns source-specific runtime setup. It does not decide when a frame
 * should be sent to pose processing; that scheduling lives in the runtime
 * controller.
 */
export function useMonitorMediaRuntime({
  targetFps,
  captureResolutionPreset,
  replayClipRef,
  inputSourceRef,
  videoRef,
  poseRef,
  streamRef,
  videoObjectUrlRef,
  setStartInfo,
  setReplayCurrentS,
  setReplayDurationS,
  handlePoseResults,
  resetVideoSource,
}: UseMonitorMediaRuntimeOptions) {
  const initPosePipeline = useCallback(async (videoEl: HTMLVideoElement, { warmup = false } = {}) => {
    /**
     * Create or reconfigure the shared MediaPipe Pose instance for the current source.
     *
     * Replay runs with more forgiving thresholds because uploaded clips often
     * contain motion blur or lower-quality frames than live webcam preview.
     */
    const isReplay = inputSourceRef.current === "video";
    if (!poseRef.current) {
      const pose = new mpPose.Pose({
        locateFile: (file) => `${MEDIAPIPE_POSE_ASSET_ROOT}/${file}`,
      });

      pose.setOptions({
        modelComplexity: isReplay ? REPLAY_POSE_MODEL_COMPLEXITY : LIVE_POSE_MODEL_COMPLEXITY,
        smoothLandmarks: true,
        minDetectionConfidence: isReplay ? REPLAY_MIN_DETECTION_CONFIDENCE : 0.5,
        minTrackingConfidence: isReplay ? REPLAY_MIN_TRACKING_CONFIDENCE : 0.5,
      });

      pose.onResults(handlePoseResults);
      poseRef.current = pose;
    } else if (poseRef.current?.setOptions) {
      try {
        poseRef.current.setOptions({
          modelComplexity: isReplay ? REPLAY_POSE_MODEL_COMPLEXITY : LIVE_POSE_MODEL_COMPLEXITY,
          smoothLandmarks: true,
          minDetectionConfidence: isReplay ? REPLAY_MIN_DETECTION_CONFIDENCE : 0.5,
          minTrackingConfidence: isReplay ? REPLAY_MIN_TRACKING_CONFIDENCE : 0.5,
        });
      } catch {
        // ignore
      }
    }

    if (warmup && videoEl?.readyState >= 2 && poseRef.current) {
      // Warm a single frame so the first visible prediction loop does not pay
      // the full model initialization cost.
      await poseRef.current.send({ image: videoEl });
    }
  }, [handlePoseResults, inputSourceRef, poseRef]);

  const prepareReplayVideo = useCallback(async (videoEl: HTMLVideoElement, clip: any, clipUrl: string) => {
    // Replay setup handles local files and backend-hosted clips through one path.
    await prepareReplayVideoSource({
      videoEl,
      clip,
      clipUrl,
      videoObjectUrlRef,
      setStartInfo,
      fetchReplayClipBlob,
      resetVideoSource,
    });
  }, [resetVideoSource, setStartInfo, videoObjectUrlRef]);

  const prepareCameraStream = useCallback(async (videoEl: HTMLVideoElement) => {
    const captureResolution =
      CAPTURE_RESOLUTIONS[captureResolutionPreset] || CAPTURE_RESOLUTIONS["720p"];
    // Resolution presets stay explicit so the UI menu matches actual constraints.
    await prepareCameraStreamSource({
      videoEl,
      captureResolution,
      targetFps,
      streamRef,
    });
  }, [captureResolutionPreset, streamRef, targetFps]);

  const syncReplayTimeState = useCallback(() => {
    /** Mirror the current replay element time into React state for the timeline UI. */
    const videoEl = videoRef.current;
    if (!videoEl) return;
    setReplayCurrentS(Number(videoEl.currentTime || 0));
    setReplayDurationS(Number(videoEl.duration || 0));
  }, [setReplayCurrentS, setReplayDurationS, videoRef]);

  const getActiveReplaySource = useCallback(() => {
    /** Resolve the currently selected replay source into file/url inputs. */
    const clip = replayClipRef.current;
    const clipFile = clip?.file instanceof File ? clip.file : null;
    const clipUrl = typeof clip?.url === "string" && clip.url ? clip.url : "";
    return { clip, clipFile, clipUrl };
  }, [replayClipRef]);

  return {
    initPosePipeline,
    prepareReplayVideo,
    prepareCameraStream,
    syncReplayTimeState,
    getActiveReplaySource,
  };
}
