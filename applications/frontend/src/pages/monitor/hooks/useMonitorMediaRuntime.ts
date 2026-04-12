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
    const isReplay = inputSourceRef.current === "video";
    if (!poseRef.current) {
      const pose = new mpPose.Pose({
        locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/pose/${file}`,
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
      await poseRef.current.send({ image: videoEl });
    }
  }, [handlePoseResults, inputSourceRef, poseRef]);

  const prepareReplayVideo = useCallback(async (videoEl: HTMLVideoElement, clip: any, clipUrl: string) => {
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
    await prepareCameraStreamSource({
      videoEl,
      captureResolution,
      targetFps,
      streamRef,
    });
  }, [captureResolutionPreset, streamRef, targetFps]);

  const syncReplayTimeState = useCallback(() => {
    const videoEl = videoRef.current;
    if (!videoEl) return;
    setReplayCurrentS(Number(videoEl.currentTime || 0));
    setReplayDurationS(Number(videoEl.duration || 0));
  }, [setReplayCurrentS, setReplayDurationS, videoRef]);

  const getActiveReplaySource = useCallback(() => {
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
