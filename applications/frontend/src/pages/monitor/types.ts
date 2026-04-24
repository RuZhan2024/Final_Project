import type { ReplayClip, SpecModel } from "../../features/monitor/types";
import type { SettingsResponse } from "../../features/settings/types";
import type { MutableRefObject } from "react";

/**
 * Shared type contracts for the monitor page and its hooks.
 *
 * These types define the data exchanged between frontend monitor subsystems:
 * raw pose frames, prediction state, pending clip uploads, and hook option
 * bundles. Keeping them here prevents each hook from drifting its own local
 * contract.
 */
export type MonitorMode = "tcn" | "gcn" | "hybrid";
export type InputSource = "camera" | "video";
export type CaptureResolutionPreset = "480p" | "540p" | "720p" | "1080p";

export interface ChosenSpecs {
  /** Deploy spec ids currently selected for each model family in hybrid mode. */
  tcn: string;
  gcn: string;
}

export interface MarkerEntry {
  /** Timeline marker rendered against replay progress or live history strips. */
  key?: string | number;
  leftPct: number;
  kind: "fall" | "uncertain" | "safe" | string;
}

export interface TimelineEntry {
  /** Deduplicated history entry keyed by monitor sequence/time rather than raw frame count. */
  kind: "fall" | "uncertain" | "safe" | string;
  t: number;
  seq: number;
}

export interface MonitorOperatingPointParams {
  /** Operating-point details surfaced from backend specs/settings for the UI. */
  opCode: string | null;
  tauLow: number | null;
  tauHigh: number | null;
  confirmK: number | null;
  confirmN: number | null;
  cooldownS: number | null;
}

export interface ApiSpecState {
  /** Current deploy-spec catalog fetch state used by monitor setup controls. */
  models: SpecModel[];
  error: string | null;
}

export interface ReplayClipsState {
  /** Replay clip catalog plus directory-availability diagnostics from the API. */
  clips: ReplayClip[];
  loading: boolean;
  error: string;
  configuredDir: string;
  available: boolean;
}

export interface MonitorControllerHandle {
  /** Imperative bridge used by the page-level start/stop controls. */
  start?: () => void | boolean | Promise<boolean | void>;
  stop?: () => void;
}

export interface MonitorRawFrame {
  /** One pose frame buffered before frontend windowing/quantization. */
  t: number;
  xy: number[][];
  conf: number[];
}

export interface PendingClipContext {
  /** Backend context needed when the delayed clip upload is finally created. */
  dataset_code: string;
  mode: string;
  op_code: string;
  use_mc: boolean;
  mc_M: number | null;
}

export interface PendingClip {
  /** Client-side plan for capturing post-roll frames after a persisted event id appears. */
  eventId: number | string;
  triggerEndTs: number;
  deadlineTs: number;
  preMs: number;
  postMs: number;
  ctx: PendingClipContext;
}

export interface MonitorPredictionState {
  /** Normalized prediction state produced from one backend monitor response. */
  safeAlert: boolean | null;
  recallAlert: boolean | null;
  safeState: string | null;
  recallState: string | null;
  stable: { fall: number; uncertain: number; safe: number; last: string };
  triageState: string;
  pFall: number | null;
  sigma: number | null;
  markerKind: "safe" | "fall" | "uncertain";
  dedupMs: number;
  eventId: number | string | null;
}

export interface UseMonitorClipCaptureOptions {
  apiBase: string;
  settingsPayload: SettingsResponse | null;
  rawFramesRef: MutableRefObject<MonitorRawFrame[]>;
  activeDatasetCode: string;
  mode: string;
  opCode: string | null;
  mcEnabled: boolean;
  mcCfg: { M: number | null; M_confirm: number | null };
}

export interface MonitorClipFlags {
  /** Effective clip-persistence policy derived from current settings payload. */
  storeEventClips: boolean;
  anonymize: boolean;
}

export interface UseMonitorPredictionLoopOptions {
  apiBase: string;
  activeDatasetCode: string;
  chosen: ChosenSpecs;
  deployS: number;
  deployW: number;
  targetFps: number;
  streamFps: number | null;
  mode: string;
  opCode: string | null;
  settingsPayload: SettingsResponse | null;
  mcEnabled: boolean;
  mcCfg: { M: number | null; M_confirm: number | null };
  selectedVideoName: string;
  replayClipRef: MutableRefObject<ReplayClip | null>;
  inputSourceRef: MutableRefObject<string>;
  monitoringOnRef: MutableRefObject<boolean>;
  rawFramesRef: MutableRefObject<MonitorRawFrame[]>;
  sessionIdRef: MutableRefObject<string>;
  windowSeqRef: MutableRefObject<number>;
  videoRef: MutableRefObject<HTMLVideoElement | null>;
  predictInFlightRef: MutableRefObject<boolean>;
  replayPredictLatencyMsRef: MutableRefObject<number>;
  replayNextWindowEndRef: MutableRefObject<number | null>;
  replayWindowQueueRef: MutableRefObject<number[]>;
  lastSentRef: MutableRefObject<number>;
  predictClientRef: MutableRefObject<any>;
  /** Applies one backend response to session UI state and clip side effects. */
  applyPredictionResponse: (data: Record<string, any>) => void;
  /** Replay uses backend latency to slow playback before the queue runs away. */
  syncReplayPlaybackRate: (videoEl: HTMLVideoElement | null) => void;
  maybeFinalizeClipUpload: () => Promise<void> | void;
  clipFlags: MonitorClipFlags;
}

export interface UseMonitorMediaRuntimeOptions {
  targetFps: number;
  captureResolutionPreset: CaptureResolutionPreset;
  replayClipRef: MutableRefObject<ReplayClip | null>;
  inputSourceRef: MutableRefObject<string>;
  videoRef: MutableRefObject<HTMLVideoElement | null>;
  poseRef: MutableRefObject<any>;
  streamRef: MutableRefObject<MediaStream | null>;
  videoObjectUrlRef: MutableRefObject<string | null>;
  setStartInfo: (value: string) => void;
  setReplayCurrentS: (value: number | ((prev: number) => number)) => void;
  setReplayDurationS: (value: number | ((prev: number) => number)) => void;
  handlePoseResults: (results: any) => void;
  resetVideoSource: () => void;
  fetchReplayClipBlob: (clipUrl: string) => Promise<Blob>;
}

export interface UseMonitorSessionStateOptions {
  mode: string;
  settingsPayload: SettingsResponse | null;
  clipFlags: MonitorClipFlags;
  pendingClipRef: MutableRefObject<PendingClip | null>;
  uploadedClipIdsRef: MutableRefObject<Set<string>>;
  maybeFinalizeClipUpload: () => Promise<void> | void;
  queueClipForEvent: (args: {
    eventId: number | string;
    endTs: number;
    clipPreS: number;
    clipPostS: number;
  }) => void;
}

export interface UseMonitorPoseProcessingOptions {
  targetFps: number;
  canvasRef: MutableRefObject<HTMLCanvasElement | null>;
  videoRef: MutableRefObject<HTMLVideoElement | null>;
  poseRef: MutableRefObject<any>;
  inputSourceRef: MutableRefObject<string>;
  isActiveRef: MutableRefObject<boolean>;
  showLivePreviewRef: MutableRefObject<boolean>;
  lastProcMsRef: MutableRefObject<number>;
  lastDrawMsRef: MutableRefObject<number>;
  adaptiveDrawFpsRef: MutableRefObject<number>;
  adaptiveInferFpsRef: MutableRefObject<number>;
  degradedModeRef: MutableRefObject<boolean>;
  lowFpsSinceMsRef: MutableRefObject<number>;
  rawFramesRef: MutableRefObject<MonitorRawFrame[]>;
  lastPoseTsRef: MutableRefObject<number | null>;
  fpsDeltasRef: MutableRefObject<number[]>;
  fpsEstimateRef: MutableRefObject<number | null>;
  ensureCanvasMatchesVideo: () => void;
  maybeFinalizeClipUpload: () => Promise<void> | void;
  maybeSendWindow: () => Promise<void> | void;
}

export interface UseMonitorRuntimeControllerOptions {
  targetFps: number;
  apiBase: string;
  autoStopMonitoring: () => void;
  buildReplayPayload: () => any;
  /** Drain queued replay windows in timestamp order until backpressure catches up. */
  drainReplayQueue: () => Promise<void>;
  ensureCanvasMatchesVideo: () => void;
  getActiveReplaySource: () => {
    clip: ReplayClip | null;
    clipFile: File | null;
    clipUrl: string;
  };
  initPosePipeline: (videoEl: HTMLVideoElement, options?: { warmup?: boolean }) => Promise<void>;
  makeSessionId: () => string;
  prepareCameraStream: (videoEl: HTMLVideoElement) => Promise<void>;
  prepareReplayVideo: (videoEl: HTMLVideoElement, clip: ReplayClip | null, clipUrl: string) => Promise<void>;
  resetFrontendSessionState: () => void;
  resetSession: () => Promise<void>;
  stopLive: () => void;
  syncReplayTimeState: () => void;
  inputSourceRef: MutableRefObject<string>;
  videoRef: MutableRefObject<HTMLVideoElement | null>;
  streamRef: MutableRefObject<MediaStream | null>;
  liveFlagRef: MutableRefObject<boolean>;
  runTokenRef: MutableRefObject<number>;
  rafRef: MutableRefObject<number | null>;
  poseRef: MutableRefObject<any>;
  poseSendBusyRef: MutableRefObject<boolean>;
  predictInFlightRef: MutableRefObject<boolean>;
  replayWindowQueueRef: MutableRefObject<number[]>;
  adaptiveInferFpsRef: MutableRefObject<number>;
  inferFpsRef: MutableRefObject<number>;
  lastInferTsRef: MutableRefObject<number>;
  lastReplayUiMsRef: MutableRefObject<number>;
  replayPredictLatencyMsRef: MutableRefObject<number>;
  sessionIdRef: MutableRefObject<string>;
  fpsEstimateRef: MutableRefObject<number | null>;
  setLiveRunning: (value: boolean) => void;
  setStartError: (value: string) => void;
  setStartInfo: (value: string) => void;
  setReplayCurrentS: (value: number | ((prev: number) => number)) => void;
  setReplayDurationS: (value: number | ((prev: number) => number)) => void;
}
