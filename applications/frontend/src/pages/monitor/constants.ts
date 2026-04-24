/** Shared monitor-page constants for capture, playback, and degraded-mode policy. */
export const NUM_JOINTS = 33;

export const CLIP_PRE_S = 3.0;
export const CLIP_POST_S = 3.0;

export const MAX_PROC_FPS = 30;

export const LIVE_CAPTURE_WIDTH = 640;
export const LIVE_CAPTURE_HEIGHT = 480;
// Keep presets explicit so the camera menu matches the actual capture constraints.
export const CAPTURE_RESOLUTIONS = {
  "480p": { w: 640, h: 480 },
  "540p": { w: 960, h: 540 },
  "720p": { w: 1280, h: 720 },
  "1080p": { w: 1920, h: 1080 },
} as const;
export const LIVE_POSE_MODEL_COMPLEXITY = 1;
export const LIVE_DRAW_FPS = 12;
export const DEGRADED_POSE_MODEL_COMPLEXITY = 0;

export const REPLAY_UI_UPDATE_MS = 200;

// Hysteresis avoids flapping between normal and degraded processing on noisy laptops.
export const LOW_FPS_ENTER = 14;
export const LOW_FPS_EXIT = 18;
export const LOW_FPS_HOLD_MS = 2000;
export const DEGRADED_DRAW_FPS = 8;
export const DEGRADED_INFER_FPS = 12;
export const MONITOR_PAYLOAD_Q_SCALE = 1000;
