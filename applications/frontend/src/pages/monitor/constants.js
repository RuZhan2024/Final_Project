export const NUM_JOINTS = 33;

// Skeleton-only event clips (saved when Settings -> Store Event Clips is enabled)
export const CLIP_PRE_S = 3.0;
export const CLIP_POST_S = 3.0;

// Cap pose processing rate so UI stays responsive even on slower laptops.
export const MAX_PROC_FPS = 30;

// Live monitor perf profile (frontend bottleneck mitigation)
export const LIVE_CAPTURE_WIDTH = 640;
export const LIVE_CAPTURE_HEIGHT = 480;
export const LIVE_POSE_MODEL_COMPLEXITY = 1;
export const LIVE_DRAW_FPS = 12;
export const DEGRADED_POSE_MODEL_COMPLEXITY = 0;

// Replay mode UI update cadence (avoid per-frame React state updates).
export const REPLAY_UI_UPDATE_MS = 200;

// Low-FPS guardrails (frontend-only load shedding).
export const LOW_FPS_ENTER = 14; // enter degraded mode below this FPS
export const LOW_FPS_EXIT = 18;  // exit degraded mode above this FPS
export const LOW_FPS_HOLD_MS = 2000;
export const DEGRADED_DRAW_FPS = 8;
export const DEGRADED_INFER_FPS = 12;
export const MONITOR_PAYLOAD_Q_SCALE = 1000;
