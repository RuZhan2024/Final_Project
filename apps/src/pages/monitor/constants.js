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
