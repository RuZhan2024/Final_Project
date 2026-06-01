/**
 * Shared monitor feature types used by both the monitor page and replay helpers.
 */
export interface OperatingPointProfile {
  tau_low?: number | null;
  tau_high?: number | null;
  [key: string]: unknown;
}

export interface SpecModel {
  id: string;
  key?: string;
  spec_key?: string;
  // Specs may expose per-op thresholds inline when a dedicated route is unavailable.
  ops?: Record<string, OperatingPointProfile>;
  tau_low?: number | null;
  tau_high?: number | null;
  [key: string]: unknown;
}

export type ReplayClipGroup = "fall" | "adl" | "other";

export interface ReplayClip {
  id: string;
  name: string;
  filename: string;
  path: string;
  category: string;
  sizeBytes: number;
  url: string;
  group: ReplayClipGroup;
  // Local file handles only exist for browser-created replay objects.
  file?: File | null;
}

export interface ReplayClipsResponse {
  clips: ReplayClip[];
  configuredDir: string;
  available: boolean;
}

export interface TriggerTestFallResponse {
  ok?: boolean;
  accepted?: boolean;
  message?: string;
  [key: string]: unknown;
}
