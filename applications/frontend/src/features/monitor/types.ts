export interface OperatingPointProfile {
  tau_low?: number | null;
  tau_high?: number | null;
  [key: string]: unknown;
}

export interface SpecModel {
  id: string;
  key?: string;
  spec_key?: string;
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

export interface LegacyOperatingPointRecord {
  id?: number | null;
  code?: string | null;
  op_code?: string | null;
  name?: string | null;
  thr_low_conf?: number | string | null;
  thr_high_conf?: number | string | null;
  threshold_low?: number | string | null;
  threshold_high?: number | string | null;
  [key: string]: unknown;
}

export interface LegacyOperatingPointsResponse {
  operating_points?: LegacyOperatingPointRecord[];
  [key: string]: unknown;
}
