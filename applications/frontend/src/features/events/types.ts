import type { CountSummary } from "../../lib/apiTypes";

export type EventType = "fall" | "uncertain" | "not_fall" | string;
export type EventStatus = "pending_review" | "confirmed_fall" | "false_alarm" | string;

export interface EventSkeletonClipMeta {
  path?: string;
  filename?: string;
  [key: string]: unknown;
}

export interface EventMeta {
  skeleton_clip?: EventSkeletonClipMeta;
  [key: string]: unknown;
}

export interface EventRecord {
  id: number;
  resident_id?: number;
  event_time?: string | null;
  type?: EventType;
  status?: EventStatus;
  model?: string | null;
  model_code?: string | null;
  dataset_code?: string | null;
  probability?: number | null;
  p_fall?: number | null;
  created_at?: string | null;
  updated_at?: string | null;
  meta?: EventMeta | null;
  [key: string]: unknown;
}

export interface EventsResponse {
  events: EventRecord[];
  total?: number;
  [key: string]: unknown;
}

export interface EventsTodaySummary {
  falls: number;
  pending: number;
  false_alarms: number;
}

export interface EventsSummaryResponse {
  today?: EventsTodaySummary;
  [key: string]: unknown;
}

export interface EventReviewUpdate {
  status: EventStatus;
}

export interface EventSkeletonClipFrame {
  t_ms?: number;
  xy?: number[][];
  conf?: number[];
  [key: string]: unknown;
}

export interface EventSkeletonClipResponse {
  event_id: number;
  frame_count: number;
  fps?: number | null;
  frames?: EventSkeletonClipFrame[];
  t_ms?: number[];
  xy?: number[][][];
  conf?: number[][];
  clip?: {
    saved_at?: string;
    n_frames?: number;
    anonymized?: boolean;
    mode?: string;
    op_code?: string;
    [key: string]: unknown;
  };
  [key: string]: unknown;
}
