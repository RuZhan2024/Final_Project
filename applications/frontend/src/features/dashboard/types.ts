import type { CountSummary } from "../../lib/apiTypes";

/**
 * Dashboard summary merges historical API fields with the current card layout.
 */
export interface DashboardTodaySummary {
  // These counters reflect the backend summary contract, not a local client-side rollup.
  falls_detected?: number | null;
  false_alarms?: number | null;
  confirmed_falls?: number | null;
  [key: string]: unknown;
}

export interface DashboardSystemSummary {
  last_latency_ms?: number | null;
  // `api_online` is derived by the backend summary route, not by direct health polling.
  api_online?: boolean | null;
  model_name?: string | null;
  [key: string]: unknown;
}

export interface DashboardSummary extends CountSummary {
  // `status` remains optional because older summary responses omitted it.
  status?: string;
  today?: DashboardTodaySummary;
  system?: DashboardSystemSummary;
  [key: string]: unknown;
}
