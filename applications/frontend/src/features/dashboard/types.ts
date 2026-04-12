import type { CountSummary } from "../../lib/apiTypes";

export interface DashboardTodaySummary {
  falls_detected?: number | null;
  false_alarms?: number | null;
  [key: string]: unknown;
}

export interface DashboardSystemSummary {
  last_latency_ms?: number | null;
  api_online?: boolean | null;
  model_name?: string | null;
  [key: string]: unknown;
}

export interface DashboardSummary extends CountSummary {
  status?: string;
  today?: DashboardTodaySummary;
  system?: DashboardSystemSummary;
  [key: string]: unknown;
}
