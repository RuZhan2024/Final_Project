/** Small shared API shapes reused across frontend features. */
export interface ApiErrorPayload {
  detail?: string;
  error?: string;
  message?: string;
  [key: string]: unknown;
}

export interface CountSummary {
  // These counters match the dashboard/events summary cards used in the UI.
  falls: number;
  pending: number;
  false_alarms: number;
}

export interface SimpleApiStatus {
  ok?: boolean;
  accepted?: boolean;
  message?: string;
  [key: string]: unknown;
}
