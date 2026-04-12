export interface ApiErrorPayload {
  detail?: string;
  error?: string;
  message?: string;
  [key: string]: unknown;
}

export interface CountSummary {
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
