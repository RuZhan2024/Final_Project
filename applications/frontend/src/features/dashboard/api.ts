/**
 * Dashboard API helpers.
 *
 * The dashboard page only needs one summary endpoint, but keeping the call here
 * avoids spreading route knowledge across page components and hooks.
 */
import { apiRequest, type ApiRequestOptions } from "../../lib/apiClient";
import type { DashboardSummary } from "./types";

export async function fetchDashboardSummary(
  apiBase: string,
  options: ApiRequestOptions = {}
): Promise<DashboardSummary> {
  /** Read the dashboard summary payload consumed by the polling hook. */
  // The summary route already merges system and today-card data, so the client keeps this call thin.
  return await apiRequest<DashboardSummary>(apiBase, "/api/dashboard/summary", options);
}
