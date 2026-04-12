import { apiRequest, type ApiRequestOptions } from "../../lib/apiClient";
import type { DashboardSummary } from "./types";

export async function fetchDashboardSummary(
  apiBase: string,
  options: ApiRequestOptions = {}
): Promise<DashboardSummary> {
  return await apiRequest<DashboardSummary>(apiBase, "/api/dashboard/summary", options);
}
