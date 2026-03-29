import { apiRequest } from "../../lib/apiClient";

export async function fetchDashboardSummary(apiBase, options = {}) {
  return await apiRequest(apiBase, "/api/dashboard/summary", options);
}
