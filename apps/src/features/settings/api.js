import { apiRequest } from "../../lib/apiClient";

export async function fetchSettings(apiBase, options = {}) {
  return await apiRequest(apiBase, "/api/settings", options);
}

export async function updateSettings(apiBase, patch) {
  return await apiRequest(apiBase, "/api/settings", {
    method: "PUT",
    body: patch || {},
  });
}
