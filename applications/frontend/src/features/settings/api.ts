import { apiRequest, type ApiRequestOptions } from "../../lib/apiClient";
import type { SettingsPatch, SettingsResponse } from "./types";

export async function fetchSettings(
  apiBase: string,
  options: ApiRequestOptions = {}
): Promise<SettingsResponse> {
  return await apiRequest<SettingsResponse>(apiBase, "/api/settings", options);
}

export async function updateSettings(
  apiBase: string,
  patch: SettingsPatch
): Promise<SettingsResponse> {
  return await apiRequest<SettingsResponse>(apiBase, "/api/settings", {
    method: "PUT",
    body: patch || {},
  });
}
