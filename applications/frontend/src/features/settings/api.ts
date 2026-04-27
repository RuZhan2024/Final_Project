/**
 * Settings API helpers.
 *
 * These calls keep the settings page on one small transport contract for
 * reading and patching the backend settings snapshot.
 */
import { apiRequest, type ApiRequestOptions } from "../../lib/apiClient";
import type { SettingsPatch, SettingsResponse } from "./types";

export async function fetchSettings(
  apiBase: string,
  options: ApiRequestOptions = {}
): Promise<SettingsResponse> {
  /** Read the merged settings snapshot used by the settings page. */
  return await apiRequest<SettingsResponse>(apiBase, "/api/settings", options);
}

export async function updateSettings(
  apiBase: string,
  patch: SettingsPatch
): Promise<SettingsResponse> {
  /** Send a partial settings update using the backend's PATCH-like PUT contract. */
  // Empty patches still go through one path so provider/pages do not need special transport branches.
  return await apiRequest<SettingsResponse>(apiBase, "/api/settings", {
    method: "PUT",
    body: patch || {},
  });
}
