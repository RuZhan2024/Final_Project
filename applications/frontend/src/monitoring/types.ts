import type { SettingsPatch, SettingsResponse } from "../features/settings/types";

export interface MonitoringContextValue {
  monitoringOn: boolean;
  monitoringDesired: boolean;
  setMonitoringOn: (next: boolean) => Promise<boolean>;
  toggleMonitoringOn: () => Promise<boolean>;
  refresh: () => Promise<void>;
  updateSettings: (patch: SettingsPatch) => Promise<boolean>;
  registerController: (controller: unknown) => void;
  loaded: boolean;
  error: string | null;
  settings: SettingsResponse | null;
  apiBase: string;
}
