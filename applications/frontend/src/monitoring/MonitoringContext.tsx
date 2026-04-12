import React, {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";

import { fetchSettings, updateSettings as persistSettings } from "../features/settings/api";
import type { SettingsPatch, SettingsResponse } from "../features/settings/types";
import { readBool } from "../lib/booleans";
import { getApiBase } from "../lib/config";
import type { MonitoringContextValue } from "./types";

const API_BASE = getApiBase();
const SETTINGS_RETRY_DELAYS_MS = [700, 1500];

interface MonitorController {
  start?: () => boolean | Promise<boolean | undefined> | undefined;
  stop?: () => void;
}

const MonitoringContext = createContext<MonitoringContextValue | null>(null);

function readDesired(data: SettingsResponse | null): boolean {
  const sys = data?.system || data || {};
  const raw =
    typeof sys.monitoring_enabled !== "undefined"
      ? sys.monitoring_enabled
      : (data as Record<string, unknown> | null)?.monitoring_enabled;
  return readBool(raw, false);
}

async function safeStart(ctrl: MonitorController | null): Promise<boolean> {
  if (!ctrl?.start) return true;
  try {
    const result = ctrl.start();
    const value = await Promise.resolve(result);
    return value ?? true;
  } catch {
    return false;
  }
}

function safeStop(ctrl: MonitorController | null): void {
  try {
    ctrl?.stop?.();
  } catch {
    // ignore
  }
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => window.setTimeout(resolve, ms));
}

async function fetchSettingsWithRetry(): Promise<SettingsResponse> {
  let lastError: unknown = null;

  for (let attempt = 0; attempt <= SETTINGS_RETRY_DELAYS_MS.length; attempt += 1) {
    try {
      return await fetchSettings(API_BASE);
    } catch (error) {
      lastError = error;
      if (attempt >= SETTINGS_RETRY_DELAYS_MS.length) break;
      await sleep(SETTINGS_RETRY_DELAYS_MS[attempt]);
    }
  }

  throw lastError || new Error("Failed to fetch settings");
}

export function MonitoringProvider({ children }: React.PropsWithChildren) {
  const [monitoringOn, setMonitoringOnState] = useState(false);
  const [monitoringDesired, setMonitoringDesired] = useState(false);
  const [loaded, setLoaded] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [settingsPayload, setSettingsPayload] = useState<SettingsResponse | null>(null);

  const controllerRef = useRef<MonitorController | null>(null);
  const togglingRef = useRef(false);
  const monitoringOnRef = useRef(false);

  const setRuntimeOn = useCallback((value: boolean) => {
    const next = Boolean(value);
    monitoringOnRef.current = next;
    setMonitoringOnState(next);
  }, []);

  const registerController = useCallback((controller: unknown) => {
    controllerRef.current = (controller as MonitorController | null) || null;
  }, []);

  const refresh = useCallback(async (): Promise<void> => {
    try {
      setError(null);

      const data = await fetchSettingsWithRetry();
      setSettingsPayload(data);

      const desired = readDesired(data);
      setMonitoringDesired(desired);

      if (!desired && monitoringOnRef.current && !togglingRef.current) {
        safeStop(controllerRef.current);
        setRuntimeOn(false);
      }

      setLoaded(true);
    } catch (err) {
      setError(String((err as Error)?.message || err));
      setLoaded(true);
    }
  }, [setRuntimeOn]);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  const setMonitoringOn = useCallback(
    async (next: boolean): Promise<boolean> => {
      if (togglingRef.current) return monitoringOnRef.current;

      togglingRef.current = true;
      const previousRuntimeOn = monitoringOnRef.current;
      try {
        const nextOn = Boolean(next);
        setRuntimeOn(nextOn);

        const ctrl = controllerRef.current;
        let startedOk = true;

        if (nextOn) {
          startedOk = await safeStart(ctrl);
          if (!startedOk) setRuntimeOn(false);
        } else {
          safeStop(ctrl);
        }

        const desired = nextOn && startedOk;

        setError(null);
        await persistSettings(API_BASE, { monitoring_enabled: desired });
        setMonitoringDesired(desired);
        await refresh();

        return desired;
      } catch (err) {
        const ctrl = controllerRef.current;
        if (previousRuntimeOn) {
          const restarted = await safeStart(ctrl);
          setRuntimeOn(Boolean(restarted));
        } else {
          safeStop(ctrl);
          setRuntimeOn(false);
        }
        setError(String((err as Error)?.message || err));
        return false;
      } finally {
        togglingRef.current = false;
      }
    },
    [refresh, setRuntimeOn]
  );

  const toggleMonitoringOn = useCallback(() => {
    return setMonitoringOn(!monitoringOnRef.current);
  }, [setMonitoringOn]);

  const updateSettings = useCallback(
    async (patch: SettingsPatch): Promise<boolean> => {
      try {
        setError(null);
        await persistSettings(API_BASE, patch || {});
        await refresh();
        return true;
      } catch (err) {
        setError(String((err as Error)?.message || err));
        return false;
      }
    },
    [refresh]
  );

  const value = useMemo<MonitoringContextValue>(
    () => ({
      monitoringOn,
      monitoringDesired,
      setMonitoringOn,
      toggleMonitoringOn,
      refresh,
      updateSettings,
      registerController,
      loaded,
      error,
      settings: settingsPayload,
      apiBase: API_BASE,
    }),
    [
      monitoringOn,
      monitoringDesired,
      setMonitoringOn,
      toggleMonitoringOn,
      refresh,
      updateSettings,
      registerController,
      loaded,
      error,
      settingsPayload,
    ]
  );

  return <MonitoringContext.Provider value={value}>{children}</MonitoringContext.Provider>;
}

export function useMonitoring(): MonitoringContextValue {
  const ctx = useContext(MonitoringContext);
  if (!ctx) {
    throw new Error("useMonitoring must be used within <MonitoringProvider>");
  }
  return ctx;
}
