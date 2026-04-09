import React, {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";

import { getApiBase } from "../lib/config";
import { readBool } from "../lib/booleans";
import { fetchSettings, updateSettings as persistSettings } from "../features/settings/api";

const API_BASE = getApiBase();
const MonitoringContext = createContext(null);
const SETTINGS_RETRY_DELAYS_MS = [700, 1500];

function readDesired(data) {
  const sys = data?.system || data || {};
  const raw =
    typeof sys.monitoring_enabled !== "undefined"
      ? sys.monitoring_enabled
      : data?.monitoring_enabled;
  return readBool(raw, false);
}

async function safeStart(ctrl) {
  if (!ctrl?.start) return true;
  try {
    const r = ctrl.start();
    const v = await Promise.resolve(r);
    return v ?? true; // undefined/null => treat as success
  } catch {
    return false;
  }
}

function safeStop(ctrl) {
  try {
    ctrl?.stop?.();
  } catch {
    // ignore
  }
}

function sleep(ms) {
  return new Promise((resolve) => window.setTimeout(resolve, ms));
}

async function fetchSettingsWithRetry() {
  let lastError = null;

  for (let attempt = 0; attempt <= SETTINGS_RETRY_DELAYS_MS.length; attempt += 1) {
    try {
      return await fetchSettings(API_BASE);
    } catch (err) {
      lastError = err;
      if (attempt >= SETTINGS_RETRY_DELAYS_MS.length) break;
      await sleep(SETTINGS_RETRY_DELAYS_MS[attempt]);
    }
  }

  throw lastError || new Error("Failed to fetch settings");
}

export function MonitoringProvider({ children }) {
  const [monitoringOn, setMonitoringOnState] = useState(false);
  const [monitoringDesired, setMonitoringDesired] = useState(false);
  const [loaded, setLoaded] = useState(false);
  const [error, setError] = useState(null);
  const [settingsPayload, setSettingsPayload] = useState(null);

  const controllerRef = useRef(null);
  const togglingRef = useRef(false);

  // single source for "latest runtime state" used by refresh/toggle
  const monitoringOnRef = useRef(false);
  const setRuntimeOn = useCallback((v) => {
    const next = Boolean(v);
    monitoringOnRef.current = next;
    setMonitoringOnState(next);
  }, []);

  const registerController = useCallback((controller) => {
    controllerRef.current = controller || null;
  }, []);

  const refresh = useCallback(async () => {
    try {
      setError(null);

      const data = await fetchSettingsWithRetry();
      setSettingsPayload(data);

      const desired = readDesired(data);
      setMonitoringDesired(desired);

      // If DB says OFF, ensure runtime is OFF unless a toggle is currently running
      if (!desired && monitoringOnRef.current && !togglingRef.current) {
        safeStop(controllerRef.current);
        setRuntimeOn(false);
      }

      setLoaded(true);
    } catch (e) {
      setError(String(e?.message || e));
      setLoaded(true);
    }
  }, [setRuntimeOn]);

  useEffect(() => {
    refresh();
  }, [refresh]);

  const setMonitoringOn = useCallback(
    async (next) => {
      if (togglingRef.current) return monitoringOnRef.current;

      togglingRef.current = true;
      const previousRuntimeOn = monitoringOnRef.current;
      try {
        const nextOn = Boolean(next);

        // optimistic runtime update
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

        // sync with latest server truth
        await refresh();

        return desired;
      } catch (e) {
        const ctrl = controllerRef.current;
        if (previousRuntimeOn) {
          const restarted = await safeStart(ctrl);
          setRuntimeOn(Boolean(restarted));
        } else {
          safeStop(ctrl);
          setRuntimeOn(false);
        }
        setError(String(e?.message || e));
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
    async (patch) => {
      try {
        setError(null);
        await persistSettings(API_BASE, patch || {});
        await refresh();
        return true;
      } catch (e) {
        setError(String(e?.message || e));
        return false;
      }
    },
    [refresh]
  );

  const value = useMemo(
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

  return (
    <MonitoringContext.Provider value={value}>
      {children}
    </MonitoringContext.Provider>
  );
}

export function useMonitoring() {
  const ctx = useContext(MonitoringContext);
  if (!ctx) {
    throw new Error("useMonitoring must be used within <MonitoringProvider>");
  }
  return ctx;
}
