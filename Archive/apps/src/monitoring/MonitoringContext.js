import React, {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";

// Prefer env var, fallback to localhost
const API_BASE =
  typeof process !== "undefined" && process.env && process.env.REACT_APP_API_BASE
    ? process.env.REACT_APP_API_BASE
    : "http://localhost:8000";

const MonitoringContext = createContext(null);

export function MonitoringProvider({ children }) {
  // `monitoringOn` is the *runtime* state (camera/pose pipeline actually running).
  // Do NOT hydrate it from DB on refresh/reload; browsers won't auto-start the camera.
  const [monitoringOn, setMonitoringOnState] = useState(false);
  // `monitoringDesired` is the persisted flag from /api/settings (useful for Settings page).
  const [monitoringDesired, setMonitoringDesired] = useState(false);
  const [loaded, setLoaded] = useState(false);
  const [error, setError] = useState(null);

  // Full /api/settings payload (so all pages can stay in sync after updates).
  const [settingsPayload, setSettingsPayload] = useState(null);

  // The camera/pose pipeline lives at the app level, so page switches don't stop it.
  // We keep a controller here so *any* page can start/stop the pipeline.
  const controllerRef = useRef(null);

  const registerController = useCallback((controller) => {
    controllerRef.current = controller || null;
  }, []);

  const refresh = useCallback(async () => {
    try {
      setError(null);
      const r = await fetch(`${API_BASE}/api/settings`);
      if (!r.ok) throw new Error(await r.text());
      const data = await r.json();
      setSettingsPayload(data);
      const sys = data?.system || data || {};
      const raw =
        typeof sys.monitoring_enabled !== "undefined"
          ? sys.monitoring_enabled
          : data?.monitoring_enabled;
      // Persisted flag (what the DB says the user wants)
      const desired = Boolean(raw);
      setMonitoringDesired(desired);
      // If DB says monitoring is disabled, ensure runtime is also off.
      if (!desired && monitoringOn) {
        try {
          controllerRef.current?.stop?.();
        } catch {
          // ignore
        }
        setMonitoringOnState(false);
      }
      setLoaded(true);
    } catch (e) {
      setError(String(e?.message || e));
      setLoaded(true);
    }
  }, [monitoringOn]);

  useEffect(() => {
    refresh();
  }, [refresh]);

  // If some other action turns monitoring off (e.g., backend sync), ensure we stop locally.
  useEffect(() => {
    if (!monitoringOn) {
      try {
        controllerRef.current?.stop?.();
      } catch {
        // ignore
      }
    }
  }, [monitoringOn]);

  const setMonitoringOn = useCallback(
    async (next) => {
      setMonitoringOnState(Boolean(next)); // optimistic runtime

      // Start/stop the live pipeline immediately (keeps browser user-gesture context)
      const ctrl = controllerRef.current;
      let startedOk = true;
      try {
        if (Boolean(next)) {
          const r = ctrl?.start?.();
          if (r && typeof r.then === "function") startedOk = await r;
        } else {
          ctrl?.stop?.();
        }
      } catch {
        startedOk = false;
      }
      if (Boolean(next) && !startedOk) {
        setMonitoringOnState(false);
      }
      const desired = Boolean(next) && startedOk;
      try {
        setError(null);
        const r = await fetch(`${API_BASE}/api/settings`, {
          method: "PUT",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ monitoring_enabled: desired }),
        });
        if (!r.ok) throw new Error(await r.text());
        // DB desired flag now matches the user's choice
        setMonitoringDesired(desired);
        // Refresh other settings (threshold, model, etc.)
        refresh();
        return desired;
      } catch (e) {
        // Keep local state (so UI stays consistent across pages) and surface the error.
        setError(String(e?.message || e));
        return false;
      }
    },
    [refresh]
  );

  // Generic settings updater used by the Settings page.
  const updateSettings = useCallback(
    async (patch) => {
      try {
        setError(null);
        const r = await fetch(`${API_BASE}/api/settings`, {
          method: "PUT",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(patch || {}),
        });
        if (!r.ok) throw new Error(await r.text());
        await refresh();
        return true;
      } catch (e) {
        setError(String(e?.message || e));
        return false;
      }
    },
    [refresh]
  );

  const toggleMonitoringOn = useCallback(async () => {
    return await setMonitoringOn(!monitoringOn);
  }, [monitoringOn, setMonitoringOn]);

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
    [monitoringOn, monitoringDesired, setMonitoringOn, toggleMonitoringOn, refresh, updateSettings, registerController, loaded, error, settingsPayload]
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
