import React, { useEffect, useMemo, useRef, useState } from "react";
import styles from "./Settings.module.css";

import { useMonitoring } from "../monitoring/MonitoringContext";

// --- Logic & Helpers (Preserved from your uploaded file) ---

// Keep this consistent with other pages (Monitor-demo uses localhost:8000 by default)
const API_BASE = process.env.REACT_APP_API_BASE || "http://localhost:8000";

// Fallback presets (used if the backend doesn't expose /api/operating_points)
const PRESET_FALL_THR = {
  "High Sensitivity": 0.75,
  Balanced: 0.85,
  "Low Sensitivity": 0.93,
};

const PRESET_COOLDOWN = {
  "High Sensitivity": 2,
  Balanced: 3,
  "Low Sensitivity": 5,
};

function modelLabelToCode(label) {
  const v = String(label || "").toLowerCase();
  if (v.includes("tcn")) return "TCN";
  if (v.includes("gcn") && !v.includes("tcn")) return "GCN";
  return "HYBRID";
}

function modelCodeToLabel(code) {
  const v = String(code || "").toUpperCase();
  if (v === "TCN") return "TCN";
  if (v === "GCN") return "GCN";
  return "Hybrid";
}

function clamp(n, lo, hi) {
  return Math.max(lo, Math.min(hi, n));
}

// Helper for gradient slider background
function sliderBg(value, min, max) {
  const v = clamp(Number(value), min, max);
  const pct = ((v - min) / (max - min)) * 100;
  // Using explicit colors to match the UI theme (Blue to Grey)
  const fill = "#4F46E5";
  const rest = "#E5E7EB";
  return `linear-gradient(to right, ${fill} 0%, ${fill} ${pct}%, ${rest} ${pct}%, ${rest} 100%)`;
}

async function apiFetch(path, opts = {}) {
  const url = `${API_BASE}${path}`;
  const res = await fetch(url, {
    headers: { "Content-Type": "application/json", ...(opts.headers || {}) },
    ...opts,
  });

  let payload = null;
  const ct = res.headers.get("content-type") || "";
  if (ct.includes("application/json")) {
    try {
      payload = await res.json();
    } catch {
      payload = null;
    }
  } else {
    try {
      payload = await res.text();
    } catch {
      payload = null;
    }
  }

  if (!res.ok) {
    const detail =
      (payload &&
        payload.detail &&
        (typeof payload.detail === "string"
          ? payload.detail
          : JSON.stringify(payload.detail))) ||
      (typeof payload === "string"
        ? payload
        : payload
        ? JSON.stringify(payload)
        : "") ||
      `${res.status} ${res.statusText}`;
    throw new Error(detail);
  }
  return payload;
}

function pickOpForPreset(ops, presetLabel) {
  const p = String(presetLabel || "").toLowerCase();
  const want = p.includes("high")
    ? "op-1"
    : p.includes("low")
    ? "op-3"
    : "op-2";

  const byCode = ops.find(
    (o) => String(o.op_code || "").toLowerCase() === want
  );
  if (byCode) return byCode;

  const byName = ops.find((o) =>
    String(o.name || "")
      .toLowerCase()
      .includes(p.includes("high") ? "high" : p.includes("low") ? "low" : "bal")
  );
  if (byName) return byName;

  const sorted = [...ops].sort((a, b) => Number(a.id) - Number(b.id));
  if (want === "op-1") return sorted[0] || null;
  if (want === "op-3") return sorted[sorted.length - 1] || null;
  return sorted[Math.floor(sorted.length / 2)] || null;
}

function presetFromOpCode(opCode) {
  const c = String(opCode || "").toLowerCase();
  if (c === "op-1") return "High Sensitivity";
  if (c === "op-3") return "Low Sensitivity";
  if (c === "op-2") return "Balanced";
  return null;
}

// --- Component ---

export default function Settings() {
  const { updateSettings: updateGlobalSettings } = useMonitoring();
  // UI state
  const [loading, setLoading] = useState(true);
  const [statusMsg, setStatusMsg] = useState("");
  const [errorMsg, setErrorMsg] = useState("");

  // Caregiver
  const [caregiverName, setCaregiverName] = useState("");
  const [caregiverEmail, setCaregiverEmail] = useState("");
  const [caregiverPhone, setCaregiverPhone] = useState("");

  // System settings
  const [monitoringEnabled, setMonitoringEnabled] = useState(false);
  const [requireConfirmation, setRequireConfirmation] = useState(false);
  const [notifyOnEveryFall, setNotifyOnEveryFall] = useState(true);

  // New: dataset selection + uncertainty toggle
  const [activeDatasetCode, setActiveDatasetCode] = useState("muvim");
  const [mcEnabled, setMcEnabled] = useState(true);

  // Detection
  const [activeModel, setActiveModel] = useState("Hybrid");
  const [activePreset, setActivePreset] = useState("Balanced");
  const [fallThreshold, setFallThreshold] = useState(85); // percent
  const [alertCooldown, setAlertCooldown] = useState(3); // seconds
  const [activeOpCode, setActiveOpCode] = useState("OP-2");

  // Privacy
  const [storeEventClips, setStoreEventClips] = useState(false);
  const [privacyMode, setPrivacyMode] = useState(true);

  // Prevent auto-saving defaults before the initial GET /api/settings completes.
  const [initialised, setInitialised] = useState(false);

  // Ops
  const [ops, setOps] = useState([]);
  const [opsErr, setOpsErr] = useState("");

  const savingRef = useRef(false);
  const modelChangedByUserRef = useRef(false);
  const statusTimerRef = useRef(null);

  useEffect(() => {
    if (!initialised) return;
    const p = presetFromOpCode(activeOpCode);
    if (p && p !== activePreset) setActivePreset(p);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [activeOpCode, initialised]);

  // Helper to show temporary status messages
  const setStatus = (msg) => {
    setStatusMsg(msg);
    if (statusTimerRef.current) window.clearTimeout(statusTimerRef.current);
    statusTimerRef.current = window.setTimeout(() => setStatusMsg(""), 2500);
  };

  const safeSavePatch = async (patch) => {
    if (savingRef.current) return;
    savingRef.current = true;
    setErrorMsg("");
    try {
      // Use the global updater so other pages immediately receive the new settings.
      const ok = await updateGlobalSettings(patch);
      if (!ok) throw new Error("Failed to save settings");
      setStatus("Saved");
    } catch (e) {
      setErrorMsg(String(e?.message || e));
    } finally {
      savingRef.current = false;
    }
  };

  const loadSettings = async () => {
    setLoading(true);
    setErrorMsg("");
    try {
      const data = await apiFetch("/api/settings", { method: "GET" });
      const sys = data?.system || data || {};

      setMonitoringEnabled(Boolean(sys.monitoring_enabled ?? false));
      setRequireConfirmation(Boolean(sys.require_confirmation ?? false));
      setNotifyOnEveryFall(Boolean(sys.notify_on_every_fall ?? true));

      if (sys.active_dataset_code) setActiveDatasetCode(String(sys.active_dataset_code));
      if (sys.mc_enabled != null) setMcEnabled(Boolean(sys.mc_enabled));

      if (sys.active_model_code)
        setActiveModel(modelCodeToLabel(sys.active_model_code));
      const opCode = (sys.active_op_code || sys.deploy_params?.ui?.op_code || sys.deployParams?.ui?.op_code || "OP-2").toString().toUpperCase();
      setActiveOpCode(opCode);

      const ui = (sys.deploy_params?.ui || sys.deployParams?.ui || null);
      const tauHigh = ui?.tau_high ?? sys.fall_threshold ?? 0.85;
      const cooldownS = ui?.cooldown_s ?? sys.alert_cooldown_sec ?? 3;
      setFallThreshold(Math.round(Number(tauHigh) * 1000) / 10);
      setAlertCooldown(Math.round(Number(cooldownS)));

      if (typeof sys.store_event_clips === "boolean")
        setStoreEventClips(sys.store_event_clips);
      if (typeof sys.anonymize_skeleton_data === "boolean")
        setPrivacyMode(sys.anonymize_skeleton_data);
    } catch (e) {
      setErrorMsg(String(e?.message || e));
    } finally {
      setLoading(false);
      setInitialised(true);
    }
  };

  const loadOperatingPoints = async (modelLabel) => {
    setOpsErr("");
    try {
      const model_code = modelLabelToCode(modelLabel); // TCN / GCN / HYBRID
      const ds = String(activeDatasetCode || "muvim").toLowerCase();
      const specData = await apiFetch(`/api/deploy/specs`, { method: "GET" });
      const models = Array.isArray(specData?.models) ? specData.models : [];
      const byKey = new Map(models.map((m) => [String(m.key || ""), m]));

      const tcn = byKey.get(`${ds}_tcn`);
      const gcn = byKey.get(`${ds}_gcn`);

      const mkList = (spec) => {
        if (!spec) return [];
        const alert_cfg = spec.alert_cfg || {};
        const cooldown_seconds = Number(alert_cfg.cooldown_s ?? 0);
        const ops = spec.ops || {};
        return Object.keys(ops).map((code) => {
          const op = ops[code] || {};
          const tauLow = Number(op.tau_low ?? 0);
          const tauHigh = Number(op.tau_high ?? 0.85);
          return {
            id: code, // stable string id
            op_code: String(code).toLowerCase(), // "op-1"
            name:
              String(code).toUpperCase() === "OP-1"
                ? "OP-1 (High recall)"
                : String(code).toUpperCase() === "OP-3"
                ? "OP-3 (Low alarm)"
                : "OP-2 (Balanced)",
            thr_low_conf: tauLow,
            thr_high_conf: tauHigh,
            thr_detect: tauHigh, // UI uses this as "Fall Threshold"
            cooldown_seconds,
          };
        });
      };

      let list = [];
      if (model_code === "TCN") list = mkList(tcn);
      else if (model_code === "GCN") list = mkList(gcn);
      else {
        // HYBRID: synthesize a single op list that matches the AND-style UI summary
        const lt = mkList(tcn);
        const lg = mkList(gcn);
        const mapT = new Map(lt.map((o) => [o.op_code, o]));
        const mapG = new Map(lg.map((o) => [o.op_code, o]));
        const codes = Array.from(new Set([...mapT.keys(), ...mapG.keys()]));
        list = codes.map((c) => {
          const a = mapT.get(c);
          const b = mapG.get(c);
          const tauLow = Math.min(Number(a?.thr_low_conf ?? 0), Number(b?.thr_low_conf ?? 0));
          const tauHigh = Math.min(Number(a?.thr_high_conf ?? 0.85), Number(b?.thr_high_conf ?? 0.85));
          const cooldown_seconds = Math.max(Number(a?.cooldown_seconds ?? 0), Number(b?.cooldown_seconds ?? 0));
          return {
            id: c.toUpperCase(),
            op_code: c,
            name:
              c === "op-1"
                ? "OP-1 (High recall)"
                : c === "op-3"
                ? "OP-3 (Low alarm)"
                : "OP-2 (Balanced)",
            thr_low_conf: tauLow,
            thr_high_conf: tauHigh,
            thr_detect: tauHigh,
            cooldown_seconds,
          };
        });
      }

      // Normalize sort order OP-1, OP-2, OP-3
      const order = { "op-1": 1, "op-2": 2, "op-3": 3 };
      list.sort((a, b) => (order[a.op_code] || 99) - (order[b.op_code] || 99));

      setOps(list);
      return list;
    } catch (e) {
      setOps([]);
      setOpsErr("Operating points unavailable. Using presets.");
      return [];
    }
  };

  useEffect(() => {
    loadSettings();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    if (!initialised) return;

    (async () => {
      const list = await loadOperatingPoints(activeModel);
      if (!Array.isArray(list) || list.length === 0) return;

      const want = String(activeOpCode || "").toLowerCase() || "op-2";
      const chosenOp =
        list.find((o) => String(o?.op_code || "").toLowerCase() === want) ||
        pickOpForPreset(list, activePreset);

      if (chosenOp?.thr_detect != null && !Number.isNaN(Number(chosenOp.thr_detect))) {
        setFallThreshold(Math.round(Number(chosenOp.thr_detect) * 1000) / 10);
      }
      if (chosenOp?.cooldown_seconds != null && !Number.isNaN(Number(chosenOp.cooldown_seconds))) {
        setAlertCooldown(Math.round(Number(chosenOp.cooldown_seconds)));
      }
    })();

    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [activeModel, activeDatasetCode, activeOpCode, initialised]);

  // Derived gradients for sliders
  const fallThrBg = useMemo(
    () => sliderBg(fallThreshold, 0, 100),
    [fallThreshold]
  );
  const cooldownBg = useMemo(
    () => sliderBg(alertCooldown, 0, 60),
    [alertCooldown]
  );

  const applyPreset = async (presetLabel) => {
    setActivePreset(presetLabel);

    // Prefer YAML-derived operating points (configs/ops/*.yaml) when available.
    const list = Array.isArray(ops) ? ops : [];
    const chosenOp = list.length > 0 ? pickOpForPreset(list, presetLabel) : null;

    const opCodeLower = String(chosenOp?.op_code || "op-2").toLowerCase();
    const opCodeUpper = opCodeLower.toUpperCase();
    setActiveOpCode(opCodeUpper);

    const thr =
      chosenOp?.thr_detect != null && !Number.isNaN(Number(chosenOp.thr_detect))
        ? Number(chosenOp.thr_detect)
        : PRESET_FALL_THR[presetLabel] ?? PRESET_FALL_THR.Balanced;

    const cd =
      chosenOp?.cooldown_seconds != null && !Number.isNaN(Number(chosenOp.cooldown_seconds))
        ? Number(chosenOp.cooldown_seconds)
        : PRESET_COOLDOWN[presetLabel] ?? PRESET_COOLDOWN.Balanced;

    setFallThreshold(Math.round(thr * 1000) / 10);
    setAlertCooldown(Math.round(cd));

    await safeSavePatch({
      // YAML is authoritative: store selected OP code so backend uses configs/ops/*.yaml
      active_op_code: opCodeUpper,
      // Keep legacy fields in sync (for backward compat / UI only)
      fall_threshold: thr,
      alert_cooldown_sec: cd,
    });
  };


  const saveContact = async () => {
    setErrorMsg("");
    try {
      await apiFetch("/api/caregivers", {
        method: "POST",
        body: JSON.stringify({
          name: caregiverName,
          email: caregiverEmail,
          phone: caregiverPhone,
        }),
      });
      setStatus("Caregiver saved");
    } catch (e) {
      setErrorMsg(`Save failed: ${String(e?.message || e)}`);
    }
  };

  // --- Render (New Style) ---

  return (
    <div className={styles.container}>
      <h2 className={styles.pageTitle}>Settings</h2>

      {/* Global Alerts / Status */}
      {/* Global Alerts / Status (Toast Style) */}
      <div className={styles.toastContainer}>
        {statusMsg && (
          <div className={`${styles.toastItem} ${styles.toastSuccess}`}>
            {statusMsg}
          </div>
        )}
        {opsErr && (
          <div className={`${styles.toastItem} ${styles.toastWarning}`}>
            {opsErr}
          </div>
        )}
        {errorMsg && (
          <div className={`${styles.toastItem} ${styles.toastError}`}>
            {errorMsg}
          </div>
        )}
      </div>

      <div className={styles.contentGrid}>
        {/* --- LEFT COLUMN --- */}
        <div className={styles.leftColumn}>
          {/* Caregiver Information */}
          <div className={styles.card}>
            <h3 className={styles.cardTitle}>Caregiver Information</h3>
            <div className={styles.formGroup}>
              <div className={styles.inputWrapper}>
                <label>Name</label>
                <input
                  type="text"
                  className={styles.textInput}
                  value={caregiverName}
                  onChange={(e) => setCaregiverName(e.target.value)}
                  placeholder="e.g. Alice"
                />
              </div>

              <div className={styles.inputWrapper}>
                <label>Email</label>
                <input
                  type="email"
                  className={styles.textInput}
                  value={caregiverEmail}
                  onChange={(e) => setCaregiverEmail(e.target.value)}
                  placeholder="alice@example.com"
                />
              </div>

              <div className={styles.inputWrapper}>
                <label>Phone</label>
                <input
                  type="tel"
                  className={styles.textInput}
                  value={caregiverPhone}
                  onChange={(e) => setCaregiverPhone(e.target.value)}
                  placeholder="+44 ..."
                />
              </div>

              <button className={styles.actionBtn} onClick={saveContact}>
                Save Caregiver
              </button>
            </div>
          </div>

          {/* System & Monitoring */}
          <div className={styles.card}>
            <h3 className={styles.cardTitle}>Monitoring System</h3>

            <div className={styles.toggleRow}>
              <span>Notify with Message</span>
              <label className={styles.switch}>
                <input
                  type="checkbox"
                  checked={monitoringEnabled}
                  onChange={(e) => {
                    const v = e.target.checked;
                    setMonitoringEnabled(v);
                    safeSavePatch({ monitoring_enabled: v });
                  }}
                />
                <span className={styles.slider}></span>
              </label>
            </div>

            <div className={styles.toggleRow}>
              <span>Notify with Call</span>
              <label className={styles.switch}>
                <input
                  type="checkbox"
                  checked={notifyOnEveryFall}
                  onChange={(e) => {
                    const v = e.target.checked;
                    setNotifyOnEveryFall(v);
                    safeSavePatch({ notify_on_every_fall: v });
                  }}
                />
                <span className={styles.slider}></span>
              </label>
            </div>

            {/* <div className={styles.toggleRow}>
              <span>Require Confirmation</span>
              <label className={styles.switch}>
                <input
                  type="checkbox"
                  checked={requireConfirmation}
                  onChange={(e) => {
                    const v = e.target.checked;
                    setRequireConfirmation(v);
                    safeSavePatch({ require_confirmation: v });
                  }}
                />
                <span className={styles.slider}></span>
              </label>
            </div> */}

            <button
              className={styles.actionBtn}
              style={{ marginTop: 24, fontSize: "0.8rem" }}
              onClick={() => loadSettings()}
              disabled={loading}
            >
              Reload System Settings
            </button>
          </div>
        </div>

        {/* --- RIGHT COLUMN --- */}
        <div className={styles.rightColumn}>
          {/* Detection Settings */}
          <div className={styles.card}>
            <h3 className={styles.cardTitle}>Detection Settings</h3>

            {/* Dataset Selection */}
            {/* <div className={styles.row}>
              <span className={styles.labelBold}>Dataset</span>
              <div className={styles.radioGroup}>
                {[
                  { code: "muvim", label: "MUVIM" },
                  { code: "le2i", label: "LE2I" },
                  { code: "urfd", label: "URFD" },
                  { code: "caucafall", label: "CAUCAFall" },
                ].map((d) => (
                  <label key={d.code} className={styles.radioLabel}>
                    {d.label}
                    <input
                      type="radio"
                      checked={activeDatasetCode === d.code}
                      onChange={() => {
                        setActiveDatasetCode(d.code);
                        safeSavePatch({ active_dataset_code: d.code });
                      }}
                    />
                    <span className={styles.radioCustom}></span>
                  </label>
                ))}
              </div>
            </div> */}

            {/* MC Dropout Toggle */}
            <div className={styles.toggleRow}>
              <span>MC Dropout (Uncertainty)</span>
              <label className={styles.switch}>
                <input
                  type="checkbox"
                  checked={mcEnabled}
                  onChange={(e) => {
                    const v = e.target.checked;
                    setMcEnabled(v);
                    safeSavePatch({ mc_enabled: v });
                  }}
                />
                <span className={styles.slider}></span>
              </label>
            </div>

            {/* Model Selection */}
            <div className={styles.row}>
              <span className={styles.labelBold}>Active Model</span>
              <div className={styles.radioGroup}>
                {["TCN", "GCN", "Hybrid"].map((m) => (
                  <label key={m} className={styles.radioLabel}>
                    {m}
                    <input
                      type="radio"
                      name="model"
                      checked={activeModel === m}
                      onChange={() => {
                        modelChangedByUserRef.current = true;
                        setActiveModel(m);
                        // Persist selection so backend resolves per-dataset/model OP params
                        safeSavePatch({ active_model_code: modelLabelToCode(m) });
                      }}
                    />
                    <span className={styles.customRadio}></span>
                  </label>
                ))}
              </div>
            </div>

            {/* Presets */}
            <div className={styles.sectionSpace}>
              <span className={styles.labelBold}>Operating Point Presets</span>
              <div className={styles.presetButtons}>
                {["High Sensitivity", "Balanced", "Low Sensitivity"].map(
                  (p) => (
                    <button
                      key={p}
                      className={`${styles.presetBtn} ${
                        activePreset === p ? styles.activePreset : ""
                      }`}
                      onClick={() => applyPreset(p)}
                    >
                      {p}
                    </button>
                  )
                )}
              </div>
            </div>

            {/* Sliders */}
            <div className={styles.sliderGroup}>
              {/* Fall Threshold */}
              <div className={styles.sliderRow}>
                <div className={styles.sliderHeader}>
                  <span>Fall Detection Threshold</span>
                  <span>{fallThreshold}%</span>
                </div>
                <input
                  type="range"
                  className={styles.rangeInput}
                  min={0}
                  max={100}
                  value={fallThreshold}
                  style={{ background: fallThrBg }}
                  disabled
                />
              </div>

              {/* Alert Cooldown */}
              <div className={styles.sliderRow}>
                <div className={styles.sliderHeader}>
                  <span>Alert Cooldown Period</span>
                  <span>{alertCooldown}s</span>
                </div>
                <input
                  type="range"
                  className={styles.rangeInput}
                  min={0}
                  max={60}
                  value={alertCooldown}
                  style={{ background: cooldownBg }}
                  disabled
                />
              </div>

              {activeOpCode != null && (
                <div
                  style={{ fontSize: "0.8rem", color: "#9CA3AF", marginTop: 8 }}
                >
                  Active Op ID: <strong>{activeOpCode}</strong>
                </div>
              )}
            </div>
          </div>

          {/* Privacy Settings */}
          <div className={styles.card}>
  <h3 className={styles.cardTitle}>Privacy & Data Persistence</h3>

  <div className={styles.privacyItem}>
    <div className={styles.privacyText}>
      {/* Updated Title and Description for Event Clips */}
      <span className={styles.itemTitle}>Skeleton-Only Event Persistence</span>
      <span className={styles.itemDesc}>
        Save detected falls as lightweight coordinate sequences (JSON) instead of video files.
      </span>
    </div>
    <label className={styles.switch}>
      <input
        type="checkbox"
        checked={storeEventClips}
        onChange={(e) => {
          const v = e.target.checked;
          setStoreEventClips(v);
          safeSavePatch({ store_event_clips: v });
        }}
      />
      <span className={styles.slider}></span>
    </label>
  </div>

 

  <div className={styles.privacyFooter}>
    <strong>Privacy Note:</strong> These settings ensure that no identifiable video footage is ever written to the device storage.
  </div>
</div>
        </div>
      </div>
    </div>
  );
}
