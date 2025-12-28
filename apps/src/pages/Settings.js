import React, { useEffect, useMemo, useRef, useState } from 'react';
import styles from './Settings.module.css';

import { useMonitoring } from '../monitoring/MonitoringContext';

// --- Logic & Helpers (Preserved from your uploaded file) ---

// Keep this consistent with other pages (Monitor-demo uses localhost:8000 by default)
const API_BASE = process.env.REACT_APP_API_BASE || 'http://localhost:8000';

// Fallback presets (used if the backend doesn't expose /api/operating_points)
const PRESET_FALL_THR = {
  'High Sensitivity': 0.75,
  'Balanced': 0.85,
  'Low Sensitivity': 0.93,
};

const PRESET_COOLDOWN = {
  'High Sensitivity': 2,
  'Balanced': 3,
  'Low Sensitivity': 5,
};

function modelLabelToCode(label) {
  const v = String(label || '').toLowerCase();
  if (v.includes('tcn')) return 'TCN';
  if (v.includes('gcn') && !v.includes('tcn')) return 'GCN';
  return 'HYBRID';
}

function modelCodeToLabel(code) {
  const v = String(code || '').toUpperCase();
  if (v === 'TCN') return 'TCN';
  if (v === 'GCN') return 'GCN';
  return 'Hybrid';
}

function clamp(n, lo, hi) {
  return Math.max(lo, Math.min(hi, n));
}

// Helper for gradient slider background
function sliderBg(value, min, max) {
  const v = clamp(Number(value), min, max);
  const pct = ((v - min) / (max - min)) * 100;
  // Using explicit colors to match the UI theme (Blue to Grey)
  const fill = '#4F46E5'; 
  const rest = '#E5E7EB';
  return `linear-gradient(to right, ${fill} 0%, ${fill} ${pct}%, ${rest} ${pct}%, ${rest} 100%)`;
}

async function apiFetch(path, opts = {}) {
  const url = `${API_BASE}${path}`;
  const res = await fetch(url, {
    headers: { 'Content-Type': 'application/json', ...(opts.headers || {}) },
    ...opts,
  });

  let payload = null;
  const ct = res.headers.get('content-type') || '';
  if (ct.includes('application/json')) {
    try { payload = await res.json(); } catch { payload = null; }
  } else {
    try { payload = await res.text(); } catch { payload = null; }
  }

  if (!res.ok) {
    const detail =
      (payload && payload.detail && (typeof payload.detail === 'string' ? payload.detail : JSON.stringify(payload.detail))) ||
      (typeof payload === 'string' ? payload : payload ? JSON.stringify(payload) : '') ||
      `${res.status} ${res.statusText}`;
    throw new Error(detail);
  }
  return payload;
}

function pickOpForPreset(ops, presetLabel) {
  const p = String(presetLabel || '').toLowerCase();
  const want =
    p.includes('high') ? 'op-1' :
    p.includes('low') ? 'op-3' :
    'op-2';

  const byCode = ops.find((o) => String(o.op_code || '').toLowerCase() === want);
  if (byCode) return byCode;

  const byName = ops.find((o) => String(o.name || '').toLowerCase().includes(p.includes('high') ? 'high' : p.includes('low') ? 'low' : 'bal'));
  if (byName) return byName;

  const sorted = [...ops].sort((a, b) => Number(a.id) - Number(b.id));
  if (want === 'op-1') return sorted[0] || null;
  if (want === 'op-3') return sorted[sorted.length - 1] || null;
  return sorted[Math.floor(sorted.length / 2)] || null;
}

function presetFromOp(ops, activeOpId) {
  if (!activeOpId) return null;

  // Try exact match by id
  const op = Array.isArray(ops) ? ops.find((o) => Number(o?.id) === Number(activeOpId)) : null;

  const code = String(op?.op_code || '').toLowerCase();
  const name = String(op?.name || '').toLowerCase();

  const looksHigh = code === 'op-1' || name.includes('high');
  const looksLow  = code === 'op-3' || name.includes('low');
  const looksBal  = code === 'op-2' || name.includes('bal');

  if (looksHigh) return 'High Sensitivity';
  if (looksLow)  return 'Low Sensitivity';
  if (looksBal)  return 'Balanced';

  // Fallback: rank by id if we have ops but no code/name
  if (Array.isArray(ops) && ops.length > 0) {
    const sorted = [...ops].sort((a, b) => Number(a.id) - Number(b.id));
    const idx = sorted.findIndex((x) => Number(x.id) === Number(activeOpId));
    if (idx === 0) return 'High Sensitivity';
    if (idx === sorted.length - 1) return 'Low Sensitivity';
    return 'Balanced';
  }

  return null;
}

// --- Component ---

export default function Settings() {
  const { updateSettings: updateGlobalSettings } = useMonitoring();
  // UI state
  const [loading, setLoading] = useState(true);
  const [statusMsg, setStatusMsg] = useState('');
  const [errorMsg, setErrorMsg] = useState('');

  // Caregiver
  const [caregiverName, setCaregiverName] = useState('');
  const [caregiverEmail, setCaregiverEmail] = useState('');
  const [caregiverPhone, setCaregiverPhone] = useState('');

  // System settings
  const [monitoringEnabled, setMonitoringEnabled] = useState(false);
  const [requireConfirmation, setRequireConfirmation] = useState(false);
  const [notifyOnEveryFall, setNotifyOnEveryFall] = useState(true);

  // Detection
  const [activeModel, setActiveModel] = useState('Hybrid');
  const [activePreset, setActivePreset] = useState('Balanced');
  const [fallThreshold, setFallThreshold] = useState(85); // percent
  const [alertCooldown, setAlertCooldown] = useState(3);  // seconds
  const [activeOpId, setActiveOpId] = useState(null);

  // Privacy
  const [storeEventClips, setStoreEventClips] = useState(false);
  const [privacyMode, setPrivacyMode] = useState(true);

  // Prevent auto-saving defaults before the initial GET /api/settings completes.
  const [initialised, setInitialised] = useState(false);

  // Ops
  const [ops, setOps] = useState([]);
  const [opsErr, setOpsErr] = useState('');

  const savingRef = useRef(false);
  const statusTimerRef = useRef(null);

  useEffect(() => {
  if (!initialised) return;

  const p = presetFromOp(ops, activeOpId);
  if (p && p !== activePreset) setActivePreset(p);
  // eslint-disable-next-line react-hooks/exhaustive-deps
}, [ops, activeOpId, initialised]);


  // Helper to show temporary status messages
  const setStatus = (msg) => {
    setStatusMsg(msg);
    if (statusTimerRef.current) window.clearTimeout(statusTimerRef.current);
    statusTimerRef.current = window.setTimeout(() => setStatusMsg(''), 2500);
  };

  const safeSavePatch = async (patch) => {
    if (savingRef.current) return;
    savingRef.current = true;
    setErrorMsg('');
    try {
      // Use the global updater so other pages immediately receive the new settings.
      const ok = await updateGlobalSettings(patch);
      if (!ok) throw new Error('Failed to save settings');
      setStatus('Saved');
    } catch (e) {
      setErrorMsg(String(e?.message || e));
    } finally {
      savingRef.current = false;
    }
  };

  const loadSettings = async () => {
    setLoading(true);
    setErrorMsg('');
    try {
      const data = await apiFetch('/api/settings', { method: 'GET' });
      const sys = data?.system || data || {};

      setMonitoringEnabled(Boolean(sys.monitoring_enabled ?? false));
      setRequireConfirmation(Boolean(sys.require_confirmation ?? false));
      setNotifyOnEveryFall(Boolean(sys.notify_on_every_fall ?? true));

      if (sys.active_model_code) setActiveModel(modelCodeToLabel(sys.active_model_code));
      if (typeof sys.active_operating_point === 'number') setActiveOpId(sys.active_operating_point);

      if (typeof sys.fall_threshold === 'number') setFallThreshold(Math.round(sys.fall_threshold * 100));
      if (typeof sys.alert_cooldown_sec === 'number') setAlertCooldown(sys.alert_cooldown_sec);

      if (typeof sys.store_event_clips === 'boolean') setStoreEventClips(sys.store_event_clips);
      if (typeof sys.anonymize_skeleton_data === 'boolean') setPrivacyMode(sys.anonymize_skeleton_data);
    } catch (e) {
      setErrorMsg(String(e?.message || e));
    } finally {
      setLoading(false);
      setInitialised(true);
    }
  };

  const loadOperatingPoints = async (modelLabel) => {
    setOpsErr('');
    try {
      const model_code = modelLabelToCode(modelLabel);
      const data = await apiFetch(`/api/operating_points?model_code=${encodeURIComponent(model_code)}`, { method: 'GET' });
      const list = Array.isArray(data) ? data : Array.isArray(data?.operating_points) ? data.operating_points : [];
      setOps(list);
    } catch (e) {
      setOps([]);
      setOpsErr('Operating points unavailable. Using presets.');
    }
  };

  useEffect(() => {
    loadSettings();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    if (!initialised) return;
    loadOperatingPoints(activeModel);
    safeSavePatch({ active_model_code: modelLabelToCode(activeModel) });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [activeModel, initialised]);

  // Derived gradients for sliders
  const fallThrBg = useMemo(() => sliderBg(fallThreshold, 50, 99), [fallThreshold]);
  const cooldownBg = useMemo(() => sliderBg(alertCooldown, 1, 10), [alertCooldown]);

  const applyPreset = async (presetLabel) => {
    setActivePreset(presetLabel);

    const thr = PRESET_FALL_THR[presetLabel] ?? PRESET_FALL_THR.Balanced;
    const cd = PRESET_COOLDOWN[presetLabel] ?? PRESET_COOLDOWN.Balanced;
    setFallThreshold(Math.round(thr * 100));
    setAlertCooldown(cd);

    let opId = null;
    if (Array.isArray(ops) && ops.length > 0) {
      const op = pickOpForPreset(ops, presetLabel);
      if (op?.id != null) {
        opId = Number(op.id);
        setActiveOpId(opId);
      }
    }

    await safeSavePatch({
      active_operating_point: opId ?? undefined,
      fall_threshold: thr,
      alert_cooldown_sec: cd,
    });
  };

  const saveContact = async () => {
    setErrorMsg('');
    try {
      await apiFetch('/api/caregivers', {
        method: 'POST',
        body: JSON.stringify({ name: caregiverName, email: caregiverEmail, phone: caregiverPhone }),
      });
      setStatus('Caregiver saved');
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
              <span>Monitoring Enabled</span>
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
              <span>Notify on Every Fall</span>
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

            <div className={styles.toggleRow}>
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
            </div>

            <button 
              className={styles.actionBtn} 
              style={{ marginTop: 24, fontSize: '0.8rem' }}
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

            {/* Model Selection */}
            <div className={styles.row}>
              <span className={styles.labelBold}>Active Model</span>
              <div className={styles.radioGroup}>
                {['TCN', 'GCN', 'Hybrid'].map((m) => (
                  <label key={m} className={styles.radioLabel}>
                    {m}
                    <input 
                      type="radio" 
                      name="model" 
                      checked={activeModel === m}
                      onChange={() => setActiveModel(m)}
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
                {['High Sensitivity', 'Balanced', 'Low Sensitivity'].map((p) => (
                  <button
                    key={p}
                    className={`${styles.presetBtn} ${activePreset === p ? styles.activePreset : ''}`}
                    onClick={() => applyPreset(p)}
                  >
                    {p}
                  </button>
                ))}
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
                  disabled
                  type="range"
                  className={styles.rangeInput}
                  min={50} max={99}
                  value={fallThreshold}
                  style={{ background: fallThrBg }}
                  onChange={(e) => setFallThreshold(Number(e.target.value))}
                  onMouseUp={() => safeSavePatch({ fall_threshold: fallThreshold / 100 })}
                  onTouchEnd={() => safeSavePatch({ fall_threshold: fallThreshold / 100 })}
                />
              </div>

              {/* Alert Cooldown */}
              <div className={styles.sliderRow}>
                <div className={styles.sliderHeader}>
                  <span>Alert Cooldown Period</span>
                  <span>{alertCooldown}s</span>
                </div>
                <input
                  disabled
                  type="range"
                  className={styles.rangeInput}
                  min={1} max={10}
                  value={alertCooldown}
                  style={{ background: cooldownBg }}
                  onChange={(e) => setAlertCooldown(Number(e.target.value))}
                  onMouseUp={() => safeSavePatch({ alert_cooldown_sec: alertCooldown })}
                  onTouchEnd={() => safeSavePatch({ alert_cooldown_sec: alertCooldown })}
                />
              </div>

              {activeOpId != null && (
                <div style={{ fontSize: "0.8rem", color: "#9CA3AF", marginTop: 8 }}>
                  Active Op ID: <strong>{activeOpId}</strong>
                </div>
              )}
            </div>
          </div>

          {/* Privacy Settings */}
          <div className={styles.card}>
            <h3 className={styles.cardTitle}>Privacy & Data</h3>

            <div className={styles.privacyItem}>
              <div className={styles.privacyText}>
                <span className={styles.itemTitle}>Store Event Clips</span>
                <span className={styles.itemDesc}>Save short video segments around detected falls</span>
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

            <div className={styles.privacyItem}>
              <div className={styles.privacyText}>
                <span className={styles.itemTitle}>Anonymize Data</span>
                <span className={styles.itemDesc}>Convert video to skeleton data before storing</span>
              </div>
              <label className={styles.switch}>
                <input 
                  type="checkbox" 
                  checked={privacyMode}
                  onChange={(e) => {
                    const v = e.target.checked;
                    setPrivacyMode(v);
                    safeSavePatch({ anonymize_skeleton_data: v });
                  }}
                />
                <span className={styles.slider}></span>
              </label>
            </div>

            <div className={styles.privacyFooter}>
              <strong>Privacy Note:</strong> These settings control how data is persisted on the device.
            </div>
          </div>

        </div>
      </div>
    </div>
  );
}