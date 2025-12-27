import React, { useEffect, useMemo, useRef, useState } from "react";
import styles from "./Settings.module.css";

// Prefer env var, fallback to localhost
const API_BASE =
  typeof process !== "undefined" && process.env && process.env.REACT_APP_API_BASE
    ? process.env.REACT_APP_API_BASE
    : "http://localhost:8000";

function modelLabelToCode(label) {
  if (label === "TCN") return "TCN";
  if (label === "GCN") return "GCN";
  if (label === "Hybrid") return "HYBRID";
  return "GCN";
}

function codeToModelLabel(code) {
  const c = (code || "").toUpperCase();
  if (c === "TCN") return "TCN";
  if (c === "GCN") return "GCN";
  if (c === "HYBRID") return "Hybrid";
  return "GCN";
}

function pickOp(ops, preset) {
  // ops: [{op_code, name, ...}]
  const norm = (s) => String(s || "").toLowerCase();
  if (!Array.isArray(ops) || ops.length === 0) return null;

  // Prefer explicit OP codes if present in name/op_code
  const isOp1 = (o) => norm(o.op_code).includes("op-1") || norm(o.name).includes("op-1") || norm(o.name).includes("high");
  const isOp2 = (o) => norm(o.op_code).includes("op-2") || norm(o.name).includes("op-2") || norm(o.name).includes("balanced");
  const isOp3 = (o) => norm(o.op_code).includes("op-3") || norm(o.name).includes("op-3") || norm(o.name).includes("low");

  if (preset === "high") return ops.find(isOp1) || ops[0];
  if (preset === "balanced") return ops.find(isOp2) || ops[0];
  return ops.find(isOp3) || ops[ops.length - 1];
}

export default function Settings() {
  // --- Section 1: Caregiver Info (sync with /api/caregivers) ---
  const [caregiverId, setCaregiverId] = useState(null);
  const [caregiverName, setCaregiverName] = useState("John Smith");
  const [caregiverEmail, setCaregiverEmail] = useState("john@example.com");
  const [caregiverPhone, setCaregiverPhone] = useState("+1 234 567 890");

  // --- Section 2: Notification Preferences ---
  const [notifyPush, setNotifyPush] = useState(true);
  const [notifyEmail, setNotifyEmail] = useState(true);
  const [notifySMS, setNotifySMS] = useState(false);

  // --- Section 3: Detection Settings ---
  const [activeModel, setActiveModel] = useState("GCN"); // "TCN" | "GCN" | "Hybrid"
  const [activePreset, setActivePreset] = useState("Balanced");

  // Sliders
  const [fallThreshold, setFallThreshold] = useState(85); // 0-100
  const [alertCooldown, setAlertCooldown] = useState(10); // 0-60 sec

  // Privacy toggles
  const [storeEventClips, setStoreEventClips] = useState(true);
  const [privacyMode, setPrivacyMode] = useState(false);

  const [ops, setOps] = useState([]);
  const [opsErr, setOpsErr] = useState(null);

  const busyRef = useRef(false);

  // Helper to create the "Filled" slider effect
  // Note: Backend logic uses 0-100 for threshold, so we calculate percentage directly
  const getSliderStyle = (value, max = 100) => {
    const percentage = (value / max) * 100;
    return {
      background: `linear-gradient(to right, #4F46E5 ${percentage}%, #E5E7EB ${percentage}%)`
    };
  };

  // --- API Effects ---

  // Load settings + caregiver + ops
  useEffect(() => {
    (async () => {
      // settings
      try {
        const rs = await fetch(`${API_BASE}/api/settings`);
        if (rs.ok) {
          const s = await rs.json();
          const sys = s?.system || {};
          if (sys?.active_model_code) setActiveModel(codeToModelLabel(sys.active_model_code));
        }
      } catch { /* ignore */ }

      // caregiver (take first caregiver for resident 1)
      try {
        const rc = await fetch(`${API_BASE}/api/caregivers?resident_id=1`);
        if (rc.ok) {
          const data = await rc.json();
          const arr = Array.isArray(data?.caregivers) ? data.caregivers : [];
          if (arr.length > 0) {
            const c = arr[0];
            setCaregiverId(c.id);
            if (c.name) setCaregiverName(c.name);
            if (c.email) setCaregiverEmail(c.email);
            if (c.phone) setCaregiverPhone(c.phone);
          }
        }
      } catch { /* ignore */ }
    })();
  }, []);

  // Load operating points when model changes
  useEffect(() => {
    (async () => {
      try {
        setOpsErr(null);
        const modelCode = modelLabelToCode(activeModel);
        const r = await fetch(`${API_BASE}/api/operating_points?model_code=${encodeURIComponent(modelCode)}`);
        if (!r.ok) throw new Error(await r.text());
        const data = await r.json();
        const arr = Array.isArray(data?.operating_points) ? data.operating_points : [];
        setOps(arr);
      } catch (e) {
        setOpsErr(String(e?.message || e));
        setOps([]);
      }
    })();
  }, [activeModel]);

  const activeModelCode = useMemo(() => modelLabelToCode(activeModel), [activeModel]);

  // --- Handlers ---

  async function setBackendSettings(patch) {
    if (busyRef.current) return;
    busyRef.current = true;
    try {
      const r = await fetch(`${API_BASE}/api/settings`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(patch),
      });
      if (!r.ok) throw new Error(await r.text());
    } catch (e) {
      console.error("Failed to update settings:", e);
    } finally {
      busyRef.current = false;
    }
  }

  async function upsertCaregiver() {
    try {
      if (!caregiverEmail && !caregiverPhone && !caregiverName) return;

      if (caregiverId == null) {
        const r = await fetch(`${API_BASE}/api/caregivers`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            resident_id: 1,
            name: caregiverName,
            email: caregiverEmail,
            phone: caregiverPhone,
          }),
        });
        if (!r.ok) throw new Error(await r.text());
        const data = await r.json();
        if (data?.caregiver?.id != null) setCaregiverId(data.caregiver.id);
      } else {
        const r = await fetch(`${API_BASE}/api/caregivers/${caregiverId}`, {
          method: "PUT",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            name: caregiverName,
            email: caregiverEmail,
            phone: caregiverPhone,
          }),
        });
        if (!r.ok) throw new Error(await r.text());
      }
    } catch (e) {
      console.error("Failed to save caregiver:", e);
    }
  }

  async function selectModel(label) {
    setActiveModel(label);
    await setBackendSettings({ active_model_code: modelLabelToCode(label) });
  }

  async function selectPreset(presetLabel) {
    setActivePreset(presetLabel);
    // Map UI preset labels to operating point logic keys
    // "High Safety" -> high, "Balanced" -> balanced, "Low Alarms" -> low
    const presetKey =
      presetLabel === "High Safety" ? "high" : presetLabel === "Balanced" ? "balanced" : "low";
    
    const op = pickOp(ops, presetKey);
    if (op?.op_code) {
      await setBackendSettings({ active_operating_point: op.op_code });
    }
  }

  async function sendTestAlert() {
    try {
      const r = await fetch(`${API_BASE}/api/notifications/test`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          resident_id: 1,
          channel: notifyEmail ? "email" : notifySMS ? "sms" : "push",
          message: "Test alert from Settings page",
        }),
      });
      if (!r.ok) throw new Error(await r.text());
      alert("Test alert request sent.");
    } catch (e) {
      alert(`Failed to send test alert: ${String(e?.message || e)}`);
    }
  }

  return (
    <div className={styles.container}>
      <h2 className={styles.pageTitle}>Settings</h2>

      <div className={styles.contentGrid}>
        
        {/* --- LEFT COLUMN: Notification Settings --- */}
        <div className={styles.leftColumn}>
          <div className={styles.card}>
            <h3 className={styles.cardTitle}>Notification Settings</h3>
            
            <div className={styles.formGroup}>
              <div className={styles.inputWrapper}>
                <label>Caregiver Name</label>
                <input 
                  type="text" 
                  value={caregiverName}
                  onChange={(e) => setCaregiverName(e.target.value)}
                  onBlur={upsertCaregiver}
                  className={styles.textInput} 
                />
              </div>

              <div className={styles.inputWrapper}>
                <label>Email</label>
                <input 
                  type="email" 
                  value={caregiverEmail}
                  onChange={(e) => setCaregiverEmail(e.target.value)}
                  onBlur={upsertCaregiver}
                  className={styles.textInput} 
                />
              </div>

              <div className={styles.inputWrapper}>
                <label>Phone</label>
                <input 
                  type="tel" 
                  value={caregiverPhone}
                  onChange={(e) => setCaregiverPhone(e.target.value)}
                  onBlur={upsertCaregiver}
                  className={styles.textInput} 
                />
              </div>

              {/* Notification Toggles */}
              <div className={styles.toggleRow}>
                <span>Notify with Push</span>
                <label className={styles.switch}>
                  <input 
                    type="checkbox" 
                    checked={notifyPush}
                    onChange={() => setNotifyPush(!notifyPush)}
                  />
                  <span className={styles.slider}></span>
                </label>
              </div>

              <div className={styles.toggleRow}>
                <span>Notify with Email</span>
                <label className={styles.switch}>
                  <input 
                    type="checkbox" 
                    checked={notifyEmail}
                    onChange={() => setNotifyEmail(!notifyEmail)}
                  />
                  <span className={styles.slider}></span>
                </label>
              </div>

              <div className={styles.toggleRow}>
                <span>Notify with SMS</span>
                <label className={styles.switch}>
                  <input 
                    type="checkbox" 
                    checked={notifySMS}
                    onChange={() => setNotifySMS(!notifySMS)}
                  />
                  <span className={styles.slider}></span>
                </label>
              </div>

              <button type="button" className={styles.actionBtn} onClick={sendTestAlert}>
                Send Test Alert
              </button>
            </div>
          </div>
        </div>

        {/* --- RIGHT COLUMN: Model & Privacy --- */}
        <div className={styles.rightColumn}>
          
          {/* Card 1: Model & Threshold Settings */}
          <div className={styles.card}>
            <h3 className={styles.cardTitle}>Model & Threshold Settings</h3>
            
            {/* Model Selection */}
            <div className={styles.row}>
              <span className={styles.labelBold}>Active Model</span>
              <div className={styles.radioGroup}>
                {["TCN", "GCN", "Hybrid"].map((model) => (
                  <label key={model} className={styles.radioLabel}>
                    {model}
                    <input 
                      type="radio" 
                      name="model" 
                      checked={activeModel === model}
                      onChange={() => selectModel(model)}
                    />
                    <span className={styles.customRadio}></span>
                  </label>
                ))}
              </div>
            </div>

            {/* Operating Point Presets */}
            <div className={styles.sectionSpace}>
              <span className={styles.labelBold}>Operating Point Presets</span>
              {opsErr && (
                <div style={{ margin: "8px 0", color: "#B45309", fontSize: "0.85rem" }}>
                  Could not load operating points: {opsErr}
                </div>
              )}
              <div className={styles.presetButtons}>
                {["High Safety", "Balanced", "Low Alarms"].map((preset) => (
                  <button
                    key={preset}
                    className={`${styles.presetBtn} ${activePreset === preset ? styles.activePreset : ''}`}
                    onClick={() => selectPreset(preset)}
                  >
                    {preset}
                  </button>
                ))}
              </div>
            </div>

            {/* Sliders Group */}
            <div className={styles.sliderGroup}>
              
              {/* Slider 1: Fall Detection Threshold */}
              <div className={styles.sliderRow}>
                <div className={styles.sliderHeader}>
                  <span>Fall Detection Threshold</span>
                  <span>{fallThreshold}%</span>
                </div>
                <input 
                  type="range" 
                  min="0" max="100" 
                  value={fallThreshold}
                  onChange={(e) => setFallThreshold(Number(e.target.value))}
                  className={styles.rangeInput}
                  style={getSliderStyle(fallThreshold, 100)} 
                />
              </div>

              {/* Slider 2: Alert Cooldown */}
              <div className={styles.sliderRow}>
                <div className={styles.sliderHeader}>
                  <span>Alert Cooldown Period</span>
                  <span>{alertCooldown}s</span>
                </div>
                <input 
                  type="range" 
                  min="0" max="60" 
                  value={alertCooldown}
                  onChange={(e) => setAlertCooldown(Number(e.target.value))}
                  className={styles.rangeInput} 
                  style={getSliderStyle(alertCooldown, 60)}
                />
              </div>

              <div style={{ fontSize: "0.8rem", color: "#9CA3AF", marginTop: 8 }}>
                Backend Active Code: {activeModelCode}
              </div>

            </div>
          </div>

          {/* Card 2: Privacy Settings */}
          <div className={styles.card}>
            <h3 className={styles.cardTitle}>Privacy & Data Settings</h3>
            
            <div className={styles.privacyItem}>
              <div className={styles.privacyText}>
                <span className={styles.itemTitle}>Store Event Clips</span>
                <span className={styles.itemDesc}>Save short segments around detected falls for review</span>
              </div>
              <label className={styles.switch}>
                <input 
                  type="checkbox" 
                  checked={storeEventClips}
                  onChange={() => setStoreEventClips(!storeEventClips)}
                />
                <span className={styles.slider}></span>
              </label>
            </div>

            <div className={styles.privacyItem}>
              <div className={styles.privacyText}>
                <span className={styles.itemTitle}>Privacy Mode</span>
                <span className={styles.itemDesc}>Do not store video, only keep anonymous skeleton data</span>
              </div>
              <label className={styles.switch}>
                <input 
                  type="checkbox" 
                  checked={privacyMode}
                  onChange={() => setPrivacyMode(!privacyMode)}
                />
                <span className={styles.slider}></span>
              </label>
            </div>

            <div className={styles.privacyFooter}>
              <strong>Privacy by Design:</strong> This system uses skeleton-based detection, which means no raw RGB video is stored long-term. All data is encrypted and stored locally on your device.
            </div>
          </div>

        </div>
      </div>
    </div>
  );
}