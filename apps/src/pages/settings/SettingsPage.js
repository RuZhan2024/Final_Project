import React, { useEffect, useMemo, useState } from "react";

import styles from "../Settings.module.css";

import { useMonitoring } from "../../monitoring/MonitoringContext";
import { readBool } from "../../lib/booleans";
import { modelCodeToLabel, modelLabelToCode } from "../../lib/modelCodes";
import { presetFromOpCode, opCodeForPreset, PRESET_LABELS } from "../../lib/operatingPoints";
import { sliderBackground } from "../../lib/ui";
import { useCaregivers } from "./hooks/useCaregivers";

export default function SettingsPage() {
  const {
    settings,
    loaded,
    error: globalError,
    updateSettings,
    refresh,
    apiBase,
    setMonitoringOn,
  } = useMonitoring();

  const sys = settings?.system || {};

  // Toast UX
  const [statusMsg, setStatusMsg] = useState("");
  const [localErr, setLocalErr] = useState("");

  const showToast = (msg) => {
    setStatusMsg(msg);
    window.setTimeout(() => setStatusMsg(""), 2500);
  };

  // Caregivers
  const { caregivers, loading: caregiversLoading, error: caregiversError, upsert } = useCaregivers(apiBase);
  const primary = caregivers?.[0] || null;
  const [cgName, setCgName] = useState("");
  const [cgTelegramChatId, setCgTelegramChatId] = useState("");
  const [editingCaregiver, setEditingCaregiver] = useState(false);

  useEffect(() => {
    if (!primary) return;
    // Re-sync even when the same caregiver row is rewritten with normalized data;
    // depending only on the id would leave the form showing stale values.
    setCgName(primary.name || "");
    setCgTelegramChatId(primary.telegram_chat_id || "");
  }, [primary?.id, primary?.name, primary?.telegram_chat_id]);

  const monitoringEnabled = readBool(sys.monitoring_enabled, false);
  const notifyOnEveryFall = readBool(sys.notify_on_every_fall, true);
  const caregiverNameReady = Boolean(String(cgName || "").trim());
  const caregiverTelegramReady = Boolean(String(cgTelegramChatId || "").trim());

  const activeDatasetCode = String(sys.active_dataset_code || "caucafall").toLowerCase();
  const mcEnabled = readBool(sys.mc_enabled, false);
  const storeAnonymizedData = readBool(
    sys.store_anonymized_data,
    readBool(sys.store_event_clips, false) && readBool(sys.anonymize_skeleton_data, true)
  );

  const activeModelLabel = modelCodeToLabel(sys.active_model_code || "TCN");
  const activeOpCode = String(sys.active_op_code || "OP-2").toUpperCase();
  const activePreset = presetFromOpCode(activeOpCode);

  // These are real params derived from configs/ops/*.yaml (server sets them in GET /api/settings)
  const fallThresholdPct = useMemo(() => {
    const v = Number(sys.fall_threshold ?? 0.71);
    return Math.round(v * 1000) / 10; // 0.1% precision
  }, [sys.fall_threshold]);
  const lowThresholdPct = useMemo(() => {
    const v = Number(sys.tau_low ?? 0.5);
    return Math.round(v * 1000) / 10;
  }, [sys.tau_low]);

  const alertCooldownSec = useMemo(() => {
    return Math.round(Number(sys.alert_cooldown_sec ?? 3));
  }, [sys.alert_cooldown_sec]);

  const fallThrBg = useMemo(() => sliderBackground(fallThresholdPct, 0, 100), [fallThresholdPct]);
  const lowThrBg = useMemo(() => sliderBackground(lowThresholdPct, 0, 100), [lowThresholdPct]);
  const cooldownBg = useMemo(() => sliderBackground(alertCooldownSec, 0, 60), [alertCooldownSec]);

  async function savePatch(patch, actionLabel = "Settings update", okMsg = "") {
    setLocalErr("");
    const ok = await updateSettings(patch);
    if (ok) {
      showToast(okMsg || `${actionLabel} saved successfully.`);
    } else {
      setLocalErr(`${actionLabel} could not be saved. Please check API/DB status and try again.`);
    }
  }

  async function handleMonitoringToggle(next) {
    setLocalErr("");
    const desired = Boolean(next);
    const actual = await setMonitoringOn(desired);
    if (actual === desired) {
      showToast(
        next
          ? "Monitoring has been enabled. Live detection is now active."
          : "Monitoring has been disabled. Live detection is now paused."
      );
      return;
    }
    setLocalErr(
      desired
        ? "Monitoring could not be enabled. Open Monitor page and check camera/permissions, then try again."
        : "Monitoring could not be disabled due to API error. Please retry."
    );
  }

  async function saveCaregiver() {
    setLocalErr("");
    try {
      const result = await upsert({ id: primary?.id, name: cgName, telegram_chat_id: cgTelegramChatId });
      setEditingCaregiver(false);
      if (result?.db_available === false) {
        setLocalErr(
          "Caregiver information was accepted in fallback mode only. Save it again after database connectivity is restored."
        );
        return;
      }
      showToast("Caregiver information saved. Telegram alerts will use this contact profile.");
    } catch (e) {
      setLocalErr(`Caregiver information could not be saved. ${String(e?.message || e)}`);
    }
  }

  async function reloadSettingsWithToast() {
    setLocalErr("");
    try {
      await refresh();
      showToast("Settings reloaded from backend. The page now reflects the latest persisted configuration.");
    } catch (e) {
      setLocalErr(`Could not reload settings from backend. ${String(e?.message || e)}`);
    }
  }

  return (
    <div className={styles.container}>
      <h2 className={styles.pageTitle}>Settings</h2>

      {/* Toasts */}
      <div className={styles.toastContainer}>
        {statusMsg && (
          <div className={`${styles.toastItem} ${styles.toastSuccess}`}>{statusMsg}</div>
        )}
        {caregiversError && (
          <div className={`${styles.toastItem} ${styles.toastWarning}`}>
            Caregiver API: {caregiversError}
          </div>
        )}
        {globalError && (
          <div className={`${styles.toastItem} ${styles.toastWarning}`}>
            Settings API: {globalError}
          </div>
        )}
        {localErr && (
          <div className={`${styles.toastItem} ${styles.toastError}`}>{localErr}</div>
        )}
      </div>

      <div className={styles.contentGrid}>
        {/* LEFT */}
        <div className={styles.leftColumn}>
          {/* Caregiver */}
          <div className={styles.card}>
            <h3 className={styles.cardTitle}>Caregiver Information</h3>

            <div className={styles.formGroup}>
              <div className={styles.inputWrapper}>
                <label>Name</label>
                <input
                  type="text"
                  className={styles.textInput}
                  value={cgName}
                  onChange={(e) => setCgName(e.target.value)}
                  placeholder="e.g. Alice"
                  disabled={caregiversLoading || !editingCaregiver}
                />
              </div>

              <div className={styles.inputWrapper}>
                <label>Telegram Chat ID</label>
                <input
                  type="text"
                  className={styles.textInput}
                  value={cgTelegramChatId}
                  onChange={(e) => setCgTelegramChatId(e.target.value)}
                  placeholder="e.g. 123456789"
                  disabled={caregiversLoading || !editingCaregiver}
                />
              </div>

              <div className={styles.buttonRow}>
                <button
                  className={styles.actionBtn}
                  onClick={() => setEditingCaregiver(true)}
                  disabled={caregiversLoading || editingCaregiver}
                >
                  Edit Caregiver
                </button>
                <button
                  className={styles.actionBtn}
                  onClick={saveCaregiver}
                  disabled={caregiversLoading || !editingCaregiver}
                >
                  Save Caregiver
                </button>
              </div>
            </div>
          </div>

          {/* Monitoring */}
          <div className={styles.card}>
            <h3 className={styles.cardTitle}>Monitoring System</h3>

            <div className={styles.toggleRow}>
              <span>Monitoring Enabled</span>
              <label className={styles.switch}>
                <input
                  type="checkbox"
                  checked={monitoringEnabled}
                  onChange={(e) => handleMonitoringToggle(e.target.checked)}
                  disabled={!loaded}
                />
                <span className={styles.slider}></span>
              </label>
            </div>

            <div className={styles.toggleRow}>
              <span>Enable Notifications</span>
              <label className={styles.switch}>
                <input
                  type="checkbox"
                  checked={notifyOnEveryFall}
                  onChange={(e) => {
                    const on = e.target.checked;
                    savePatch(
                      on
                        ? { notify_on_every_fall: true }
                        : { notify_on_every_fall: false, notify_sms: false, notify_phone: false },
                      "Notification switch",
                      on
                        ? "Notifications enabled. Telegram alerts will be sent when falls are detected."
                        : "Notifications disabled. Telegram alerts are now paused."
                    );
                  }}
                  disabled={!loaded}
                />
                <span className={styles.slider}></span>
              </label>
            </div>

            <div className={styles.inlineInfo}>
              <div className={styles.inlineInfoTitle}>Notification Policy</div>
              <div>Telegram is the active notification channel for the current build.</div>
              <div>Each alert includes a concise generated caregiver summary when available.</div>
              <div className={styles.subtleMeta}>
                Caregiver status:
                {` Name ${caregiverNameReady ? "ready" : "missing"}, Telegram Chat ID ${caregiverTelegramReady ? "ready" : "missing"}.`}
              </div>
              {!caregiverTelegramReady && (
                <div className={styles.inlineInfoWarning}>
                  Telegram alerts cannot be delivered until a caregiver Telegram chat ID is saved.
                </div>
              )}
            </div>

            <button
              className={styles.actionBtn}
              onClick={reloadSettingsWithToast}
              disabled={!loaded}
            >
              Reload System Settings
            </button>
          </div>
        </div>

        {/* RIGHT */}
        <div className={styles.rightColumn}>
          {/* Detection */}
          <div className={styles.card}>
            <h3 className={styles.cardTitle}>Detection Settings</h3>

            {/* Dataset */}
            <div className={styles.row}>
              <span className={styles.labelBold}>Dataset</span>
              <div className={styles.radioGroup}>
                {[
                  { code: "le2i", label: "LE2I" },
                  { code: "caucafall", label: "CAUCAFall" },
                ].map((d) => (
                  <label
                    key={d.code}
                    className={`${styles.radioLabel} ${activeDatasetCode === d.code ? styles.radioLabelActive : ""}`}
                  >
                    {d.label}
                    <input
                      type="radio"
                      checked={activeDatasetCode === d.code}
                      onChange={() =>
                        savePatch(
                          { active_dataset_code: d.code },
                          "Dataset selection",
                          `Dataset switched to ${d.label}. Model/profile compatibility will follow this dataset.`
                        )
                      }
                      disabled={!loaded}
                    />
                    <span className={styles.radioCustom}></span>
                  </label>
                ))}
              </div>
            </div>

            {/* Uncertainty-aware live gate */}
            <div className={styles.toggleRow}>
              <span>Live Uncertainty Gate</span>
              <label className={styles.switch}>
                <input
                  type="checkbox"
                  checked={mcEnabled}
                  onChange={(e) =>
                    savePatch(
                      { mc_enabled: e.target.checked },
                      "Live uncertainty gate",
                      e.target.checked
                        ? "Live uncertainty gate enabled. Boundary windows can use MC dropout and high-uncertainty falls will be downgraded to uncertain."
                        : "Live uncertainty gate disabled. Live monitoring now uses deterministic single-pass scoring only."
                    )
                  }
                  disabled={!loaded}
                />
                <span className={styles.slider}></span>
              </label>
            </div>

            {/* Model */}
            <div className={styles.row}>
              <span className={styles.labelBold}>Active Model</span>
              <div className={styles.radioGroup}>
                {["TCN", "GCN", "HYBRID"].map((m) => (
                  <label
                    key={m}
                    className={`${styles.radioLabel} ${activeModelLabel === m ? styles.radioLabelActive : ""}`}
                  >
                    {m}
                    <input
                      type="radio"
                      name="model"
                      checked={activeModelLabel === m}
                      onChange={() =>
                        savePatch(
                          { active_model_code: modelLabelToCode(m) },
                          "Active model",
                          `Active model set to ${m}. New monitoring windows will use this model choice.`
                        )
                      }
                      disabled={!loaded}
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
                {PRESET_LABELS.map((p) => (
                  <button
                    key={p}
                    className={`${styles.presetBtn} ${activePreset === p ? styles.activePreset : ""}`}
                    onClick={() =>
                      savePatch(
                        { active_op_code: opCodeForPreset(p) },
                        "Operating point preset",
                        `${p} preset applied. Detection thresholds are now loaded from deploy config.`
                      )
                    }
                    disabled={!loaded}
                  >
                    {p}
                  </button>
                ))}
              </div>
            </div>

            {/* Sliders (read-only; derived from YAML configs on the server) */}
            <div className={styles.sliderGroup}>
              <div className={styles.sliderRow}>
                <div className={styles.sliderHeader}>
                  <span>High Threshold (tau_high)</span>
                  <span>{fallThresholdPct}%</span>
                </div>
                <input
                  type="range"
                  className={styles.rangeInput}
                  min={0}
                  max={100}
                  value={fallThresholdPct}
                  style={{ background: fallThrBg }}
                  disabled
                />
              </div>

              <div className={styles.sliderRow}>
                <div className={styles.sliderHeader}>
                  <span>Low Threshold (tau_low)</span>
                  <span>{lowThresholdPct}%</span>
                </div>
                <input
                  type="range"
                  className={styles.rangeInput}
                  min={0}
                  max={100}
                  value={lowThresholdPct}
                  style={{ background: lowThrBg }}
                  disabled
                />
              </div>

              <div className={styles.sliderRow}>
                <div className={styles.sliderHeader}>
                  <span>Alert Cooldown Period</span>
                  <span>{alertCooldownSec}s</span>
                </div>
                <input
                  type="range"
                  className={styles.rangeInput}
                  min={0}
                  max={60}
                  value={alertCooldownSec}
                  style={{ background: cooldownBg }}
                  disabled
                />
              </div>

              <div className={styles.subtleMeta}>
                Active Op Code: <strong>{activeOpCode}</strong>
              </div>
            </div>
          </div>

          {/* Privacy */}
          <div className={styles.card}>
            <h3 className={styles.cardTitle}>Privacy & Data</h3>

            <div className={styles.privacyItem}>
              <div className={styles.privacyText}>
                <span className={styles.itemTitle}>Store Anonymized Data</span>
                <span className={styles.itemDesc}>Persist event data only in anonymized skeleton form</span>
              </div>
              <label className={styles.switch}>
                <input
                  type="checkbox"
                  checked={storeAnonymizedData}
                  onChange={(e) =>
                    savePatch(
                      { store_anonymized_data: e.target.checked },
                      "Privacy storage mode",
                      e.target.checked
                        ? "Anonymized storage enabled. Event data will be saved as skeleton-only records."
                        : "Anonymized storage disabled. Event data privacy mode has been turned off."
                    )
                  }
                  disabled={!loaded}
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
