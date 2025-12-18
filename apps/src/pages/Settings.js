import React, { useState } from "react";
import styles from "./Settings.module.css";

function Settings() {
  // State for toggles/inputs (visual only for this demo)
  const [activeModel, setActiveModel] = useState("TCN");
  const [activePreset, setActivePreset] = useState("Balanced");

  return (
    <div className={styles.container}>
      <h2 className={styles.pageTitle}>Settings</h2>

      <div className={styles.contentGrid}>
        
        {/* --- LEFT COLUMN: Notification Settings --- */}
        <div className={styles.leftColumn}>
          <div className={styles.card}>
            <h3 className={styles.cardTitle}>Notification Settings</h3>
            
            <form className={styles.formGroup}>
              <div className={styles.inputWrapper}>
                <label>Caregiver Name</label>
                <input type="text" placeholder="" className={styles.textInput} />
              </div>

              <div className={styles.inputWrapper}>
                <label>Email</label>
                <input type="email" placeholder="" className={styles.textInput} />
              </div>

              <div className={styles.inputWrapper}>
                <label>Phone</label>
                <input type="tel" placeholder="" className={styles.textInput} />
              </div>

              {/* Toggles Section */}
              <div className={styles.toggleRow}>
                <span>Notify with Call</span>
                <label className={styles.switch}>
                  <input type="checkbox" />
                  <span className={styles.slider}></span>
                </label>
              </div>

              <div className={styles.toggleRow}>
                <span>Notify with Msg</span>
                <label className={styles.switch}>
                  <input type="checkbox" defaultChecked />
                  <span className={styles.slider}></span>
                </label>
              </div>

              <button type="button" className={styles.actionBtn}>
                Send Test Alert
              </button>
            </form>
          </div>
        </div>

        {/* --- RIGHT COLUMN: Model & Privacy --- */}
        <div className={styles.rightColumn}>
          
          {/* Card 1: Model & Thresholds */}
          <div className={styles.card}>
            <h3 className={styles.cardTitle}>Model & Threshold Settings</h3>
            
            {/* Model Selection (Radio look-alike) */}
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
                      onChange={() => setActiveModel(model)}
                    />
                    <span className={styles.customRadio}></span>
                  </label>
                ))}
              </div>
            </div>

            {/* Operating Point Presets */}
            <div className={styles.sectionSpace}>
              <span className={styles.labelBold}>Operating Point Presets</span>
              <div className={styles.presetButtons}>
                {["High Safety", "Balanced", "Low Alarms"].map((preset) => (
                  <button
                    key={preset}
                    className={`${styles.presetBtn} ${activePreset === preset ? styles.activePreset : ''}`}
                    onClick={() => setActivePreset(preset)}
                  >
                    {preset}
                  </button>
                ))}
              </div>
            </div>

            {/* Sliders */}
            <div className={styles.sliderGroup}>
              <div className={styles.sliderRow}>
                <div className={styles.sliderHeader}>
                  <span>Fall Detection Threshold</span>
                  <span>0.47</span>
                </div>
                <input type="range" min="0" max="1" step="0.01" defaultValue="0.47" className={styles.rangeInput} />
              </div>

              <div className={styles.sliderRow}>
                <div className={styles.sliderHeader}>
                  <span>Low Confidence Bound</span>
                  <span>0.47</span>
                </div>
                <input type="range" min="0" max="1" step="0.01" defaultValue="0.47" className={styles.rangeInput} />
              </div>

              <div className={styles.sliderRow}>
                <div className={styles.sliderHeader}>
                  <span>High Confidence Bound</span>
                  <span>0.47</span>
                </div>
                <input type="range" min="0" max="1" step="0.01" defaultValue="0.47" className={styles.rangeInput} />
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
                <input type="checkbox" />
                <span className={styles.slider}></span>
              </label>
            </div>

            <div className={styles.privacyItem}>
              <div className={styles.privacyText}>
                <span className={styles.itemTitle}>Anonymize Skeleton Data</span>
                <span className={styles.itemDesc}>Discard RGB frames, only keep joint coordinates</span>
              </div>
              <label className={styles.switch}>
                <input type="checkbox" defaultChecked />
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

export default Settings;