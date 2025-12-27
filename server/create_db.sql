CREATE DATABASE IF NOT EXISTS elder_fall_monitor;
USE elder_fall_monitor;

CREATE USER IF NOT EXISTS 'fall_app'@'localhost' IDENTIFIED BY 'strong_password_here';
GRANT ALL PRIVILEGES ON elder_fall_monitor.* TO 'fall_app'@'localhost';
FLUSH PRIVILEGES;


-- 1) Create database
CREATE DATABASE IF NOT EXISTS elder_fall_monitor
  CHARACTER SET utf8mb4
  COLLATE utf8mb4_unicode_ci;

USE elder_fall_monitor;

-- 2) (Optional) Create dedicated DB user
--    Change 'strong_password_here' before using in real setup.
CREATE USER IF NOT EXISTS 'fall_app'@'localhost'
  IDENTIFIED BY 'strong_password_here';

GRANT ALL PRIVILEGES ON elder_fall_monitor.* TO 'fall_app'@'localhost';
FLUSH PRIVILEGES;

-- =========================================================
-- 3) Tables
-- =========================================================

-- -----------------------------
-- residents
-- One row per monitored person / flat
-- -----------------------------
CREATE TABLE residents (
    id              INT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    display_name    VARCHAR(100) NOT NULL,          -- e.g. "Mrs. Chen"
    date_of_birth   DATE NULL,
    notes           TEXT,
    event_metadata      JSON NULL,

    created_at      TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at      TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
                    ON UPDATE CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- -----------------------------
-- caregivers
-- Contact details for alerts (Settings → Notification Settings)
-- -----------------------------
CREATE TABLE caregivers (
    id              INT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    resident_id     INT UNSIGNED NOT NULL,
    name            VARCHAR(100) NOT NULL,
    email           VARCHAR(255) NOT NULL,
    phone           VARCHAR(50),
    is_primary      TINYINT(1) NOT NULL DEFAULT 1,
    created_at      TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at      TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
                    ON UPDATE CURRENT_TIMESTAMP,
    CONSTRAINT fk_caregivers_resident
      FOREIGN KEY (resident_id) REFERENCES residents(id)
      ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- -----------------------------
-- models
-- Different ML models (TCN LE2I, TCN URFD, CAUCAFall, future GCN/Hybrid)
-- -----------------------------
CREATE TABLE models (
    id              INT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    code            VARCHAR(50) NOT NULL UNIQUE,    -- 'tcn_le2i', 'tcn_urfd', 'tcn_caucafall', ...
    name            VARCHAR(100) NOT NULL,          -- "TCN (LE2I)" etc.
    family          VARCHAR(50) NOT NULL,           -- 'TCN', 'GCN', 'Hybrid'
    dataset         VARCHAR(100),                   -- 'LE2I', 'URFD', 'CAUCAFall'
    window_size     INT UNSIGNED,                   -- W
    stride          INT UNSIGNED,                   -- S
    default_thr     DECIMAL(4,3),                   -- recommended threshold
    description     TEXT,
    created_at      TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at      TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
                    ON UPDATE CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- Stable DB codes (recommended)
-- Insert these once so the server can reference them:
-- INSERT INTO models (code, name, family) VALUES ('TCN','TCN','TCN'),('GCN','GCN','GCN'),('HYBRID','HYBRID','Hybrid');


-- -----------------------------
-- operating_points
-- Per-model operating point presets (High Safety / Balanced / Low Alarms)
-- and their thresholds
-- -----------------------------
CREATE TABLE operating_points (
    id              INT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    model_id        INT UNSIGNED NOT NULL,
    name            VARCHAR(50) NOT NULL,           -- e.g. "High Safety"
    code            VARCHAR(50) NOT NULL,           -- e.g. 'OP1', 'OP2', 'OP3'
    thr_detect      DECIMAL(4,3) NOT NULL,          -- main detection threshold
    thr_low_conf    DECIMAL(4,3) NULL,              -- low confidence bound (slider)
    thr_high_conf   DECIMAL(4,3) NULL,              -- high confidence bound (slider)
    est_fa24h       DECIMAL(7,3) NULL,              -- optional: FA/24h estimate
    est_recall      DECIMAL(4,3) NULL,              -- optional: recall at this OP
    created_at      TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at      TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
                    ON UPDATE CURRENT_TIMESTAMP,
    CONSTRAINT fk_operating_points_model
      FOREIGN KEY (model_id) REFERENCES models(id)
      ON DELETE CASCADE,
    CONSTRAINT uq_operating_points_model_code
      UNIQUE (model_id, code)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- -----------------------------
-- system_settings
-- Per-resident system configuration
-- (active model, OP, monitoring + privacy + notification flags)
-- -----------------------------
CREATE TABLE system_settings (
    id                      INT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    resident_id             INT UNSIGNED NOT NULL UNIQUE,
    active_model_id         INT UNSIGNED NULL,
    active_operating_point  INT UNSIGNED NULL,

    notify_on_every_fall    TINYINT(1) NOT NULL DEFAULT 1,
    require_confirmation    TINYINT(1) NOT NULL DEFAULT 0,

    monitoring_enabled      TINYINT(1) NOT NULL DEFAULT 1,

    store_event_clips       TINYINT(1) NOT NULL DEFAULT 0,
    anonymize_skeleton_data TINYINT(1) NOT NULL DEFAULT 1,

    api_online              TINYINT(1) NOT NULL DEFAULT 1,
    last_latency_ms         INT NULL,
    last_health_check_at    DATETIME NULL,

    created_at              TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at              TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
                            ON UPDATE CURRENT_TIMESTAMP,

    CONSTRAINT fk_system_settings_resident
      FOREIGN KEY (resident_id) REFERENCES residents(id)
      ON DELETE CASCADE,
    CONSTRAINT fk_system_settings_model
      FOREIGN KEY (active_model_id) REFERENCES models(id)
      ON DELETE SET NULL,
    CONSTRAINT fk_system_settings_operating_point
      FOREIGN KEY (active_operating_point) REFERENCES operating_points(id)
      ON DELETE SET NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- -----------------------------
-- events
-- Detected falls / alarms / test events
-- Drives Dashboard counts + Events page
-- -----------------------------
CREATE TABLE events (
    id                  INT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    resident_id         INT UNSIGNED NOT NULL,
    model_id            INT UNSIGNED NOT NULL,
    operating_point_id  INT UNSIGNED NULL,

    event_time          DATETIME NOT NULL,
    type                VARCHAR(30) NOT NULL,       -- 'fall', 'test_fall', 'system', etc.

    p_fall              DECIMAL(4,3) NOT NULL,      -- model probability at trigger
    delay_seconds       DECIMAL(5,2) NULL,          -- detection delay
    fa24h_snapshot      DECIMAL(7,3) NULL,          -- optional FA/24h estimate at this point

    status              VARCHAR(30) NOT NULL DEFAULT 'pending_review',
                        -- 'pending_review', 'confirmed_fall', 'false_alarm', 'dismissed'
    reviewed_by         INT UNSIGNED NULL,
    reviewed_at         DATETIME NULL,

    notes               TEXT,

    created_at          TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at          TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
                        ON UPDATE CURRENT_TIMESTAMP,

    CONSTRAINT fk_events_resident
      FOREIGN KEY (resident_id) REFERENCES residents(id)
      ON DELETE CASCADE,
    CONSTRAINT fk_events_model
      FOREIGN KEY (model_id) REFERENCES models(id)
      ON DELETE RESTRICT,
    CONSTRAINT fk_events_operating_point
      FOREIGN KEY (operating_point_id) REFERENCES operating_points(id)
      ON DELETE SET NULL,
    CONSTRAINT fk_events_reviewed_by
      FOREIGN KEY (reviewed_by) REFERENCES caregivers(id)
      ON DELETE SET NULL,

    INDEX idx_events_resident_time (resident_id, event_time)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- -----------------------------
-- notifications_log
-- Records alerts sent to caregivers (incl. "Send Test Alert")
-- -----------------------------
CREATE TABLE notifications_log (
    id              INT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    resident_id     INT UNSIGNED NOT NULL,
    caregiver_id    INT UNSIGNED NULL,
    event_id        INT UNSIGNED NULL,
    channel         VARCHAR(20) NOT NULL,          -- 'email', 'sms', 'push'
    type            VARCHAR(30) NOT NULL,          -- 'test_alert', 'fall_alert'
    sent_at         DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    success         TINYINT(1) NOT NULL DEFAULT 1,
    error_message   TEXT,

    CONSTRAINT fk_notifications_resident
      FOREIGN KEY (resident_id) REFERENCES residents(id)
      ON DELETE CASCADE,
    CONSTRAINT fk_notifications_caregiver
      FOREIGN KEY (caregiver_id) REFERENCES caregivers(id)
      ON DELETE SET NULL,
    CONSTRAINT fk_notifications_event
      FOREIGN KEY (event_id) REFERENCES events(id)
      ON DELETE SET NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- =========================================================
-- 4) (Optional) Seed a default resident + system_settings
--    You can modify or remove this section.
-- =========================================================
INSERT INTO residents (display_name, date_of_birth, notes)
VALUES ('Default Resident', NULL, 'Demo resident for fall detection system')
ON DUPLICATE KEY UPDATE display_name = VALUES(display_name);

-- ensure a system_settings row exists for resident id=1
INSERT INTO system_settings (resident_id)
SELECT id FROM residents WHERE id = 1
ON DUPLICATE KEY UPDATE resident_id = resident_id;
