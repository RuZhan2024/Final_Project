-- create_db_fixed.sql
-- MySQL schema + demo seed data for Elder Fall Monitor.

DROP DATABASE IF EXISTS elder_fall_monitor;
CREATE DATABASE elder_fall_monitor CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
USE elder_fall_monitor;

-- Optional: create a dedicated user (recommended for local dev)
-- CREATE USER IF NOT EXISTS 'fall_app'@'127.0.0.1' IDENTIFIED WITH mysql_native_password BY 'strong_password_here';
-- GRANT ALL PRIVILEGES ON elder_fall_monitor.* TO 'fall_app'@'127.0.0.1';
-- FLUSH PRIVILEGES;

CREATE TABLE residents (
  id INT AUTO_INCREMENT PRIMARY KEY,
  name VARCHAR(100) NOT NULL,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE caregivers (
  id INT AUTO_INCREMENT PRIMARY KEY,
  resident_id INT NOT NULL,
  name VARCHAR(100) NOT NULL,
  email VARCHAR(255) NULL,
  phone VARCHAR(50) NULL,
  telegram_chat_id VARCHAR(120) NULL,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  CONSTRAINT fk_caregiver_resident FOREIGN KEY (resident_id) REFERENCES residents(id) ON DELETE CASCADE
);

CREATE TABLE models (
  id INT AUTO_INCREMENT PRIMARY KEY,
  code VARCHAR(16) NOT NULL UNIQUE,
  name VARCHAR(100) NOT NULL,
  description TEXT NULL,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE operating_points (
  id INT AUTO_INCREMENT PRIMARY KEY,
  model_id INT NOT NULL,
  code VARCHAR(16) NOT NULL,
  name VARCHAR(100) NOT NULL,
  thr_detect DECIMAL(6,4) NOT NULL,
  thr_low_conf DECIMAL(6,4) NULL,
  thr_high_conf DECIMAL(6,4) NULL,
  est_fa24h DECIMAL(10,4) NULL,
  est_recall DECIMAL(6,4) NULL,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  UNIQUE KEY uq_model_code (model_id, code),
  CONSTRAINT fk_op_model FOREIGN KEY (model_id) REFERENCES models(id) ON DELETE CASCADE
);

CREATE TABLE system_settings (
  id INT AUTO_INCREMENT PRIMARY KEY,
  resident_id INT NOT NULL,
  monitoring_enabled TINYINT(1) NOT NULL DEFAULT 0,
  api_online TINYINT(1) NOT NULL DEFAULT 1,
  last_latency_ms INT NULL,
  active_model_code VARCHAR(16) NOT NULL DEFAULT 'TCN',
  active_operating_point INT NULL,
  alert_cooldown_sec INT NOT NULL DEFAULT 3,
  notify_on_every_fall TINYINT(1) NOT NULL DEFAULT 1,
  fall_threshold DECIMAL(6,4) NULL DEFAULT 0.7100,
  store_event_clips TINYINT(1) NOT NULL DEFAULT 0,
  anonymize_skeleton_data TINYINT(1) NOT NULL DEFAULT 1,
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  CONSTRAINT fk_settings_resident FOREIGN KEY (resident_id) REFERENCES residents(id) ON DELETE CASCADE,
  CONSTRAINT fk_settings_op FOREIGN KEY (active_operating_point) REFERENCES operating_points(id) ON DELETE SET NULL
);

CREATE TABLE events (
  id BIGINT AUTO_INCREMENT PRIMARY KEY,
  resident_id INT NOT NULL,
  ts TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  type VARCHAR(32) NOT NULL,
  severity VARCHAR(16) NULL,
  model_code VARCHAR(16) NULL,
  operating_point_id INT NULL,
  score DECIMAL(6,4) NULL,
  meta JSON NULL,
  INDEX idx_events_ts (ts),
  INDEX idx_events_resident_ts (resident_id, ts),
  CONSTRAINT fk_event_resident FOREIGN KEY (resident_id) REFERENCES residents(id) ON DELETE CASCADE,
  CONSTRAINT fk_event_op FOREIGN KEY (operating_point_id) REFERENCES operating_points(id) ON DELETE SET NULL
);

CREATE TABLE notifications_log (
  id BIGINT AUTO_INCREMENT PRIMARY KEY,
  resident_id INT NOT NULL,
  ts TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  channel VARCHAR(16) NOT NULL,
  status VARCHAR(16) NOT NULL,
  message TEXT NULL,
  event_id BIGINT NULL,
  INDEX idx_notify_resident_ts (resident_id, ts),
  CONSTRAINT fk_notify_resident FOREIGN KEY (resident_id) REFERENCES residents(id) ON DELETE CASCADE,
  CONSTRAINT fk_notify_event FOREIGN KEY (event_id) REFERENCES events(id) ON DELETE SET NULL
);

-- Seed demo rows (id = 1)
INSERT INTO residents (name) VALUES ('Demo Resident');

INSERT INTO caregivers (resident_id, name, email, phone)
VALUES (1, 'Demo Caregiver', 'caregiver@example.com', '0000000000');

INSERT INTO models (code, name, description) VALUES
  ('TCN', 'TCN', 'Temporal Convolution Network'),
  ('GCN', 'GCN', 'Graph Convolution Network'),
  ('HYBRID', 'Hybrid', 'GCN + TCN (hybrid)');

-- Seed 3 demo operating points for TCN (model id = 1) using current deployment lock
INSERT INTO operating_points (model_id, code, name, thr_detect, thr_low_conf, thr_high_conf, est_fa24h, est_recall) VALUES
  (1, 'OP-1', 'High Sensitivity', 0.20, 0.1560, 0.20, NULL, NULL),
  (1, 'OP-2', 'Balanced',         0.71, 0.5538, 0.71, NULL, NULL),
  (1, 'OP-3', 'Low Sensitivity',  0.95, 0.7410, 0.95, NULL, NULL);

-- Default settings row (Balanced, TCN)
INSERT INTO system_settings (resident_id, active_model_code, active_operating_point, alert_cooldown_sec, notify_on_every_fall)
VALUES (
  1,
  'TCN',
  (SELECT id FROM operating_points WHERE model_id=1 AND code='OP-2' LIMIT 1),
  3,
  1
);
