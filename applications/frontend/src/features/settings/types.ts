export interface CaregiverRecord {
  id?: number | null;
  name?: string | null;
  telegram_chat_id?: string | null;
  [key: string]: unknown;
}

export interface SystemSettings {
  monitoring_enabled?: boolean | number | string | null;
  notify_on_every_fall?: boolean | number | string | null;
  notify_sms?: boolean | number | string | null;
  notify_phone?: boolean | number | string | null;
  active_dataset_code?: string | null;
  active_model_code?: string | null;
  active_op_code?: string | null;
  active_operating_point?: number | null;
  mc_enabled?: boolean | number | string | null;
  store_anonymized_data?: boolean | number | string | null;
  store_event_clips?: boolean | number | string | null;
  anonymize_skeleton_data?: boolean | number | string | null;
  fall_threshold?: number | string | null;
  tau_low?: number | string | null;
  alert_cooldown_sec?: number | string | null;
  deploy_params?: {
    ui?: {
      op_code?: string | null;
      tau_high?: number | string | null;
      tau_low?: number | string | null;
      k?: number | string | null;
      n?: number | string | null;
      confirm?: number | string | null;
      confirm_k?: number | string | null;
      confirm_n?: number | string | null;
      cooldown_s?: number | string | null;
      cooldownSec?: number | string | null;
      cooldown_sec?: number | string | null;
      [key: string]: unknown;
    };
    [key: string]: unknown;
  };
  [key: string]: unknown;
}

export interface DeploySettings {
  window?: {
    W?: number | string | null;
    S?: number | string | null;
    [key: string]: unknown;
  };
  mc?: {
    M?: number | string | null;
    M_confirm?: number | string | null;
    [key: string]: unknown;
  };
  [key: string]: unknown;
}

export interface SettingsResponse {
  system?: SystemSettings;
  deploy?: DeploySettings;
  caregivers?: CaregiverRecord[];
  [key: string]: unknown;
}

export interface SettingsPatch {
  monitoring_enabled?: boolean;
  notify_on_every_fall?: boolean;
  notify_sms?: boolean;
  notify_phone?: boolean;
  active_dataset_code?: string;
  active_model_code?: string;
  active_op_code?: string;
  mc_enabled?: boolean;
  store_anonymized_data?: boolean;
  store_event_clips?: boolean;
  anonymize_skeleton_data?: boolean;
  fall_threshold?: number;
  tau_low?: number;
  alert_cooldown_sec?: number;
  [key: string]: unknown;
}
