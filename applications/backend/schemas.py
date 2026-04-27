from __future__ import annotations

from typing import Any, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class SettingsUpdatePayload(BaseModel):
    """Settings update payload sent by the frontend."""

    model_config = ConfigDict(extra="ignore")

    monitoring_enabled: Optional[bool] = None
    api_online: Optional[bool] = None
    alert_cooldown_sec: Optional[int] = None
    notify_on_every_fall: Optional[bool] = None

    fall_threshold: Optional[float] = Field(
        default=None,
        description="Probability threshold for fall decision (usually 0.0-1.0).",
    )

    store_event_clips: Optional[bool] = None
    anonymize_skeleton_data: Optional[bool] = None
    store_anonymized_data: Optional[bool] = None

    active_model_code: Optional[str] = Field(default=None, description="TCN | GCN | HYBRID")
    active_operating_point: Optional[int] = Field(default=None, description="operating_points.id")
    active_dataset_code: Optional[str] = Field(default=None, description="le2i | caucafall")
    active_op_code: Optional[str] = Field(default=None, description="OP-1 | OP-2 | OP-3")

    mc_enabled: Optional[bool] = Field(default=None, description="Enable the live uncertainty gate")
    mc_M: Optional[int] = Field(default=None, description="MC samples for boundary-window live inference")
    mc_M_confirm: Optional[int] = Field(default=None, description="Reserved MC sample count for confirm-stage logic")

    risk_profile: Optional[str] = None
    notify_sms: Optional[bool] = None
    notify_phone: Optional[bool] = None


class SkeletonClipPayload(BaseModel):
    """Skeleton-only event clip payload."""

    model_config = ConfigDict(extra="ignore")

    resident_id: int = 1
    dataset_code: Optional[str] = None
    mode: Optional[str] = None
    op_code: Optional[str] = None
    use_mc: Optional[bool] = None
    mc_M: Optional[int] = None
    mc_sigma_tol: Optional[float] = None
    mc_se_tol: Optional[float] = None
    pre_s: Optional[float] = None
    post_s: Optional[float] = None

    t_ms: List[float]
    xy: Optional[List[List[List[float]]]] = None
    conf: Optional[List[List[float]]] = None
    xy_flat: Optional[List[float]] = None
    conf_flat: Optional[List[float]] = None
    raw_joints: Optional[int] = None


class MonitorPredictPayload(BaseModel):
    """Live monitor inference payload."""

    model_config = ConfigDict(extra="ignore")

    session_id: Optional[str] = None
    input_source: Optional[str] = None
    mode: Optional[str] = None
    dataset_code: Optional[str] = None
    dataset: Optional[str] = None
    op_code: Optional[str] = None
    op: Optional[str] = None

    model_tcn: Optional[str] = None
    model_gcn: Optional[str] = None
    model_id: Optional[str] = None

    resident_id: Optional[int] = None
    location: Optional[str] = None
    use_mc: Optional[bool] = None
    mc_M: Optional[int] = None
    persist: Optional[bool] = None
    compact_response: Optional[bool] = None

    target_T: Optional[int] = None
    target_fps: Optional[float] = None
    fps: Optional[float] = None
    capture_fps: Optional[float] = None
    timestamp_ms: Optional[float] = None
    window_end_t_ms: Optional[float] = None
    window_seq: Optional[int] = None

    raw_t_ms: Any = None
    raw_xy: Any = None
    raw_conf: Any = None
    raw_xy_q: Any = None
    raw_conf_q: Any = None
    raw_shape: Any = None
    xy: Any = None
    conf: Any = None


class CaregiverUpsertPayload(BaseModel):
    """Create or update a caregiver record."""

    model_config = ConfigDict(extra="ignore")

    id: Optional[int] = None
    resident_id: int = 1
    name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    telegram_chat_id: Optional[str] = None
