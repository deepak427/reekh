"""Pydantic schemas for API requests and responses."""
from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


# ── Enums ─────────────────────────────────────────────────────────────────────

class Phase(str, Enum):
    idle     = "idle"
    sweeping = "sweeping"
    tracking = "tracking"


class Action(str, Enum):
    none           = "none"
    bark_sound     = "BARK_SOUND"
    flash_light    = "FLASH_LIGHT"
    alert_owner    = "ALERT_OWNER"
    continue_sweep = "CONTINUE_SWEEP"


# ── Detection ─────────────────────────────────────────────────────────────────

class Detection(BaseModel):
    class_id:    int
    class_name:  str
    confidence:  float
    bbox:        list[float]    # [x1, y1, x2, y2] in original image pixels
    bbox_center: list[float]    # [cx, cy]


# ── Servo movement ────────────────────────────────────────────────────────────

class ServoMovement(BaseModel):
    """Absolute servo target.

    Always absolute angles (never deltas) so a mid-move update simply
    redirects without accumulating error.
    skip=True means the animal is within the deadzone — hold position.
    """
    pan_absolute:  float = Field(..., description="Target pan in degrees (absolute)")
    tilt_absolute: float = Field(..., description="Target tilt in degrees (absolute)")
    skip:          bool  = Field(False, description="True = animal centred, no move needed")


# ── Inference responses ───────────────────────────────────────────────────────

class InferResponse(BaseModel):
    camera_id:    str
    phase:        Phase
    action:       Action
    movement:     ServoMovement
    detection:    Optional[Detection] = None
    camera_state: Optional[dict]      = None
    inference_ms: float = Field(..., description="Inference time in milliseconds")
    total_ms:     float = Field(..., description="Total processing time in milliseconds")


class HealthResponse(BaseModel):
    status:         str
    model_loaded:   bool
    opencv_version: str
    engine:         str


# ── Push / Poll decoupled ─────────────────────────────────────────────────────

class FrameAcceptedResponse(BaseModel):
    status:    str = "accepted"
    camera_id: str
    seq:       int = 0
    dropped:   int = 0


class ServoPollResponse(BaseModel):
    camera_id:    str
    phase:        Phase
    movement:     ServoMovement
    detection:    Optional[Detection] = None
    camera_state: Optional[dict]      = None
    seq:          int   = 0
    latency_ms:   float = 0.0


class ActionPollResponse(BaseModel):
    camera_id:               str
    action:                  Action
    action_frames_remaining: int
    camera_state:            Optional[dict] = None


# ── Log query ─────────────────────────────────────────────────────────────────

class LogEntry(BaseModel):
    ts:    float
    iso:   str
    level: str
    data:  dict[str, Any]


class LogQueryResponse(BaseModel):
    total_matched: int
    returned:      int
    offset:        int
    order:         str
    entries:       list[LogEntry]


class LogStatsResponse(BaseModel):
    total_lines:             int
    by_level:                dict[str, int]
    file_size_bytes:         int
    avg_inference_ms:        Optional[float] = None
    avg_total_ms:            Optional[float] = None
    inference_frames_logged: int = 0
