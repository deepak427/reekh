"""In-process state for the single connected ESP32-CAM.

No database, no Redis.  If the process restarts the camera re-initialises
from its first frame, which is acceptable for a single-device deployment.
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Optional

from app.schemas import Action, ActionPollResponse, Phase, ServoMovement, ServoPollResponse


# ── Frame data container ──────────────────────────────────────────────────────

@dataclass
class FrameData:
    raw_bytes:    bytes
    pan_angle:    float
    tilt_angle:   float
    image_width:  int
    image_height: int
    seq:          int
    received_ts:  float   # time.time() at arrival


# ── Latest-frame-wins slot ────────────────────────────────────────────────────

class FrameSlot:
    """Holds at most one pending frame.

    If a new frame arrives before the previous one is processed the old frame
    is silently overwritten.  Stale frames cause laggy tracking — always
    inferring the freshest frame is the correct trade-off.

    asyncio.Event is created lazily on first use (inside the running event
    loop) to avoid binding to the wrong loop at module import time.
    """

    def __init__(self) -> None:
        self._frame:        Optional[FrameData] = None
        self.dropped_count: int = 0
        self._event:        Optional[object] = None  # asyncio.Event, lazy

    def _get_event(self):
        if self._event is None:
            import asyncio
            self._event = asyncio.Event()
        return self._event

    def put(self, frame: FrameData) -> int:
        """Store a new frame.  Returns cumulative drop count."""
        if self._frame is not None:
            self.dropped_count += 1
        self._frame = frame
        self._get_event().set()
        return self.dropped_count

    def pop(self) -> Optional[FrameData]:
        """Consume and return the stored frame (None if empty)."""
        self._get_event().clear()
        frame, self._frame = self._frame, None
        return frame


# ── Per-camera tracking state ─────────────────────────────────────────────────

@dataclass
class CameraState:
    # Phase / detection hysteresis
    phase:                    Phase  = Phase.sweeping
    frames_without_detection: int    = 0

    # Action debounce
    active_action:           Action = Action.none
    action_frames_remaining: int    = 0

    # Sweep
    sweep_step: int = 0

    # Velocity compensation state
    prev_bbox_center: Optional[list[float]] = None
    prev_frame_ts:    float = field(default_factory=time.monotonic)

    # Last absolute servo position commanded
    last_pan_absolute:  float = 0.0
    last_tilt_absolute: float = 0.0

    def to_dict(self) -> dict:
        return {
            "phase":                    self.phase.value,
            "frames_without_detection": self.frames_without_detection,
            "active_action":            self.active_action.value,
            "action_frames_remaining":  self.action_frames_remaining,
            "sweep_step":               self.sweep_step,
            "prev_bbox_center":         self.prev_bbox_center,
            "last_pan_absolute":        round(self.last_pan_absolute, 2),
            "last_tilt_absolute":       round(self.last_tilt_absolute, 2),
        }


# ── Single-device store ───────────────────────────────────────────────────────

class DeviceStore:
    """All runtime state for the single connected ESP32-CAM."""

    def __init__(self) -> None:
        self._state:          CameraState               = CameraState()
        self._slot:           FrameSlot                 = FrameSlot()
        self._latest_servo:   Optional[ServoPollResponse]  = None
        self._latest_action:  Optional[ActionPollResponse] = None
        # WebSocket result queue — set when ESP32 connects, cleared on disconnect.
        # Background worker puts JSON strings here; ws_handler sends them to ESP32.
        self._ws_result_queue: Optional[asyncio.Queue] = None

    # ── Accessors ─────────────────────────────────────────────────────────────

    @property
    def camera_state(self) -> CameraState:
        return self._state

    @property
    def slot(self) -> FrameSlot:
        return self._slot

    def reset(self) -> None:
        """Reset all tracking, servo coordinates, and action states back to default (0, 0)."""
        self._state = CameraState()
        self._latest_servo = None
        self._latest_action = None

    # ── State mutations ───────────────────────────────────────────────────────

    def update_detected(
        self,
        *,
        new_action:    Action,
        debounce:      int,
        bbox_center:   list[float],
        frame_ts:      float,
        pan_absolute:  float,
        tilt_absolute: float,
    ) -> CameraState:
        s = self._state
        s.phase                    = Phase.tracking
        s.frames_without_detection = 0

        if new_action != Action.none:
            s.active_action           = new_action
            s.action_frames_remaining = debounce
        elif s.action_frames_remaining > 0:
            s.action_frames_remaining -= 1
        else:
            s.active_action = Action.none

        s.prev_bbox_center   = bbox_center
        s.prev_frame_ts      = frame_ts
        s.last_pan_absolute  = pan_absolute
        s.last_tilt_absolute = tilt_absolute
        return s

    def update_not_detected(self, *, lost_threshold: int) -> CameraState:
        s = self._state
        s.frames_without_detection += 1

        if s.action_frames_remaining > 0:
            s.action_frames_remaining -= 1
        else:
            s.active_action = Action.none

        if s.frames_without_detection >= lost_threshold:
            s.phase = Phase.sweeping

        s.prev_bbox_center = None   # next detection is a fresh start
        return s

    def advance_sweep(self, next_step: int) -> None:
        self._state.sweep_step = next_step

    # ── Poll response store ───────────────────────────────────────────────────

    def get_latest_servo(self, camera_id: str) -> ServoPollResponse:
        if self._latest_servo is None:
            s = self._state
            self._latest_servo = ServoPollResponse(
                camera_id=camera_id,
                phase=s.phase,
                movement=ServoMovement(
                    pan_absolute=s.last_pan_absolute,
                    tilt_absolute=s.last_tilt_absolute,
                    skip=True,
                ),
                camera_state=s.to_dict(),
            )
        return self._latest_servo

    def set_latest_servo(self, resp: ServoPollResponse) -> None:
        self._latest_servo = resp

    def get_latest_action(self, camera_id: str) -> ActionPollResponse:
        if self._latest_action is None:
            s = self._state
            self._latest_action = ActionPollResponse(
                camera_id=camera_id,
                action=s.active_action,
                action_frames_remaining=s.action_frames_remaining,
                camera_state=s.to_dict(),
            )
        return self._latest_action

    def set_latest_action(self, resp: ActionPollResponse) -> None:
        self._latest_action = resp

    # ── WebSocket result queue ────────────────────────────────────────────────

    def set_ws_result_queue(self, q: Optional[asyncio.Queue]) -> None:
        self._ws_result_queue = q

    def push_ws_result(self, result_json: str) -> None:
        """Called by background worker after inference. Non-blocking."""
        if self._ws_result_queue is None:
            return
        try:
            self._ws_result_queue.put_nowait(result_json)
        except asyncio.QueueFull:
            pass  # drop if ESP32 isn't reading fast enough


# ── Global singleton ──────────────────────────────────────────────────────────

store = DeviceStore()
