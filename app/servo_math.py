"""Pure servo math: detection bbox → absolute pan/tilt target.

Four layers applied in order:
  1. Deadzone            — hold position if animal is already near-centre
  2. Velocity compensation — predict where the animal will be at command execution
  3. Absolute target     — proportional gain, per-cycle delta clamp, range clamp
  4. Angular deadband    — suppress sub-threshold oscillations

No state, no I/O, no side effects.  All inputs are passed explicitly.
"""
from __future__ import annotations

import math
from typing import Optional

from app.config import (
    COMMAND_LATENCY_MS,
    DEADZONE_PX,
    MAX_PAN_DELTA,
    MAX_TILT_DELTA,
    MAX_VELOCITY_COMP_PX,
    MIN_SERVO_DELTA_DEG,
    PAN_GAIN,
    PAN_INVERT,
    PAN_MAX,
    PAN_MIN,
    SWEEP_SEQUENCE,
    TILT_GAIN,
    TILT_MAX,
    TILT_MIN,
)
from app.schemas import ServoMovement


def tracking_movement(
    bbox_center:      list[float],
    image_width:      int,
    image_height:     int,
    last_pan_abs:     float,
    last_tilt_abs:    float,
    prev_bbox_center: Optional[list[float]] = None,
    frame_dt_ms:      float = 0.0,
) -> ServoMovement:
    """Compute absolute pan/tilt target to keep the detected animal centred.

    Returns ServoMovement with pan_absolute + tilt_absolute in degrees,
    or skip=True if the animal is already within the deadzone.
    """
    img_cx = image_width  / 2.0
    img_cy = image_height / 2.0

    # ── Layer 1: Deadzone ─────────────────────────────────────────────────────
    raw_error_x = bbox_center[0] - img_cx
    raw_error_y = bbox_center[1] - img_cy

    if math.hypot(raw_error_x, raw_error_y) <= DEADZONE_PX:
        return ServoMovement(
            pan_absolute=round(last_pan_abs, 2),
            tilt_absolute=0.0,   # V1: Tilt locked to 0.0
            skip=True,
        )

    # ── Layer 2: Velocity compensation ───────────────────────────────────────
    if prev_bbox_center is not None and frame_dt_ms > 1.0:
        vel_x  = (bbox_center[0] - prev_bbox_center[0]) / frame_dt_ms
        vel_y  = (bbox_center[1] - prev_bbox_center[1]) / frame_dt_ms
        comp_x = _clamp(vel_x * COMMAND_LATENCY_MS, -MAX_VELOCITY_COMP_PX, MAX_VELOCITY_COMP_PX)
        comp_y = _clamp(vel_y * COMMAND_LATENCY_MS, -MAX_VELOCITY_COMP_PX, MAX_VELOCITY_COMP_PX)
        predicted_cx = bbox_center[0] + comp_x
        predicted_cy = bbox_center[1] + comp_y
    else:
        predicted_cx = bbox_center[0]
        predicted_cy = bbox_center[1]

    error_x = predicted_cx - img_cx
    error_y = predicted_cy - img_cy

    # ── Layer 3: Absolute target ──────────────────────────────────────────────
    dir_multiplier = -1.0 if PAN_INVERT else 1.0
    pan_correction  = _clamp(error_x * PAN_GAIN * dir_multiplier,  -MAX_PAN_DELTA,  MAX_PAN_DELTA)
    tilt_correction = 0.0  # V1: Tilt locked to 0.0

    pan_absolute  = _clamp(last_pan_abs  + pan_correction,  PAN_MIN,  PAN_MAX)
    tilt_absolute = 0.0   # V1: Tilt locked to 0.0

    # ── Layer 4: Angular deadband ─────────────────────────────────────────────
    if abs(pan_absolute  - last_pan_abs)  < MIN_SERVO_DELTA_DEG:
        pan_absolute  = last_pan_abs
    if abs(tilt_absolute - last_tilt_abs) < MIN_SERVO_DELTA_DEG:
        tilt_absolute = last_tilt_abs

    skip = (pan_absolute == last_pan_abs and tilt_absolute == last_tilt_abs)

    return ServoMovement(
        pan_absolute=round(pan_absolute, 2),
        tilt_absolute=round(tilt_absolute, 2),
        skip=skip,
    )


def sweep_movement(sweep_step: int) -> tuple[ServoMovement, int]:
    """Return the absolute position for the next sweep waypoint.

    Returns:
        movement:  ServoMovement for this step.
        next_step: Index of the following step (wraps around).
    """
    step = sweep_step % len(SWEEP_SEQUENCE)
    pan, tilt = SWEEP_SEQUENCE[step]
    return (
        ServoMovement(
            pan_absolute=round(pan, 2),
            tilt_absolute=round(tilt, 2),
            skip=False,
        ),
        (step + 1) % len(SWEEP_SEQUENCE),
    )


def is_centred(
    bbox_center:  list[float],
    image_width:  int,
    image_height: int,
    tolerance_px: float = 30.0,
) -> bool:
    """Return True if the animal is within tolerance_px of the image centre."""
    return math.hypot(
        bbox_center[0] - image_width  / 2.0,
        bbox_center[1] - image_height / 2.0,
    ) <= tolerance_px


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))
