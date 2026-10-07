"""JEV (TypeSafe AI) client — decides the deterrent action for a detection.

JEV is stateless: it receives the current detection + camera context and
returns one structured action.  All history tracking stays in state.py.

If JEV is not configured (JEV_API_URL is empty) or the call fails / times
out, the function falls back to deterministic local logic and never blocks
the inference worker.
"""
from __future__ import annotations

import time

import httpx

from app.config import JEV_API_KEY, JEV_API_URL, JEV_TIMEOUT_S
import app.logging_service as svclog
from app.schemas import Action


async def decide_action(
    *,
    detection:               dict,
    camera_id:               str,
    pan_angle:               float,
    tilt_angle:              float,
    active_action:           Action,
    action_frames_remaining: int,
) -> Action:
    """Return the Action to take for the current detection.

    Args:
        detection:               Best-scoring detection dict from inference.
        camera_id:               Identifier used for logging.
        pan_angle / tilt_angle:  Current servo position (sent to JEV for context).
        active_action:           Action currently in its debounce window.
        action_frames_remaining: Frames left on the debounce countdown.
    """
    # Debounce: if an action is already playing out, don't restart it.
    if action_frames_remaining > 0 and active_action != Action.none:
        svclog.log_jev(camera_id=camera_id, outcome="skipped")
        return active_action

    if not JEV_API_URL:
        action = _local_fallback(detection)
        svclog.log_jev(camera_id=camera_id, outcome="fallback")
        return action

    payload = {
        "detection": {
            "class_name":  detection["class_name"],
            "confidence":  round(detection["score"], 3),
            "bbox_center": detection["bbox_center"],
        },
        "camera_id":  camera_id,
        "pan_angle":  pan_angle,
        "tilt_angle": tilt_angle,
    }

    t0 = time.perf_counter()
    try:
        async with httpx.AsyncClient(timeout=JEV_TIMEOUT_S) as client:
            resp = await client.post(
                JEV_API_URL,
                json=payload,
                headers={"Authorization": f"Bearer {JEV_API_KEY}"},
            )
            resp.raise_for_status()
            data = resp.json()
            action_str = data.get("action", "none").upper()
            action = (
                Action(action_str)
                if action_str in Action._value2member_map_
                else Action.bark_sound
            )
            svclog.log_jev(
                camera_id=camera_id,
                outcome="success",
                latency_ms=(time.perf_counter() - t0) * 1000.0,
            )
            return action

    except httpx.TimeoutException:
        svclog.log_jev(
            camera_id=camera_id,
            outcome="timeout",
            latency_ms=(time.perf_counter() - t0) * 1000.0,
        )
        return _local_fallback(detection)

    except Exception:
        svclog.log_jev(
            camera_id=camera_id,
            outcome="fallback",
            latency_ms=(time.perf_counter() - t0) * 1000.0,
        )
        return _local_fallback(detection)


def _local_fallback(detection: dict) -> Action:
    """Deterministic fallback used when JEV is unavailable."""
    dangerous = {"AmurTiger", "Leopard", "LeopardCat", "BlackBear", "WildBoar"}
    if detection["class_name"] in dangerous:
        return Action.flash_light
    return Action.bark_sound
