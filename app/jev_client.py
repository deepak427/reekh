"""JEV (TypeSafe AI on OpenRouter) client — decides deterrent action for detections.

Uses OpenRouter Decisions API (~typesafe/jev-latest).
Enforces strict 800ms timeout with deterministic per-class local fallback.
Never blocks or crashes inference if OpenRouter is unreachable or unconfigured.
"""
from __future__ import annotations

import time
from typing import Any

import httpx

from app.config import JEV_API_KEY, JEV_API_URL, JEV_MODEL, JEV_TIMEOUT_S
import app.logging_service as svclog
from app.schemas import Action


# ── Exhaustive per-class fallback for all 17 dataset classes ─────────────────
FALLBACK_MAP: dict[str, Action] = {
    # All species trigger flash_light in V1 (spotlight wired, speaker not wired)
    "AmurTiger":   Action.flash_light,
    "Leopard":     Action.flash_light,
    "LeopardCat":  Action.flash_light,
    "BlackBear":   Action.flash_light,
    "WildBoar":    Action.flash_light,
    "Cow":         Action.flash_light,
    "SikaDeer":    Action.flash_light,
    "RoeDeer":     Action.flash_light,
    "MuskDeer":    Action.flash_light,
    "Dog":         Action.flash_light,
    "Badger":      Action.flash_light,
    "RedFox":      Action.flash_light,
    "RaccoonDog":  Action.flash_light,
    "Hare":        Action.flash_light,
    "Sable":       Action.flash_light,
    "Weasel":      Action.flash_light,
    "Y-T-Marten":  Action.flash_light,
}


async def decide_action(
    *,
    detection:               dict[str, Any],
    camera_id:               str,
    pan_angle:               float,
    tilt_angle:              float,
    active_action:           Action,
    action_frames_remaining: int,
) -> Action:
    """Return the Action to take for the current detection.

    V1 mode: Always commands FLASH_LIGHT for any detected animal.
    """
    # Debounce: if action is actively playing out, don't restart it
    if action_frames_remaining > 0 and active_action != Action.none:
        svclog.log_jev(camera_id=camera_id, outcome="skipped")
        return active_action

    svclog.log_jev(camera_id=camera_id, outcome="flash_light_v1")
    return Action.flash_light

    class_name = detection.get("class_name", "Unknown")
    confidence = float(detection.get("score", 0.0))

    # Formulate typed decision payload for OpenRouter Decisions API
    payload = {
        "model": JEV_MODEL,
        "state": (
            f"Agricultural field alert: animal detected. Species='{class_name}', "
            f"confidence={confidence:.2f}. Camera orientation pan={pan_angle:.1f} deg, "
            f"tilt={tilt_angle:.1f} deg."
        ),
        "questions": {
            "action": {
                "type": "choice",
                "instructions": "Which automated deterrent action should be triggered to protect the farm?",
                "criteria": {
                    "BARK_SOUND": "Acoustic predator barking sound for herbivores, deer, dogs, foxes, raccoons, hares, badgers",
                    "FLASH_LIGHT": "High-intensity strobe light burst for dangerous apex predators (tigers, leopards, bears, boars)",
                    "ALERT_OWNER": "Silently alert owner without field siren",
                    "none": "Harmless observation, no deterrent"
                }
            }
        }
    }

    headers = {
        "Authorization": f"Bearer {JEV_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://wildlife-surveillance.local",
        "X-OpenRouter-Title": "Wildlife Agricultural Surveillance",
    }

    t0 = time.perf_counter()
    try:
        async with httpx.AsyncClient(timeout=JEV_TIMEOUT_S) as client:
            resp = await client.post(
                JEV_API_URL,
                json=payload,
                headers=headers,
            )
            resp.raise_for_status()
            data = resp.json()
            answers = data.get("answers", {})
            action_choice = answers.get("action", {}).get("choice", "").upper()

            action = (
                Action(action_choice)
                if action_choice in Action._value2member_map_
                else _local_fallback(detection)
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


def _local_fallback(detection: dict[str, Any]) -> Action:
    """Instant deterministic fallback matching all 17 species."""
    class_name = detection.get("class_name", "")
    return FALLBACK_MAP.get(class_name, Action.bark_sound)
