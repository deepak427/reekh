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


# ── Per-class primary action map ──────────────────────────────────────────────
#
# Logic:
#   Apex predators  → human shout + flash light + alert owner (most aggressive)
#   Large dangerous → dog bark + flash light
#   Medium wild     → dog bark or raptor screech (no flash, no alert)
#   Deer            → wolf howl (their hardwired flight trigger)
#   Small mustelids → raptor screech (birds of prey are their natural predator)
#   Domestic/pet    → friendly voice (no need to scare, just redirect)
#
# FLASH_LIGHT and ALERT_OWNER are sent as secondary actions via the
# `secondary_actions` field in the WS payload — see _build_actions() below.

# Primary sound action per species
_PRIMARY: dict[str, Action] = {
    # Apex predators
    "AmurTiger":   Action.sound_human_shout,
    "Leopard":     Action.sound_human_shout,

    # Large dangerous
    "BlackBear":   Action.sound_dog_bark,
    "WildBoar":    Action.sound_dog_bark,

    # Medium wild cats / canids
    "LeopardCat":  Action.sound_dog_bark,
    "RedFox":      Action.sound_dog_bark,
    "RaccoonDog":  Action.sound_dog_bark,

    # Small mustelids + hare
    "Badger":      Action.sound_raptor_screech,
    "Sable":       Action.sound_raptor_screech,
    "Weasel":      Action.sound_raptor_screech,
    "Y-T-Marten":  Action.sound_raptor_screech,
    "Hare":        Action.sound_raptor_screech,

    # Deer — wolf howl is their hardwired flight trigger
    "SikaDeer":    Action.sound_wolf_howl,
    "RoeDeer":     Action.sound_wolf_howl,
    "MuskDeer":    Action.sound_wolf_howl,

    # Domestic / pet — calm redirect, no scare needed
    "Dog":         Action.sound_friendly_voice,
    "Cow":         Action.sound_friendly_voice,
}

# Which species also get the strobe light fired alongside the sound
_ALSO_FLASH: frozenset[str] = frozenset({
    "AmurTiger", "Leopard",
    "BlackBear", "WildBoar",
})

# Which species also silently alert the owner
_ALSO_ALERT: frozenset[str] = frozenset({
    "AmurTiger", "Leopard", "BlackBear",
})


def get_actions_for_species(class_name: str) -> tuple[Action, list[str]]:
    """Return (primary_action, list_of_secondary_action_strings) for a species.

    Secondary actions are returned as plain strings so the WS payload stays
    forward-compatible without firmware needing to parse a nested object.
    """
    primary = _PRIMARY.get(class_name, Action.sound_dog_bark)

    secondary: list[str] = []
    if class_name in _ALSO_FLASH:
        secondary.append(Action.flash_light.value)
    if class_name in _ALSO_ALERT:
        secondary.append(Action.alert_owner.value)

    return primary, secondary


async def decide_action(
    *,
    detection:               dict[str, Any],
    camera_id:               str,
    pan_angle:               float,
    tilt_angle:              float,
    active_action:           Action,
    action_frames_remaining: int,
) -> Action:
    """Return the primary Action to take for the current detection.

    Secondary actions (flash, alert) are resolved separately via
    get_actions_for_species() and injected into the WS payload by the caller.

    Falls back to the local species map instantly if JEV is unavailable.
    """
    class_name = detection.get("class_name", "Unknown")
    confidence = float(detection.get("score", 0.0))

    # Debounce: if an action is actively playing out, don't restart it
    if action_frames_remaining > 0 and active_action != Action.none:
        svclog.log_jev(camera_id=camera_id, outcome="skipped")
        return active_action

    # ── Try JEV / OpenRouter if configured ───────────────────────────────────
    if not JEV_API_KEY or JEV_API_KEY == "change-me-in-production":
        svclog.log_jev(camera_id=camera_id, outcome="local_fallback_no_key")
        return _local_fallback(detection)

    payload = {
        "model": JEV_MODEL,
        "state": (
            f"Wildlife field alert: animal detected. Species='{class_name}', "
            f"confidence={confidence:.2f}. Camera orientation pan={pan_angle:.1f} deg, "
            f"tilt={tilt_angle:.1f} deg."
        ),
        "questions": {
            "action": {
                "type": "choice",
                "instructions": (
                    "Which primary deterrent sound should be triggered "
                    "to drive this animal away from the field?"
                ),
                "criteria": {
                    "SOUND_HUMAN_SHOUT":    "Human shouting — apex predators (tiger, leopard) that fear human presence",
                    "SOUND_DOG_BARK":       "Dog pack barking — bear, boar, fox, raccoon dog, leopard cat",
                    "SOUND_WOLF_HOWL":      "Wolf howl — deer (sika, roe, musk) hardwired flight response",
                    "SOUND_RAPTOR_SCREECH": "Raptor/hawk screech — small mustelids (sable, weasel, marten, badger) and hare",
                    "SOUND_FRIENDLY_VOICE": "Calm human voice — domestic animals (dog, cow) redirect without scaring",
                    "none":                 "Harmless observation, no deterrent needed",
                },
            }
        },
    }

    headers = {
        "Authorization": f"Bearer {JEV_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://wildlife-surveillance.local",
        "X-OpenRouter-Title": "Wildlife Field Surveillance",
    }

    t0 = time.perf_counter()
    try:
        async with httpx.AsyncClient(timeout=JEV_TIMEOUT_S) as client:
            resp = await client.post(JEV_API_URL, json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()
            action_choice = data.get("answers", {}).get("action", {}).get("choice", "").upper()

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
    """Instant deterministic fallback — uses the species map directly."""
    class_name = detection.get("class_name", "")
    return _PRIMARY.get(class_name, Action.sound_dog_bark)
