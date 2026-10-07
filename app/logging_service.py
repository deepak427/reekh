"""Structured file-based logging.

Every significant event is written as a single JSON line (JSONL) to
LOG_FILE_PATH.  The log file is the single source of truth for latency
metrics and system events.

Log levels / event types
------------------------
  inference   — one entry per processed frame (latency, detection result)
  action      — deterrent action triggered or continued
  jev         — JEV call outcome (success / fallback / timeout / skipped)
  frame_drop  — frame overwritten in slot before inference ran (every 50)
  phase       — phase transition (sweeping ↔ tracking)
  error       — unexpected exceptions in the worker
  startup     — service started / model loaded
  shutdown    — service stopping
"""
from __future__ import annotations

import json
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.config import LOG_FILE_PATH, LOG_MAX_LINES

# ── Setup ─────────────────────────────────────────────────────────────────────

_log_path = Path(LOG_FILE_PATH)
_log_path.parent.mkdir(parents=True, exist_ok=True)
_lock = threading.Lock()


# ── Write helpers ─────────────────────────────────────────────────────────────

def log_inference(
    *,
    camera_id:     str,
    seq:           int,
    inference_ms:  float,
    total_ms:      float,
    detected:      bool,
    class_name:    str | None = None,
    confidence:    float | None = None,
    phase:         str,
    action:        str,
    dropped_total: int = 0,
) -> None:
    _write("inference", {
        "camera_id":     camera_id,
        "seq":           seq,
        "inference_ms":  round(inference_ms, 2),
        "total_ms":      round(total_ms, 2),
        "detected":      detected,
        "class_name":    class_name,
        "confidence":    round(confidence, 4) if confidence is not None else None,
        "phase":         phase,
        "action":        action,
        "dropped_total": dropped_total,
    })


def log_action(*, camera_id: str, action: str, class_name: str, trigger: str) -> None:
    """trigger: 'new' | 'debounce' | 'fallback'"""
    _write("action", {
        "camera_id":  camera_id,
        "action":     action,
        "class_name": class_name,
        "trigger":    trigger,
    })


def log_jev(*, camera_id: str, outcome: str, latency_ms: float | None = None) -> None:
    """outcome: 'success' | 'fallback' | 'timeout' | 'skipped'"""
    _write("jev", {
        "camera_id":  camera_id,
        "outcome":    outcome,
        "latency_ms": round(latency_ms, 2) if latency_ms is not None else None,
    })


def log_frame_drop(*, camera_id: str, dropped_count: int) -> None:
    _write("frame_drop", {"camera_id": camera_id, "dropped_count": dropped_count})


def log_phase(*, camera_id: str, from_phase: str, to_phase: str, reason: str) -> None:
    _write("phase", {
        "camera_id":  camera_id,
        "from_phase": from_phase,
        "to_phase":   to_phase,
        "reason":     reason,
    })


def log_error(*, camera_id: str | None = None, message: str, exc: str | None = None) -> None:
    _write("error", {"camera_id": camera_id, "message": message, "exc": exc})


def log_startup(*, model_path: str, engine: str, opencv_version: str) -> None:
    _write("startup", {
        "model_path":     model_path,
        "engine":         engine,
        "opencv_version": opencv_version,
    })


def log_shutdown() -> None:
    _write("shutdown", {})


# ── Query API ─────────────────────────────────────────────────────────────────

def query_logs(
    *,
    level:  str | None = None,
    since:  float | None = None,
    until:  float | None = None,
    search: str | None = None,
    limit:  int = 200,
    offset: int = 0,
    order:  str = "desc",
) -> tuple[int, list[dict]]:
    """Read and filter log entries from the JSONL file.

    Args:
        level:   Comma-separated event types, e.g. "inference,error".
                 None = all levels.
        since:   Unix timestamp — include events at or after.
        until:   Unix timestamp — include events at or before.
        search:  Substring match across the raw JSON line.
        limit:   Maximum entries to return (hard cap: 2000).
        offset:  Skip first N matching entries (pagination).
        order:   "asc" = oldest first, "desc" = newest first.

    Returns:
        (total_matched, entries)
    """
    limit = min(limit, 2000)

    levels: set[str] | None = None
    if level:
        levels = {lv.strip().lower() for lv in level.split(",") if lv.strip()}

    if not _log_path.exists():
        return 0, []

    with _lock:
        try:
            raw_lines = _log_path.read_text(encoding="utf-8").splitlines()
        except OSError:
            return 0, []

    matched: list[dict] = []
    for raw in raw_lines:
        raw = raw.strip()
        if not raw:
            continue
        if search and search.lower() not in raw.lower():
            continue
        try:
            entry = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if levels and entry.get("level", "").lower() not in levels:
            continue
        ts = entry.get("ts", 0.0)
        if since is not None and ts < since:
            continue
        if until is not None and ts > until:
            continue
        matched.append(entry)

    if order == "desc":
        matched.reverse()

    return len(matched), matched[offset: offset + limit]


def log_stats() -> dict:
    """Aggregate statistics over the entire log file."""
    if not _log_path.exists():
        return {"total_lines": 0, "by_level": {}, "file_size_bytes": 0}

    with _lock:
        try:
            raw_lines = _log_path.read_text(encoding="utf-8").splitlines()
        except OSError:
            return {"total_lines": 0, "by_level": {}, "file_size_bytes": 0}

    by_level: dict[str, int] = {}
    inf_ms_sum = 0.0
    tot_ms_sum = 0.0
    inf_count  = 0

    for raw in raw_lines:
        raw = raw.strip()
        if not raw:
            continue
        try:
            entry = json.loads(raw)
        except json.JSONDecodeError:
            continue
        lvl = entry.get("level", "unknown")
        by_level[lvl] = by_level.get(lvl, 0) + 1
        if lvl == "inference":
            d = entry.get("data", {})
            if "inference_ms" in d:
                inf_ms_sum += d["inference_ms"]
                tot_ms_sum += d.get("total_ms", 0)
                inf_count  += 1

    return {
        "total_lines":             len(raw_lines),
        "by_level":                by_level,
        "file_size_bytes":         _log_path.stat().st_size,
        "avg_inference_ms":        round(inf_ms_sum / inf_count, 2) if inf_count else None,
        "avg_total_ms":            round(tot_ms_sum / inf_count, 2) if inf_count else None,
        "inference_frames_logged": inf_count,
    }


# ── Internal ──────────────────────────────────────────────────────────────────

def _write(level: str, data: dict[str, Any]) -> None:
    now = time.time()
    entry = {
        "ts":    now,
        "iso":   datetime.fromtimestamp(now, tz=timezone.utc).isoformat(),
        "level": level,
        "data":  data,
    }
    line = json.dumps(entry, separators=(",", ":")) + "\n"

    with _lock:
        try:
            with _log_path.open("a", encoding="utf-8") as fh:
                fh.write(line)
            _maybe_trim()
        except OSError as exc:
            print(f"[logging_service] write error: {exc}")


def _maybe_trim() -> None:
    """Drop oldest 10% of lines once file exceeds LOG_MAX_LINES."""
    try:
        if _log_path.stat().st_size < LOG_MAX_LINES * 100:
            return
        lines = _log_path.read_text(encoding="utf-8").splitlines(keepends=True)
        if len(lines) <= LOG_MAX_LINES:
            return
        keep_from = len(lines) - int(LOG_MAX_LINES * 0.9)
        _log_path.write_text("".join(lines[keep_from:]), encoding="utf-8")
    except OSError:
        pass
