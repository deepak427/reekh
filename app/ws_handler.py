"""WebSocket handler for the ESP32-CAM connection.

Protocol (one persistent connection per camera):

  ESP32 → backend (text frame):
    {"type":"frame","seq":42,"pan":0.0,"tilt":0.0,"w":640,"h":480}

  ESP32 → backend (binary frame, immediately after text):
    [raw JPEG bytes — no encoding, no wrapping]

  ESP32 → backend (text frame, status):
    {"type":"status","state":"ACTIVE","message":"...","camera_id":"...",
     "uptime_ms":1234,"wifi_rssi":-65,"free_heap":245000}

  backend → ESP32 (text frame, after inference completes):
    {"pan":15.2,"tilt":-3.1,"skip":false,"action":"BARK_SOUND","phase":"tracking"}

The background inference worker (_background_frame_processor in main.py) puts
results into a per-connection asyncio.Queue stored in DeviceStore. This handler
reads from that queue and pushes results back to the ESP32.
"""
from __future__ import annotations

import asyncio
import base64
import datetime
import json
import logging
import time
from typing import Optional

import cv2
import numpy as np
from fastapi import WebSocket, WebSocketDisconnect

import app.logging_service as svclog
from app.config import MAX_FRAME_BYTES
from app.state import FrameData, store

logger = logging.getLogger(__name__)

# ── SSE broadcast hook (set by main.py at startup) ───────────────────────────
# We call this to push frame thumbnails to the /monitor dashboard,
# same as the HTTP POST /frame endpoint did before.
_sse_broadcast_fn = None

def set_sse_broadcast(fn):
    global _sse_broadcast_fn
    _sse_broadcast_fn = fn


def _broadcast_frame_to_sse(raw: bytes, seq: int, camera_id: str) -> None:
    """Decode JPEG, make thumbnail, push to SSE monitor dashboard."""
    if _sse_broadcast_fn is None:
        return
    try:
        arr = np.frombuffer(raw, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            return
        h, w = img.shape[:2]
        scale = 160.0 / w
        small = cv2.resize(img, (160, int(h * scale)))
        _, enc_small = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 60])
        thumb_b64 = base64.b64encode(enc_small.tobytes()).decode()
        _, enc_full = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
        full_b64 = base64.b64encode(enc_full.tobytes()).decode()
        _sse_broadcast_fn("frame", {
            "ts":        datetime.datetime.now().isoformat(timespec="milliseconds"),
            "seq":       seq,
            "camera_id": camera_id,
            "bytes":     len(raw),
            "thumb":     thumb_b64,
            "full":      full_b64,
        })
    except Exception:
        pass  # never let dashboard break the camera pipeline


async def handle_camera_ws(websocket: WebSocket, camera_id: str) -> None:
    """Main handler for one ESP32 WebSocket connection.

    Runs for the lifetime of the connection. Accepts frames from the ESP32,
    feeds them into the existing FrameSlot (background worker unchanged),
    and pushes inference results back to the ESP32 as they arrive.
    """
    await websocket.accept()
    logger.info("WS_CONNECT camera=%s", camera_id)

    # Register this connection's result queue in the store.
    # Background worker puts results here after inference.
    result_queue: asyncio.Queue = asyncio.Queue(maxsize=10)
    store.set_ws_result_queue(result_queue)

    # Pending metadata from the last text frame (pan, tilt, seq, w, h)
    pending_meta: Optional[dict] = None

    try:
        # Run two coroutines concurrently inside this connection:
        #   receive_loop — reads frames from ESP32, feeds FrameSlot
        #   send_loop    — reads inference results, pushes back to ESP32
        await asyncio.gather(
            _receive_loop(websocket, camera_id, result_queue, pending_meta),
            _send_loop(websocket, camera_id, result_queue),
        )
    except WebSocketDisconnect:
        logger.info("WS_DISCONNECT camera=%s", camera_id)
    except Exception as exc:
        logger.exception("WS_ERROR camera=%s: %s", camera_id, exc)
    finally:
        store.set_ws_result_queue(None)
        logger.info("WS_CLEANUP camera=%s", camera_id)


async def _receive_loop(
    websocket: WebSocket,
    camera_id: str,
    result_queue: asyncio.Queue,
    pending_meta,
) -> None:
    """Receive messages from the ESP32 and feed the FrameSlot."""
    pending_meta_holder: dict = {}  # mutable container to share state

    while True:
        message = await websocket.receive()

        if "text" in message:
            # Text frame — either metadata prefix for a frame, or status message
            raw_text = message["text"]
            try:
                data = json.loads(raw_text)
            except json.JSONDecodeError:
                logger.warning("WS bad JSON from %s: %s", camera_id, raw_text[:80])
                continue

            msg_type = data.get("type", "frame")

            if msg_type == "reset":
                store.reset()
                logger.info("STATE_RESET camera=%s (explicit reset command)", camera_id)
                if _sse_broadcast_fn:
                    _sse_broadcast_fn("status", {
                        "ts":        datetime.datetime.now().isoformat(timespec="milliseconds"),
                        "camera_id": camera_id,
                        "state":     "IDLE",
                        "message":   "Camera state reset to (0,0)",
                        "uptime_ms": 0,
                        "wifi_rssi": 0,
                        "free_heap": 0,
                    })

            elif msg_type == "status":
                if data.get("state") in ("IDLE", "BOOT"):
                    store.reset()
                    logger.info("STATE_RESET camera=%s (state=%s)", camera_id, data.get("state"))

                # ESP32 status report — broadcast to SSE dashboard
                logger.info(
                    "ESP_STATUS camera=%s state=%s msg=%s rssi=%d heap=%d",
                    camera_id,
                    data.get("state", "unknown"),
                    data.get("message", ""),
                    data.get("wifi_rssi", 0),
                    data.get("free_heap", 0),
                )
                if _sse_broadcast_fn:
                    _sse_broadcast_fn("status", {
                        "ts":        datetime.datetime.now().isoformat(timespec="milliseconds"),
                        "camera_id": camera_id,
                        "state":     data.get("state", "unknown"),
                        "message":   data.get("message", ""),
                        "uptime_ms": data.get("uptime_ms", 0),
                        "wifi_rssi": data.get("wifi_rssi", 0),
                        "free_heap": data.get("free_heap", 0),
                    })

            elif msg_type == "frame":
                # Metadata for the JPEG binary frame that follows
                pending_meta_holder["meta"] = {
                    "seq":    data.get("seq", 0),
                    "pan":    float(data.get("pan", 0.0)),
                    "tilt":   float(data.get("tilt", 0.0)),
                    "width":  int(data.get("w", 640)),
                    "height": int(data.get("h", 480)),
                }

        elif "bytes" in message:
            # Binary frame — raw JPEG bytes
            raw: bytes = message["bytes"]

            if len(raw) > MAX_FRAME_BYTES:
                logger.warning("WS frame too large from %s: %d bytes", camera_id, len(raw))
                pending_meta_holder.clear()
                continue

            # Use metadata from preceding text frame, or defaults
            meta = pending_meta_holder.pop("meta", {
                "seq": 0, "pan": 0.0, "tilt": 0.0, "width": 640, "height": 480,
            })

            seq = meta["seq"]

            # Feed into the existing FrameSlot — background worker unchanged
            dropped = store.slot.put(FrameData(
                raw_bytes=raw,
                pan_angle=meta["pan"],
                tilt_angle=meta["tilt"],
                image_width=meta["width"],
                image_height=meta["height"],
                seq=seq,
                received_ts=time.time(),
            ))

            # Push thumbnail to /monitor dashboard
            _broadcast_frame_to_sse(raw, seq, camera_id)

            svclog.log_frame_received(
                camera_id=camera_id,
                seq=seq,
                frame_bytes=len(raw),
                pan_angle=meta["pan"],
                tilt_angle=meta["tilt"],
                image_width=meta["width"],
                image_height=meta["height"],
            )
            logger.info(
                "WS_FRAME camera=%s seq=%d bytes=%d dropped_total=%d",
                camera_id, seq, len(raw), dropped,
            )

            if dropped > 0 and dropped % 50 == 0:
                svclog.log_frame_drop(camera_id=camera_id, dropped_count=dropped)

        elif message.get("type") == "websocket.disconnect":
            break


async def _send_loop(
    websocket: WebSocket,
    camera_id: str,
    result_queue: asyncio.Queue,
) -> None:
    """Read inference results from the queue and push them to the ESP32."""
    while True:
        try:
            # Wait up to 1 second for a result
            result_json: str = await asyncio.wait_for(
                result_queue.get(), timeout=1.0
            )
            await websocket.send_text(result_json)
            logger.debug("WS_SEND camera=%s payload=%s", camera_id, result_json[:80])
        except asyncio.TimeoutError:
            # No result in 1s — send a keepalive ping (WebSocket ping, not our ping)
            # This also catches connection drops quickly
            try:
                await websocket.send_text('{"type":"ping"}')
            except Exception:
                break
        except Exception:
            break
