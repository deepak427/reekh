"""Wildlife inference service — FastAPI entry point.

Deployment: single ESP32-CAM, single uvicorn worker on AWS Graviton.

    uvicorn app.main:app --host 0.0.0.0 --port 8000

Push / Poll flow (primary):
  1. ESP32 POST /frame  → FrameSlot (latest-frame-wins) → 202 in <5ms
  2. Background worker  → inference → servo math → JEV → stores result
  3. ESP32 GET  /servo  → latest absolute pan/tilt target (<1ms)
  4. ESP32 GET  /action → latest deterrent action        (<1ms)

Synchronous /infer endpoint is kept for the browser simulator and debugging.
"""
from __future__ import annotations

import asyncio
import base64
import collections
import logging
from pathlib import Path
import sys
import time
import traceback
from contextlib import asynccontextmanager
from typing import Annotated, Optional

# Ensure inference_service root is on sys.path so 'app' package imports cleanly
_APP_DIR = Path(__file__).resolve().parent
_SVC_ROOT = _APP_DIR.parent
if str(_SVC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SVC_ROOT))

# stdout logging so every log line appears in AWS CloudWatch / console
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    stream=sys.stdout,
    force=True,
)

import cv2
import numpy as np
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Query, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

from app.config import (
    ACTION_DEBOUNCE_FRAMES,
    API_KEY,
    CONFIDENCE_THRESHOLD,
    LOST_TRACKING_FRAMES,
    MAX_FRAME_BYTES,
    MODEL_PATH,
)
from app.inference import RFDETRInference
from app.jev_client import decide_action
import app.logging_service as svclog
from app.schemas import (
    Action,
    ActionPollResponse,
    Detection,
    FrameAcceptedResponse,
    HealthResponse,
    InferResponse,
    LogQueryResponse,
    LogStatsResponse,
    Phase,
    ServoMovement,
    ServoPollResponse,
)
from app.servo_math import sweep_movement, tracking_movement
from app.state import FrameData, store

logger = logging.getLogger(__name__)

# ── Globals ───────────────────────────────────────────────────────────────────

_model: Optional[RFDETRInference] = None
_worker_task: Optional[asyncio.Task] = None

# Default camera_id used for single-device deployment
DEFAULT_CAMERA_ID = "esp32-cam-01"

# ── Frame preview ring buffer ─────────────────────────────────────────────────
# Keeps the last 20 frames received from the camera, purely in memory.
# Each entry: {"ts": float, "seq": int, "camera_id": str, "bytes": int, "b64": str}

_PREVIEW_MAX = 20
_preview_frames: collections.deque = collections.deque(maxlen=_PREVIEW_MAX)


# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(_: FastAPI):
    global _model, _worker_task

    import cv2 as _cv2
    _model = RFDETRInference(MODEL_PATH)
    svclog.log_startup(
        model_path=MODEL_PATH,
        engine=_model.engine_name,
        opencv_version=_cv2.__version__,
    )
    logger.info("Model loaded from %s (%s)", MODEL_PATH, _model.engine_name)

    _worker_task = asyncio.create_task(_background_frame_processor())

    yield

    svclog.log_shutdown()
    if _worker_task:
        _worker_task.cancel()
        try:
            await _worker_task
        except asyncio.CancelledError:
            pass


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Wildlife Inference Service",
    version="1.0.0",
    lifespan=lifespan,
)

# Open CORS for development and simulator
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Auth ──────────────────────────────────────────────────────────────────────

def verify_api_key(x_api_key: Annotated[Optional[str], Header()] = None) -> None:
    if not API_KEY or API_KEY == "change-me-in-production":
        return
    if x_api_key != API_KEY:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key")


# ── Health ────────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
def health():
    import cv2 as _cv2
    return HealthResponse(
        status="ok",
        model_loaded=_model is not None,
        opencv_version=_cv2.__version__,
        engine=_model.engine_name if _model else "not loaded",
    )


# ── State ─────────────────────────────────────────────────────────────────────

@app.get("/state")
@app.get("/state/{camera_id}")
def get_state(camera_id: str = DEFAULT_CAMERA_ID):
    """Current in-memory tracking state for the connected camera."""
    return store.camera_state.to_dict()


# ── Synchronous inference (simulator / debug) ─────────────────────────────────

@app.post("/infer", response_model=InferResponse, dependencies=[Depends(verify_api_key)])
async def infer(
    pan_angle:    Annotated[float, Form()],
    tilt_angle:   Annotated[float, Form()],
    camera_id:    Annotated[str,   Form()] = DEFAULT_CAMERA_ID,
    image_width:  Annotated[int,   Form()] = 640,
    image_height: Annotated[int,   Form()] = 480,
    image:        UploadFile = File(..., description="JPEG from ESP32-CAM"),
):
    """Synchronous inference — waits for the full result before responding.

    Used by the browser simulator and for debugging. In production the ESP32
    uses the push/poll endpoints instead.
    """
    t_start = time.perf_counter()

    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    raw = await image.read()
    if len(raw) > MAX_FRAME_BYTES:
        raise HTTPException(status_code=413, detail=f"Frame too large (max {MAX_FRAME_BYTES} bytes)")

    bgr = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        raise HTTPException(status_code=400, detail="Could not decode image")

    detections, inference_ms = _model.run(bgr, confidence=CONFIDENCE_THRESHOLD)

    state       = store.camera_state
    frame_ts    = time.monotonic()
    frame_dt_ms = (frame_ts - state.prev_frame_ts) * 1000.0

    if detections:
        best = detections[0]

        movement = tracking_movement(
            bbox_center=best["bbox_center"],
            image_width=image_width,
            image_height=image_height,
            last_pan_abs=state.last_pan_absolute,
            last_tilt_abs=state.last_tilt_absolute,
            prev_bbox_center=state.prev_bbox_center,
            frame_dt_ms=frame_dt_ms,
        )

        action = await decide_action(
            detection=best,
            camera_id=camera_id,
            pan_angle=pan_angle,
            tilt_angle=tilt_angle,
            active_action=state.active_action,
            action_frames_remaining=state.action_frames_remaining,
        )

        store.update_detected(
            new_action=action,
            debounce=ACTION_DEBOUNCE_FRAMES,
            bbox_center=best["bbox_center"],
            frame_ts=frame_ts,
            pan_absolute=movement.pan_absolute,
            tilt_absolute=movement.tilt_absolute,
        )

        detection_out = Detection(
            class_id=best["class_id"],
            class_name=best["class_name"],
            confidence=round(best["score"], 4),
            bbox=best["bbox"],
            bbox_center=best["bbox_center"],
        )
        phase = Phase.tracking

    else:
        prev_phase = store.camera_state.phase
        state = store.update_not_detected(lost_threshold=LOST_TRACKING_FRAMES)

        if state.phase != prev_phase:
            svclog.log_phase(
                camera_id=camera_id,
                from_phase=prev_phase.value,
                to_phase=state.phase.value,
                reason="lost_tracking_threshold",
            )

        if state.phase == Phase.tracking:
            movement = ServoMovement(
                pan_absolute=state.last_pan_absolute,
                tilt_absolute=state.last_tilt_absolute,
                skip=True,
            )
            action        = state.active_action
            detection_out = None
            phase         = Phase.tracking
        else:
            movement, next_step = sweep_movement(state.sweep_step)
            store.advance_sweep(next_step)
            action        = Action.continue_sweep
            detection_out = None
            phase         = Phase.sweeping

    total_ms = (time.perf_counter() - t_start) * 1000.0

    svclog.log_inference(
        camera_id=camera_id,
        seq=0,
        inference_ms=inference_ms,
        total_ms=total_ms,
        detected=detection_out is not None,
        class_name=detection_out.class_name if detection_out else None,
        confidence=detection_out.confidence if detection_out else None,
        phase=phase.value,
        action=action.value,
        dropped_total=store.slot.dropped_count,
    )

    return InferResponse(
        camera_id=camera_id,
        phase=phase,
        action=action,
        movement=movement,
        detection=detection_out,
        camera_state=store.camera_state.to_dict(),
        inference_ms=round(inference_ms, 2),
        total_ms=round(total_ms, 2),
    )


# ── Push / Poll: frame ingestion ──────────────────────────────────────────────

@app.post(
    "/frame",
    response_model=FrameAcceptedResponse,
    status_code=status.HTTP_202_ACCEPTED,
    dependencies=[Depends(verify_api_key)],
)
@app.post(
    "/frame/{camera_id}",
    response_model=FrameAcceptedResponse,
    status_code=status.HTTP_202_ACCEPTED,
    dependencies=[Depends(verify_api_key)],
)
async def push_frame(
    pan_angle:    Annotated[float, Form()],
    tilt_angle:   Annotated[float, Form()],
    camera_id:    str = DEFAULT_CAMERA_ID,
    image_width:  Annotated[int,   Form()] = 640,
    image_height: Annotated[int,   Form()] = 480,
    seq:          Annotated[int,   Form()] = 0,
    image:        UploadFile = File(..., description="JPEG frame from ESP32-CAM"),
):
    """Fire-and-forget frame push. Returns 202 Accepted in <5ms.

    The frame is stored in the latest-frame-wins slot and processed by the
    background worker. If inference is still running on a previous frame,
    that frame is dropped and replaced with this one.
    """
    raw = await image.read()
    if len(raw) > MAX_FRAME_BYTES:
        raise HTTPException(status_code=413, detail=f"Frame too large (max {MAX_FRAME_BYTES} bytes)")

    dropped = store.slot.put(FrameData(
        raw_bytes=raw,
        pan_angle=pan_angle,
        tilt_angle=tilt_angle,
        image_width=image_width,
        image_height=image_height,
        seq=seq,
        received_ts=time.time(),
    ))

    # Store in preview ring buffer (no DB, stays in memory)
    _preview_frames.append({
        "ts":        time.time(),
        "seq":       seq,
        "camera_id": camera_id,
        "bytes":     len(raw),
        "b64":       base64.b64encode(raw).decode(),
    })

    svclog.log_frame_received(
        camera_id=camera_id,
        seq=seq,
        frame_bytes=len(raw),
        pan_angle=pan_angle,
        tilt_angle=tilt_angle,
        image_width=image_width,
        image_height=image_height,
    )

    logger.info(
        "FRAME_RECEIVED camera=%s seq=%d bytes=%d dropped_total=%d",
        camera_id, seq, len(raw), dropped,
    )

    if dropped > 0 and dropped % 50 == 0:
        # Log every 50th drop to avoid flooding the log file
        svclog.log_frame_drop(camera_id=camera_id, dropped_count=dropped)

    return FrameAcceptedResponse(
        status="accepted",
        camera_id=camera_id,
        seq=seq,
        dropped=dropped,
    )


# ── Push / Poll: servo and action poll ───────────────────────────────────────

@app.get("/servo", response_model=ServoPollResponse)
@app.get("/servo/{camera_id}", response_model=ServoPollResponse)
def poll_servo(camera_id: str = DEFAULT_CAMERA_ID):
    """Fast poll (~80ms cadence). Returns latest absolute servo target."""
    return store.get_latest_servo(camera_id)


@app.get("/action", response_model=ActionPollResponse)
@app.get("/action/{camera_id}", response_model=ActionPollResponse)
def poll_action(camera_id: str = DEFAULT_CAMERA_ID):
    """Slow poll (~400ms cadence). Returns latest deterrent action."""
    return store.get_latest_action(camera_id)


# ── Background frame processor ────────────────────────────────────────────────

async def _background_frame_processor() -> None:
    """Single-camera background worker.

    Wakes whenever the slot has a frame, runs inference, updates the poll
    stores. Sleeps 10ms when the slot is empty to avoid busy-spinning.
    """
    while True:
        try:
            frame = store.slot.pop()

            if frame is None or _model is None:
                await asyncio.sleep(0.01)
                continue

            await _process_frame(frame)

        except asyncio.CancelledError:
            break
        except Exception as exc:
            svclog.log_error(
                camera_id=DEFAULT_CAMERA_ID,
                message="Background worker exception",
                exc=traceback.format_exc(),
            )
            logger.exception("Background worker error: %s", exc)
            await asyncio.sleep(0.05)


async def _process_frame(frame: FrameData) -> None:
    """Run inference + servo math + JEV for one frame, update poll stores."""
    t_start = time.perf_counter()

    bgr = cv2.imdecode(np.frombuffer(frame.raw_bytes, np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        svclog.log_error(camera_id=DEFAULT_CAMERA_ID, message="Failed to decode JPEG frame")
        return

    detections, inference_ms = _model.run(bgr, confidence=CONFIDENCE_THRESHOLD)  # type: ignore[union-attr]

    state       = store.camera_state
    frame_ts    = time.monotonic()
    frame_dt_ms = (frame_ts - state.prev_frame_ts) * 1000.0

    if detections:
        best = detections[0]

        movement = tracking_movement(
            bbox_center=best["bbox_center"],
            image_width=frame.image_width,
            image_height=frame.image_height,
            last_pan_abs=state.last_pan_absolute,
            last_tilt_abs=state.last_tilt_absolute,
            prev_bbox_center=state.prev_bbox_center,
            frame_dt_ms=frame_dt_ms,
        )

        action = await decide_action(
            detection=best,
            camera_id=DEFAULT_CAMERA_ID,
            pan_angle=frame.pan_angle,
            tilt_angle=frame.tilt_angle,
            active_action=state.active_action,
            action_frames_remaining=state.action_frames_remaining,
        )

        store.update_detected(
            new_action=action,
            debounce=ACTION_DEBOUNCE_FRAMES,
            bbox_center=best["bbox_center"],
            frame_ts=frame_ts,
            pan_absolute=movement.pan_absolute,
            tilt_absolute=movement.tilt_absolute,
        )

        detection_out = Detection(
            class_id=best["class_id"],
            class_name=best["class_name"],
            confidence=round(best["score"], 4),
            bbox=best["bbox"],
            bbox_center=best["bbox_center"],
        )
        phase = Phase.tracking

    else:
        prev_phase = store.camera_state.phase
        state = store.update_not_detected(lost_threshold=LOST_TRACKING_FRAMES)

        if state.phase != prev_phase:
            svclog.log_phase(
                camera_id=DEFAULT_CAMERA_ID,
                from_phase=prev_phase.value,
                to_phase=state.phase.value,
                reason="lost_tracking_threshold",
            )

        if state.phase == Phase.tracking:
            movement = ServoMovement(
                pan_absolute=state.last_pan_absolute,
                tilt_absolute=state.last_tilt_absolute,
                skip=True,
            )
            action        = state.active_action
            detection_out = None
            phase         = Phase.tracking
        else:
            movement, next_step = sweep_movement(state.sweep_step)
            store.advance_sweep(next_step)
            action        = Action.continue_sweep
            detection_out = None
            phase         = Phase.sweeping

    total_ms  = (time.perf_counter() - t_start) * 1000.0
    latency_ms = (time.time() - frame.received_ts) * 1000.0

    svclog.log_inference(
        camera_id=DEFAULT_CAMERA_ID,
        seq=frame.seq,
        inference_ms=inference_ms,
        total_ms=total_ms,
        detected=detection_out is not None,
        class_name=detection_out.class_name if detection_out else None,
        confidence=detection_out.confidence if detection_out else None,
        phase=phase.value,
        action=action.value,
        dropped_total=store.slot.dropped_count,
    )

    if detection_out:
        logger.info(
            "INFERENCE_RESULT seq=%d detected=True class=%s confidence=%.4f "
            "action=%s phase=%s inference_ms=%.1f total_ms=%.1f latency_ms=%.1f",
            frame.seq,
            detection_out.class_name,
            detection_out.confidence,
            action.value,
            phase.value,
            inference_ms,
            total_ms,
            latency_ms,
        )
    else:
        logger.info(
            "INFERENCE_RESULT seq=%d detected=False phase=%s action=%s "
            "inference_ms=%.1f total_ms=%.1f latency_ms=%.1f",
            frame.seq,
            phase.value,
            action.value,
            inference_ms,
            total_ms,
            latency_ms,
        )

    store.set_latest_servo(ServoPollResponse(
        camera_id=DEFAULT_CAMERA_ID,
        phase=phase,
        movement=movement,
        detection=detection_out,
        camera_state=store.camera_state.to_dict(),
        seq=frame.seq,
        latency_ms=round(latency_ms, 1),
    ))

    store.set_latest_action(ActionPollResponse(
        camera_id=DEFAULT_CAMERA_ID,
        action=action,
        action_frames_remaining=store.camera_state.action_frames_remaining,
        camera_state=store.camera_state.to_dict(),
    ))


# ── Frame preview (debug) ─────────────────────────────────────────────────────

@app.get("/frames/preview", response_class=HTMLResponse)
def frames_preview():
    """Browser-viewable page showing the last 20 frames received from the camera.

    No auth required — intended for local debugging.
    Frames are stored in memory only; cleared on server restart.
    """
    import datetime

    if not _preview_frames:
        body = "<p style='color:#888;font-size:1.1em'>No frames received yet. Is the camera active?</p>"
    else:
        cards = []
        for f in reversed(_preview_frames):   # newest first
            dt = datetime.datetime.fromtimestamp(f["ts"]).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            cards.append(
                f"""<div class="card">
  <div class="meta">
    <span class="ts">📅 {dt}</span>
    <span class="badge">seq&nbsp;{f['seq']}</span>
    <span class="badge cam">📷 {f['camera_id']}</span>
    <span class="badge sz">{f['bytes']:,}&nbsp;B</span>
  </div>
  <img src="data:image/jpeg;base64,{f['b64']}" alt="frame seq {f['seq']}" />
</div>"""
            )
        body = "\n".join(cards)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta http-equiv="refresh" content="3"/>
  <title>Camera Frame Preview</title>
  <style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{ background: #111; color: #eee; font-family: monospace; padding: 16px; }}
    h1 {{ font-size: 1.2em; margin-bottom: 4px; color: #7cf; }}
    .subtitle {{ font-size: 0.8em; color: #888; margin-bottom: 16px; }}
    .card {{ background: #1e1e1e; border: 1px solid #333; border-radius: 6px;
             padding: 10px; margin-bottom: 14px; max-width: 700px; }}
    .meta {{ display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 8px;
             align-items: center; }}
    .ts {{ color: #adf; font-size: 0.85em; }}
    .badge {{ background: #2a2a2a; border: 1px solid #444; border-radius: 3px;
              padding: 2px 6px; font-size: 0.75em; color: #ccc; }}
    .cam {{ color: #fca; border-color: #654; }}
    .sz  {{ color: #afc; border-color: #464; }}
    img  {{ max-width: 100%; border-radius: 4px; display: block; border: 1px solid #333; }}
    .empty {{ color: #666; margin-top: 40px; text-align: center; }}
  </style>
</head>
<body>
  <h1>📡 Camera Frame Preview</h1>
  <p class="subtitle">Last {_PREVIEW_MAX} frames &mdash; auto-refreshes every 3 s &mdash;
     {len(_preview_frames)} frame(s) buffered</p>
  {body}
</body>
</html>"""
    return HTMLResponse(content=html)


# ── Log query endpoints ───────────────────────────────────────────────────────

@app.get("/logs", response_model=LogQueryResponse)
def get_logs(
    level:  Optional[str]   = Query(None,    description="Comma-separated event types: inference,error,action,jev,frame_drop,phase,startup,shutdown"),
    since:  Optional[float] = Query(None,    description="Unix timestamp — include events at or after"),
    until:  Optional[float] = Query(None,    description="Unix timestamp — include events at or before"),
    search: Optional[str]   = Query(None,    description="Substring match across raw JSON line"),
    limit:  int             = Query(200,     ge=1, le=2000, description="Max entries to return"),
    offset: int             = Query(0,       ge=0,          description="Skip first N matching entries"),
    order:  str             = Query("desc",  pattern="^(asc|desc)$", description="asc = oldest first, desc = newest first"),
):
    """Query structured log entries with filtering, search, and pagination.

    Common usage examples:
      GET /logs                              — last 200 entries (newest first)
      GET /logs?level=inference&limit=50    — last 50 inference events
      GET /logs?level=error                 — all errors
      GET /logs?search=WildBoar             — all lines mentioning WildBoar
      GET /logs?since=1700000000            — events after a Unix timestamp
      GET /logs?level=inference&order=asc&limit=1000&offset=0  — paginate
    """
    total_matched, entries = svclog.query_logs(
        level=level,
        since=since,
        until=until,
        search=search,
        limit=limit,
        offset=offset,
        order=order,
    )
    return LogQueryResponse(
        total_matched=total_matched,
        returned=len(entries),
        offset=offset,
        order=order,
        entries=entries,  # type: ignore[arg-type]
    )


@app.get("/logs/stats", response_model=LogStatsResponse)
def get_log_stats():
    """Aggregate statistics over the entire log file.

    Returns total event counts by type, file size, and average inference
    and end-to-end latency figures.
    """
    return svclog.log_stats()
