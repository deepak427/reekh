# Inference Service — Context & Architecture

## What We Are Building

An agricultural wildlife surveillance system. A single ESP32-CAM is deployed
beside farming land. A passive motion sensor wakes the camera when something
moves. The camera streams JPEG frames to this inference service running on
AWS Graviton. The service detects which animal is in the frame, computes an
absolute servo target to keep the animal centred, and decides what deterrent
action to take (bark sound, flash light, alert owner). The result is stored
in memory and the ESP32 polls for it independently.

## Why This Stack

**RF-DETR Medium fine-tuned on 17 custom wildlife classes** via Roboflow.
Weights exported from the Roboflow fine-tuning pipeline as a `.pt` file,
then converted to ONNX for production inference.

17 classes: AmurTiger, Badger, BlackBear, Cow, Dog, Hare, Leopard,
LeopardCat, MuskDeer, RaccoonDog, RedFox, RoeDeer, Sable, SikaDeer,
Weasel, WildBoar, Y-T-Marten.

**OpenCV 5 DNN engine** — not PyTorch, not ONNX Runtime. OpenCV 5 has
significantly improved ONNX support and better ARM performance. At inference
time the only heavy dependencies are `opencv-python-headless` and `numpy`.

**AWS Graviton (m6g family)**.  OpenCV's NEON/SVE SIMD paths map well to
Graviton's wide vector units.  RF-DETR Medium at 576×576 runs in ~80–150ms
on a single Graviton vCPU.

---

## Critical: Use This Repo's Code, Not the Internet

OpenCV 5 DNN is new. Most tutorials and Stack Overflow answers describe the
OpenCV 4.x API which is completely different.  This repo is ground truth.

### What Changed in OpenCV 5 DNN

**Engine selection** — must be explicit:

```python
if hasattr(cv2.dnn, "ENGINE_OPENCV"):
    DNN_ENGINE = cv2.dnn.ENGINE_OPENCV   # 5.x mainline
elif hasattr(cv2.dnn, "ENGINE_NEW"):
    DNN_ENGINE = cv2.dnn.ENGINE_NEW      # 5.0.0 wheel
else:
    raise RuntimeError("Requires OpenCV 5's built-in DNN engine.")
```

Never hardcode one constant. Never use `cv2.dnn.ENGINE_DEFAULT`.

**Environment variable guard** — check before importing cv2:

```python
if "OPENCV_FORCE_DNN_ENGINE" in os.environ:
    raise RuntimeError("Unset OPENCV_FORCE_DNN_ENGINE before running.")
```

**Loading the model** — second argument is required in OpenCV 5:

```python
cv2.dnn.readNetFromONNX(path, engine)
```

**Multiple named outputs:**

```python
dets, logits = net.forward(["dets", "labels"])
```

**No letterbox for RF-DETR** — direct square resize only.

### Reference Files

| File | What it shows |
|---|---|
| `rfdetr-fine-tuned/rfdetr_finetuned_detect.py` | Complete RF-DETR inference: engine selection, preprocessing, forward pass, decoding |
| `rfdetr-fine-tuned/evaluate.py` | Same inference in evaluation context |
| `rfdetr/rfdetr_segment.py` | RF-DETR Seg Nano (COCO) — engine pattern reference |
| `yolo26/detect_yolo26.py` | YOLO26n inference: letterbox, output shape (1,300,6), class decoding |
| `yolo26/yolo26_dnn.cpp` | C++ YOLO26 — confirms Python patterns |
| `rfdetr/rfdetr_segment.cpp` | C++ RF-DETR — same engine selection pattern |
| `dnn_engine.hpp` | C++ engine selection header |
| `evaluate.py` | Cross-model evaluation with all three models |

### Model Output Contract (Fine-tuned RF-DETR Medium)

```
Input tensor:
  name:  "input"
  shape: (1, 3, 576, 576)
  dtype: float32
  preprocessing: BGR→RGB, resize to 576×576, divide by 255,
                 subtract [0.485, 0.456, 0.406], divide by [0.229, 0.224, 0.225]
  layout: NCHW

Output tensors:
  "dets"   shape: (1, 300, 4)   — normalised cx, cy, w, h (0..1 range)
  "labels" shape: (1, 300, 18)  — raw logits, NOT softmax, NOT sigmoid

Decoding:
  1. sigmoid(labels[0])                    → (300, 18) confidence scores
  2. flatten → (5400,) → argsort descending → top 300 indices
  3. divmod(index, 18)                     → (query_idx, class_id)
  4. skip if class_id >= 17               → index 17 is unused background slot
  5. box = dets[0, query_idx]             → [cx, cy, w, h] normalised
  6. scale by [img_w, img_h, img_w, img_h] → pixel coords
  7. convert centre format to [x1,y1,x2,y2]

No NMS needed — RF-DETR uses a one-to-one matching head.
Class IDs are dense 0-indexed (0=AmurTiger … 16=Y-T-Marten).
```

---

## System Architecture

```
                          [ ESP32-CAM ]
                           │         │
   1. Fire-and-forget       │         │  2. Fast servo poll (~80ms)
   POST /frame              │         │     GET /servo
   Returns 202 in <5ms      │         │     Returns target angles in <1ms
                            ▼         ▼
            ┌─────────────────┐   ┌────────────────────┐
            │   FrameSlot     │   │  Latest Servo Store │
            │ (latest-frame-  │   │  { pan, tilt, skip }│
            │  wins)          │   └──────────▲──────────┘
            └───────┬─────────┘              │
                    │                        │  3. Slow action poll (~400ms)
          Worker    │                        │     GET /action
          wakes     ▼                        │
        ┌──────────────────────────────────────────────┐
        │  Background Async Worker (_process_frame)    │
        │    1. OpenCV 5 DNN — RF-DETR inference       │
        │    2. 4-layer servo math  (<1ms)             │
        │    3. JEV / local-fallback action decision   │
        └──────────────────────────────────────────────┘
```

### Endpoints

| Endpoint | Method | Latency | Purpose |
|---|---|---|---|
| `/frame` | POST | < 5ms | Push JPEG (multipart). Returns 202 Accepted. |
| `/servo` | GET | < 1ms | Latest absolute servo target (`pan_absolute`, `tilt_absolute`, `skip`). |
| `/action` | GET | < 1ms | Latest deterrent action (`action`, `action_frames_remaining`). |
| `/state` | GET | < 1ms | Full in-memory `CameraState` dict. |
| `/infer` | POST | ~100ms | Synchronous endpoint — used by the browser simulator. |
| `/health` | GET | < 1ms | Service health, model status, OpenCV version, active engine. |
| `/logs` | GET | < 50ms | Query structured log file with filters. |
| `/logs/stats` | GET | < 50ms | Aggregate latency stats and event counts. |

### Latest-Frame-Wins Slot

- ESP32 sends frames at ~15 FPS (~67ms interval). Inference takes ~80–120ms.
- One `FrameSlot` holds the latest unprocessed frame.
- A new frame arriving while the worker is busy overwrites the slot.
- Stale frames are discarded intentionally — they cause laggy tracking.
  The freshest frame always wins.

---

## Servo Math — Four-Layer Design (`servo_math.py`)

Pure math, no I/O, no side effects.

### Layer 1 — Deadzone
If the animal centre is within `DEADZONE_PX` (20 px) of image centre,
return `skip=True` and hold position. Protects gears from micro-corrections.

### Layer 2 — Velocity Compensation
Compensates for end-to-end hardware latency `COMMAND_LATENCY_MS` (220ms).
```python
vel_x = (bbox_center[0] - prev_bbox_center[0]) / frame_dt_ms
comp_x = clamp(vel_x * COMMAND_LATENCY_MS, -MAX_VELOCITY_COMP_PX, MAX_VELOCITY_COMP_PX)
predicted_center_x = bbox_center[0] + comp_x
```

### Layer 3 — Absolute Target Generation
Pixel error → degree correction via `PAN_GAIN = 0.05` and `TILT_GAIN = 0.05`.
Clamped to `MAX_PAN_DELTA` (15°) and `MAX_TILT_DELTA` (10°) per cycle.
Result is added to `last_pan_abs` / `last_tilt_abs`.
Commands are **always absolute angles**, never deltas.

### Layer 4 — Angular Deadband (Anti-Jitter)
If the angle change is within ± `MIN_SERVO_DELTA_DEG` (1.5°), normalise back
to the current angle.  Eliminates high-frequency oscillation from per-frame
bounding-box detection noise.

---

## State Design & Hysteresis (`state.py`)

### 9 Fields in `CameraState`

```python
phase:                    Phase    # "sweeping" | "tracking"
frames_without_detection: int      # consecutive no-detection counter
active_action:            Action   # "BARK_SOUND" | "FLASH_LIGHT" | "none"
action_frames_remaining:  int      # debounce countdown (~10 frames ≈ 1s)
sweep_step:               int      # waypoint index in SWEEP_SEQUENCE (0..11)
prev_bbox_center:         [cx,cy] | None
prev_frame_ts:            float    # monotonic timestamp of last frame
last_pan_absolute:        float    # last commanded pan (degrees)
last_tilt_absolute:       float    # last commanded tilt (degrees)
```

### Tracking Hysteresis
- During brief occlusion (1–4 frames): `frames_without_detection` increments,
  `phase` stays `"tracking"`, servos hold last position.
- After `LOST_TRACKING_FRAMES = 5` consecutive no-detections: switch to
  `"sweeping"` and resume the perimeter sweep sequence.

---

## JEV Deterrent Decision (`jev_client.py`)

JEV (TypeSafe AI) decides the high-level deterrent action:
- Dangerous predators (`AmurTiger`, `Leopard`, `LeopardCat`, `BlackBear`,
  `WildBoar`): `Action.flash_light`.
- Other species: `Action.bark_sound`.
- `ACTION_DEBOUNCE_FRAMES = 10` prevents rapid re-triggering.
- All JEV outcomes (success / fallback / timeout / skipped) are logged.

---

## Logging (`logging_service.py`)

All significant events are written as JSONL to `LOG_FILE_PATH`
(default `logs/inference.jsonl`).

### Event types
| Level | When |
|---|---|
| `inference` | Every processed frame — latency, detection result, phase, action |
| `action` | Deterrent action triggered or continued |
| `jev` | JEV call outcome (success / fallback / timeout / skipped) |
| `frame_drop` | Logged every 50 drops |
| `phase` | Phase transition (sweeping ↔ tracking) |
| `error` | Unexpected exceptions |
| `startup` | Service started, model loaded |
| `shutdown` | Service stopping |

### Querying via API
```
GET /logs                              # last 200 entries (newest first)
GET /logs?level=inference&limit=50    # last 50 inference events
GET /logs?level=error                 # all errors
GET /logs?search=WildBoar             # lines mentioning WildBoar
GET /logs?since=1700000000            # events after Unix timestamp
GET /logs?level=inference&order=asc&offset=200&limit=200  # page 2
GET /logs/stats                       # counts + avg latency
```

---

## Simulator (`simulator.html`)

High-fidelity browser simulation of the field deployment:
- Three.js 3D farm scene with 17 wildlife species textures and a physical
  pan/tilt surveillance post.
- Dedicated `THREE.WebGLRenderTarget` for the sensor camera viewport
  (640×480, 50° VFOV / 66° HFOV, OV2640 pinhole optics).
- Supports both `Push/Poll` and `Synchronous /infer` modes.

---

## Deployment

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

- Model is loaded once at startup via FastAPI `lifespan`.
- One background asyncio task processes frames sequentially.
- All tuning constants come from environment variables (see `.env.example`).
- Target instance: `m6g.large` (2 vCPU, 8 GB) on AWS Graviton.
- Build OpenCV from source with `ENABLE_NEON=ON` for full SIMD performance on
  Graviton, or verify the PyPI `aarch64` wheel enables NEON before using it.
