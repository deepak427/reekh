# Wildlife Inference Service

Agricultural wildlife surveillance inference backend running on AWS Graviton (ARM64) with OpenCV 5 DNN and fine-tuned RF-DETR Medium.

Designed for single ESP32-CAM deployment with low-latency push/poll architecture, 4-layer servo tracking math, and structured JSONL logging.

---

## Architecture Overview

```
                      [ ESP32-CAM ]
                       │         │
1. Fire-and-forget     │         │  2. Fast servo poll (~80ms)
   POST /frame         │         │     GET /servo
   Returns 202 (<5ms)  │         │     Returns target angles (<1ms)
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
    │  Background Worker (_background_processor)   │
    │    1. OpenCV 5 DNN — RF-DETR inference       │
    │    2. 4-layer servo math  (<1ms)             │
    │    3. JEV / local-fallback action decision   │
    └──────────────────────────────────────────────┘
```

---

## Directory Structure

```
inference_service/
├── .gitignore              # Git ignore rules for independent repo deployment
├── .env.example            # Template for runtime environment variables
├── requirements.txt        # Runtime dependencies (pinned)
├── README.md               # Quickstart and API documentation
├── CONTEXT.md              # Deep system specifications & mathematical design
│
├── app/                    # Application source code
│   ├── __init__.py
│   ├── main.py             # FastAPI entrypoint, routes, background worker
│   ├── inference.py        # OpenCV 5 DNN RF-DETR wrapper & NCHW pre/post-processing
│   ├── servo_math.py       # 4-layer pure servo math (deadzone, vel comp, clamp, deadband)
│   ├── state.py            # DeviceState, FrameSlot (latest-frame-wins, lazy asyncio.Event)
│   ├── config.py           # Environment config & dynamic path resolution
│   ├── schemas.py          # Pydantic request/response schemas & enums
│   ├── jev_client.py       # TypeSafe AI JEV client with local heuristic fallback
│   └── logging_service.py  # Thread-safe JSONL structured logger, query & stats
│
├── models/                 # Model artifacts
│   └── inference_model.onnx # Fine-tuned RF-DETR Medium ONNX weights
│
├── Dockerfile              # Container definition (ARM64 / Graviton compatible)
├── docker-compose.yml      # Container orchestration, healthcheck, volume mounts
│
├── deploy/                 # Production deployment scripts
│   ├── setup_ec2_docker.sh # Automated Docker + Systemd setup on fresh EC2
│   ├── wildlife-inference-docker.service # Systemd unit managing Docker Compose
│   ├── wildlife-inference.service        # Native Systemd unit (optional)
│   └── setup_ec2.sh                      # Native EC2 setup script (optional)
│
├── simulator.html          # Web-based 3D simulation & testing tool
│
└── logs/                   # Structured runtime logs
    └── .gitkeep
```

---

## Quickstart

### 1. Setup Virtual Environment
```bash
python -m venv .venv
# Linux / macOS / Graviton
source .venv/bin/activate
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure Environment
```bash
cp .env.example .env
# Edit .env with your API keys and configuration
```

### 4. Run the Service
Run directly inside `inference_service/`:
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Or from the workspace root:
```bash
uvicorn inference_service.app.main:app --host 0.0.0.0 --port 8000
```

---

## API Reference

| Endpoint | Method | Purpose |
|---|---|---|
| `/health` | GET | Service status, model loaded flag, OpenCV engine info |
| `/frame` or `/frame/{camera_id}` | POST | Push JPEG frame (multipart/form-data). Returns `202 Accepted` |
| `/servo` or `/servo/{camera_id}` | GET | Fast poll (<1ms) returning latest absolute pan/tilt targets |
| `/action` or `/action/{camera_id}` | GET | Slow poll (<1ms) returning active deterrent action |
| `/state` or `/state/{camera_id}` | GET | Current in-memory camera state (phase, sweep step, last angles) |
| `/infer` | POST | Synchronous inference endpoint (used by simulator & debugging) |
| `/logs` | GET | Query structured JSONL log entries with filters & pagination |
| `/logs/stats` | GET | Aggregated log statistics, latency averages, and event counts |

### Querying Logs
```bash
# Get last 50 inference events
curl "http://localhost:8000/logs?level=inference&limit=50"

# Search for specific class detection
curl "http://localhost:8000/logs?search=WildBoar"

# View system errors
curl "http://localhost:8000/logs?level=error"

# Check latency stats
curl "http://localhost:8000/logs/stats"
```
