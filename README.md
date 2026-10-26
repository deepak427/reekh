# Autonomous Agricultural Wildlife Deterrent — Inference Service

Edge-to-cloud wildlife surveillance and deterrence inference service built with **OpenCV 5 DNN**, fine-tuned **RF-DETR Medium (17 classes)**, and **FastAPI**, deployed on **AWS Graviton (t4g.medium ARM64)**.

---

## Architecture Overview

```
                      ┌──────────────────────────┐
                      │ ESP32-CAM Hardware Node  │
                      │  (PIR Wake · FreeRTOS)   │
                      └─────────────┬────────────┘
                                    │
                                    │ WebSocket /ws/esp32-cam-01 (Primary)
                                    │ or HTTP POST /frame + GET /servo (Fallback)
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                 AWS Graviton (t4g.medium) Inference Service                 │
│                                                                             │
│  1. Ingestion:                                                              │
│     - Text metadata: {"type":"frame", "seq":42, "pan":0.0, "tilt":0.0}      │
│     - Binary JPEG: Zero-copy in-memory buffer decode via cv2.imdecode       │
│                                                                             │
│  2. Latest-Frame-Wins FrameSlot:                                            │
│     - Dropped stale frame counter; prevents queue latency accumulation      │
│                                                                             │
│  3. OpenCV 5 DNN Engine (ENGINE_OPENCV / ENGINE_NEW):                       │
│     - Fine-tuned RF-DETR Medium ONNX (17 wildlife classes)                  │
│     - 576×576 NCHW normalized input tensor with ARM NEON vectorization      │
│     - Named multi-output forward pass: ["dets", "labels"]                   │
│                                                                             │
│  4. 4-Layer Kinematic Servo Math:                                           │
│     - Layer 1: Center deadzone (20px) — eliminates jitter                   │
│     - Layer 2: Velocity compensation — predicts animal lead angle (220ms)   │
│     - Layer 3: Absolute angle clamp — [-90°, +90°] zero cumulative drift    │
│     - Layer 4: Angular deadband (1.5°) — suppresses sub-threshold oscillation│
│                                                                             │
│  5. Deterrence Policy:                                                      │
│     - Escalation: Strobe Flash → Audio Alarm → Farmer Notification          │
│     - Action debounce (10 frames) — prevents animal harassment              │
│                                                                             │
│  6. Telemetry & Observability:                                              │
│     - WebSocket push back to ESP32: {"pan":X, "tilt":Y, "action":"..."}     │
│     - Server-Sent Events (SSE) broadcast to /monitor dashboard              │
│     - Structured JSONL audit logging in logs/inference.jsonl                │
└─────────────────────────────────────┬───────────────────────────────────────┘
                                      │
                                      ▼
                      ┌───────────────────────────────┐
                      │ Real-time Web Observability   │
                      │  - /simulate : 3D Twin        │
                      │  - /monitor  : Live Telemetry │
                      └───────────────────────────────┘
```

---

## Technical Deep Dive: OpenCV 5 DNN Integration

1. **Explicit Engine Resolution**:
   OpenCV 5 DNN requires explicit engine binding. This service uses runtime detection to bind the active engine without relying on legacy fallbacks:
   ```python
   if hasattr(cv2.dnn, "ENGINE_OPENCV"):
       DNN_ENGINE = cv2.dnn.ENGINE_OPENCV   # OpenCV 5.x mainline
   elif hasattr(cv2.dnn, "ENGINE_NEW"):
       DNN_ENGINE = cv2.dnn.ENGINE_NEW      # OpenCV 5.0.0
   else:
       raise RuntimeError("Requires OpenCV 5 DNN built-in engine")
   ```

2. **Environment Variable Guard**:
   Prevents runtime engine override corruption by inspecting environment variables before `import cv2`:
   ```python
   if "OPENCV_FORCE_DNN_ENGINE" in os.environ:
       raise RuntimeError("Unset OPENCV_FORCE_DNN_ENGINE before starting service")
   ```

3. **Explicit Named Multi-Output Tensors**:
   OpenCV 5 DNN forward passes require explicit output tensor names:
   ```python
   detections, logits = net.forward(["dets", "labels"])
   ```

4. **Zero-Copy In-Memory Decoding**:
   Frames streamed over WebSocket or HTTP are decoded directly from raw memory buffers without writing temporary files to disk:
   ```python
   bgr = cv2.imdecode(np.frombuffer(raw_bytes, np.uint8), cv2.IMREAD_COLOR)
   ```

5. **Hardware Vectorization on AWS Graviton**:
   Leverages OpenCV 5's native ARM NEON SIMD compute kernels on AWS Graviton (`t4g.medium`), delivering sustained sub-second inference latency on a standard cloud CPU without requiring expensive GPU instances.

---

## Cloud Architecture & AWS Deployment

### Primary Target: AWS Graviton `t4g.medium` (Standard Ubuntu AMI)

The production service runs on an **AWS EC2 `t4g.medium`** instance:
- **Processor**: AWS Graviton2 (64-bit ARM Neoverse N1 cores, 2 vCPUs)
- **Memory**: 4 GB RAM
- **OS**: Ubuntu 24.04 LTS (aarch64)
- **OpenCV Runtime**: Pinned `opencv-python-headless==5.0.0.93`

#### Deployment Steps:

```bash
# 1. SSH into t4g.medium instance
ssh -i ~/.ssh/your-key.pem ubuntu@<ec2-ip-address>

# 2. Clone repository & enter service folder
git clone https://github.com/deepak427/reekh.git
cd reekh/inference_service

# 3. Download model weights (if not using automated setup script)
mkdir -p models
curl -L -o models/inference_model.onnx "https://github.com/deepak427/reekh/releases/download/weights/inference_model.onnx"

# 4. Run automated setup script (auto-provisions dependencies, downloads model if missing, sets systemd)
chmod +x deploy/setup_ec2.sh
./deploy/setup_ec2.sh

# 5. Verify service status
sudo systemctl status wildlife-inference.service
```

The script automatically detects standard Ubuntu ARM64, provisions a Python virtual environment, installs pinned dependencies, downloads model weights if missing, configures systemd, and starts the service.

### Alternative Target: AWS COOL AMI (Pre-compiled OpenCV 5.1)

If using the official AWS COOL (Computer Vision on AWS) AMI with pre-compiled OpenCV 5.1:
```bash
cd /home/ubuntu/reekh/inference_service
chmod +x deploy/setup_ec2.sh
./deploy/setup_ec2.sh
```
`setup_ec2.sh` automatically detects `/opt/cool/venvs/python_3.12` and activates the optimized hardware environment.

### Docker Deployment

A hardened multi-stage Docker setup running as non-root user (`appuser`, UID 1000) with container `HEALTHCHECK`:
```bash
# 1. Enter service directory
cd reekh/inference_service

# 2. Download model weights
mkdir -p models
curl -L -o models/inference_model.onnx "https://github.com/deepak427/reekh/releases/download/weights/inference_model.onnx"

# 3. Setup environment and build
cp .env.example .env
mkdir -p logs
docker compose up -d --build
docker compose ps
docker compose logs -f
```

---

## Local Quickstart

```bash
# 1. Create venv and install pinned dependencies
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2. Configure environment
cp .env.example .env
# Edit .env — set API_KEY at minimum

# 3. Verify OpenCV 5 engine + model + servo math
python verify_installation.py    # all 5 checks should print PASS

# 4. Run automated test suite (zero external deps)
python run_tests.py              # 15 tests, all should pass

# 5. Benchmark RF-DETR latency on your hardware
python benchmark_opencv5.py --iterations 30   # exports benchmark_results.json

# 6. Start the service
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Open in browser:
- `/simulate` — 3D digital twin, trigger detections without hardware
- `/monitor`  — live telemetry dashboard
- `/health`   — engine name, model status, uptime

On AWS Graviton t4g.medium, step 5 should report **80–150ms mean latency, 6–12 FPS**.
To benchmark on EC2: `python benchmark_opencv5.py --iterations 50 --output benchmark_graviton.json`

---

## Transport Protocols

### 1. WebSocket Protocol (`/ws/{camera_id}`) — Primary
Matches real ESP32-CAM firmware (`firmware_service/wildlife_cam/ws_client.cpp`):
- **On Connect**: Camera sends `{"type":"reset"}` to wipe stale state, followed by `{"type":"status","state":"BOOT"}`.
- **Frame Transmission**:
  1. Text metadata frame: `{"type":"frame","seq":42,"pan":0.0,"tilt":0.0,"w":640,"h":480}`
  2. Binary JPEG frame immediately following.
- **Inference Push**: Backend pushes results back over the same connection:
  `{"pan":12.5,"tilt":0.0,"skip":false,"action":"FLASH_LIGHT","phase":"tracking","seq":42}`
- **Remote Commands**: Dashboard sends `{"type":"simulate_pir"}` and `{"type":"simulate_idle"}` to trigger hardware remotely.

### 2. HTTP Push/Poll (`/frame`, `/servo`, `/action`) — Fallback
For low-bandwidth or unreliable networks:
- `POST /frame/{camera_id}`: Fire-and-forget JPEG upload; returns `202 Accepted` in <5ms.
- `GET /servo/{camera_id}`: Fast poll (~80ms cadence) returning latest absolute pan/tilt target.
- `GET /action/{camera_id}`: Slow poll (~400ms cadence) returning deterrent action.

---

## Responsible AI & Operational Ethics

1. **Animal Welfare & Humane Escalation**:
   Deterrence follows a humane multi-tier progression:
   - Level 1: Soft spotlight illumination (optical awareness).
   - Level 2: Acoustic deterrence (dog bark / warning sounds) if animal persists.
   - Level 3: Push alert to orchard owner.
   - **Anti-Harassment Protection**: Action debounce (10 frames) and angular deadband (1.5°) prevent continuous strobe or acoustic harassment.

2. **Privacy by Design**:
   Model is restricted exclusively to 17 wild and domestic animal species. No human facial recognition, biometric analysis, or cleartext surveillance data of bystanders is ever recorded or transmitted.

3. **Edge & Cloud Energy Efficiency**:
   ESP32 hardware sleeps in microamp standby until physical PIR interrupt wakes it. Inference worker drops obsolete intermediate frames (`latest-frame-wins`), eliminating wasted compute on AWS Graviton.
