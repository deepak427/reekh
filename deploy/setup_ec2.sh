#!/usr/bin/env bash
set -e

echo "=== Setting up Wildlife Inference Service on AWS EC2 (t4g.medium / Graviton ARM64) ==="

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SVC_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$SVC_DIR"

# 1. Detect Python Environment: COOL AMI vs Standard Ubuntu ARM64
if [ -d "/opt/cool/venvs/python_3.12" ]; then
    echo "--- Detected AWS COOL 5.1 Environment ---"
    source /opt/cool/venvs/python_3.12/bin/activate
    PYTHON_EXEC="/opt/cool/venvs/python_3.12/bin/python3"
    UVICORN_EXEC="/opt/cool/venvs/python_3.12/bin/uvicorn"
else
    echo "--- Standard Ubuntu ARM64 (t4g.medium) Detected ---"
    sudo apt-get update -y
    sudo apt-get install -y python3-pip python3-venv libgl1 libglib2.0-0 libgomp1 curl

    if [ ! -d ".venv" ]; then
        echo "Creating Python virtual environment in .venv..."
        python3 -m venv .venv
    fi
    source .venv/bin/activate
    PYTHON_EXEC="$SVC_DIR/.venv/bin/python3"
    UVICORN_EXEC="$SVC_DIR/.venv/bin/uvicorn"
fi

# 2. Install dependencies from pinned requirements.txt
echo "--- Installing dependencies from requirements.txt ---"
pip install --upgrade pip
pip install -r requirements.txt

# 3. Verify OpenCV 5 Engine & Hardware Dispatch
echo "--- Verifying OpenCV 5 Engine & SIMD Dispatch ---"
$PYTHON_EXEC -c "import cv2; print('Active OpenCV:', cv2.__version__, 'from', cv2.__file__)"
$PYTHON_EXEC -c "import cv2; print(cv2.getBuildInformation())" | grep -E 'Baseline|Custom HAL|CPU/HW' || true

# 4. Initialize .env if missing
if [ ! -f .env ]; then
    echo "--- Initializing .env from .env.example ---"
    cp .env.example .env
fi

# 5. Download model weights if missing
mkdir -p models
if [ ! -f models/inference_model.onnx ]; then
    echo "--- Downloading RF-DETR ONNX model weights from GitHub Releases ---"
    curl -L -o models/inference_model.onnx "https://github.com/deepak427/reekh/releases/download/weights/inference_model.onnx"
fi

# 6. Create logs directory
mkdir -p logs

# 6. Configure Systemd Service dynamically based on environment
echo "--- Configuring systemd service ---"
cat <<EOF | sudo tee /etc/systemd/system/wildlife-inference.service > /dev/null
[Unit]
Description=Wildlife Surveillance Inference Service (AWS Graviton t4g.medium)
After=network.target

[Service]
Type=simple
User=ubuntu
Group=ubuntu
WorkingDirectory=$SVC_DIR
EnvironmentFile=-$SVC_DIR/.env
Environment=PYTHONUNBUFFERED=1
ExecStart=$UVICORN_EXEC app.main:app --host 0.0.0.0 --port 8000 --workers 1
Restart=always
RestartSec=3
StandardOutput=journal
StandardError=journal

LimitNOFILE=65536
TimeoutStopSec=10

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable wildlife-inference.service
sudo systemctl restart wildlife-inference.service

# 7. Check service status
echo "--- Service Status ---"
sudo systemctl status wildlife-inference.service --no-pager

echo ""
echo "=== Setup Complete! ==="
echo "Test endpoint: curl http://localhost:8000/health"
echo "View logs:     journalctl -u wildlife-inference.service -f"
