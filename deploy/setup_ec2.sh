#!/usr/bin/env bash
set -e

echo "=== Setting up Wildlife Inference Service on AWS Graviton5 with COOL 5.1 ==="

SVC_DIR="/home/ubuntu/inference_service"
cd "$SVC_DIR"

# 1. Activate official COOL Python 3.12 environment
echo "--- Activating /opt/cool/venvs/python_3.12 ---"
source /opt/cool/venvs/python_3.12/bin/activate

# 2. Verify pre-compiled COOL OpenCV build
echo "--- Verifying COOL OpenCV 5.1 & Graviton5 Hardware Optimizations ---"
python3 -c "import cv2; print('Active OpenCV:', cv2.__version__, 'from', cv2.__file__)"
python3 -c "import cv2; print(cv2.getBuildInformation())" | grep -E 'Baseline|Custom HAL' || true

# 3. Install FastAPI & web dependencies into the COOL environment
echo "--- Installing FastAPI & dependencies into COOL Python 3.12 ---"
pip install --upgrade pip
pip install fastapi "uvicorn[standard]" httpx pydantic python-multipart python-dotenv

# 4. Initialize .env if missing
if [ ! -f .env ]; then
    echo "--- Initializing .env from .env.example ---"
    cp .env.example .env
fi

# 5. Create logs directory
mkdir -p logs

# 6. Install, enable, and start systemd service
echo "--- Configuring systemd service ---"
sudo cp deploy/wildlife-inference.service /etc/systemd/system/wildlife-inference.service
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
