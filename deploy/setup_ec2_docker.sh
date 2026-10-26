#!/usr/bin/env bash
set -e

echo "=== Setting up Wildlife Inference Service with Docker + Systemd on AWS Graviton ==="

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SVC_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$SVC_DIR"

# 1. Update OS and install Docker + Docker Compose
echo "--- Installing Docker & Docker Compose plugin ---"
sudo apt-get update -y
sudo apt-get install -y ca-certificates curl gnupg docker.io docker-compose-v2

# 2. Add ubuntu user to docker group & start Docker daemon
echo "--- Configuring Docker service ---"
sudo usermod -aG docker ubuntu || true
sudo systemctl enable --now docker

# 3. Create .env if not exists
if [ ! -f .env ]; then
    echo "--- Initializing .env from .env.example ---"
    cp .env.example .env
fi

# 4. Download model weights if not present
mkdir -p models
if [ ! -f models/inference_model.onnx ]; then
    echo "--- Downloading RF-DETR ONNX model weights from GitHub Releases ---"
    curl -L -o models/inference_model.onnx "https://github.com/deepak427/reekh/releases/download/weights/inference_model.onnx"
fi

# 5. Create logs directory
mkdir -p logs

# 6. Build and launch container via Docker Compose
echo "--- Building and launching container ---"
sudo docker compose build
sudo docker compose up -d

# 7. Install and enable Systemd service for auto-start across reboots
echo "--- Configuring Systemd Service ---"
cat <<EOF | sudo tee /etc/systemd/system/wildlife-inference-docker.service > /dev/null
[Unit]
Description=Wildlife Inference Service (Docker Compose)
Requires=docker.service
After=docker.service network.target

[Service]
Type=oneshot
RemainAfterExit=yes
WorkingDirectory=$SVC_DIR
ExecStart=/usr/bin/docker compose up -d
ExecStop=/usr/bin/docker compose down
TimeoutStartSec=0

[Install]
WantedBy=multi-user.target
EOF
sudo systemctl daemon-reload
sudo systemctl enable wildlife-inference-docker.service

# 8. Verification
echo "--- Waiting for service warmup ---"
sleep 5
sudo docker ps
echo ""
echo "--- Health check ---"
curl -s http://localhost:8000/health || echo "Still initializing, check logs."

echo ""
echo "=== Deployment Complete ==="
echo "Container logs: sudo docker compose logs -f"
echo "Systemd status: sudo systemctl status wildlife-inference-docker.service"
