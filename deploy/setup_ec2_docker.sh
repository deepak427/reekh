#!/usr/bin/env bash
set -e

echo "=== Setting up Wildlife Inference Service with Docker + Systemd on AWS Graviton ==="

SVC_DIR="/home/ubuntu/inference_service"
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

# 4. Create logs directory
mkdir -p logs

# 5. Build and launch container via Docker Compose
echo "--- Building and launching container ---"
sudo docker compose build
sudo docker compose up -d

# 6. Install and enable Systemd service for auto-start across reboots
echo "--- Configuring Systemd Service ---"
sudo cp deploy/wildlife-inference-docker.service /etc/systemd/system/wildlife-inference-docker.service
sudo systemctl daemon-reload
sudo systemctl enable wildlife-inference-docker.service

# 7. Verification
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
