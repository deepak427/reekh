#!/usr/bin/env python3
"""One-command environment and verification script for OpenCV 5 Hackathon judges.

Verifies:
  1. Python version (3.11+ / 3.12+ recommended)
  2. OpenCV 5 installation & built-in DNN engine availability
  3. RF-DETR 17-class ONNX model integrity
  4. 4-layer kinematics math correctness
  5. Configuration & environment validity
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

# Ensure inference_service root is on path
SVC_ROOT = Path(__file__).resolve().parent
if str(SVC_ROOT) not in sys.path:
    sys.path.insert(0, str(SVC_ROOT))

# Safety guard
if "OPENCV_FORCE_DNN_ENGINE" in os.environ:
    del os.environ["OPENCV_FORCE_DNN_ENGINE"]


def verify_all():
    print("=" * 70)
    print("  AGRICULTURAL WILDLIFE SURVEILLANCE — OPENCV 5 VERIFICATION")
    print("=" * 70)

    passes = 0
    total = 5

    # 1. Python runtime
    py_ver = sys.version.split()[0]
    print(f"[1/5] Python Runtime: {py_ver} ... ", end="")
    if sys.version_info >= (3, 10):
        print("PASS")
        passes += 1
    else:
        print("FAIL (Python 3.10+ required)")

    # 2. OpenCV 5 DNN engine check
    print(f"[2/5] OpenCV 5 DNN Engine ... ", end="")
    try:
        import cv2
        ver = cv2.__version__
        engine = None
        if hasattr(cv2.dnn, "ENGINE_OPENCV"):
            engine = "ENGINE_OPENCV"
        elif hasattr(cv2.dnn, "ENGINE_NEW"):
            engine = "ENGINE_NEW"

        if ver.startswith("5.") and engine:
            print(f"PASS (OpenCV {ver}, {engine})")
            passes += 1
        else:
            print(f"FAIL (Requires OpenCV 5.x with ENGINE_OPENCV/ENGINE_NEW; found {ver})")
    except Exception as exc:
        print(f"FAIL ({exc})")

    # 3. Model weights presence & load
    from app.config import MODEL_PATH
    print(f"[3/5] RF-DETR ONNX Model ({MODEL_PATH}) ... ", end="")
    model_path = Path(MODEL_PATH)
    if model_path.exists():
        size_mb = model_path.stat().st_size / (1024 * 1024)
        try:
            from app.inference import RFDETRInference
            inf = RFDETRInference(str(model_path))
            print(f"PASS ({size_mb:.1f} MB, warmup OK)")
            passes += 1
        except Exception as exc:
            print(f"FAIL (Model load failed: {exc})")
    else:
        print(f"FAIL (File not found at {model_path})")

    # 4. Kinematics Servo Math
    print(f"[4/5] 4-Layer Servo Tracking Kinematics ... ", end="")
    try:
        from app.servo_math import tracking_movement, sweep_movement
        # Test deadzone
        mv_deadzone = tracking_movement([320.0, 240.0], 640, 480, 0.0, 0.0)
        # Test offset tracking
        mv_offset = tracking_movement([420.0, 240.0], 640, 480, 0.0, 0.0)
        # Test sweep sequence
        mv_sweep, next_step = sweep_movement(0)

        if mv_deadzone.skip and not mv_offset.skip and next_step == 1:
            print("PASS (Deadzone + Offset + Sweep verified)")
            passes += 1
        else:
            print("FAIL (Math constraint violation)")
    except Exception as exc:
        print(f"FAIL ({exc})")

    # 5. Core dependencies check
    print(f"[5/5] Core Dependencies (FastAPI, Uvicorn, WebSockets) ... ", end="")
    try:
        import fastapi
        import uvicorn
        import websockets
        import pydantic
        import numpy
        print("PASS")
        passes += 1
    except Exception as exc:
        print(f"FAIL (Missing module: {exc})")

    print("=" * 70)
    if passes == total:
        print(f"  SYSTEM VERIFICATION PASSED ({passes}/{total}) — READY FOR EVALUATION [OK]")
        print("=" * 70)
        print("\nCommands to run:")
        print("  1. Run unit test suite:       python run_tests.py")
        print("  2. Run OpenCV 5 benchmark:    python benchmark_opencv5.py")
        print("  3. Start live service:        uvicorn app.main:app --port 8000")
        print("  4. View digital twin:         http://localhost:8000/simulate")
        print("  5. View live telemetry:       http://localhost:8000/monitor")
        return 0
    else:
        print(f"  SYSTEM VERIFICATION FAILED ({passes}/{total})")
        print("=" * 70)
        return 1


if __name__ == "__main__":
    sys.exit(verify_all())
