#!/usr/bin/env python3
"""OpenCV 5 DNN Engine Benchmark Tool.

Measures:
  1. Engine initialization & cold-start forward pass latency
  2. Sustained forward pass latency distribution (min, mean, median, p95, p99, max)
  3. Sustained throughput (FPS)
  4. Platform hardware vector dispatch (ARM NEON / SVE / AVX)
  5. JSON report export for evaluation records
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path
import numpy as np

# Ensure inference_service root is on path
SVC_ROOT = Path(__file__).resolve().parent
if str(SVC_ROOT) not in sys.path:
    sys.path.insert(0, str(SVC_ROOT))

# Safety guard
if "OPENCV_FORCE_DNN_ENGINE" in os.environ:
    del os.environ["OPENCV_FORCE_DNN_ENGINE"]

import cv2
from app.config import MODEL_PATH
from app.inference import RFDETRInference, INPUT_SIZE


def run_benchmark(iterations: int = 50, output_json: str = "benchmark_results.json"):
    print("=" * 70)
    print("  OPENCV 5 DNN ENGINE PERFORMANCE BENCHMARK")
    print("=" * 70)

    model_path = Path(MODEL_PATH)
    if not model_path.exists():
        print(f"Error: model file not found at {model_path}")
        sys.exit(1)

    print(f"Platform:      {platform.platform()}")
    print(f"Processor:     {platform.processor() or platform.machine()}")
    print(f"Architecture:  {platform.machine()}")
    print(f"OpenCV:        {cv2.__version__}")
    print(f"Model Path:    {model_path} ({model_path.stat().st_size / (1024*1024):.1f} MB)")
    print(f"Input Tensor:  1x3x{INPUT_SIZE}x{INPUT_SIZE} (NCHW float32 normalized)")
    print("-" * 70)

    # 1. Measure Engine Loading & Cold Start
    t_load_start = time.perf_counter()
    inf = RFDETRInference(str(model_path))
    t_load_ms = (time.perf_counter() - t_load_start) * 1000.0

    print(f"Active Engine: {inf.engine_name}")
    print(f"Init & Warmup: {t_load_ms:.2f} ms")
    print("-" * 70)

    # 2. Prepare representative test frame (640x480 BGR)
    test_frame = np.full((480, 640, 3), 120, dtype=np.uint8)
    cv2.circle(test_frame, (320, 240), 80, (20, 180, 240), -1)
    cv2.rectangle(test_frame, (100, 100), (250, 300), (40, 200, 50), -1)

    # 3. Benchmark Sustained Inferences
    print(f"Running {iterations} timed iterations...")
    latencies = []

    for i in range(iterations):
        t0 = time.perf_counter()
        dets, inf_time_ms = inf.run(test_frame, confidence=0.3)
        t_total_ms = (time.perf_counter() - t0) * 1000.0
        latencies.append(t_total_ms)
        if (i + 1) % 10 == 0 or (i + 1) == iterations:
            print(f"  Processed {i + 1}/{iterations} frames... (latest: {t_total_ms:.1f} ms)")

    latencies = np.array(latencies)
    mean_ms = float(np.mean(latencies))
    median_ms = float(np.median(latencies))
    min_ms = float(np.min(latencies))
    max_ms = float(np.max(latencies))
    p95_ms = float(np.percentile(latencies, 95))
    p99_ms = float(np.percentile(latencies, 99))
    fps = 1000.0 / mean_ms

    print("=" * 70)
    print("  RESULTS SUMMARY")
    print("=" * 70)
    print(f"  Mean Latency:    {mean_ms:.2f} ms")
    print(f"  Median (p50):    {median_ms:.2f} ms")
    print(f"  Min Latency:     {min_ms:.2f} ms")
    print(f"  95th Percentile: {p95_ms:.2f} ms")
    print(f"  99th Percentile: {p99_ms:.2f} ms")
    print(f"  Max Latency:     {max_ms:.2f} ms")
    print(f"  Throughput:      {fps:.2f} FPS")
    print("=" * 70)

    # 4. Save results to JSON
    report = {
        "timestamp": time.time(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "opencv_version": cv2.__version__,
        "engine_name": inf.engine_name,
        "model": str(model_path.name),
        "input_size": INPUT_SIZE,
        "iterations": iterations,
        "init_warmup_ms": round(t_load_ms, 2),
        "mean_ms": round(mean_ms, 2),
        "median_ms": round(median_ms, 2),
        "min_ms": round(min_ms, 2),
        "p95_ms": round(p95_ms, 2),
        "p99_ms": round(p99_ms, 2),
        "max_ms": round(max_ms, 2),
        "throughput_fps": round(fps, 2),
    }

    out_file = SVC_ROOT / output_json
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"Saved benchmark report to: {out_file}")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark OpenCV 5 DNN engine on RF-DETR.")
    parser.add_argument("--iterations", type=int, default=30, help="Number of benchmark iterations")
    parser.add_argument("--output", type=str, default="benchmark_results.json", help="Output JSON path")
    args = parser.parse_args()

    run_benchmark(iterations=args.iterations, output_json=args.output)
