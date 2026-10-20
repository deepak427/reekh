"""Validation tests for OpenCV 5 DNN engine correctness and model contract."""
from __future__ import annotations

import os
from pathlib import Path
import unittest
import numpy as np

import cv2
from app.config import MODEL_PATH
from app.inference import RFDETRInference, INPUT_SIZE, NUM_CLASSES, CLASS_NAMES


class TestOpenCV5Engine(unittest.TestCase):
    def test_opencv5_engine_resolution(self):
        """Verify runtime resolved OpenCV 5 built-in DNN engine correctly."""
        has_engine = hasattr(cv2.dnn, "ENGINE_OPENCV") or hasattr(cv2.dnn, "ENGINE_NEW")
        self.assertTrue(has_engine, "OpenCV 5 DNN requires cv2.dnn.ENGINE_OPENCV or ENGINE_NEW")

    def test_opencv_version(self):
        """Verify OpenCV major version is 5.x."""
        ver = cv2.__version__
        self.assertTrue(ver.startswith("5."), f"Expected OpenCV 5.x, detected {ver}")

    def test_model_loading_and_warmup(self):
        """Verify ONNX model can be read with explicit OpenCV 5 engine."""
        model_file = Path(MODEL_PATH)
        if not model_file.exists():
            self.skipTest(f"Model file not found at {MODEL_PATH}")

        engine = RFDETRInference(str(model_file))
        self.assertIn(engine.engine_name, ("ENGINE_OPENCV", "ENGINE_NEW"))

    def test_inference_forward_pass(self):
        """Verify forward pass on synthetic BGR image returns structured detections."""
        model_file = Path(MODEL_PATH)
        if not model_file.exists():
            self.skipTest(f"Model file not found at {MODEL_PATH}")

        engine = RFDETRInference(str(model_file))
        dummy_bgr = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.circle(dummy_bgr, (320, 240), 50, (200, 200, 200), -1)

        dets, latency_ms = engine.run(dummy_bgr, confidence=0.1)
        self.assertIsInstance(dets, list)
        self.assertIsInstance(latency_ms, float)
        self.assertGreater(latency_ms, 0.0)

        for det in dets:
            self.assertIn("class_id", det)
            self.assertIn("class_name", det)
            self.assertIn("score", det)
            self.assertIn("bbox", det)
            self.assertIn("bbox_center", det)
            self.assertEqual(len(det["bbox"]), 4)
            self.assertEqual(len(det["bbox_center"]), 2)


if __name__ == "__main__":
    unittest.main()
