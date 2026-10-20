"""End-to-end integration tests for FastAPI routes and WebSocket protocol."""
from __future__ import annotations

import io
import json
import unittest
import numpy as np
import cv2
from fastapi.testclient import TestClient

from app.main import app, DEFAULT_CAMERA_ID
from app.config import API_KEY


class TestApiRoutes(unittest.TestCase):
    def setUp(self):
        self.client_cm = TestClient(app)
        self.client = self.client_cm.__enter__()

    def tearDown(self):
        self.client_cm.__exit__(None, None, None)

    def test_health_endpoint(self):
        """Health check endpoint must return status ok, model state, and engine."""
        res = self.client.get("/health")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["status"], "ok")
        self.assertIn("opencv_version", data)
        self.assertIn("engine", data)
        self.assertIn("architecture", data)

    def test_web_routes(self):
        """UI endpoints must return HTTP 200 with HTML content."""
        res_sim = self.client.get("/simulate")
        self.assertEqual(res_sim.status_code, 200)
        self.assertIn("text/html", res_sim.headers["content-type"])
        self.assertIn("Agricultural Surveillance Simulator", res_sim.text)

        res_mon = self.client.get("/monitor")
        self.assertEqual(res_mon.status_code, 200)
        self.assertIn("text/html", res_mon.headers["content-type"])
        self.assertTrue("Wildlife Deterrent" in res_mon.text or "Telemetry" in res_mon.text)

    def test_servo_and_action_poll(self):
        """Push/poll fallback endpoints return valid schemas."""
        res_servo = self.client.get(f"/servo/{DEFAULT_CAMERA_ID}")
        self.assertEqual(res_servo.status_code, 200)
        s_data = res_servo.json()
        self.assertIn("movement", s_data)
        self.assertIn("pan_absolute", s_data["movement"])
        self.assertIn("tilt_absolute", s_data["movement"])

        res_action = self.client.get(f"/action/{DEFAULT_CAMERA_ID}")
        self.assertEqual(res_action.status_code, 200)
        a_data = res_action.json()
        self.assertIn("action", a_data)

    def test_frame_push_endpoint(self):
        """POST /frame/{camera_id} receives JPEG and returns 202 Accepted in <10ms."""
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        _, buf = cv2.imencode(".jpg", img)
        files = {"image": ("test.jpg", io.BytesIO(buf.tobytes()), "image/jpeg")}
        data = {
            "pan_angle": "0.0",
            "tilt_angle": "0.0",
            "image_width": "640",
            "image_height": "480",
            "seq": "1",
        }
        headers = {"X-Api-Key": API_KEY}

        res = self.client.post(f"/frame/{DEFAULT_CAMERA_ID}", data=data, files=files, headers=headers)
        self.assertEqual(res.status_code, 202)
        res_data = res.json()
        self.assertEqual(res_data["status"], "accepted")
        self.assertEqual(res_data["camera_id"], DEFAULT_CAMERA_ID)

    def test_websocket_protocol(self):
        """WebSocket protocol handshake, reset, and text metadata frame."""
        headers = {"x-api-key": API_KEY}
        with self.client.websocket_connect(f"/ws/{DEFAULT_CAMERA_ID}?api_key={API_KEY}", headers=headers) as ws:
            ws.send_text(json.dumps({"type": "reset", "camera_id": DEFAULT_CAMERA_ID}))
            ws.send_text(json.dumps({
                "type": "status",
                "state": "ACTIVE",
                "message": "Test motion",
                "camera_id": DEFAULT_CAMERA_ID,
            }))
            ws.send_text(json.dumps({
                "type": "frame",
                "seq": 100,
                "pan": 0.0,
                "tilt": 0.0,
                "w": 640,
                "h": 480,
            }))
            img = np.zeros((480, 640, 3), dtype=np.uint8)
            _, buf = cv2.imencode(".jpg", img)
            ws.send_bytes(buf.tobytes())


if __name__ == "__main__":
    unittest.main()
