"""Unit tests for 4-layer kinematics servo math."""
from __future__ import annotations

import unittest
from app.servo_math import tracking_movement, sweep_movement, is_centred
from app.config import DEADZONE_PX, PAN_MIN, PAN_MAX, MIN_SERVO_DELTA_DEG, SWEEP_SEQUENCE


class TestServoMath(unittest.TestCase):
    def test_layer1_deadzone_skip(self):
        """If target bbox is within DEADZONE_PX of center, movement must be skipped."""
        cx, cy = 320.0, 240.0
        mv = tracking_movement(
            bbox_center=[cx, cy],
            image_width=640,
            image_height=480,
            last_pan_abs=0.0,
            last_tilt_abs=0.0,
        )
        self.assertTrue(mv.skip)
        self.assertEqual(mv.pan_absolute, 0.0)
        self.assertEqual(mv.tilt_absolute, 0.0)

    def test_layer2_velocity_compensation(self):
        """Fast-moving target should predict future coordinates and apply lead angle."""
        prev_cx = 300.0
        curr_cx = 380.0
        dt_ms = 50.0

        mv = tracking_movement(
            bbox_center=[curr_cx, 240.0],
            image_width=640,
            image_height=480,
            last_pan_abs=0.0,
            last_tilt_abs=0.0,
            prev_bbox_center=[prev_cx, 240.0],
            frame_dt_ms=dt_ms,
        )
        self.assertFalse(mv.skip)
        self.assertGreater(mv.pan_absolute, 0.0)

    def test_layer3_clamping_limits(self):
        """Target far off-screen must clamp within [PAN_MIN, PAN_MAX]."""
        mv = tracking_movement(
            bbox_center=[2000.0, 240.0],
            image_width=640,
            image_height=480,
            last_pan_abs=85.0,
            last_tilt_abs=0.0,
        )
        self.assertLessEqual(mv.pan_absolute, PAN_MAX)
        self.assertGreaterEqual(mv.pan_absolute, PAN_MIN)

    def test_layer4_angular_deadband(self):
        """Small corrections less than MIN_SERVO_DELTA_DEG should be held to prevent jitter."""
        cx = 320.0 + (DEADZONE_PX + 2.0)
        mv = tracking_movement(
            bbox_center=[cx, 240.0],
            image_width=640,
            image_height=480,
            last_pan_abs=10.0,
            last_tilt_abs=0.0,
        )
        self.assertTrue(mv.skip)
        self.assertEqual(mv.pan_absolute, 10.0)

    def test_sweep_sequence_progression(self):
        """Sweep waypoints must cycle through sequence and wrap cleanly."""
        step = 0
        for expected_pan, expected_tilt in SWEEP_SEQUENCE:
            mv, next_step = sweep_movement(step)
            self.assertEqual(mv.pan_absolute, expected_pan)
            self.assertEqual(mv.tilt_absolute, expected_tilt)
            self.assertFalse(mv.skip)
            step = next_step
        self.assertEqual(step, 0)

    def test_is_centred(self):
        """Tolerance checks correctly around optical center."""
        self.assertTrue(is_centred([320.0, 240.0], 640, 480, tolerance_px=10.0))
        self.assertFalse(is_centred([350.0, 240.0], 640, 480, tolerance_px=10.0))


if __name__ == "__main__":
    unittest.main()
