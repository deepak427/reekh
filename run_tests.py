"""Automated test runner for Wildlife Inference Service."""
from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

# Ensure inference_service root is on sys.path
SVC_ROOT = Path(__file__).resolve().parent
if str(SVC_ROOT) not in sys.path:
    sys.path.insert(0, str(SVC_ROOT))

# Safety check for OpenCV 5 DNN engine
if "OPENCV_FORCE_DNN_ENGINE" in os.environ:
    del os.environ["OPENCV_FORCE_DNN_ENGINE"]


def main():
    print("=" * 65)
    print("  WILDLIFE INFERENCE SERVICE — AUTOMATED VALIDATION SUITE")
    print("=" * 65)
    loader = unittest.TestLoader()
    suite = loader.discover(start_dir=str(SVC_ROOT / "tests"), pattern="test_*.py")
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    print("=" * 65)
    if result.wasSuccessful():
        print(f"  ALL {result.testsRun} TESTS PASSED SUCCESSFULLY! [OK]")
        print("=" * 65)
        sys.exit(0)
    else:
        print(f"  FAILURES: {len(result.failures)} | ERRORS: {len(result.errors)}")
        print("=" * 65)
        sys.exit(1)


if __name__ == "__main__":
    main()
