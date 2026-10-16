"""Runtime configuration.

All values are read from environment variables (or .env in the
inference_service root).  Defaults work out-of-the-box when the service is
launched from inside the inference_service/ directory.
"""
import os
from pathlib import Path

# Load .env from the inference_service root (one level above this file).
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass

# ── Helpers ───────────────────────────────────────────────────────────────────

# Absolute path to the inference_service/ root, independent of cwd.
_SVC_ROOT = Path(__file__).resolve().parent.parent


def _resolve_path(raw: str | None, default_subpath: str) -> str:
    if not raw:
        return str((_SVC_ROOT / default_subpath).resolve())
    p = Path(raw)
    if p.is_absolute():
        return str(p)
    if p.exists():
        return str(p.resolve())
    if (_SVC_ROOT / p).exists():
        return str((_SVC_ROOT / p).resolve())
    return str((_SVC_ROOT / p).resolve())


# ── Model ─────────────────────────────────────────────────────────────────────
MODEL_PATH = _resolve_path(
    os.getenv("MODEL_PATH"),
    "models/inference_model.onnx",
)
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.45"))

# ── Camera / Image ────────────────────────────────────────────────────────────
DEFAULT_IMAGE_WIDTH  = 640
DEFAULT_IMAGE_HEIGHT = 480

# Maximum accepted JPEG payload size (bytes).
# OV2640 at 640×480 JPEG typically produces 30–80 KB.
MAX_FRAME_BYTES = int(os.getenv("MAX_FRAME_BYTES", str(500_000)))  # 500 KB hard cap

# ── Servo math ────────────────────────────────────────────────────────────────
PAN_GAIN   = float(os.getenv("PAN_GAIN",   "0.05"))
PAN_INVERT = os.getenv("PAN_INVERT", "false").lower() in ("true", "1", "yes")
TILT_GAIN  = float(os.getenv("TILT_GAIN",  "0.0"))   # V1: Tilt locked to 0.0

MAX_PAN_DELTA  = float(os.getenv("MAX_PAN_DELTA",  "15.0"))
MAX_TILT_DELTA = float(os.getenv("MAX_TILT_DELTA", "0.0"))

PAN_MIN  = float(os.getenv("PAN_MIN",  "-90.0"))
PAN_MAX  = float(os.getenv("PAN_MAX",   "90.0"))
TILT_MIN = float(os.getenv("TILT_MIN",   "0.0"))   # V1: Tilt locked to 0.0
TILT_MAX = float(os.getenv("TILT_MAX",   "0.0"))

# Pixel deadzone: if animal centre is within this radius of image centre,
# hold position and skip the servo write.
DEADZONE_PX = float(os.getenv("DEADZONE_PX", "20.0"))

# Angular deadband: minimum degree change needed to issue a servo move.
MIN_SERVO_DELTA_DEG = float(os.getenv("MIN_SERVO_DELTA_DEG", "1.5"))

# End-to-end hardware latency (ms) used for velocity compensation.
# Breakdown: ESP32 capture→send (~10ms) + network (~20ms) + inference (~100ms)
#            + ESP32 poll interval (~80ms) + ESP32 execution (~10ms) ≈ 220ms.
COMMAND_LATENCY_MS = float(os.getenv("COMMAND_LATENCY_MS", "220.0"))

# Maximum velocity compensation clamp in pixels.
MAX_VELOCITY_COMP_PX = float(os.getenv("MAX_VELOCITY_COMP_PX", "150.0"))

# ── Phase / State machine ─────────────────────────────────────────────────────
LOST_TRACKING_FRAMES   = int(os.getenv("LOST_TRACKING_FRAMES",   "5"))
ACTION_DEBOUNCE_FRAMES = int(os.getenv("ACTION_DEBOUNCE_FRAMES", "10"))

# ── Sweep sequence ────────────────────────────────────────────────────────────
# (pan_absolute_deg, tilt_absolute_deg) waypoints visited when no animal is seen.
SWEEP_SEQUENCE: list[tuple[float, float]] = [
    (  0.0, 0.0),
    ( 30.0, 0.0),
    ( 60.0, 0.0),
    ( 90.0, 0.0),
    ( 60.0, 0.0),
    ( 30.0, 0.0),
    (  0.0, 0.0),
    (-30.0, 0.0),
    (-60.0, 0.0),
    (-90.0, 0.0),
    (-60.0, 0.0),
    (-30.0, 0.0),
]

# ── JEV / OpenRouter (TypeSafe AI) ─────────────────────────────────────────────
JEV_API_URL   = os.getenv("JEV_API_URL", "https://openrouter.ai/api/alpha/decisions")
JEV_API_KEY   = os.getenv("OPENROUTER_API_KEY") or os.getenv("JEV_API_KEY") or ""
JEV_MODEL     = os.getenv("JEV_MODEL", "~typesafe/jev-latest")
JEV_TIMEOUT_S = float(os.getenv("JEV_TIMEOUT_S", "0.8"))

# ── API auth ──────────────────────────────────────────────────────────────────
API_KEY = os.getenv("API_KEY", "change-me-in-production")

# ── Logging ───────────────────────────────────────────────────────────────────
LOG_FILE_PATH = _resolve_path(
    os.getenv("LOG_FILE_PATH"),
    "logs/inference.jsonl",
)
LOG_MAX_LINES = int(os.getenv("LOG_MAX_LINES", "50000"))
