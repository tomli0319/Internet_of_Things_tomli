from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "hydration.db"

STATIC_DIR = BASE_DIR / "static"
SNAPSHOT_DIR = STATIC_DIR / "snapshots"
LIVE_DIR = STATIC_DIR / "live"
SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
LIVE_DIR.mkdir(parents=True, exist_ok=True)

HOST = "0.0.0.0"
PORT = 5000

# Camera settings
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# Full-frame motion detection settings
BACKGROUND_WARMUP_FRAMES = 90
BG_HISTORY = 300
BG_VAR_THRESHOLD = 24
BG_DETECT_SHADOWS = False
BG_LEARNING_RATE_IDLE = 0.015
BG_LEARNING_RATE_EVENT = 0.0
FG_THRESHOLD = 200
MIN_CONTOUR_AREA_PX = 22000
MIN_CONTOUR_WIDTH_PX = 130
MIN_CONTOUR_HEIGHT_PX = 130
MIN_OCCUPANCY_SCORE = 0.040
MORPH_KERNEL_SIZE = 5

# Ignore-zone for fountain/water motion. Off by default in this version.
# Format: (x1, y1, x2, y2)
USE_IGNORE_ZONE = False
IGNORE_ZONE = (230, 170, 410, 470)
DRAW_IGNORE_ZONE = True

# Detector timing
IDLE_CHECK_INTERVAL_S = 1.0
ACTIVE_CHECK_INTERVAL_S = 0.5
CONFIRM_FRAMES = 6
MIN_POSITIVE_FRAMES = 4
IDLE_TRIGGER_WINDOWS = 2
END_LOST_WINDOWS = 2
STATUS_UPDATE_PERIOD_S = 1.0

# Live-feed writing cadence
LIVE_FEED_CAPTURE_INTERVAL_S = 1.0
JPEG_QUALITY = 88
