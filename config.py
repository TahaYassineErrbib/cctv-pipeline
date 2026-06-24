"""
Central configuration for the CCTV attribute pipeline.
Edit the paths in this file to match your local machine — nowhere else.
"""

import os

# =====================================================================
# PATHS
# =====================================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Input source. One of:
#   - local video file path (str)
#   - 0 (int) for webcam
#   - a YouTube URL (str) — resolved to a direct stream URL at runtime via
#     yt-dlp, only when USE_YOUTUBE_STREAM is True below
VIDEO_SOURCE = "https://www.youtube.com/watch?v=6MMXJrzT5c0"

# Set True and put a YouTube URL in VIDEO_SOURCE to pull from a live stream
# instead of a local file. Requires: pip install yt-dlp
USE_YOUTUBE_STREAM = True
YOUTUBE_STREAM_QUALITY = "best[ext=mp4]/best"  # yt-dlp format selector

# Model checkpoints — EDIT to your local paths
CHECKPOINTS_DIR = os.path.join(BASE_DIR, "checkpoints")
C1_CKPT_PATH = os.path.join(CHECKPOINTS_DIR, "garment_type_v2.pth")
C2_CKPT_PATH = os.path.join(CHECKPOINTS_DIR, "C2_long_type_resnet50.pth")
C3_CKPT_PATH = os.path.join(CHECKPOINTS_DIR, "C3_upper_resnet50.pth")
C4_CKPT_PATH = os.path.join(CHECKPOINTS_DIR, "C4_lower_resnet50.pth")

YOLO_WEIGHTS = "yolov8n.pt"  # auto-downloads if missing

# Output locations
DATA_DIR = os.path.join(BASE_DIR, "data")
SNAPSHOT_DIR = os.path.join(BASE_DIR, "snapshots")
PROFILES_JSON_PATH = os.path.join(DATA_DIR, "profiles.json")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(SNAPSHOT_DIR, exist_ok=True)


# =====================================================================
# PERFORMANCE / SAMPLING
# =====================================================================

# Weak-GPU friendly: only run the full classification tree every Nth frame.
# Tracking itself still runs every frame (it's cheap) so IDs stay consistent;
# only the expensive classifier calls are skipped on non-sampled frames.
PROCESS_EVERY_N_FRAMES = 10

# Save an annotated snapshot image every time a frame is actually processed
SAVE_SNAPSHOTS = True


# =====================================================================
# DETECTION / TRACKING
# =====================================================================

PERSON_CLASS_ID = 0          # COCO "person" class
PERSON_CONF_THRESHOLD = 0.4
TRACKER_CONFIG = "bytetrack.yaml"   # built into ultralytics


# =====================================================================
# ACTIVE ZONE (avoid camera borders)
# =====================================================================

# When enabled, only detections whose bbox CENTER falls inside this zone
# are classified/tracked for attribute purposes. People near the frame
# edges (entering/leaving, partially visible, distorted by lens edge
# effects) are ignored. This is expressed as a percentage of frame
# width/height so it automatically adapts to any video resolution.
USE_ACTIVE_ZONE = True

# Margins cut off from each side, as a fraction of frame width/height.
# e.g. 0.15 means the outer 15% on each side is excluded, leaving a
# centered zone covering the middle 70% of the frame (both axes).
ACTIVE_ZONE_MARGIN_X = 0.15
ACTIVE_ZONE_MARGIN_Y = 0.15

# Draw the active zone boundary on the frame for visual confirmation
DRAW_ACTIVE_ZONE = True
ACTIVE_ZONE_COLOR_BGR = (0, 255, 255)  # yellow, distinct from class colors


# =====================================================================
# CLASSIFICATION
# =====================================================================

CLASSIFIER_IMG_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Geometric upper/lower split ratios (slight overlap around the waist)
UPPER_SPLIT_RATIO = 0.55
LOWER_SPLIT_RATIO = 0.45

# C1 class list fallback if checkpoint has no embedded class names.
# Confirmed via 03_inspect_C1_model.ipynb / runtime logs: ['long', 'standard']
C1_CLASSES_FALLBACK = ["long", "standard"]


# =====================================================================
# UI / COLOR CODING
# =====================================================================

# BGR colors (cv2 convention) per garment class, for bounding box + label
# background. Add entries here if your classifier class names ever change.
CLASS_COLORS_BGR = {
    # long-garment classes (C1/C2)
    "long":        (180, 105, 255),   # pink-ish
    "jellaba":     (180, 105, 255),
    "manteau":     (147, 20, 255),    # deep pink
    # standard / upper (C3)
    "standard":    (255, 200, 0),     # cyan-ish
    "tshirt":      (0, 200, 255),     # yellow-orange
    "longmanche":  (0, 165, 255),     # orange
    "jacket":      (0, 100, 255),     # darker orange
    # lower (C4)
    "pantalon":    (0, 200, 0),       # green
    "jupe":        (255, 0, 150),     # magenta
    "short":       (0, 255, 0),       # bright green
    # fallback
    "unknown":     (200, 200, 200),   # gray
}

DEFAULT_BOX_COLOR_BGR = (0, 255, 0)
UPPER_BOX_COLOR_BGR = (0, 165, 255)   # used for the upper-region sub-box outline
LOWER_BOX_COLOR_BGR = (0, 200, 0)     # used for the lower-region sub-box outline   