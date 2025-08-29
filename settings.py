import os

# base directory: folder tempat settings.py berada
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# --------------------------------------------------------------------------------
# DEFAULT PATHS —> pastikan highway.mp4 & yolo11m.pt ada di folder ini
# --------------------------------------------------------------------------------

DEFAULT_MODEL_PATH = os.path.join(BASE_DIR, "yolo11m.pt")
DEFAULT_VIDEO_PATH = os.path.join(BASE_DIR, "highway.mp4")

# --------------------------------------------------------------------------------
# CONFIDENCE, BATCH, CLASSES
# --------------------------------------------------------------------------------

DEFAULT_CONFIDENCE = 0.2
BATCH_SIZE        = 2

# Sesuai COCO: bicycle=1, car=2, motorcycle=3, bus=5, truck=7
VEHICLE_CLASS_IDS = [1, 2, 3, 5, 7]

# --------------------------------------------------------------------------------
# CANVAS SETTINGS (streamlit‐drawable‐canvas)
# --------------------------------------------------------------------------------

CANVAS_WIDTH       = 800
CANVAS_HEIGHT      = 450
CANVAS_FILL_COLOR  = ""    # transparan
CANVAS_STROKE_WIDTH= 2
CANVAS_BG_COLOR    = ""    # kosong = putih