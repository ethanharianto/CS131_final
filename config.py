"""Paths and defaults for SportsMOT basketball experiments."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT / "data" / "dataset"
OUTPUT_ROOT = ROOT / "outputs"

SEQUENCE_NAME = "v_-6Os86HzwCs_c001"
# Human-readable label for reports (SportsMOT internal IDs are not reader-friendly).
SEQUENCE_TITLE = (
    "Women's Asia Cup 2019 basketball clip from SportsMOT "
    "(825 frames, about 33 seconds at 1280×720)"
)
SEQUENCE_SPLIT = "train"

# Process a contiguous segment (frames are 1-indexed in MOT)
FRAME_START = 1
FRAME_END = 825
SAMPLE_EVERY = 50

# MOG2 / morphology (kept for the deliberately-broken panning-failure baseline)
MIN_BOX_AREA = 800
MAX_BOX_AREA = 80_000
MIN_ASPECT = 0.15
MAX_ASPECT = 1.2
MORPH_KERNEL = (5, 5)

# YOLO detection
YOLO_MODEL = "yolov8n.pt"
YOLO_CONF = 0.25
YOLO_IOU = 0.5
YOLO_PERSON_CLASS = 0  # COCO person class id
YOLO_IMG_SIZE = 640

# Tracker (tuned via scripts/sweep_tracker.py for track-internal appearance coherence)
TRACK_IOU_THRESH = 0.5
TRACK_MAX_MISSED = 3

# LAB torso histogram
HIST_BINS = (4, 8, 8)  # L x A x B = 256 bins
TORSO_TOP = 0.15  # skip head/skin
TORSO_BOT = 0.50  # focus on chest/upper jersey
TORSO_LEFT = 0.20
TORSO_RIGHT = 0.80
# Saturation mask: drop pixels whose chroma in LAB AB-plane is below threshold
# (kills grey hardwood, white walls, neutral skin); 0 disables.
MIN_CHROMA = 12.0

# K-means baseline
KMEANS_K = 3
KMEANS_SEED = 0

# BiLSTM team-id model
LSTM_HIDDEN = 64
LSTM_EPOCHS = 50
LSTM_LR = 1e-3
LSTM_MAX_SEQ_LEN = 500
LSTM_SEED = 0
LSTM_USE_POSITION = True  # concat (cx, cy, w, h) normalized to LAB embedding

# M3: end-to-end CNN encoder over raw torso crops
CROP_H = 48
CROP_W = 24
CNN_FEATURE_DIM = 64
E2E_EPOCHS = 30
E2E_LR = 1e-3
E2E_DROPOUT = 0.3
E2E_WEIGHT_DECAY = 1e-4

# SportsMOT broadcast resolution (used to normalize position features)
SEQ_WIDTH = 1280
SEQ_HEIGHT = 720
