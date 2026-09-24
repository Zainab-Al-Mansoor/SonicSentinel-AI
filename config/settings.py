"""
Central configuration for SonicSentinel AI.

Everything that a team member might need to change during the competition
(e.g. the "surprise modification": new class, new segment duration, new
threshold) lives here or in the admin Settings page (which overrides the
DEFAULT_RUNTIME_SETTINGS stored in the database).
"""
import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("SONIC_DATA_DIR", BASE_DIR / "data"))   # override for tests
UPLOAD_DIR = DATA_DIR / "uploads"          # original uploaded files
SEGMENT_DIR = DATA_DIR / "segments"        # per-event segment WAVs (GTM input + reviewer playback)
IMAGE_DIR = DATA_DIR / "images"            # waveform / spectrogram PNGs
FEATURE_DIR = DATA_DIR / "features"        # extracted-feature files (.npz) for training
REPORT_DIR = DATA_DIR / "reports"
EXPORT_DIR = DATA_DIR / "exports"
DATABASE_PATH = Path(os.environ.get("SONIC_DB_PATH", DATA_DIR / "sonicsentinel.db"))

DATASET_DIR = BASE_DIR / "audio_dataset"
RAW_DATASET_DIR = DATASET_DIR / "raw"              # raw/<Class Name>/*.wav|mp3|...
PROCESSED_DATASET_DIR = DATASET_DIR / "processed"  # processed/<split>/<Class Name>/<AudioID>.wav
DATASET_METADATA_CSV = DATA_DIR / "dataset_metadata.csv"

PYTHON_MODEL_DIR = BASE_DIR / "python_models" / "saved"
PYTHON_MODEL_PATH = Path(os.environ.get("SONIC_MODEL_PATH", PYTHON_MODEL_DIR / "sonic_model.joblib"))
GTM_MODEL_DIR = Path(os.environ.get("SONIC_GTM_DIR", BASE_DIR / "gtm_model" / "model"))  # model.json, metadata.json, weights.bin
ALERT_RULES_PATH = Path(os.environ.get("SONIC_RULES_PATH", BASE_DIR / "alert_rules" / "alert_rules.json"))

for _d in (DATA_DIR, UPLOAD_DIR, SEGMENT_DIR, IMAGE_DIR, FEATURE_DIR, REPORT_DIR, EXPORT_DIR,
           RAW_DATASET_DIR, PROCESSED_DATASET_DIR, PYTHON_MODEL_DIR, GTM_MODEL_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Sound classes  (names MUST be identical in the Python model and the GTM model)
# ---------------------------------------------------------------------------
CLASSES = [
    "Machinery Fault",
    "Glass Breaking",
    "Alarm or Siren",
    "Vehicle Horn",
    "Animal Sound",
    "Gunshot",
    "Panic Scream",
    "Aggression",
    "Person Asking for Help",
    "Background Noise",
]
BACKGROUND_CLASS = "Background Noise"
UNKNOWN_CLASS = "Unknown"
CRITICAL_RECALL_CLASSES = ["Gunshot", "Glass Breaking", "Panic Scream", "Aggression",
                           "Person Asking for Help"]

# ---------------------------------------------------------------------------
# Audio processing
# ---------------------------------------------------------------------------
SUPPORTED_FORMATS = ["wav", "mp3", "flac", "ogg", "m4a"]
TARGET_SR = 22050            # Python pipeline sample rate
GTM_SR = 44100               # Google Teachable Machine (speech-commands BROWSER_FFT) sample rate
MAX_UPLOAD_MB = 25
MIN_DURATION_S = 0.5
MAX_DURATION_S = 300
N_MFCC = 40
N_MELS = 64

SEVERITY_LEVELS = ["Informational", "Low", "Medium", "High", "Critical"]
QUALITY_LEVELS = ["Good", "Acceptable", "Poor", "Unusable"]
EVENT_STATUSES = ["Uploaded", "Classified", "Uncertain", "Alert Generated", "Manual Review",
                  "Reviewed", "Closed"]
ROLES = {
    "user": "Normal User",
    "reviewer": "Audio Reviewer",
    "security": "Security Operator",
    "maintenance": "Maintenance Operator",
    "admin": "Administrator",
}

# Values below are defaults; admins can change them at runtime (Settings page).
DEFAULT_RUNTIME_SETTINGS = {
    "segment_seconds": 2.0,          # fixed-duration segment for uploads
    "segment_hop_seconds": 1.0,      # hop between segments (overlap = segment - hop)
    "live_window_seconds": 2.0,      # live microphone window
    "min_confidence": 0.60,          # minimum confidence for automatic decisions
    "top2_margin": 0.15,             # minimum gap between best and 2nd-best class
    "acceptable_match_diff": 0.20,   # |Python top conf - GTM top conf| for "Acceptable Match"
    "unknown_threshold": 0.35,       # below this combined confidence -> Unknown
    "overlap_threshold": 0.25,       # >=2 non-background classes above this -> overlap
    "python_weight": 0.5,            # share of the Python model in the combined score (GTM gets 1 - this)
    "lookalike_margin": 0.20,        # min gap between a class and its look-alike (SRS Step 14) before it is trusted
    "repeat_window_seconds": 10,     # time period used for repeated-detection confirmation
    "noise_reduction": True,
    "background_noise_limit_db": -20.0,   # estimated noise floor above this -> alert
    "audio_retention_days": 90,
    "record_retention_days": 365,
}

# SRS Step 14 – similar events the application must try to tell apart.
# class -> [(look-alike class in our label set, what that look-alike is in the real world)]
# The real-world look-alikes are trained as Background Noise (fireworks, voices, normal machinery …)
# or are another class (alarm <-> horn).
LOOKALIKE_PAIRS = {
    "Gunshot": [("Background Noise", "fireworks / vehicle backfire / door slam")],
    "Panic Scream": [("Background Noise", "normal shouting or loud voices")],
    "Aggression": [("Background Noise", "normal conversation")],
    "Glass Breaking": [("Background Noise", "metal impact or other impact sounds")],
    "Alarm or Siren": [("Vehicle Horn", "a vehicle horn")],
    "Vehicle Horn": [("Alarm or Siren", "an alarm or siren")],
    "Machinery Fault": [("Background Noise", "normal machinery")],
    "Person Asking for Help": [("Background Noise", "ordinary speech")],
}

SECRET_KEY = os.environ.get("SONIC_SECRET_KEY", "change-this-secret-key-in-production")
