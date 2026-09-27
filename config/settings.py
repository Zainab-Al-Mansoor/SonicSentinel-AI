import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("SONIC_DATA_DIR", BASE_DIR / "data"))
UPLOAD_DIR = DATA_DIR / "uploads"
SEGMENT_DIR = DATA_DIR / "segments"
IMAGE_DIR = DATA_DIR / "images"
FEATURE_DIR = DATA_DIR / "features"
REPORT_DIR = DATA_DIR / "reports"
EXPORT_DIR = DATA_DIR / "exports"
DATABASE_PATH = Path(os.environ.get("SONIC_DB_PATH", DATA_DIR / "sonicsentinel.db"))

DATASET_DIR = BASE_DIR / "audio_dataset"
RAW_DATASET_DIR = DATASET_DIR / "raw"
PROCESSED_DATASET_DIR = DATASET_DIR / "processed"
DATASET_METADATA_CSV = DATA_DIR / "dataset_metadata.csv"

PYTHON_MODEL_DIR = BASE_DIR / "python_models" / "saved"
PYTHON_MODEL_PATH = Path(os.environ.get("SONIC_MODEL_PATH", PYTHON_MODEL_DIR / "sonic_model.joblib"))
GTM_MODEL_DIR = Path(os.environ.get("SONIC_GTM_DIR", BASE_DIR / "gtm_model" / "model"))
ALERT_RULES_PATH = Path(os.environ.get("SONIC_RULES_PATH", BASE_DIR / "alert_rules" / "alert_rules.json"))

for _d in (DATA_DIR, UPLOAD_DIR, SEGMENT_DIR, IMAGE_DIR, FEATURE_DIR, REPORT_DIR, EXPORT_DIR,
           RAW_DATASET_DIR, PROCESSED_DATASET_DIR, PYTHON_MODEL_DIR, GTM_MODEL_DIR):
    _d.mkdir(parents=True, exist_ok=True)

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

SUPPORTED_FORMATS = ["wav", "mp3", "flac", "ogg", "m4a"]
TARGET_SR = 22050
GTM_SR = 44100
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

DEFAULT_RUNTIME_SETTINGS = {
    "segment_seconds": 2.0,
    "segment_hop_seconds": 1.0,
    "live_window_seconds": 2.0,
    "min_confidence": 0.60,
    "top2_margin": 0.15,
    "acceptable_match_diff": 0.20,
    "unknown_threshold": 0.35,
    "overlap_threshold": 0.25,
    "python_weight": 0.75,
    "agreement_fusion": True,
    "lookalike_margin": 0.20,
    "repeat_window_seconds": 10,
    "noise_reduction": True,
    "background_noise_limit_db": -20.0,
    "audio_retention_days": 90,
    "record_retention_days": 365,
}

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
