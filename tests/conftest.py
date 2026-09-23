"""
Test fixtures. Everything runs in a temporary data folder with a temporary
database, a temporary copy of the alert rules and a TINY test-only model
trained on synthetic tones (it is never used by the real application).
"""
import os
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

_TMP = Path(tempfile.mkdtemp(prefix="sonic_test_"))
os.environ["SONIC_DATA_DIR"] = str(_TMP / "data")
os.environ["SONIC_DB_PATH"] = str(_TMP / "test.db")
os.environ["SONIC_MODEL_PATH"] = str(_TMP / "test_model.joblib")
os.environ["SONIC_GTM_DIR"] = str(_TMP / "gtm")               # empty -> GTM "not installed"
shutil.copy(ROOT / "alert_rules" / "alert_rules.json", _TMP / "rules.json")
os.environ["SONIC_RULES_PATH"] = str(_TMP / "rules.json")

import joblib
import numpy as np
import pytest
import soundfile as sf
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config.settings import CLASSES, TARGET_SR

SR = 44100


def tone(freq=440.0, seconds=2.0, sr=SR, amp=0.4, noise=0.01, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * sr)) / sr
    return (amp * np.sin(2 * np.pi * freq * t) + noise * rng.standard_normal(len(t))).astype(np.float32)


def class_signal(ci, seconds=2.0, seed=0):
    """Synthetic, clearly separable signal per class index (test-only)."""
    if CLASSES[ci] == "Background Noise":
        return (0.05 * np.random.default_rng(seed).standard_normal(int(seconds * SR))).astype(np.float32)
    return tone(300 + 400 * ci, seconds, seed=seed)


@pytest.fixture(scope="session", autouse=True)
def tiny_model():
    from audio_preprocessing import preprocess_signal, segment
    from feature_extraction import extract_features, feature_names
    X, y = [], []
    for ci in range(len(CLASSES)):
        for k in range(6):
            clean, _ = preprocess_signal(class_signal(ci, 2.5, seed=k), SR)
            for _, _, s in segment(clean, TARGET_SR, 2.0, 1.0):
                X.append(extract_features(s)); y.append(ci)
    pipe = Pipeline([("scaler", StandardScaler()), ("clf", RandomForestClassifier(n_estimators=50, random_state=0))])
    pipe.fit(np.array(X), np.array(y))
    joblib.dump({"pipeline": pipe, "classes": CLASSES, "algorithm": "rf-test", "version": "py-test-model",
                 "feature_names": feature_names(), "settings": {"segment_seconds": 2.0}, "metrics": {}},
                os.environ["SONIC_MODEL_PATH"])
    yield
    shutil.rmtree(_TMP, ignore_errors=True)


@pytest.fixture()
def wav_file(tmp_path):
    def make(signal, name="clip.wav", sr=SR, subtype="PCM_16"):
        p = tmp_path / name
        sf.write(p, signal, sr, subtype=subtype)
        return p
    return make


@pytest.fixture(scope="session")
def app(tiny_model):
    from src import create_app
    app = create_app({"TESTING": True, "WARM_UP": False})
    return app


@pytest.fixture()
def client(app):
    return app.test_client()


def csrf(client):
    client.get("/login")
    with client.session_transaction() as s:
        return s["_csrf"]


def login(client, username="admin", password="Admin@12345"):
    token = csrf(client)
    r = client.post("/login", data={"username": username, "password": password, "csrf_token": token})
    return token, r
