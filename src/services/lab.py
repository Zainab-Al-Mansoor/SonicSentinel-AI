"""
Robustness Lab – "what happens to the prediction if the recording gets worse?"

A user loads one clip (own upload, or a random recording from the unseen TEST split) and moves sliders for
noise, real background noise, echo, distance, volume, microphone type and a cut-off start. The server applies
the same augmentation functions used for training (augmentation/augment.py), runs the Python model with the
SAME pipeline as a normal upload, and returns a 44.1 kHz WAV so the browser can run the Google Teachable
Machine model on exactly the same modified audio. Nothing is written to the event database.
"""
from __future__ import annotations

import hashlib
import json
import random
import re
import time
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

from config.settings import (BASE_DIR, BACKGROUND_CLASS, CLASSES, DATA_DIR, DATASET_METADATA_CSV, GTM_SR, TARGET_SR)
from audio_preprocessing import (load_audio, preprocess_signal, segment, to_mono, resample, analyze_quality)
from augmentation.augment import add_noise, reverb, volume, distance, device
from feature_extraction import extract_features
from python_models.inference import aggregate_scores, top_k, top_margin
from .settings_service import get_settings

LAB_DIR = Path(DATA_DIR) / "lab"
MAX_SECONDS = 20.0
TOKEN_RE = re.compile(r"^[a-f0-9]{16}$")
NAME_RE = re.compile(r"^[a-f0-9]{16}_[a-f0-9]{10}\.wav$")

DEFAULTS = {"noise_snr": None, "bg_snr": None, "echo_rt60": 0.0, "distance_m": 0.0, "gain_db": 0.0,
            "device": False, "cut_start_pct": 0}
LIMITS = {"noise_snr": (-5.0, 40.0), "bg_snr": (-5.0, 40.0), "echo_rt60": (0.0, 1.2), "distance_m": (0.0, 40.0),
          "gain_db": (-45.0, 12.0), "cut_start_pct": (0, 80)}
SWEEP_SNR = [None, 30, 20, 15, 10, 5, 0]


class LabError(Exception):
    pass


def _cleanup(max_age_s: int = 3 * 3600):
    LAB_DIR.mkdir(parents=True, exist_ok=True)
    cutoff = time.time() - max_age_s
    for p in LAB_DIR.iterdir():
        try:
            if p.stat().st_mtime < cutoff:
                p.unlink()
        except OSError:
            pass


def clean_params(raw: dict | None) -> dict:
    p = dict(DEFAULTS)
    for k, v in (raw or {}).items():
        if k not in DEFAULTS:
            continue
        if k == "device":
            p[k] = bool(v)
        elif v is None or v == "" or v == "off":
            p[k] = DEFAULTS[k]
        else:
            lo, hi = LIMITS[k]
            p[k] = float(min(hi, max(lo, float(v))))
    p["cut_start_pct"] = int(p["cut_start_pct"])
    return p


# ---------------------------------------------------------------- clip storage
def _meta_path(token):
    return LAB_DIR / f"{token}.json"


def _store(y44: np.ndarray, meta: dict) -> str:
    _cleanup()
    token = uuid.uuid4().hex[:16]
    np.save(LAB_DIR / f"{token}.npy", y44.astype(np.float32))
    _meta_path(token).write_text(json.dumps(meta), encoding="utf-8")
    return token


def load_token(token: str) -> tuple[np.ndarray, dict]:
    if not token or not TOKEN_RE.match(token):
        raise LabError("Unknown lab clip – load a clip first.")
    npy = LAB_DIR / f"{token}.npy"
    if not npy.exists():
        raise LabError("This lab clip has expired – load it again.")
    return np.load(npy), json.loads(_meta_path(token).read_text(encoding="utf-8"))


def _to44(samples, sr) -> np.ndarray:
    y = resample(to_mono(samples), sr, GTM_SR).astype(np.float32)
    return y[: int(MAX_SECONDS * GTM_SR)]


def load_file(path: Path, filename: str, user_id: int) -> tuple[str, dict]:
    a = load_audio(path)
    if a.duration < 0.5:
        raise LabError("The clip is shorter than 0.5 s.")
    meta = {"source": "upload", "filename": filename, "actual_class": None, "duration": round(min(a.duration, MAX_SECONDS), 2),
            "user_id": user_id, "truncated": a.duration > MAX_SECONDS}
    return _store(_to44(a.samples, a.sample_rate), meta), meta


def test_split_available() -> bool:
    try:
        return DATASET_METADATA_CSV.exists()
    except Exception:
        return False


def load_test_clip(class_label: str | None, user_id: int) -> tuple[str, dict]:
    if not test_split_available():
        raise LabError("The dataset metadata is not available on this server – upload a clip instead.")
    m = pd.read_csv(DATASET_METADATA_CSV)
    t = m[(m["split"] == "test") & (m["is_augmented"] == 0)]
    if class_label and class_label in CLASSES:
        t = t[t["class_label"] == class_label]
    t = t[t["path"].map(lambda p: (BASE_DIR / p).exists())]
    if t.empty:
        raise LabError("No unseen test recordings found for this class.")
    r = t.sample(1).iloc[0]
    a = load_audio(BASE_DIR / r["path"])
    meta = {"source": "test split", "filename": f"{r['audio_id']}.wav", "audio_id": r["audio_id"],
            "actual_class": r["class_label"], "duration": round(min(a.duration, MAX_SECONDS), 2), "user_id": user_id,
            "truncated": a.duration > MAX_SECONDS}
    return _store(_to44(a.samples, a.sample_rate), meta), meta


# ---------------------------------------------------------------- degradation
def _background(n: int, seed: int) -> np.ndarray:
    """A real Background Noise recording from the TEST split if available, else pink-ish noise."""
    rng = np.random.default_rng(seed)
    try:
        if test_split_available():
            m = pd.read_csv(DATASET_METADATA_CSV)
            b = m[(m["split"] == "test") & (m["class_label"] == BACKGROUND_CLASS) & (m["is_augmented"] == 0)]
            b = b[b["path"].map(lambda p: (BASE_DIR / p).exists())]
            if len(b):
                r = b.iloc[int(rng.integers(len(b)))]
                a = load_audio(BASE_DIR / r["path"])
                return np.resize(_to44(a.samples, a.sample_rate), n)
    except Exception:
        pass
    white = rng.standard_normal(n).astype(np.float32)
    return np.cumsum(white) * 0.02 + white * 0.3          # brown + white mix


def degrade(y: np.ndarray, params: dict, seed: int = 0, sr: int = GTM_SR) -> np.ndarray:
    """Apply the selected conditions in a physically sensible order (source -> room -> mic -> recording)."""
    random.seed(seed); np.random.seed(seed)
    out = y.astype(np.float32).copy()
    if params["cut_start_pct"] > 0 and len(out) > sr // 2:
        out = out[int(len(out) * params["cut_start_pct"] / 100):]
    if params["distance_m"] > 0:
        out = distance(out, sr, params["distance_m"])
    if params["echo_rt60"] > 0:
        out = reverb(out, sr, params["echo_rt60"])
    if params["bg_snr"] is not None:
        out = add_noise(out, params["bg_snr"], _background(len(out), seed + 1))
    if params["device"]:
        out = device(out, sr)
    if params["noise_snr"] is not None:
        out = add_noise(out, params["noise_snr"])
    if params["gain_db"]:
        out = volume(out, params["gain_db"])
    return np.clip(out, -1.0, 1.0).astype(np.float32)


# ---------------------------------------------------------------- analysis
def python_analyse(y: np.ndarray, sr: int, model) -> dict:
    """Same Python-side steps as a normal upload (analysis.analyze_upload)."""
    s = get_settings()
    q = analyze_quality(y, sr)
    clean, _ = preprocess_signal(y, sr, trim=True, denoise=s["noise_reduction"])
    segs = segment(clean, TARGET_SR, s["segment_seconds"], s["segment_hop_seconds"])
    if not segs:
        return {"prediction": None, "confidence": 0.0, "scores": {c: 0.0 for c in CLASSES}, "quality": q.get("label"),
                "quality_issues": q.get("issues", []), "snr_db": q.get("snr_db"), "segments": 0}
    seg_scores = model.segment_scores([seg for _, _, seg in segs])
    clip, idx = aggregate_scores(seg_scores, s["min_confidence"])
    pred, conf = top_k(clip, 1)[0]
    return {"prediction": pred, "confidence": round(conf, 4), "scores": {k: round(v, 4) for k, v in clip.items()},
            "top3": top_k(clip, 3), "margin": round(top_margin(clip), 4),
            "uncertain": conf < s["min_confidence"], "quality": q.get("label"), "quality_issues": q.get("issues", []),
            "snr_db": q.get("snr_db"), "rms_dbfs": q.get("rms_dbfs"), "segments": len(segs)}


def _write_wav(token: str, y: np.ndarray, params: dict) -> str:
    h = hashlib.sha1(json.dumps(params, sort_keys=True).encode()).hexdigest()[:10]
    name = f"{token}_{h}.wav"
    p = LAB_DIR / name
    if not p.exists():
        sf.write(p, y, GTM_SR, subtype="PCM_16")
    return name


def audio_path(name: str) -> Path:
    if not NAME_RE.match(name or ""):
        raise LabError("Invalid file name.")
    p = LAB_DIR / name
    if not p.exists():
        raise LabError("File expired.")
    return p


def run(token: str, raw_params: dict, model) -> dict:
    y0, meta = load_token(token)
    params = clean_params(raw_params)
    t0 = time.perf_counter()
    y = degrade(y0, params, seed=int(token[:6], 16))
    res = python_analyse(y, GTM_SR, model)
    res.update({"params": params, "audio_name": _write_wav(token, y, params), "meta": meta,
                "duration": round(len(y) / GTM_SR, 2), "python_ms": int((time.perf_counter() - t0) * 1000),
                "model_version": model.version})
    if meta.get("actual_class"):
        res["correct"] = res["prediction"] == meta["actual_class"]
    return res


def sweep(token: str, raw_params: dict, model, kind: str = "noise_snr") -> dict:
    """Confidence of the clean prediction while white noise gets stronger (other sliders kept)."""
    if kind not in ("noise_snr", "bg_snr"):
        raise LabError("Unknown sweep.")
    y0, meta = load_token(token)
    base = clean_params(raw_params)
    ref = meta.get("actual_class")
    points = []
    for snr in SWEEP_SNR:
        p = dict(base); p[kind] = snr
        y = degrade(y0, p, seed=int(token[:6], 16))
        r = python_analyse(y, GTM_SR, model)
        ref = ref or r["prediction"]
        points.append({"snr": snr, "label": "clean" if snr is None else f"{snr:g} dB", "prediction": r["prediction"],
                       "confidence": r["confidence"], "ref_confidence": r["scores"].get(ref, 0.0), "quality": r["quality"],
                       "audio_name": _write_wav(token, y, p)})
    return {"kind": kind, "reference_class": ref, "points": points}
