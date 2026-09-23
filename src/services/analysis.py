"""
Analysis orchestration: upload / live window  ->  validation -> quality ->
pre-processing -> segmentation -> Python model -> (GTM in browser) ->
comparison -> rules -> alerts -> storage.

GTM runs in the browser (TensorFlow.js, the official GTM export). The server
never sends Python predictions to the browser before GTM has produced its own
scores for a segment, so the two models stay independent.
"""
import hashlib
import json
import secrets
import time
from datetime import timedelta
from pathlib import Path

import numpy as np
import soundfile as sf

from config.settings import (TARGET_SR, GTM_SR, SEGMENT_DIR, IMAGE_DIR, GTM_MODEL_DIR, CLASSES,
                             PYTHON_MODEL_PATH)
from audio_preprocessing import (validate_file, validate_audio, load_bytes, analyze_quality,
                                 preprocess_signal, segment, prepare_for_gtm, to_mono, resample)
from feature_extraction import extract_features, save_waveform, save_spectrogram, compute_fingerprint, similarity
from python_models.inference import PythonSoundModel, ModelNotAvailable, aggregate_scores, top_k

from ..extensions import db
from ..models import AudioEvent, Segment, Alert, ModelVersion, now
from . import audit
from .decision import decide
from .rules import load_rules
from .settings_service import get_settings


class AnalysisError(Exception):
    pass


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------
_py_model = {"obj": None, "mtime": None}


def python_model() -> PythonSoundModel:
    p = Path(PYTHON_MODEL_PATH)
    if not p.exists():
        raise ModelNotAvailable("No trained Python model. Run: python -m python_models.train_models")
    m = p.stat().st_mtime
    if _py_model["obj"] is None or _py_model["mtime"] != m:
        _py_model["obj"] = PythonSoundModel(p)
        _py_model["mtime"] = m
        _register_version("python", _py_model["obj"].version,
                          {"algorithm": _py_model["obj"].algorithm, "metrics": _py_model["obj"].metrics.get("test")})
    return _py_model["obj"]


def gtm_info() -> dict:
    meta_p = GTM_MODEL_DIR / "metadata.json"
    model_p = GTM_MODEL_DIR / "model.json"
    if not (meta_p.exists() and model_p.exists()):
        return {"available": False, "reason": "GTM model files not found in gtm_model/model/"}
    meta = json.loads(meta_p.read_text(encoding="utf-8"))
    labels = meta.get("wordLabels") or meta.get("labels") or []
    version = f"gtm-{meta.get('modelName', 'model')}-{str(meta.get('timeStamp', ''))[:19]}".replace(" ", "_")
    missing = [c for c in CLASSES if c not in labels]
    extra = [l for l in labels if l not in CLASSES]
    return {"available": True, "version": version, "labels": labels, "missing": missing, "extra": extra}


def _register_version(kind, version, details):
    try:
        if not ModelVersion.query.filter_by(model_type=kind, version=version).first():
            db.session.add(ModelVersion(model_type=kind, version=version, details=details))
            db.session.commit()
            audit.log("model_update", "model", version, f"{kind} model registered", commit=True)
    except Exception:
        db.session.rollback()


def warm_up():
    """Run the pipeline once on synthetic audio so the first real request is fast (numba JIT)."""
    y = (0.1 * np.sin(2 * np.pi * 440 * np.arange(TARGET_SR) / TARGET_SR)).astype(np.float32)
    clean, _ = preprocess_signal(y, TARGET_SR)
    extract_features(clean)
    resample(y, TARGET_SR, GTM_SR)


# ---------------------------------------------------------------------------
def new_audio_id(prefix="AUD") -> str:
    return f"{prefix}-{now():%Y%m%d-%H%M%S}-{secrets.token_hex(2).upper()}"


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _python_scores(segments: list[np.ndarray]) -> list[dict]:
    model = python_model()
    X = np.vstack([extract_features(s) for s in segments])
    return model.predict_proba(X)


def _rms_db(x):
    return float(20 * np.log10(np.sqrt(np.mean(x ** 2)) + 1e-10))


def make_images(event: AudioEvent, y_raw_mono: np.ndarray | None = None, sr: int | None = None):
    """Waveform + spectrogram PNGs (called eagerly for uploads, lazily for live windows)."""
    if y_raw_mono is None:
        from audio_preprocessing import load_audio
        a = load_audio(event.stored_path)
        y_raw_mono, sr = to_mono(a.samples), a.sample_rate
    y = resample(y_raw_mono, sr, TARGET_SR)
    wave = IMAGE_DIR / f"{event.audio_id}_wave.png"
    spec = IMAGE_DIR / f"{event.audio_id}_spec.png"
    hl = None
    idx = event.python_segment_index
    if idx is not None and idx >= 0 and idx < len(event.segments):
        hl = (event.segments[idx].start_s, event.segments[idx].end_s)
    save_waveform(y, TARGET_SR, wave, segments=[(s.start_s, s.end_s) for s in event.segments], highlight=hl)
    save_spectrogram(y, TARGET_SR, spec)
    event.waveform_path, event.spectrogram_path = str(wave), str(spec)


# ---------------------------------------------------------------------------
# Uploaded files
# ---------------------------------------------------------------------------
def analyze_upload(path: Path, original_filename: str, user, source="upload", actual_class=None):
    """Returns (event, errors, duplicate_event)."""
    t0 = time.time()
    s = get_settings()
    v = validate_file(path, original_filename)
    if not v.ok:
        audit.log("upload_rejected", "file", original_filename, "; ".join(v.errors), user=user)
        audit.check_failed_uploads(user)
        return None, v.errors, None

    digest = sha256_file(path)
    dup = AudioEvent.query.filter_by(sha256=digest).first()
    if dup:
        audit.log("duplicate_upload", "event", dup.audio_id, f"identical to {dup.audio_id} ({original_filename})", user=user)
        audit.notify("duplicate_file", f"Identical file uploaded again: {original_filename} = {dup.audio_id}")
        return None, [], dup

    audio = v.audio
    ev = AudioEvent(audio_id=new_audio_id(), user_id=user.id, source=source, actual_class=actual_class,
                    original_filename=original_filename, stored_path=str(path), sha256=digest,
                    **{k: val for k, val in audio.metadata(original_filename).items() if k != "filename"})
    ev.status = "Uploaded"
    db.session.add(ev)
    db.session.flush()

    raw_mono = to_mono(audio.samples)
    q = analyze_quality(audio.samples, audio.sample_rate)
    ev.quality, ev.quality_label, ev.noise_level_db = q, q["label"], q["noise_floor_dbfs"]

    # near-duplicate check (re-encoded / trimmed / volume-changed copies)
    fp = compute_fingerprint(raw_mono, audio.sample_rate)
    ev.fingerprint = fp
    best, best_sim = None, 0.0
    for other in AudioEvent.query.filter(AudioEvent.id != ev.id, AudioEvent.source != "live",
                                         AudioEvent.fingerprint.isnot(None)).order_by(AudioEvent.id.desc()).limit(2000):
        sim = similarity(fp, other.fingerprint)
        if sim > best_sim:
            best, best_sim = other, sim
    if best is not None and best_sim >= 0.90:
        ev.near_duplicate_of_id, ev.near_duplicate_score = best.id, round(best_sim, 3)
        audit.notify("duplicate_file", f"{original_filename} looks like a copy of {best.audio_id} (similarity {best_sim:.2f})")

    # pre-processing + segmentation (timestamps are on the ORIGINAL timeline)
    clean, offset = preprocess_signal(audio.samples, audio.sample_rate, trim=True, denoise=s["noise_reduction"])
    segs = segment(clean, TARGET_SR, s["segment_seconds"], s["segment_hop_seconds"])
    gtm_audio = prepare_for_gtm(audio.samples, audio.sample_rate, GTM_SR)
    seg_dir = SEGMENT_DIR / ev.audio_id
    seg_dir.mkdir(parents=True, exist_ok=True)
    seg_len_gtm = int(s["segment_seconds"] * GTM_SR)
    for i, (st, en, _) in enumerate(segs):
        a, b = st + offset, en + offset
        chunk = gtm_audio[int(a * GTM_SR): int(a * GTM_SR) + seg_len_gtm]
        chunk = np.pad(chunk, (0, max(0, seg_len_gtm - len(chunk))))
        p = seg_dir / f"seg_{i:03d}.wav"
        sf.write(p, chunk, GTM_SR, subtype="PCM_16")
        ev.segments.append(Segment(index=i, start_s=round(a, 3), end_s=round(b, 3), audio_path=str(p),
                                   rms_db=round(_rms_db(segs[i][2]), 1)))

    # Python model (independent of GTM)
    try:
        scores = _python_scores([x for _, _, x in segs])
        for seg, sc in zip(ev.segments, scores):
            seg.python_scores = sc
            seg.python_prediction, seg.python_confidence = top_k(sc, 1)[0]
        ev.python_model_version = python_model().version
    except ModelNotAvailable as exc:
        audit.notify("model_failure", str(exc))
        ev.python_prediction = None
    except Exception as exc:  # model failure must not crash the app
        audit.notify("model_failure", f"Python model failed on {ev.audio_id}: {exc}")
        ev.python_prediction = None
    ev.status = "Classified" if ev.segments and ev.segments[0].python_scores else "Uploaded"

    make_images(ev, raw_mono, audio.sample_rate)
    g = gtm_info()
    ev.gtm_status = "pending" if g["available"] else "unavailable"
    ev.gtm_model_version = g.get("version")
    ev.processing_ms = int((time.time() - t0) * 1000)
    db.session.commit()
    audit.log("upload", "event", ev.audio_id, f"{original_filename} ({ev.duration:.1f}s, {len(segs)} segments)", user=user)
    if not g["available"]:
        finalize(ev)
    return ev, v.warnings, None


def submit_gtm(ev: AudioEvent, segment_scores: list | None, error: str | None = None):
    """Store GTM scores per segment (from the browser), then produce the final decision."""
    if error or not segment_scores:
        ev.gtm_status = "error" if error else "unavailable"
        if error:
            audit.notify("model_failure", f"GTM failed on {ev.audio_id}: {error}")
    else:
        for seg, sc in zip(ev.segments, segment_scores):
            sc = {c: float(sc.get(c, 0.0)) for c in CLASSES}
            seg.gtm_scores = sc
            seg.gtm_prediction, seg.gtm_confidence = top_k(sc, 1)[0]
        ev.gtm_status = "done"
    db.session.commit()
    return finalize(ev)


# ---------------------------------------------------------------------------
# Final decision
# ---------------------------------------------------------------------------
def _upload_repeated(ev: AudioEvent):
    def f(category):
        best = run = 0
        for seg in ev.segments:
            comb = seg.python_scores or {}
            if seg.gtm_scores:
                comb = {c: (comb.get(c, 0) + seg.gtm_scores.get(c, 0)) / 2 for c in CLASSES}
            top = top_k(comb, 1)[0][0] if comb else None
            run = run + 1 if top == category else 0
            best = max(best, run)
        return max(best, 1)
    return f


def _live_repeated(ev: AudioEvent, window_s: int):
    def f(category):
        since = ev.uploaded_at - timedelta(seconds=window_s)
        prev = AudioEvent.query.filter(AudioEvent.live_session_id == ev.live_session_id,
                                       AudioEvent.id != ev.id, AudioEvent.uploaded_at >= since) \
            .order_by(AudioEvent.uploaded_at.desc()).all()
        n = 1
        for p in prev:                       # consecutive windows, newest first
            if p.final_category == category:
                n += 1
            else:
                break
        return n
    return f


def finalize(ev: AudioEvent) -> AudioEvent:
    s = get_settings()
    rules = load_rules()
    py_seg = [seg.python_scores for seg in ev.segments if seg.python_scores]
    if not py_seg:
        ev.status = "Manual Review"
        ev.manual_review_required = True
        ev.review_reasons = ["Python model unavailable – no automatic classification"]
        ev.final_category = "Unknown"
        ev.alert_status = "No Alert"
        ev.severity = "Low"
        db.session.commit()
        return ev
    py, py_idx = aggregate_scores(py_seg, s["min_confidence"])
    gtm_seg = [seg.gtm_scores for seg in ev.segments if seg.gtm_scores]
    gtm, _ = aggregate_scores(gtm_seg, s["min_confidence"]) if gtm_seg else (None, -1)

    repeated = _live_repeated(ev, s["repeat_window_seconds"]) if ev.source == "live" else _upload_repeated(ev)
    d = decide(py, gtm, quality=ev.quality_label or "Acceptable", noise_db=ev.noise_level_db,
               repeated_for=repeated, rules=rules, settings=s)

    ev.python_scores, ev.gtm_scores, ev.python_segment_index = py, gtm, py_idx
    for k in ("python_prediction", "python_confidence", "python_margin", "gtm_prediction", "gtm_confidence",
              "gtm_margin", "class_match", "confidence_diff", "consistency_status", "combined_scores",
              "final_category", "final_confidence", "confidence_level", "overlap_detected", "overlap_classes",
              "uncertain", "uncertain_reasons", "repeated_count", "severity", "alert_status",
              "recommended_action", "manual_review_required", "review_reasons", "status"):
        setattr(ev, k, d[k])
    ev.decision_trace = d["trace"]
    ev.processed_at = now()

    if d["alert_status"] == "Alert Generated" and not ev.alerts:
        msg = f"{d['severity']} – {d['final_category']} detected ({d['final_confidence']:.0%})"
        if ev.source == "live":
            msg += " on live microphone"
        ev.alerts.append(Alert(category=d["final_category"], severity=d["severity"], message=msg,
                               recommended_action=d["recommended_action"], audience=d["audience"]))
    db.session.commit()
    audit.log("prediction", "event", ev.audio_id,
              f"py={d['python_prediction']} gtm={d['gtm_prediction']} final={d['final_category']} "
              f"sev={d['severity']} {d['alert_status']}", user=ev.user)
    if d["alert_status"] == "Alert Generated":
        audit.log("alert", "event", ev.audio_id, f"{d['severity']} {d['final_category']}", user=ev.user)
    audit.check_after_event()
    return ev


# ---------------------------------------------------------------------------
# Live microphone windows
# ---------------------------------------------------------------------------
def analyze_live_window(raw: bytes, session, user, gtm_scores: dict | None, gtm_error: str | None = None):
    """One live window. GTM scores were computed in the browser on the SAME window before upload."""
    t0 = time.time()
    s = get_settings()
    audio = load_bytes(raw, ".wav")
    v = validate_audio(audio, check_duration=False)
    q = analyze_quality(audio.samples, audio.sample_rate, check_duration=False)
    if q["is_silent"] or not v.ok:
        return {"silent": True, "quality": q, "message": "Silence – nothing to classify"}

    ev = AudioEvent(audio_id=new_audio_id("LIV"), user_id=user.id, source="live", live_session_id=session.id,
                    original_filename=f"live-window-{session.session_code}.wav", format="wav",
                    file_size=len(raw), duration=round(audio.duration, 3), sample_rate=audio.sample_rate,
                    channels=audio.channels, bit_depth=16, quality=q, quality_label=q["label"],
                    noise_level_db=q["noise_floor_dbfs"], status="Uploaded")
    db.session.add(ev)
    db.session.flush()
    seg_dir = SEGMENT_DIR / ev.audio_id
    seg_dir.mkdir(parents=True, exist_ok=True)
    wav_path = seg_dir / "seg_000.wav"
    sf.write(wav_path, prepare_for_gtm(audio.samples, audio.sample_rate, GTM_SR), GTM_SR, subtype="PCM_16")
    ev.stored_path = str(wav_path)
    ev.sha256 = hashlib.sha256(raw).hexdigest()

    clean, _ = preprocess_signal(audio.samples, audio.sample_rate, trim=False, denoise=s["noise_reduction"])
    from audio_preprocessing import pad_or_truncate
    clean = pad_or_truncate(clean, int(s["live_window_seconds"] * TARGET_SR))
    seg = Segment(index=0, start_s=0.0, end_s=round(audio.duration, 3), audio_path=str(wav_path),
                  rms_db=round(_rms_db(clean), 1))
    ev.segments.append(seg)
    try:
        sc = _python_scores([clean])[0]
        seg.python_scores = sc
        seg.python_prediction, seg.python_confidence = top_k(sc, 1)[0]
        ev.python_model_version = python_model().version
    except Exception as exc:
        audit.notify("model_failure", f"Python model failed on live window: {exc}")
    if gtm_scores:
        g = {c: float(gtm_scores.get(c, 0.0)) for c in CLASSES}
        seg.gtm_scores = g
        seg.gtm_prediction, seg.gtm_confidence = top_k(g, 1)[0]
        ev.gtm_status = "done"
    else:
        ev.gtm_status = "error" if gtm_error else "unavailable"
    ev.gtm_model_version = gtm_info().get("version")
    session.windows = (session.windows or 0) + 1
    db.session.commit()
    finalize(ev)
    ev.processing_ms = int((time.time() - t0) * 1000)
    db.session.commit()
    return {"silent": False, "event": ev}
