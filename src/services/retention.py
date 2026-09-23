"""Data retention (SRS lxxx): delete old audio files and old event records."""
import shutil
from datetime import timedelta
from pathlib import Path

from config.settings import SEGMENT_DIR
from ..extensions import db
from ..models import AudioEvent, now
from .settings_service import get_settings


def _remove(p):
    try:
        if p and Path(p).exists():
            Path(p).unlink()
    except OSError:
        pass


def apply_retention() -> dict:
    s = get_settings()
    audio_cut = now() - timedelta(days=int(s["audio_retention_days"]))
    rec_cut = now() - timedelta(days=int(s["record_retention_days"]))
    files = recs = 0
    for ev in AudioEvent.query.filter(AudioEvent.uploaded_at < audio_cut, AudioEvent.stored_path.isnot(None)).all():
        _remove(ev.stored_path)
        shutil.rmtree(SEGMENT_DIR / ev.audio_id, ignore_errors=True)
        ev.stored_path = None
        for seg in ev.segments:
            seg.audio_path = None
        files += 1
    for ev in AudioEvent.query.filter(AudioEvent.uploaded_at < rec_cut).all():
        _remove(ev.waveform_path); _remove(ev.spectrogram_path)
        db.session.delete(ev)
        recs += 1
    db.session.commit()
    return {"audio_files_removed": files, "records_removed": recs}
