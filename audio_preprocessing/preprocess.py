"""
Audio pre-processing used identically for training and for inference:

    resample -> mono -> amplitude normalisation -> silence trimming
    -> noise reduction -> fixed-duration segmentation -> padding / truncation
"""
import numpy as np
import librosa

from config.settings import TARGET_SR

try:
    import noisereduce as nr
except ImportError:  # noise reduction is optional
    nr = None


def to_mono(samples: np.ndarray) -> np.ndarray:
    """Stereo (n, ch) -> mono (n,) by averaging channels."""
    if samples.ndim == 2:
        return samples.mean(axis=1).astype(np.float32)
    return samples.astype(np.float32)


def resample(y: np.ndarray, orig_sr: int, target_sr: int = TARGET_SR) -> np.ndarray:
    if orig_sr == target_sr:
        return y
    return librosa.resample(y, orig_sr=orig_sr, target_sr=target_sr, res_type="soxr_hq").astype(np.float32)


def peak_normalize(y: np.ndarray, peak: float = 0.95) -> np.ndarray:
    """Scale so the loudest sample is at `peak` (removes recording-volume differences)."""
    m = float(np.max(np.abs(y))) if y.size else 0.0
    if m < 1e-8:
        return y
    return (y / m * peak).astype(np.float32)


def trim_silence(y: np.ndarray, top_db: float = 40.0) -> tuple[np.ndarray, float]:
    """Remove leading/trailing silence. Returns (trimmed, offset_seconds_of_start)."""
    if y.size == 0:
        return y, 0.0
    trimmed, idx = librosa.effects.trim(y, top_db=top_db)
    if trimmed.size < 0.25 * TARGET_SR:      # never trim a clip down to almost nothing
        return y, 0.0
    return trimmed, idx[0] / TARGET_SR


def reduce_noise(y: np.ndarray, sr: int = TARGET_SR) -> np.ndarray:
    """Spectral-gating noise reduction (stationary). Mild setting keeps event transients."""
    if nr is None or y.size < sr // 4:
        return y
    try:
        return nr.reduce_noise(y=y, sr=sr, stationary=True, prop_decrease=0.75).astype(np.float32)
    except Exception:
        return y


def pad_or_truncate(y: np.ndarray, length: int) -> np.ndarray:
    if len(y) >= length:
        return y[:length]
    return np.pad(y, (0, length - len(y)))


def segment(y: np.ndarray, sr: int, seg_seconds: float, hop_seconds: float) -> list[tuple[float, float, np.ndarray]]:
    """
    Split into fixed-duration segments. Returns [(start_s, end_s, samples)].
    The last partial segment is zero-padded; a clip shorter than one segment
    becomes a single padded segment.
    """
    seg_len = int(round(seg_seconds * sr))
    hop = max(1, int(round(hop_seconds * sr)))
    out = []
    if len(y) <= seg_len:
        return [(0.0, len(y) / sr, pad_or_truncate(y, seg_len))]
    start = 0
    while start < len(y):
        chunk = y[start:start + seg_len]
        # skip a tiny trailing remainder that is mostly covered by the previous segment
        if out and len(chunk) < 0.5 * seg_len:
            break
        out.append((start / sr, min(len(y), start + seg_len) / sr, pad_or_truncate(chunk, seg_len)))
        start += hop
    return out


def preprocess_signal(samples: np.ndarray, orig_sr: int, *, trim: bool = True,
                      denoise: bool = True) -> tuple[np.ndarray, float]:
    """Full clean-up chain before segmentation. Returns (clean_signal@TARGET_SR, trim_offset_s)."""
    y = to_mono(samples)
    y = resample(y, orig_sr, TARGET_SR)
    y = peak_normalize(y)
    offset = 0.0
    if trim:
        y, offset = trim_silence(y)
    if denoise:
        y = reduce_noise(y, TARGET_SR)
        y = peak_normalize(y)
    return y, offset


def prepare_for_gtm(samples: np.ndarray, orig_sr: int, gtm_sr: int = 44100) -> np.ndarray:
    """
    Audio sent to Google Teachable Machine: mono, 44.1 kHz, peak-normalised.
    No noise reduction – GTM was trained on raw clips and has its own
    Background Noise class.
    """
    y = to_mono(samples)
    y = resample(y, orig_sr, gtm_sr)
    return peak_normalize(y)
