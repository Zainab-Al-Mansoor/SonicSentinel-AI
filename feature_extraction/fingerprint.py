"""
Audio fingerprint for near-duplicate detection.

Catches re-encoded (MP3/OGG), trimmed and volume-adjusted copies:
  * spectral profile  – mean log-Mel spectrum (volume-independent after normalisation)
  * energy envelope   – 10 Hz RMS envelope, z-normalised, compared with sliding
                        cross-correlation so a trimmed copy still aligns.
"""
import numpy as np
import librosa

FP_SR = 11025


def compute_fingerprint(y: np.ndarray, sr: int) -> dict:
    y = librosa.resample(y.astype(np.float32), orig_sr=sr, target_sr=FP_SR) if sr != FP_SR else y
    m = np.max(np.abs(y)) or 1.0
    y = y / m
    mel = librosa.feature.melspectrogram(y=y, sr=FP_SR, n_mels=32, n_fft=1024, hop_length=512, fmax=5000)
    prof = librosa.power_to_db(mel.mean(axis=1) + 1e-10)
    prof = prof - prof.mean()
    hop = FP_SR // 10
    rms = librosa.feature.rms(y=y, frame_length=hop * 2, hop_length=hop)[0]
    env = np.log(rms + 1e-4)
    return {"profile": [round(float(v), 3) for v in prof],
            "envelope": [round(float(v), 3) for v in env[:3000]],
            "duration": round(len(y) / FP_SR, 2)}


def _cos(a, b):
    a, b = np.asarray(a), np.asarray(b)
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / d) if d else 0.0


def _best_xcorr(a, b) -> float:
    """Max normalised cross-correlation of the shorter envelope slid over the longer one."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) > len(b):
        a, b = b, a
    if len(a) < 5:
        return 0.0
    a = (a - a.mean()) / (a.std() + 1e-9)
    best = -1.0
    n = len(a)
    for off in range(0, len(b) - n + 1):
        w = b[off:off + n]
        w = (w - w.mean()) / (w.std() + 1e-9)
        best = max(best, float(np.dot(a, w) / n))
    return best


def similarity(fp1: dict, fp2: dict) -> float:
    """0..1 similarity. >= 0.9 is treated as a near-duplicate."""
    if not fp1 or not fp2:
        return 0.0
    spec = _cos(fp1["profile"], fp2["profile"])
    if spec < 0.85:
        return max(0.0, spec * 0.5)       # clearly different spectrum – skip expensive check
    shorter = min(fp1["duration"], fp2["duration"]) / max(fp1["duration"], fp2["duration"], 1e-6)
    if shorter < 0.3:                      # far too different in length
        return spec * 0.6
    env = _best_xcorr(fp1["envelope"], fp2["envelope"])
    return round(0.3 * spec + 0.7 * max(env, 0.0), 4)
