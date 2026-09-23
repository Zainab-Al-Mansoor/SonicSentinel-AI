"""
Audio-quality analysis: silence, clipping, excessive noise, low signal
strength, unsuitable duration, encoding problems and missing frames.

Result label: Good / Acceptable / Poor / Unusable
"""
import numpy as np

from config.settings import MIN_DURATION_S, MAX_DURATION_S


def _db(x: float) -> float:
    return float(20 * np.log10(max(x, 1e-10)))


def frame_rms(y: np.ndarray, frame: int = 1024, hop: int = 512) -> np.ndarray:
    if len(y) < frame:
        y = np.pad(y, (0, frame - len(y)))
    n = 1 + (len(y) - frame) // hop
    idx = np.arange(frame)[None, :] + hop * np.arange(n)[:, None]
    return np.sqrt(np.mean(y[idx] ** 2, axis=1) + 1e-12)


def analyze_quality(samples: np.ndarray, sr: int, *, check_duration: bool = True) -> dict:
    """`samples` = raw (un-normalised) audio, mono or (n, ch)."""
    y = samples.mean(axis=1) if samples.ndim == 2 else samples
    y = y.astype(np.float32)
    duration = len(y) / sr if sr else 0.0
    issues = []

    peak = float(np.max(np.abs(y))) if y.size else 0.0
    rms = float(np.sqrt(np.mean(y ** 2))) if y.size else 0.0
    rms_db = _db(rms)
    peak_db = _db(peak)

    frames_db = 20 * np.log10(frame_rms(y))
    silence_ratio = float(np.mean(frames_db < -60)) if frames_db.size else 1.0

    # Clipping: samples sitting at (or extremely near) full scale.
    clip_ratio = float(np.mean(np.abs(y) >= 0.999)) if y.size else 0.0

    # Noise floor / SNR estimate from quiet vs loud frames.
    if frames_db.size >= 4:
        noise_floor_db = float(np.percentile(frames_db, 10))
        signal_db = float(np.percentile(frames_db, 95))
    else:
        noise_floor_db = signal_db = rms_db
    snr_db = signal_db - noise_floor_db

    # Missing frames / dropouts: runs of exact digital zero (>= 20 ms) inside the clip.
    dropouts = 0
    if y.size:
        zero = (y == 0.0).astype(np.int8)
        min_run = int(0.02 * sr)
        interior = zero[int(0.05 * len(zero)): int(0.95 * len(zero))]
        if interior.size:
            edges = np.diff(np.concatenate(([0], interior, [0])))
            starts, ends = np.where(edges == 1)[0], np.where(edges == -1)[0]
            dropouts = int(np.sum((ends - starts) >= min_run))

    # Spectral flatness (0 = tonal, 1 = white noise), median over short frames.
    flatness = 0.0
    if y.size >= 2048:
        n = min(len(y) // 2048, 200)
        frames = y[: n * 2048].reshape(n, 2048) * np.hanning(2048)
        mag = np.abs(np.fft.rfft(frames, axis=1)) ** 2 + 1e-12   # power spectrum
        flat = np.exp(np.mean(np.log(mag), axis=1)) / np.mean(mag, axis=1)
        loud = frames.std(axis=1) > 1e-4
        flatness = float(np.median(flat[loud])) if loud.any() else 0.0

    encoding_problem = bool(not np.all(np.isfinite(y)))

    # ---- classification rules ------------------------------------------------
    label = "Good"

    def worsen(to):
        nonlocal label
        order = ["Good", "Acceptable", "Poor", "Unusable"]
        if order.index(to) > order.index(label):
            label = to

    if encoding_problem:
        issues.append("Encoding problem (non-finite samples)")
        worsen("Unusable")
    if peak < 1e-4 or silence_ratio > 0.97:
        issues.append("Silent or near-silent recording")
        worsen("Unusable")
    if check_duration and duration < MIN_DURATION_S:
        issues.append(f"Too short ({duration:.2f}s < {MIN_DURATION_S}s)")
        worsen("Unusable")
    if check_duration and duration > MAX_DURATION_S:
        issues.append(f"Too long ({duration:.0f}s > {MAX_DURATION_S}s)")
        worsen("Poor")
    if clip_ratio > 0.01:
        issues.append(f"Severe clipping ({clip_ratio*100:.1f}% of samples)")
        worsen("Poor")
    elif clip_ratio > 0.001:
        issues.append(f"Some clipping ({clip_ratio*100:.2f}% of samples)")
        worsen("Acceptable")
    if rms_db < -50:
        issues.append(f"Very low signal strength ({rms_db:.0f} dBFS)")
        worsen("Poor")
    elif rms_db < -38:
        issues.append(f"Low signal strength ({rms_db:.0f} dBFS)")
        worsen("Acceptable")
    # Low SNR alone is normal for continuous sounds (sirens, machinery hum).
    # Only treat it as "noise" when the spectrum is also noise-like (flat).
    noise_like = flatness > 0.30
    if snr_db < 6 and noise_like and label != "Unusable":
        issues.append(f"Excessive background noise (SNR ≈ {snr_db:.1f} dB, noise-like spectrum)")
        worsen("Poor" if rms_db < -30 else "Acceptable")
    elif snr_db < 12 and noise_like:
        issues.append(f"Noticeable background noise (SNR ≈ {snr_db:.1f} dB)")
        worsen("Acceptable")
    if dropouts:
        issues.append(f"{dropouts} missing-frame gap(s) detected")
        worsen("Poor" if dropouts > 2 else "Acceptable")

    return {
        "label": label,
        "issues": issues,
        "duration": round(duration, 3),
        "rms_dbfs": round(rms_db, 1),
        "peak_dbfs": round(peak_db, 1),
        "silence_ratio": round(silence_ratio, 3),
        "clipping_ratio": round(clip_ratio, 5),
        "noise_floor_dbfs": round(noise_floor_db, 1),
        "snr_db": round(snr_db, 1),
        "dropouts": dropouts,
        "spectral_flatness": round(flatness, 3),
        "is_silent": peak < 1e-4 or silence_ratio > 0.97,
    }
