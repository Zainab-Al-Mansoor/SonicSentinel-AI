"""
Acoustic feature extraction for the Python classification model.

Each fixed-duration segment is turned into ONE fixed-length vector made of
summary statistics (mean / std / max) of frame-level features:

  MFCC (40) mean+std, delta-MFCC (40) mean, log-Mel spectrogram (64) mean+std,
  chroma (12) mean+std, spectral contrast (7) mean, zero-crossing rate,
  RMS energy, spectral centroid, bandwidth, roll-off, flatness, onset strength,
  tempo.
"""
import numpy as np
import librosa

from config.settings import TARGET_SR, N_MFCC, N_MELS

FEATURE_VERSION = "v1"   # bump when feature code changes -> invalidates cached features
N_FFT = 2048
HOP = 512


def feature_names() -> list[str]:
    names = []
    names += [f"mfcc{i}_mean" for i in range(N_MFCC)] + [f"mfcc{i}_std" for i in range(N_MFCC)]
    names += [f"dmfcc{i}_mean" for i in range(N_MFCC)]
    names += [f"mel{i}_mean" for i in range(N_MELS)] + [f"mel{i}_std" for i in range(N_MELS)]
    names += [f"chroma{i}_mean" for i in range(12)] + [f"chroma{i}_std" for i in range(12)]
    names += [f"contrast{i}_mean" for i in range(7)]
    for f in ("zcr", "rms", "centroid", "bandwidth", "rolloff", "onset"):
        names += [f"{f}_mean", f"{f}_std", f"{f}_max"]
    names += ["flatness_mean", "tempo"]
    return names


def extract_features(y: np.ndarray, sr: int = TARGET_SR) -> np.ndarray:
    """y: mono float32 segment (already pre-processed). Returns 1-D float32 vector."""
    y = y.astype(np.float32)
    if not np.any(y):
        y = y + 1e-6  # avoid NaNs on digital silence

    S = np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP)) ** 2
    mel = librosa.feature.melspectrogram(S=S, sr=sr, n_mels=N_MELS)
    log_mel = librosa.power_to_db(mel, ref=np.max)
    mfcc = librosa.feature.mfcc(S=log_mel, n_mfcc=N_MFCC)
    n_frames = mfcc.shape[1]
    if n_frames >= 3:
        width = 9 if n_frames >= 9 else (n_frames if n_frames % 2 else n_frames - 1)
        dmfcc = librosa.feature.delta(mfcc, width=width)
    else:
        dmfcc = np.zeros_like(mfcc)
    chroma = librosa.feature.chroma_stft(S=S, sr=sr, tuning=0.0)
    contrast = librosa.feature.spectral_contrast(S=np.sqrt(S), sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y, frame_length=N_FFT, hop_length=HOP)[0]
    rms = librosa.feature.rms(y=y, frame_length=N_FFT, hop_length=HOP)[0]
    centroid = librosa.feature.spectral_centroid(S=np.sqrt(S), sr=sr)[0]
    bandwidth = librosa.feature.spectral_bandwidth(S=np.sqrt(S), sr=sr)[0]
    rolloff = librosa.feature.spectral_rolloff(S=np.sqrt(S), sr=sr)[0]
    flatness = librosa.feature.spectral_flatness(S=np.sqrt(S))[0]
    onset = librosa.onset.onset_strength(y=y, sr=sr, hop_length=HOP)
    try:
        tempo = float(np.atleast_1d(librosa.feature.tempo(onset_envelope=onset, sr=sr, hop_length=HOP))[0])
    except Exception:
        tempo = 0.0

    def stats3(v):
        return [float(np.mean(v)), float(np.std(v)), float(np.max(v))]

    vec = []
    vec += list(mfcc.mean(axis=1)) + list(mfcc.std(axis=1))
    vec += list(dmfcc.mean(axis=1))
    vec += list(log_mel.mean(axis=1)) + list(log_mel.std(axis=1))
    vec += list(chroma.mean(axis=1)) + list(chroma.std(axis=1))
    vec += list(contrast.mean(axis=1))
    for v in (zcr, rms, centroid / 1000.0, bandwidth / 1000.0, rolloff / 1000.0, onset):
        vec += stats3(v)
    vec += [float(np.mean(flatness)), tempo / 100.0]
    out = np.nan_to_num(np.asarray(vec, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    return out


def extract_batch(segments: list[np.ndarray], sr: int = TARGET_SR) -> np.ndarray:
    return np.vstack([extract_features(s, sr) for s in segments]) if segments else np.zeros((0, len(feature_names())))
