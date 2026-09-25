"""
Extra hand-crafted features (feature set v2, step 3 of the feature-engineering plan).

They target the confusions seen in the confusion matrix and the look-alike report:

  impulsiveness   crest factor, attack / decay time, spectral flux, onset rate
                  -> gunshot, glass and metal impacts vs. steady sounds
  voice           pitch (F0) median / spread / range, harmonic and percussive energy share
                  -> panic scream vs. normal shouting, help phrases vs. ordinary speech
  frequency bands energy share in 7 bands + high/low ratio
                  -> glass (very high) vs. metal (lower), machinery hum vs. alarms
  PCEN            per-channel energy-normalised Mel spectrum (32 bands, mean + std)
                  -> more robust to background noise and recording level
  shape stats     percentiles of loudness and brightness, skew / kurtosis of the spectrum and envelope
"""
import numpy as np
import librosa
from scipy.stats import kurtosis, skew

from config.settings import TARGET_SR

N_FFT = 2048
HOP = 512
BANDS = [0, 250, 500, 1000, 2000, 4000, 8000, 11025]
N_PCEN = 32


def extra_names() -> list[str]:
    n = ["crest_log", "attack_s", "decay_s", "flux_mean", "flux_std", "flux_max", "onset_rate",
         "f0_median", "f0_std", "f0_range", "harmonic_share", "percussive_share"]
    n += [f"band{i}_share" for i in range(len(BANDS) - 1)] + ["band_high_low_log"]
    n += [f"pcen{i}_mean" for i in range(N_PCEN)] + [f"pcen{i}_std" for i in range(N_PCEN)]
    n += ["rmsdb_p10", "rmsdb_p50", "rmsdb_p90", "centroid_p10", "centroid_p50", "centroid_p90",
          "spec_skew", "spec_kurt", "env_skew", "env_kurt"]
    return n


def _safe(v, default=0.0):
    v = float(v)
    return v if np.isfinite(v) else default


def extract_extra(y: np.ndarray, sr: int = TARGET_SR) -> np.ndarray:
    y = y.astype(np.float32)
    if not np.any(y):
        y = y + 1e-6
    S = np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP))
    P = S ** 2
    rms = librosa.feature.rms(S=S, frame_length=N_FFT)[0] + 1e-9
    f = []

    # ---- impulsiveness ---------------------------------------------------
    peak = float(np.max(np.abs(y))) + 1e-9
    f.append(np.log(peak / (float(np.sqrt(np.mean(y ** 2))) + 1e-9)))
    i_pk = int(np.argmax(rms))
    thr = 0.1 * rms[i_pk]
    start = i_pk
    while start > 0 and rms[start - 1] >= thr:
        start -= 1
    end = i_pk
    while end < len(rms) - 1 and rms[end + 1] >= thr:
        end += 1
    f += [(i_pk - start) * HOP / sr, (end - i_pk) * HOP / sr]
    flux = np.sqrt(np.sum(np.clip(np.diff(S, axis=1), 0, None) ** 2, axis=0)) if S.shape[1] > 1 else np.zeros(1)
    flux = flux / (np.max(flux) + 1e-9) if np.max(flux) > 0 else flux
    f += [float(np.mean(flux)), float(np.std(flux)), float(np.max(flux))]
    onset_env = librosa.onset.onset_strength(S=librosa.amplitude_to_db(S, ref=np.max), sr=sr)
    onsets = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr, hop_length=HOP)
    f.append(len(onsets) / (len(y) / sr))

    # ---- voice -----------------------------------------------------------
    try:
        f0 = librosa.yin(y, fmin=80, fmax=1000, sr=sr, frame_length=N_FFT, hop_length=HOP)
        n = min(len(f0), len(rms))
        loud = rms[:n] >= 0.2 * rms.max()
        v = f0[:n][loud] if loud.any() else f0[:n]
        f += [float(np.median(v)) / 1000, float(np.std(v)) / 1000,
              float(np.percentile(v, 90) - np.percentile(v, 10)) / 1000]
    except Exception:
        f += [0.0, 0.0, 0.0]
    Sv = S[:372]                                   # 0–4 kHz: where voice harmonics live; small kernel keeps it fast
    H, Pc = librosa.decompose.hpss(Sv, kernel_size=9)
    tot = float(np.sum(Sv ** 2)) + 1e-9
    f += [float(np.sum(H ** 2)) / tot, float(np.sum(Pc ** 2)) / tot]

    # ---- frequency bands -------------------------------------------------
    freqs = librosa.fft_frequencies(sr=sr, n_fft=N_FFT)
    spec = P.mean(axis=1)
    etot = float(spec.sum()) + 1e-12
    shares = [float(spec[(freqs >= lo) & (freqs < hi)].sum()) / etot for lo, hi in zip(BANDS[:-1], BANDS[1:])]
    f += shares
    f.append(np.log((sum(shares[5:]) + 1e-6) / (sum(shares[:3]) + 1e-6)))

    # ---- PCEN ------------------------------------------------------------
    mel = librosa.feature.melspectrogram(S=P, sr=sr, n_mels=N_PCEN * 2)
    pc = librosa.pcen(mel * (2 ** 31), sr=sr, hop_length=HOP)
    pc = pc.reshape(N_PCEN, 2, -1).mean(axis=1)
    f += list(pc.mean(axis=1)) + list(pc.std(axis=1))

    # ---- shape statistics ------------------------------------------------
    rmsdb = 20 * np.log10(rms)
    cent = librosa.feature.spectral_centroid(S=S, sr=sr)[0] / 1000
    f += list(np.percentile(rmsdb, [10, 50, 90]) / 100) + list(np.percentile(cent, [10, 50, 90]))
    lspec = np.log(spec + 1e-12)
    f += [_safe(skew(lspec)), _safe(kurtosis(lspec)) / 10, _safe(skew(rms)), _safe(kurtosis(rms)) / 10]

    out = np.nan_to_num(np.asarray([_safe(x) for x in f], dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    return out
