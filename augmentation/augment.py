"""
Training-audio augmentation (TRAINING SPLIT ONLY).

Augmented clips keep the parent's Audio ID in `parent_audio_id`, stay in the
same split, and are never counted as unique original clips.

Usage (after scripts/build_dataset.py):
    python -m augmentation.augment --per-clip 2
"""
import argparse
import random

import numpy as np
import pandas as pd
import soundfile as sf
import librosa
from scipy.signal import fftconvolve, butter, sosfilt

from config.progress import progress as _progress
from config.settings import DATASET_METADATA_CSV, BASE_DIR, BACKGROUND_CLASS


def add_noise(y, snr_db=None, noise=None):
    snr_db = snr_db if snr_db is not None else random.uniform(5, 25)
    if noise is None or len(noise) == 0:
        noise = np.random.randn(len(y)).astype(np.float32)
    else:
        noise = np.resize(noise, len(y))
    p_sig = np.mean(y ** 2) + 1e-12
    p_noise = np.mean(noise ** 2) + 1e-12
    k = np.sqrt(p_sig / (p_noise * 10 ** (snr_db / 10)))
    return (y + k * noise).astype(np.float32)


def time_shift(y, max_frac=0.3):
    return np.roll(y, int(random.uniform(-max_frac, max_frac) * len(y))).astype(np.float32)


def pitch_shift(y, sr, steps=None):
    return librosa.effects.pitch_shift(y, sr=sr, n_steps=steps if steps is not None else random.uniform(-2, 2)).astype(np.float32)


def time_stretch(y, rate=None):
    return librosa.effects.time_stretch(y, rate=rate or random.uniform(0.85, 1.15)).astype(np.float32)


def volume(y, gain_db=None):
    return (y * 10 ** ((gain_db if gain_db is not None else random.uniform(-12, 6)) / 20)).astype(np.float32)


def reverb(y, sr, rt60=None):
    """Limited synthetic room reverberation (exponentially decaying noise IR)."""
    rt60 = rt60 or random.uniform(0.15, 0.6)
    n = int(rt60 * sr)
    ir = np.random.randn(n) * np.exp(-6.9 * np.arange(n) / n)
    ir[0] = 1.0
    wet = fftconvolve(y, ir)[: len(y)]
    wet = wet / (np.max(np.abs(wet)) + 1e-9) * (np.max(np.abs(y)) + 1e-9)
    return (0.6 * y + 0.4 * wet).astype(np.float32)


def distance(y, sr, metres=None):
    """Far-away source: quieter + high frequencies absorbed (low-pass)."""
    metres = metres or random.uniform(5, 30)
    cutoff = max(1500, 9000 - metres * 200)
    sos = butter(4, cutoff, btype="low", fs=sr, output="sos")
    return (sosfilt(sos, y) / max(1.0, metres / 5)).astype(np.float32)


def device(y, sr):
    """Cheap-microphone simulation: band-pass 300 Hz–4 kHz + light saturation."""
    sos = butter(3, [300, min(4000, sr / 2 - 100)], btype="band", fs=sr, output="sos")
    return np.tanh(2.0 * sosfilt(sos, y)).astype(np.float32) / 2.0


def random_augment(y, sr, bg_pool=None):
    ops = [
        ("noise", lambda a: add_noise(a, noise=random.choice(bg_pool) if bg_pool else None)),
        ("shift", time_shift),
        ("pitch", lambda a: pitch_shift(a, sr)),
        ("stretch", time_stretch),
        ("volume", volume),
        ("reverb", lambda a: reverb(a, sr)),
        ("distance", lambda a: distance(a, sr)),
        ("device", lambda a: device(a, sr)),
    ]
    chosen = random.sample(ops, k=random.randint(1, 3))
    out = y
    for _, f in chosen:
        out = f(out)
    peak = np.max(np.abs(out)) or 1.0
    return (out / peak * 0.9).astype(np.float32), "+".join(n for n, _ in chosen)


def main():
    ap = argparse.ArgumentParser(description="Augment TRAINING clips only")
    ap.add_argument("--per-clip", type=int, default=1, help="augmented copies per original training clip")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    random.seed(args.seed); np.random.seed(args.seed)

    meta = pd.read_csv(DATASET_METADATA_CSV)
    meta = meta[meta["is_augmented"] == 0]  # never augment an augmented clip
    train = meta[meta["split"] == "train"]

    # real background recordings make the most realistic noise
    bg_pool = []
    for p in train[train["class_label"] == BACKGROUND_CLASS]["path"].head(100):
        a, _ = librosa.load(BASE_DIR / p, sr=None, mono=True)
        bg_pool.append(a)

    new_rows = []
    for _, row in train.iterrows():
        src = BASE_DIR / row["path"]
        y, sr = librosa.load(src, sr=None, mono=True)
        for k in range(args.per_clip):
            aug, ops = random_augment(y, sr, bg_pool)
            aug_id = f"{row['audio_id']}-A{k+1}"
            out = src.with_name(f"{aug_id}.wav")
            sf.write(out, aug, sr)
            r = row.to_dict()
            r.update({"audio_id": aug_id, "filename": out.name, "is_augmented": 1,
                      "parent_audio_id": row["audio_id"], "augmentation": ops,
                      "duration": round(len(aug) / sr, 3), "sha256": "",
                      "path": str(out.relative_to(BASE_DIR)).replace("\\", "/")})
            new_rows.append(r)
        _progress(f"augmented {row['audio_id']}")

    full = pd.read_csv(DATASET_METADATA_CSV)
    full = full[~full["audio_id"].isin([r["audio_id"] for r in new_rows])]
    full = pd.concat([full, pd.DataFrame(new_rows)], ignore_index=True)
    full.to_csv(DATASET_METADATA_CSV, index=False)
    print(f"\nCreated {len(new_rows)} augmented training clips (all in the TRAIN split).")


if __name__ == "__main__":
    main()
