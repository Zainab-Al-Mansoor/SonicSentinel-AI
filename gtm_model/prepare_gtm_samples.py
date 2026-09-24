"""
Prepare Google Teachable Machine (GTM) training samples from the SAME
training recordings used by the Python model.

GTM audio projects work on 1-second samples at 44.1 kHz. This script cuts
every TRAIN-split recording into 1-second windows, keeps the windows that
actually contain the event (energy check), and writes

    gtm_model/training_samples/<Class Name>/<AudioID>_w<k>.wav
    gtm_model/training_samples/gtm_samples_metadata.csv   (sample -> Audio ID)
    gtm_model/training_samples/gtm_sample_counts.csv

Validation and test recordings are NEVER exported.

    python -m gtm_model.prepare_gtm_samples --max-per-class 400 --zip

--zip (recommended) also writes one Teachable Machine sample archive per class:

    gtm_model/training_samples/_tm_upload/<Class Name>.zip

Teachable Machine's audio "Upload" button only accepts archives in its own
format (samples.json + .webm audio). These zips use exactly that format, and the
spectrogram of every sample is computed exactly like Teachable Machine's own
microphone recorder (Web Audio AnalyserNode: 44.1 kHz, FFT 2048, Blackman
window, smoothing 0, one frame every 1024 samples, first 232 bins, 43 frames).
So the samples go into GTM digitally: no speaker -> microphone recording,
and hundreds of samples per class instead of a few dozen.

--playlist writes one long WAV per class instead (old method: play it into
GTM's microphone recorder).
"""
import argparse
import json
import random
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

from config.settings import BASE_DIR, DATASET_METADATA_CSV, GTM_SR, BACKGROUND_CLASS, CLASSES
from audio_preprocessing import load_audio, prepare_for_gtm

OUT = BASE_DIR / "gtm_model" / "training_samples"

# Teachable Machine / speech-commands BROWSER_FFT settings (same as static/js/gtm.js)
TM_FFT = 2048
TM_HOP = 1024
TM_FRAMES = 43
TM_COLS = 232
_n = np.arange(TM_FFT)
_BLACKMAN = 0.42 - 0.5 * np.cos(2 * np.pi * _n / TM_FFT) + 0.08 * np.cos(4 * np.pi * _n / TM_FFT)


def tm_frequency_frames(y: np.ndarray, start: int = 0) -> np.ndarray:
    """43 x 232 dB spectrogram, identical to AnalyserNode.getFloatFrequencyData() frames
    (and to spectrogram() in static/js/gtm.js)."""
    out = np.empty((TM_FRAMES, TM_COLS), np.float32)
    for f in range(TM_FRAMES):
        off = start + f * TM_HOP
        buf = np.zeros(TM_FFT, np.float64)
        seg = y[off: off + TM_FFT]
        buf[: len(seg)] = seg
        mag = np.abs(np.fft.rfft(buf * _BLACKMAN)) / TM_FFT
        out[f] = 20 * np.log10(np.maximum(mag[:TM_COLS], 1e-10))
    return out


def _encode_webm(wav_path: Path, webm_path: Path) -> bool:
    """Opus/WebM audio for playback inside Teachable Machine (needs FFmpeg)."""
    try:
        r = subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(wav_path),
                            "-c:a", "libopus", "-b:a", "48k", str(webm_path)],
                           capture_output=True, timeout=600)
        return r.returncode == 0 and webm_path.exists() and webm_path.stat().st_size > 0
    except (OSError, subprocess.SubprocessError):
        return False


def write_tm_zip(zip_path: Path, items: list, sr: int = GTM_SR) -> int:
    """items: list of (full_signal, start_sample). Writes a Teachable Machine sample archive.

    Format (as read by teachablemachine.withgoogle.com):
      samples.json : [{frequencyFrames: [[232 floats] x 43], blob: null, blobFilePath: "sample-1.webm",
                       startTime, endTime, recordingDuration}, ...]
      sample-1.webm: the audio of all samples one after another (used only for playback)
    """
    span = TM_FRAMES * TM_HOP
    samples, audio = [], []
    t = 0.0
    for y, s in items:
        frames = tm_frequency_frames(y, s)
        clip = np.zeros(span, np.float32)
        seg = y[s: s + span]
        clip[: len(seg)] = seg
        audio.append(clip)
        dur = span / sr
        samples.append({"frequencyFrames": np.round(frames.astype(np.float64), 1).tolist(), "blob": None,
                        "blobFilePath": "sample-1.webm",
                        "startTime": round(t, 4), "endTime": round(t + dur, 4)})
        t += dur
    for smp in samples:
        smp["recordingDuration"] = round(t, 4)

    with tempfile.TemporaryDirectory() as tmp:
        wav = Path(tmp) / "all.wav"
        webm = Path(tmp) / "all.webm"
        sf.write(wav, np.concatenate(audio), sr, subtype="PCM_16")
        audio_file = webm if _encode_webm(wav, webm) else wav   # WAV bytes still play in Chrome
        zip_path.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
            z.write(audio_file, "sample-1.webm")
            z.writestr("samples.json", json.dumps(samples, separators=(",", ":")))
    return len(samples)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-per-class", type=int, default=400,
                    help="GTM becomes slow with very many samples; 200-500 per class is plenty")
    ap.add_argument("--include-augmented", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--zip", action="store_true",
                    help="write Teachable Machine upload archives to training_samples/_tm_upload/ (recommended)")
    ap.add_argument("--playlist", action="store_true",
                    help="also write one long WAV per class (for recording through GTM's microphone input)")
    args = ap.parse_args()
    random.seed(args.seed)
    if OUT.exists():
        shutil.rmtree(OUT)          # always rebuild from the current TRAIN split

    meta = pd.read_csv(DATASET_METADATA_CSV)
    train = meta[meta["split"] == "train"]
    if not args.include_augmented:
        train = train[train["is_augmented"] == 0]

    rows, summary = [], []
    for cls in CLASSES:
        sub = train[train["class_label"] == cls]
        windows = []
        for _, r in sub.iterrows():
            try:
                a = load_audio(BASE_DIR / r["path"])
            except Exception as e:          # unreadable file: skip, keep going
                print(f"  skipped {r['path']}: {e}")
                continue
            y = prepare_for_gtm(a.samples, a.sample_rate, GTM_SR).astype(np.float32)
            n = GTM_SR
            if len(y) < n:
                y = np.pad(y, (0, n - len(y)))
            starts = list(range(0, len(y) - n + 1, n // 2))   # 50 % overlap
            energies = [float(np.sqrt(np.mean(y[s:s + n] ** 2))) for s in starts]
            emax = max(energies) or 1.0
            for k, (s, e) in enumerate(zip(starts, energies)):
                if cls == BACKGROUND_CLASS or e >= 0.3 * emax:
                    windows.append((r["audio_id"], k, y, s))
        random.shuffle(windows)
        windows = windows[: args.max_per_class]
        d = OUT / cls
        d.mkdir(parents=True, exist_ok=True)
        for aid, k, y, s in windows:
            name = f"{aid}_w{k}.wav"
            sf.write(d / name, y[s:s + GTM_SR], GTM_SR, subtype="PCM_16")
            rows.append({"sample": f"{cls}/{name}", "audio_id": aid, "class_label": cls, "split": "train"})
        if args.playlist and windows:
            pl = OUT / "_playlists"
            pl.mkdir(exist_ok=True)
            gap = np.zeros(int(0.25 * GTM_SR), np.float32)
            sf.write(pl / f"{cls}.wav",
                     np.concatenate([np.concatenate([y[s:s + GTM_SR], gap]) for _, _, y, s in windows]), GTM_SR)
        n_zip = 0
        if args.zip and windows:
            n_zip = write_tm_zip(OUT / "_tm_upload" / f"{cls}.zip", [(y, s) for _, _, y, s in windows])
        summary.append({"class": cls, "samples": len(windows), "recordings": len({w[0] for w in windows})})
        print(f"{cls:<25} {len(windows):>4} GTM samples" + (f"  -> _tm_upload/{cls}.zip" if n_zip else ""))
    pd.DataFrame(rows).to_csv(OUT / "gtm_samples_metadata.csv", index=False)
    pd.DataFrame(summary).to_csv(OUT / "gtm_sample_counts.csv", index=False)
    if args.zip:
        print(f"\nDone. In Teachable Machine, for every class: Upload -> choose "
              f"{OUT / '_tm_upload'} / <Class>.zip  (see gtm_model/README.md).")
    else:
        print(f"\nDone. Samples in {OUT}. Tip: add --zip to create Teachable Machine upload files.")


if __name__ == "__main__":
    main()
