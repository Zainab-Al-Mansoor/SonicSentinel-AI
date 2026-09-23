"""
Prepare Google Teachable Machine (GTM) training samples from the SAME
training recordings used by the Python model.

GTM audio projects work on 1-second samples at 44.1 kHz. This script cuts
every TRAIN-split recording (original + augmented) into 1-second windows,
keeps the windows that actually contain the event (energy check), and writes

    gtm_model/training_samples/<Class Name>/<AudioID>_w<k>.wav
    gtm_model/training_samples/gtm_samples_metadata.csv   (sample -> Audio ID)

Validation and test recordings are NEVER exported.

    python -m gtm_model.prepare_gtm_samples --max-per-class 400
"""
import argparse
import random
import shutil

import numpy as np
import pandas as pd
import soundfile as sf

from config.settings import BASE_DIR, DATASET_METADATA_CSV, GTM_SR, BACKGROUND_CLASS, CLASSES
from audio_preprocessing import load_audio, prepare_for_gtm

OUT = BASE_DIR / "gtm_model" / "training_samples"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-per-class", type=int, default=400,
                    help="GTM becomes slow with very many samples; 200-500 per class is plenty")
    ap.add_argument("--include-augmented", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
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

    rows = []
    for cls in CLASSES:
        sub = train[train["class_label"] == cls]
        windows = []
        for _, r in sub.iterrows():
            a = load_audio(BASE_DIR / r["path"])
            y = prepare_for_gtm(a.samples, a.sample_rate, GTM_SR)
            n = GTM_SR
            if len(y) < n:
                y = np.pad(y, (0, n - len(y)))
            starts = list(range(0, len(y) - n + 1, n // 2))   # 50 % overlap
            energies = [float(np.sqrt(np.mean(y[s:s + n] ** 2))) for s in starts]
            emax = max(energies) or 1.0
            for k, (s, e) in enumerate(zip(starts, energies)):
                if cls == BACKGROUND_CLASS or e >= 0.3 * emax:
                    windows.append((r["audio_id"], k, y[s:s + n]))
        random.shuffle(windows)
        windows = windows[: args.max_per_class]
        d = OUT / cls
        d.mkdir(parents=True, exist_ok=True)
        for aid, k, w in windows:
            name = f"{aid}_w{k}.wav"
            sf.write(d / name, w, GTM_SR, subtype="PCM_16")
            rows.append({"sample": f"{cls}/{name}", "audio_id": aid, "class_label": cls, "split": "train"})
        if args.playlist and windows:
            pl = OUT / "_playlists"
            pl.mkdir(exist_ok=True)
            gap = np.zeros(int(0.25 * GTM_SR), np.float32)
            sf.write(pl / f"{cls}.wav", np.concatenate([np.concatenate([w, gap]) for _, _, w in windows]), GTM_SR)
        print(f"{cls:<25} {len(windows):>4} GTM samples")
    pd.DataFrame(rows).to_csv(OUT / "gtm_samples_metadata.csv", index=False)
    print(f"\nDone. Upload each folder in {OUT} to the matching GTM class (see gtm_model/README.md).")


if __name__ == "__main__":
    main()
