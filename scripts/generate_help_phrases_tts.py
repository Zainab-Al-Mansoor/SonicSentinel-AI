"""
Synthetic "Person Asking for Help" clips using the OFFLINE text-to-speech voices
installed on Windows (SAPI) via pyttsx3. The SRS allows synthetic samples
"where permitted". Mix these with real voluntary recordings – TTS alone will
not generalise well to real voices.

    pip install pyttsx3
    python scripts/generate_help_phrases_tts.py --per-phrase 20

Each clip varies the voice, speaking rate and volume.
"""
import argparse
import csv
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import RAW_DATASET_DIR

PHRASES = ["Help me", "Somebody help", "Please help", "Call for help", "Emergency",
           "Help me!", "Somebody help me", "Please help me", "Someone call for help", "Emergency, help"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-phrase", type=int, default=10)
    args = ap.parse_args()
    import pyttsx3

    engine = pyttsx3.init()
    voices = engine.getProperty("voices")
    out_dir = RAW_DATASET_DIR / "Person Asking for Help"
    out_dir.mkdir(parents=True, exist_ok=True)
    ann_path = RAW_DATASET_DIR / "annotations.csv"
    new = not ann_path.exists()
    rows = []
    n = 0
    for p_i, phrase in enumerate(PHRASES):
        for k in range(args.per_phrase):
            v = random.choice(voices)
            engine.setProperty("voice", v.id)
            engine.setProperty("rate", random.randint(130, 220))
            engine.setProperty("volume", random.uniform(0.6, 1.0))
            name = f"tts_help_{p_i:02d}_{k:03d}.wav"
            engine.save_to_file(phrase, str(out_dir / name))
            rows.append({"filename": name, "class_label": "Person Asking for Help",
                         "source": f"synthetic-tts:{v.name}", "license": "synthetic (generated)",
                         "environment": "synthetic", "device": "tts", "distance_m": ""})
            n += 1
    engine.runAndWait()
    with open(ann_path, "a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        if new:
            w.writeheader()
        w.writerows(rows)
    print(f"Generated {n} synthetic help-phrase clips in {out_dir}")


if __name__ == "__main__":
    main()
