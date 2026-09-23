"""
Voluntary recording helper for classes that public datasets cover poorly
(Person Asking for Help, Panic Scream, Aggression).

Every speaker must agree to be recorded. Recordings go to
audio_dataset/raw/<Class>/ and are logged in annotations.csv.

    pip install sounddevice
    python scripts/record_samples.py --class "Person Asking for Help" --speaker S01 --count 20 \
        --environment indoor --device laptop-mic --distance 1

Only the defined safety phrases are used for the help class:
    "Help me", "Somebody help", "Please help", "Call for help", "Emergency"
"""
import argparse
import csv
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import soundfile as sf

from config.settings import RAW_DATASET_DIR, CLASSES

HELP_PHRASES = ["Help me", "Somebody help", "Please help", "Call for help", "Emergency"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--class", dest="cls", required=True, choices=CLASSES)
    ap.add_argument("--speaker", required=True, help="anonymous speaker code, e.g. S01")
    ap.add_argument("--count", type=int, default=10)
    ap.add_argument("--seconds", type=float, default=3.0)
    ap.add_argument("--environment", default="indoor")
    ap.add_argument("--device", default="unknown")
    ap.add_argument("--distance", default="1")
    ap.add_argument("--sr", type=int, default=44100)
    args = ap.parse_args()

    import sounddevice as sd  # imported here so the rest of the project does not need it

    out_dir = RAW_DATASET_DIR / args.cls
    out_dir.mkdir(parents=True, exist_ok=True)
    ann_path = RAW_DATASET_DIR / "annotations.csv"
    new = not ann_path.exists()
    fh = open(ann_path, "a", newline="", encoding="utf-8")
    w = csv.DictWriter(fh, fieldnames=["filename", "class_label", "source", "license", "environment", "device", "distance_m"])
    if new:
        w.writeheader()

    print("Consent reminder: the speaker has agreed to this recording.  Ctrl+C to stop.\n")
    for i in range(args.count):
        prompt = HELP_PHRASES[i % len(HELP_PHRASES)] if args.cls == "Person Asking for Help" else args.cls
        input(f"[{i+1}/{args.count}] Press Enter, then perform: \"{prompt}\" ")
        for c in (3, 2, 1):
            print(c, end=" ", flush=True); time.sleep(0.4)
        print("REC")
        audio = sd.rec(int(args.seconds * args.sr), samplerate=args.sr, channels=1, dtype="float32")
        sd.wait()
        name = f"rec_{args.speaker}_{datetime.now():%Y%m%d_%H%M%S}_{i:03d}.wav"
        sf.write(out_dir / name, audio, args.sr)
        w.writerow({"filename": name, "class_label": args.cls, "source": f"team-recording:{args.speaker}",
                    "license": "own recording (consent given)", "environment": args.environment,
                    "device": args.device, "distance_m": args.distance})
        fh.flush()
        print("  saved", name)
    fh.close()


if __name__ == "__main__":
    main()
