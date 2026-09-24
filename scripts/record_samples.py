"""
Voluntary recording helper for classes that public datasets cover poorly
(Person Asking for Help, Panic Scream, Aggression).

Every speaker must agree to be recorded. Recordings go to
audio_dataset/raw/<Class>/ and are logged in annotations.csv.

    pip install sounddevice
    python scripts/record_samples.py --list-devices          # which microphone is used?
    python scripts/record_samples.py --class "Person Asking for Help" --speaker S01 --count 20 \
        --environment indoor --device laptop-mic --distance 1 [--input 2]

Clips that are almost silent (the microphone recorded nothing) are not saved – you are asked to repeat them.

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
    ap.add_argument("--class", dest="cls", choices=CLASSES)
    ap.add_argument("--speaker", help="anonymous speaker code, e.g. S01")
    ap.add_argument("--count", type=int, default=10)
    ap.add_argument("--seconds", type=float, default=3.0)
    ap.add_argument("--environment", default="indoor")
    ap.add_argument("--device", default="unknown")
    ap.add_argument("--distance", default="1")
    ap.add_argument("--sr", type=int, default=44100)
    ap.add_argument("--input", type=int, default=None, help="microphone device number (see --list-devices)")
    ap.add_argument("--list-devices", action="store_true", help="show microphones and exit")
    ap.add_argument("--min-level", type=float, default=-45.0,
                    help="clips whose peak is below this dBFS are not saved (the mic probably recorded nothing)")
    args = ap.parse_args()

    import numpy as np
    import sounddevice as sd  # imported here so the rest of the project does not need it

    if args.list_devices:
        for i, d in enumerate(sd.query_devices()):
            if d["max_input_channels"] > 0:
                mark = "  <- default" if i == sd.default.device[0] else ""
                print(f"{i:>3}  {d['name']}{mark}")
        return
    if not args.cls or not args.speaker:
        ap.error("--class and --speaker are required when recording")
    if args.input is not None:
        sd.default.device = (args.input, sd.default.device[1])
    print(f"Microphone: {sd.query_devices(sd.default.device[0])['name']}")

    out_dir = RAW_DATASET_DIR / args.cls
    out_dir.mkdir(parents=True, exist_ok=True)
    ann_path = RAW_DATASET_DIR / "annotations.csv"
    new = not ann_path.exists()
    fh = open(ann_path, "a", newline="", encoding="utf-8")
    w = csv.DictWriter(fh, fieldnames=["filename", "class_label", "source", "license", "environment", "device", "distance_m"])
    if new:
        w.writeheader()

    print("Consent reminder: the speaker has agreed to this recording.  Ctrl+C to stop.\n")
    i = 0
    while i < args.count:
        prompt = HELP_PHRASES[i % len(HELP_PHRASES)] if args.cls == "Person Asking for Help" else args.cls
        input(f"[{i+1}/{args.count}] Press Enter, then perform: \"{prompt}\" ")
        for c in (3, 2, 1):
            print(c, end=" ", flush=True); time.sleep(0.4)
        print("REC")
        audio = sd.rec(int(args.seconds * args.sr), samplerate=args.sr, channels=1, dtype="float32")
        sd.wait()
        peak_db = 20 * np.log10(float(np.max(np.abs(audio))) + 1e-9)
        if peak_db < args.min_level:
            print(f"  !! too quiet (peak {peak_db:.0f} dBFS) – NOT saved. Check the microphone "
                  f"(Windows sound settings, or pick another one with --list-devices / --input N) and try again.")
            continue
        name = f"rec_{args.speaker}_{datetime.now():%Y%m%d_%H%M%S}_{i:03d}.wav"
        sf.write(out_dir / name, audio, args.sr)
        w.writerow({"filename": name, "class_label": args.cls, "source": f"team-recording:{args.speaker}",
                    "license": "own recording (consent given)", "environment": args.environment,
                    "device": args.device, "distance_m": args.distance})
        fh.flush()
        print(f"  saved {name}  (peak {peak_db:.0f} dBFS)")
        i += 1
    fh.close()


if __name__ == "__main__":
    main()
