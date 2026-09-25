"""
Hidden-test readiness check (SRS 1.8 – competition integrity, item 9).

Evaluators test with UNSEEN recordings that may be noisy, echoing, quiet, recorded on other
devices, far away, cut in the middle, overlapping, confusable or re-encoded. This script
simulates each of those conditions on the unseen TEST split and runs the real Python model
through the SAME pipeline the web app uses (pre-processing -> 2-s segments -> 299 features
-> model -> clip aggregation -> quality analysis -> overlap check).

    python scripts/hidden_test_robustness.py                 # 30 test clips per class (~15 min)
    python scripts/hidden_test_robustness.py --per-class 10  # quick run

Results: reports/hidden_test_robustness.csv and reports/hidden_test_robustness.md
"""
import argparse
import random
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import soundfile as sf

from config.settings import (BACKGROUND_CLASS, CLASSES, DATASET_METADATA_CSV, DEFAULT_RUNTIME_SETTINGS,
                             TARGET_SR)
from audio_preprocessing import load_audio
from audio_preprocessing.preprocess import preprocess_signal, segment, to_mono, resample
from audio_preprocessing.quality import analyze_quality
from augmentation.augment import add_noise, reverb, volume, distance, device
from feature_extraction.features import extract_batch
from python_models.inference import PythonSoundModel, aggregate_scores

S = DEFAULT_RUNTIME_SETTINGS

# SRS "similar classes" pairs, as represented in our label set
CONFUSABLE = [("Gunshot", "Background Noise", "gunshot vs fireworks / door slam (Background)"),
              ("Panic Scream", "Background Noise", "scream vs normal shouting / voices (Background)"),
              ("Aggression", "Background Noise", "aggression vs normal conversation (Background)"),
              ("Glass Breaking", "Background Noise", "glass vs other impacts (Background)"),
              ("Alarm or Siren", "Vehicle Horn", "alarm vs horn"),
              ("Machinery Fault", "Background Noise", "faulty vs normal machinery (Background)"),
              ("Person Asking for Help", "Background Noise", "help phrase vs normal speech (Background)")]


def classify(y: np.ndarray, sr: int, model: PythonSoundModel) -> tuple[str, dict, str, bool]:
    """Same steps as src/services/analysis.analyze_upload (Python side)."""
    q = analyze_quality(y, sr)
    clean, _ = preprocess_signal(y, sr, trim=True, denoise=S["noise_reduction"])
    segs = segment(clean, TARGET_SR, S["segment_seconds"], S["segment_hop_seconds"])
    scores = model.segment_scores([s for _, _, s in segs])
    clip, _ = aggregate_scores(scores, S["min_confidence"])
    pred = max(clip, key=clip.get)
    strong = [c for c, v in clip.items() if c != BACKGROUND_CLASS and v >= S["overlap_threshold"]]
    return pred, clip, q.get("quality_label", q.get("label", "")), len(strong) >= 2


def reencode_mp3(y: np.ndarray, sr: int, kbps: int = 64) -> np.ndarray:
    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg not found")
    with tempfile.TemporaryDirectory() as d:
        w, m = Path(d) / "a.wav", Path(d) / "a.mp3"
        sf.write(w, y, sr, subtype="PCM_16")
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(w), "-b:a", f"{kbps}k", str(m)], check=True)
        a = load_audio(m)
        return to_mono(a.samples).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=30, help="test clips per class (all if fewer)")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    random.seed(args.seed); np.random.seed(args.seed)

    model = PythonSoundModel()
    meta = pd.read_csv(DATASET_METADATA_CSV)
    test = meta[(meta["split"] == "test") & (meta["is_augmented"] == 0)]
    test = test[test["path"].map(lambda p: (ROOT / p).exists())]
    pick = pd.concat([g.sample(min(len(g), args.per_class), random_state=args.seed)
                      for _, g in test.groupby("class_label")])
    bg_pool = [p for p in test[test["class_label"] == BACKGROUND_CLASS]["path"]]

    def load(p):
        a = load_audio(ROOT / p)
        return resample(to_mono(a.samples), a.sample_rate, TARGET_SR).astype(np.float32)

    conditions = {
        "clean": lambda y, r: y,
        "white noise 20 dB": lambda y, r: add_noise(y, 20),
        "white noise 10 dB": lambda y, r: add_noise(y, 10),
        "real background 10 dB": lambda y, r: add_noise(y, 10, load(random.choice(bg_pool))),
        "echo / reverb (RT60 0.6 s)": lambda y, r: reverb(y, TARGET_SR, 0.6),
        "low volume (-30 dB)": lambda y, r: volume(y, -30),
        "other device (band-pass 300 Hz-4 kHz)": lambda y, r: device(y, TARGET_SR),
        "distant source (15 m)": lambda y, r: distance(y, TARGET_SR, 15),
        "partial event (first 40 % cut)": lambda y, r: y[int(0.4 * len(y)):] if len(y) > TARGET_SR else y,
        "re-encoded MP3 64 kbps": lambda y, r: reencode_mp3(y, TARGET_SR, 64),
        "overlap with another class (0 dB)": None,     # handled below
    }
    rows, t0 = [], time.time()
    items = list(pick.itertuples())
    for i, r in enumerate(items):
        y = load(r.path)
        for name, fn in conditions.items():
            other = None
            if fn is None:
                o = pick[(pick["class_label"] != r.class_label) & (pick["class_label"] != BACKGROUND_CLASS)].sample(
                    1, random_state=i).iloc[0]
                other = o["class_label"]
                z = np.resize(load(o["path"]), len(y))
                z = z * (np.sqrt(np.mean(y ** 2) + 1e-12) / (np.sqrt(np.mean(z ** 2)) + 1e-12))
                yy = (y + z) / 2
            else:
                yy = fn(y, r)
            try:
                pred, clip, qual, ovl = classify(np.asarray(yy, np.float32), TARGET_SR, model)
            except Exception as e:                      # a crash would itself be a failed hidden test
                rows.append({"condition": name, "audio_id": r.audio_id, "actual": r.class_label, "error": str(e)})
                continue
            ok = pred == r.class_label or (other is not None and pred == other)
            rows.append({"condition": name, "audio_id": r.audio_id, "actual": r.class_label, "other": other,
                         "predicted": pred, "confidence": round(clip[pred], 3), "correct": ok,
                         "quality": qual, "overlap_flag": ovl})
        if i % 25 == 0:
            print(f"  {i + 1}/{len(items)} clips  {time.time() - t0:.0f} s", flush=True)

    df = pd.DataFrame(rows)
    out = ROOT / "reports"; out.mkdir(exist_ok=True)
    df.to_csv(out / "hidden_test_robustness.csv", index=False)

    ok_df = df[df.get("error").isna()] if "error" in df else df
    summ = ok_df.groupby("condition", sort=False).agg(
        clips=("correct", "size"), accuracy=("correct", "mean"),
        poor_or_unusable=("quality", lambda q: float(np.isin(q, ["Poor", "Unusable"]).mean())),
        overlap_flagged=("overlap_flag", "mean")).reset_index()
    crashes = int(df["error"].notna().sum()) if "error" in df else 0

    clean = ok_df[ok_df["condition"] == "clean"]
    conf = []
    for a, b, label in CONFUSABLE:
        sa, sb = clean[clean["actual"] == a], clean[clean["actual"] == b]
        conf.append({"pair": label, "A→B confusions": f"{int((sa['predicted'] == b).sum())}/{len(sa)}",
                     "B→A confusions": f"{int((sb['predicted'] == a).sum())}/{len(sb)}"})

    md = ["# Hidden-test readiness – robustness results", "",
          f"Python model `{model.version}` · {len(items)} unseen TEST clips ({args.per_class} per class max) · "
          f"same pipeline as the web app · crashes: **{crashes}**", "",
          "| Condition | Clips | Accuracy | Rated Poor/Unusable (→ manual review) | Overlap flagged |",
          "|---|---|---|---|---|"]
    md += [f"| {r.condition} | {r.clips} | {r.accuracy:.3f} | {r.poor_or_unusable:.0%} | {r.overlap_flagged:.0%} |"
           for r in summ.itertuples()]
    md += ["", "For the overlap condition a prediction counts as correct if it is either of the two mixed classes.", "",
           "## Confusable pairs from the SRS (clean test clips)", "", "| Pair | A predicted as B | B predicted as A |",
           "|---|---|---|"]
    md += [f"| {c['pair']} | {c['A→B confusions']} | {c['B→A confusions']} |" for c in conf]
    md += ["", "How the application handles hard cases: low confidence → *Uncertain*; Poor/Unusable quality, close top-2 "
               "scores, overlapping sounds, model disagreement or an unconfirmed critical class → manual review; "
               "re-encoded copies → near-duplicate notice (SHA-256 + fingerprint)."]
    (out / "hidden_test_robustness.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
