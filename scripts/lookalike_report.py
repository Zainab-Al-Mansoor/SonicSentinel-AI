"""
Similar-event evaluation (SRS Step 14): can the Python model tell each critical sound from its look-alike?

    python scripts/lookalike_report.py            # validation + test clips (never used for training)

For every pair in config.settings.LOOKALIKE_PAIRS the script reports
  * recall of the real event,
  * how often the look-alike recordings are mistaken for the event (false alarm rate),
  * how many look-alike training clips the model has seen (data coverage),
using the same pipeline and the same look-alike gap check as the web application.
Look-alike recordings are found by their original dataset category (e.g. ESC-50 "fireworks",
MIMII "normal", VSD "noviolence", Kaggle "NotScreaming"). Pairs without any look-alike recordings
are listed as "no data yet" – add clips to downloads/extra/Background Noise/ (e.g. backfire_*.wav,
metal_*.wav) and rebuild the dataset.

Results: reports/lookalike_pairs.md and reports/lookalike_pairs.csv
"""
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from config.settings import CLASSES, DATASET_METADATA_CSV, DEFAULT_RUNTIME_SETTINGS, TARGET_SR
from audio_preprocessing import load_audio, preprocess_signal, segment
from feature_extraction import extract_batch
from python_models.inference import PythonSoundModel, aggregate_scores

S = DEFAULT_RUNTIME_SETTINGS
# pair name -> (event class, look-alike class, keywords that identify look-alike recordings in `source`/`original_filename`)
PAIRS = [
    ("Gunshot vs fireworks", "Gunshot", "Background Noise", ["fireworks"]),
    ("Gunshot vs vehicle backfire", "Gunshot", "Background Noise", ["backfire"]),
    ("Gunshot vs door slam / knock", "Gunshot", "Background Noise", ["door_wood_knock", "door_slam", "door slam"]),
    ("Panic scream vs normal shouting / voices", "Panic Scream", "Background Noise", ["NotScreaming", "shout", "laughing"]),
    ("Aggression vs normal conversation", "Aggression", "Background Noise", ["noviolence", "conversation"]),
    ("Glass breaking vs metal impact", "Glass Breaking", "Background Noise", ["metal", "clang", "can_opening"]),
    ("Alarm vs vehicle horn", "Alarm or Siren", "Vehicle Horn", ["car_horn"]),
    ("Vehicle horn vs alarm", "Vehicle Horn", "Alarm or Siren", ["siren", "clock_alarm"]),
    ("Machinery fault vs normal machinery", "Machinery Fault", "Background Noise", [":normal", "normal_"]),
    ("Help request vs ordinary speech", "Person Asking for Help", "Background Noise", ["NotScreaming", "speech", "talk"]),
]


def classify(path, model):
    a = load_audio(ROOT / path)
    clean, _ = preprocess_signal(a.samples, a.sample_rate, trim=True, denoise=S["noise_reduction"])
    segs = segment(clean, TARGET_SR, S["segment_seconds"], S["segment_hop_seconds"])
    sc, _ = aggregate_scores(model.predict_proba(extract_batch([s for _, _, s in segs])), S["min_confidence"])
    return sc


def main():
    model = PythonSoundModel()
    meta = pd.read_csv(DATASET_METADATA_CSV)
    meta["text"] = (meta["source"].astype(str) + " " + meta["original_filename"].astype(str))
    orig = meta[meta["is_augmented"] == 0]
    evals = orig[orig["split"].isin(["validation", "val", "test"])]
    evals = evals[evals["path"].map(lambda p: (ROOT / p).exists())]
    cache = {}

    def scores(row):
        if row.audio_id not in cache:
            cache[row.audio_id] = classify(row.path, model)
        return cache[row.audio_id]

    rows = []
    lm = S.get("lookalike_margin", 0.2)
    for name, cls, alt, keys in PAIRS:
        match = lambda df: df[(df["class_label"] == alt) & df["text"].str.contains("|".join(keys), case=False, regex=True)]
        look = match(evals)
        pos = evals[evals["class_label"] == cls]
        train_look = len(match(orig[orig["split"] == "train"]))
        r = {"pair": name, "event": cls, "event_clips": len(pos), "lookalike_clips": len(look),
             "lookalike_train_clips": train_look}
        if len(pos):
            ps = [scores(x) for x in pos.itertuples()]
            r["event_recall"] = float(np.mean([max(s, key=s.get) == cls for s in ps]))
            r["event_passed_gap"] = float(np.mean([s[cls] - s[alt] >= lm for s in ps]))
        if len(look):
            ls = [scores(x) for x in look.itertuples()]
            r["false_alarm_rate"] = float(np.mean([max(s, key=s.get) == cls for s in ls]))
            r["lookalike_correct"] = float(np.mean([max(s, key=s.get) == alt for s in ls]))
            r["flagged_for_review"] = float(np.mean([max(s, key=s.get) != cls and s[cls] >= S["unknown_threshold"] for s in ls]))
        rows.append(r)
        print(f"  {name:<42} event {len(pos):>3} · look-alike {len(look):>3} (train {train_look})", flush=True)

    df = pd.DataFrame(rows)
    out = ROOT / "reports"; out.mkdir(exist_ok=True)
    df.to_csv(out / "lookalike_pairs.csv", index=False)
    f = lambda v: "–" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.0%}"
    md = ["# Similar-event evaluation (SRS Step 14)", "",
          f"Python model `{model.version}` · validation + test clips only (never used for training) · "
          f"look-alike gap threshold {lm:.2f}", "",
          "| Pair | Event clips | Event recognised | Look-alike clips (train) | Look-alike mistaken for the event | Look-alike recognised as itself |",
          "|---|---|---|---|---|---|"]
    for r in rows:
        la = f"{r['lookalike_clips']} ({r['lookalike_train_clips']})" if r["lookalike_clips"] or r["lookalike_train_clips"] else "**no data yet**"
        md.append(f"| {r['pair']} | {r['event_clips']} | {f(r.get('event_recall'))} | {la} | "
                  f"{f(r.get('false_alarm_rate'))} | {f(r.get('lookalike_correct'))} |")
    md += ["", "In the application every decision for these classes also runs the **look-alike check**: if the gap between "
               "the event and its look-alike is below the threshold (Admin → Settings → `lookalike_margin`), the event is "
               "sent to manual review with the reason “Possible look-alike: could be …”; a Background decision with a "
               "strong critical-class score is flagged as a possible hidden event.",
           "", "Pairs marked *no data yet* need licensed recordings of the look-alike (e.g. Freesound CC0 “car backfire”, "
               "“metal hit”) in `downloads/extra/Background Noise/` followed by import → build → augment → train."]
    (out / "lookalike_pairs.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
