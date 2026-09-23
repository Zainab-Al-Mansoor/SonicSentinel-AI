"""
Copy clips from FREE, LICENSED public datasets into audio_dataset/raw/<Class>/
and record source + licence in audio_dataset/raw/annotations.csv.

Download the datasets yourself (links in documentation/DATASET_GUIDE.md), unzip,
then run for example:

    python scripts/import_public_datasets.py --urbansound8k D:/data/UrbanSound8K --max-per-class 300
    python scripts/import_public_datasets.py --esc50 D:/data/ESC-50-master
    python scripts/import_public_datasets.py --fsd50k D:/data/FSD50K
    python scripts/import_public_datasets.py --mimii D:/data/mimii

Label mapping lives in config/dataset_mapping.json.
The importer never exceeds --max-per-class clips in any class folder.
"""
import argparse
import csv
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import pandas as pd

from config.settings import RAW_DATASET_DIR, BASE_DIR, CLASSES

MAPPING = json.loads((BASE_DIR / "config" / "dataset_mapping.json").read_text())
ANN_PATH = RAW_DATASET_DIR / "annotations.csv"
ANN_FIELDS = ["filename", "class_label", "source", "license", "environment", "device", "distance_m"]


def existing_counts():
    return {c: len(list((RAW_DATASET_DIR / c).glob("*"))) if (RAW_DATASET_DIR / c).exists() else 0 for c in CLASSES}


def append_annotations(rows):
    new = not ANN_PATH.exists()
    with open(ANN_PATH, "a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=ANN_FIELDS)
        if new:
            w.writeheader()
        w.writerows(rows)


def copy_clip(src: Path, cls: str, prefix: str, source: str, lic: str, counts, max_n, ann, env="unknown"):
    if cls not in CLASSES or counts[cls] >= max_n:
        return False
    dst_dir = RAW_DATASET_DIR / cls
    dst_dir.mkdir(parents=True, exist_ok=True)
    name = f"{prefix}_{src.parent.name}_{src.name}".replace(" ", "_")
    dst = dst_dir / name
    if dst.exists():
        return False
    shutil.copy2(src, dst)
    counts[cls] += 1
    ann.append({"filename": name, "class_label": cls, "source": source, "license": lic,
                "environment": env, "device": "unknown", "distance_m": ""})
    return True


def import_urbansound(root: Path, counts, max_n, ann):
    cfg = MAPPING["urbansound8k"]
    meta = pd.read_csv(root / "metadata" / "UrbanSound8K.csv")
    for _, r in meta.iterrows():
        cls = cfg["map"].get(r["class"])
        if cls:
            src = root / "audio" / f"fold{r['fold']}" / r["slice_file_name"]
            if src.exists():
                copy_clip(src, cls, "us8k", f"UrbanSound8K:{r['fsID']}", cfg["license"], counts, max_n, ann, "outdoor-urban")


def import_esc50(root: Path, counts, max_n, ann):
    cfg = MAPPING["esc50"]
    meta = pd.read_csv(root / "meta" / "esc50.csv")
    for _, r in meta.iterrows():
        cls = cfg["map"].get(r["category"])
        if cls:
            copy_clip(root / "audio" / r["filename"], cls, "esc50", f"ESC-50:{r['src_file']}", cfg["license"], counts, max_n, ann)


def import_fsd50k(root: Path, counts, max_n, ann):
    cfg = MAPPING["fsd50k"]
    keys = {k.lower(): v for k, v in cfg["map"].items()}
    for split, audio_dir in (("dev", "FSD50K.dev_audio"), ("eval", "FSD50K.eval_audio")):
        csv_path = root / "FSD50K.ground_truth" / f"{split}.csv"
        if not csv_path.exists():
            continue
        meta = pd.read_csv(csv_path)
        for _, r in meta.iterrows():
            labels = [l.lower() for l in str(r["labels"]).split(",")]
            hits = {keys[l] for l in labels if keys.get(l)}
            if len(hits) == 1:           # skip clips that map to several of our classes
                src = root / audio_dir / f"{r['fname']}.wav"
                if src.exists():
                    copy_clip(src, hits.pop(), "fsd50k", f"FSD50K:{r['fname']}", cfg["license"], counts, max_n, ann)


def import_mimii(root: Path, counts, max_n, ann):
    cfg = MAPPING["mimii"]
    for src in sorted(root.rglob("abnormal/*.wav")):
        copy_clip(src, cfg["abnormal_class"], "mimii", f"MIMII:{src.parent.parent.name}", cfg["license"], counts, max_n, ann, "factory")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--urbansound8k", type=Path)
    ap.add_argument("--esc50", type=Path)
    ap.add_argument("--fsd50k", type=Path)
    ap.add_argument("--mimii", type=Path)
    ap.add_argument("--max-per-class", type=int, default=300)
    args = ap.parse_args()

    counts, ann = defaultdict(int, existing_counts()), []
    if args.urbansound8k: import_urbansound(args.urbansound8k, counts, args.max_per_class, ann)
    if args.esc50: import_esc50(args.esc50, counts, args.max_per_class, ann)
    if args.fsd50k: import_fsd50k(args.fsd50k, counts, args.max_per_class, ann)
    if args.mimii: import_mimii(args.mimii, counts, args.max_per_class, ann)
    append_annotations(ann)
    print(f"Imported {len(ann)} clips. Current counts:")
    for c in CLASSES:
        print(f"  {c:<25}{counts[c]:>5}")


if __name__ == "__main__":
    main()
