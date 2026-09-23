"""
Build the COMMON sound dataset used by BOTH the Python model and GTM.

Input layout (put your collected / recorded / imported clips here):
    audio_dataset/raw/<Class Name>/<any file>.wav|mp3|flac|ogg|m4a
Optional per-file annotations (all columns optional except `filename`):
    audio_dataset/raw/annotations.csv
        filename,class_label,source,license,environment,device,distance_m

What it does
  1. validates every clip (decodable, not silent, >= 0.5 s)
  2. removes exact duplicates (SHA-256)
  3. assigns a unique Audio ID (AUD-000001 ...)
  4. converts to 44.1 kHz mono 16-bit WAV (keeps full quality for GTM)
  5. stratified 70 / 15 / 15 split (train / val / test) BY ORIGINAL CLIP
  6. writes data/dataset_metadata.csv, data/dataset_statistics.json,
     data/dataset_quality_report.csv and a class-balance check

Usage:
    python scripts/build_dataset.py            # full rebuild
    python scripts/build_dataset.py --min-per-class 300
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import soundfile as sf
from sklearn.model_selection import train_test_split

from config.progress import progress as _progress
from config.settings import (CLASSES, RAW_DATASET_DIR, PROCESSED_DATASET_DIR, DATASET_METADATA_CSV,
                             DATA_DIR, BASE_DIR, SUPPORTED_FORMATS, GTM_SR)
from audio_preprocessing import load_audio, analyze_quality, to_mono, resample

COLUMNS = ["audio_id", "filename", "original_filename", "class_label", "source", "license",
           "duration", "sample_rate", "channels", "original_sample_rate", "original_channels",
           "environment", "device", "distance_m", "is_augmented", "parent_audio_id", "augmentation",
           "split", "quality", "sha256", "path"]


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-per-class", type=int, default=300)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    ann_path = RAW_DATASET_DIR / "annotations.csv"
    ann = pd.read_csv(ann_path).set_index("filename").to_dict("index") if ann_path.exists() else {}

    rows, seen_hashes, rejected = [], {}, []
    counter = 0
    for cls in CLASSES:
        folder = RAW_DATASET_DIR / cls
        if not folder.exists():
            print(f"[warn] missing folder: {folder}")
            continue
        files = sorted(p for p in folder.rglob("*") if p.suffix.lower().lstrip(".") in SUPPORTED_FORMATS)
        for f in files:
            digest = sha256_of(f)
            if digest in seen_hashes:
                rejected.append((str(f), f"exact duplicate of {seen_hashes[digest]}"))
                continue
            try:
                audio = load_audio(f)
            except Exception as exc:
                rejected.append((str(f), f"decode error: {exc}"))
                continue
            q = analyze_quality(audio.samples, audio.sample_rate)
            if q["label"] == "Unusable":
                rejected.append((str(f), "unusable: " + "; ".join(q["issues"])))
                continue
            seen_hashes[digest] = f.name
            counter += 1
            a = ann.get(f.name, {})
            rows.append({
                "audio_id": f"AUD-{counter:06d}",
                "original_filename": f.name,
                "class_label": a.get("class_label", cls) if isinstance(a.get("class_label"), str) else cls,
                "source": a.get("source", "team-collected"),
                "license": a.get("license", "own recording"),
                "original_sample_rate": audio.sample_rate,
                "original_channels": audio.channels,
                "environment": a.get("environment", "unknown"),
                "device": a.get("device", "unknown"),
                "distance_m": a.get("distance_m", ""),
                "is_augmented": 0, "parent_audio_id": "", "augmentation": "",
                "quality": q["label"], "sha256": digest,
                "_src": f, "_audio": None,
            })
            _progress(f"scanned {counter} clips")
    if not rows:
        print("No clips found. Put audio in audio_dataset/raw/<Class Name>/ first.")
        return

    df = pd.DataFrame(rows)
    # ---- class balance check ----------------------------------------------
    counts = df["class_label"].value_counts().reindex(CLASSES, fill_value=0)
    print("\nClips per class:")
    for c, n in counts.items():
        flag = "" if n >= args.min_per_class else f"   <-- below {args.min_per_class}"
        print(f"  {c:<25}{n:>5}{flag}")
    ratio = counts.max() / max(counts[counts > 0].min(), 1)
    if ratio > 1.5:
        print(f"[warn] class imbalance ratio {ratio:.2f} (> 1.5). Collect more clips for the small classes.")

    # ---- stratified 70/15/15 split by ORIGINAL clip -----------------------
    strat = df["class_label"] if counts[counts > 0].min() >= 4 else None
    train_idx, rest_idx = train_test_split(df.index, test_size=0.30, random_state=args.seed, stratify=strat)
    rest = df.loc[rest_idx]
    strat2 = rest["class_label"] if strat is not None and rest["class_label"].value_counts().min() >= 2 else None
    val_idx, test_idx = train_test_split(rest.index, test_size=0.50, random_state=args.seed, stratify=strat2)
    df.loc[train_idx, "split"] = "train"
    df.loc[val_idx, "split"] = "val"
    df.loc[test_idx, "split"] = "test"

    # ---- write standardised WAVs ------------------------------------------
    for i, r in df.iterrows():
        audio = load_audio(r["_src"])
        y = resample(to_mono(audio.samples), audio.sample_rate, GTM_SR)
        out_dir = PROCESSED_DATASET_DIR / r["split"] / r["class_label"]
        out_dir.mkdir(parents=True, exist_ok=True)
        out = out_dir / f"{r['audio_id']}.wav"
        sf.write(out, y, GTM_SR, subtype="PCM_16")
        df.at[i, "filename"] = out.name
        df.at[i, "duration"] = round(len(y) / GTM_SR, 3)
        df.at[i, "sample_rate"] = GTM_SR
        df.at[i, "channels"] = 1
        df.at[i, "path"] = str(out.relative_to(BASE_DIR)).replace("\\", "/")
        _progress(f"wrote {i+1}/{len(df)}")

    df = df[COLUMNS]
    df.to_csv(DATASET_METADATA_CSV, index=False)
    pd.DataFrame(rejected, columns=["file", "reason"]).to_csv(DATA_DIR / "dataset_rejected.csv", index=False)
    df[["audio_id", "class_label", "split", "quality", "duration"]].to_csv(DATA_DIR / "dataset_quality_report.csv", index=False)

    stats = {
        "total_original_clips": int(len(df)),
        "per_class": counts.to_dict(),
        "per_split": df["split"].value_counts().to_dict(),
        "per_class_split": df.groupby(["class_label", "split"]).size().unstack(fill_value=0).to_dict("index"),
        "duration_seconds": {"total": float(df["duration"].sum()), "mean": float(df["duration"].mean()),
                             "min": float(df["duration"].min()), "max": float(df["duration"].max())},
        "quality": df["quality"].value_counts().to_dict(),
        "rejected": len(rejected),
        "imbalance_ratio": float(ratio),
    }
    (DATA_DIR / "dataset_statistics.json").write_text(json.dumps(stats, indent=2, default=int))
    print(f"\nDataset ready: {len(df)} original clips -> {DATASET_METADATA_CSV}")
    print(f"Split: {stats['per_split']}   Rejected: {len(rejected)} (see data/dataset_rejected.csv)")


if __name__ == "__main__":
    main()
