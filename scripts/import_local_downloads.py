"""
Import the datasets that are ALREADY in the project's  downloads/  folder
into audio_dataset/raw/<Class>/ and append source + licence to
audio_dataset/raw/annotations.csv.

Why this script exists: scripts/import_public_datasets.py expects the
*official* folder layouts (UrbanSound8K/metadata + audio/foldN,
ESC-50-master/meta + audio). The copies in downloads/ use the Kaggle layouts,
and the Gunshot, Scream and VSD (violence) sets are not supported there at all.

Expected layout (what is on disk now):
    downloads/urbansound8k/UrbanSound8K.csv + fold1..fold10/
    downloads/esc50/esc50.csv + audio/audio/*.wav   (or audio/*.wav)
    downloads/mimii/abnormal/*.wav                   (normal/ is ignored)
    downloads/gunshot/<weapon>/*.wav                 -> Gunshot
    downloads/scream/Screaming/*.wav                 -> Panic Scream
    downloads/violence/VSD.xlsx + audios_VSD/audios_VSD/angry_*.wav -> Aggression
    downloads/glass_extra/*.wav                      -> Glass Breaking (any extra clips you add)

Usage (from the project folder):
    python scripts/import_local_downloads.py
    python scripts/import_local_downloads.py --max-per-class 500 --per-source 250
    python scripts/import_local_downloads.py --dry-run      # only print what would be imported

Each source adds at most --per-source clips per class (random, seeded), so a
class is a mix of several datasets instead of being filled by the first one.
Re-running is safe: files that already exist in raw/ are skipped.
"""
import argparse
import csv
import json
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import pandas as pd
import soundfile as sf

from config.settings import BASE_DIR, CLASSES, RAW_DATASET_DIR

DL = BASE_DIR / "downloads"
MAPPING = json.loads((BASE_DIR / "config" / "dataset_mapping.json").read_text(encoding="utf-8"))
ANN_PATH = RAW_DATASET_DIR / "annotations.csv"
ANN_FIELDS = ["filename", "class_label", "source", "license", "environment", "device", "distance_m"]

# (class, src_path, dst_name, source, license, environment, cut=(start_s, end_s) or None)
Candidate = tuple


def _safe(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in name)


# ---------------------------------------------------------------- sources
def urbansound8k():
    root = DL / "urbansound8k"
    csv_path = next((p for p in (root / "UrbanSound8K.csv", root / "metadata" / "UrbanSound8K.csv") if p.exists()), None)
    if not csv_path:
        return []
    cfg = MAPPING["urbansound8k"]
    out = []
    for _, r in pd.read_csv(csv_path).iterrows():
        cls = cfg["map"].get(r["class"])
        if not cls:
            continue
        for src in (root / f"fold{r['fold']}" / r["slice_file_name"],
                    root / "audio" / f"fold{r['fold']}" / r["slice_file_name"]):
            if src.exists():
                out.append((cls, src, f"us8k_{src.name}", f"UrbanSound8K:{r['fsID']}",
                            cfg["license"], "outdoor-urban", None))
                break
    return out


def esc50():
    root = DL / "esc50"
    csv_path = next((p for p in (root / "esc50.csv", root / "meta" / "esc50.csv") if p.exists()), None)
    audio_dir = next((p for p in (root / "audio" / "audio", root / "audio") if p.exists() and any(p.glob("*.wav"))), None)
    if not csv_path or not audio_dir:
        return []
    cfg = MAPPING["esc50"]
    out = []
    for _, r in pd.read_csv(csv_path).iterrows():
        cls = cfg["map"].get(r["category"])
        src = audio_dir / r["filename"]
        if cls and src.exists():
            out.append((cls, src, f"esc50_{src.name}", f"ESC-50:{r['src_file']}", cfg["license"], "unknown", None))
    return out


def mimii():
    root = DL / "mimii"
    cfg = MAPPING["mimii"]
    out = []
    for src in sorted(root.rglob("abnormal/*.wav")):
        # keep machine type / id in the name when the official nested layout is used
        tag = "_".join(p for p in src.parent.parent.relative_to(root).parts) or "mimii"
        out.append((cfg["abnormal_class"], src, _safe(f"mimii_{tag}_{src.name}"), f"MIMII:{tag}",
                    cfg["license"], "factory", None))
    return out


def gunshot():
    root = DL / "gunshot"
    out = []
    for src in sorted(root.rglob("*.wav")):
        weapon = src.parent.name
        out.append(("Gunshot", src, _safe(f"gun_{weapon}_{src.name}"), f"Kaggle gunshot dataset:{weapon}",
                    "check Kaggle dataset licence", "unknown", None))
    return out


def scream():
    root = DL / "scream" / "Screaming"
    return [("Panic Scream", src, _safe(f"scream_{src.name}"), "Kaggle Human Screaming Detection",
             "check Kaggle dataset licence", "unknown", None) for src in sorted(root.glob("*.wav"))]


def vsd(min_len=1.5, max_len=6.0):
    """One clip per annotated violence interval (from the angry_XXX segment files),
    at most `max_len` seconds, centred in the interval."""
    root = DL / "violence"
    xlsx = root / "VSD.xlsx"
    audio_dir = next((p for p in (root / "audios_VSD" / "audios_VSD", root / "audios_VSD") if p.exists()), None)
    if not xlsx.exists() or not audio_dir:
        return []
    df = pd.read_excel(xlsx, sheet_name="read_dataset")
    out = []
    for i, r in df.iterrows():
        dur = float(r["Violence_end"]) - float(r["Violence_start"])
        src = audio_dir / f"{r['File_segment_name']}.wav"
        if dur < min_len or not src.exists():
            continue
        mid = (float(r["Violence_start"]) + float(r["Violence_end"])) / 2
        half = min(dur, max_len) / 2
        start, end = max(0.0, mid - half), mid + half
        out.append(("Aggression", src, f"vsd_{src.stem}_{int(start*10):05d}.wav",
                    f"VSD:{src.stem}@{start:.1f}-{end:.1f}s", "check VSD dataset licence", "film/acted", (start, end)))
    return out


def glass_extra():
    root = DL / "glass_extra"
    exts = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
    return [("Glass Breaking", src, _safe(f"glass_{src.name}"), "glass_extra (manual)", "record the licence!",
             "unknown", None) for src in sorted(root.rglob("*")) if src.suffix.lower() in exts]


SOURCES = {"urbansound8k": urbansound8k, "esc50": esc50, "mimii": mimii, "gunshot": gunshot,
           "scream": scream, "vsd": vsd, "glass_extra": glass_extra}


# ---------------------------------------------------------------- copy
def write_clip(cls, src, dst_name, cut):
    dst = RAW_DATASET_DIR / cls / dst_name
    dst.parent.mkdir(parents=True, exist_ok=True)
    if cut is None:
        shutil.copy2(src, dst)
    else:
        info = sf.info(str(src))
        a, b = int(cut[0] * info.samplerate), min(int(cut[1] * info.samplerate), info.frames)
        data, sr = sf.read(str(src), start=a, stop=b, dtype="float32", always_2d=True)
        sf.write(str(dst), data, sr, subtype="PCM_16")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", default=",".join(SOURCES), help="comma list: " + ",".join(SOURCES))
    ap.add_argument("--max-per-class", type=int, default=500, help="total cap per class folder")
    ap.add_argument("--per-source", type=int, default=300, help="cap per class from ONE source")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    counts = defaultdict(int)
    for c in CLASSES:
        d = RAW_DATASET_DIR / c
        counts[c] = sum(1 for p in d.iterdir() if p.is_file()) if d.exists() else 0

    ann = []
    for name in [s.strip() for s in args.sources.split(",") if s.strip()]:
        cands = SOURCES[name]()
        by_cls = defaultdict(list)
        for c in cands:
            by_cls[c[0]].append(c)
        print(f"[{name}] found {len(cands)} candidate clips")
        for cls, items in by_cls.items():
            rng.shuffle(items)
            added = 0
            for cls_, src, dst_name, source, lic, env, cut in items:
                if added >= args.per_source or counts[cls] >= args.max_per_class:
                    break
                if (RAW_DATASET_DIR / cls / dst_name).exists():
                    continue
                if not args.dry_run:
                    try:
                        write_clip(cls, src, dst_name, cut)
                    except Exception as exc:
                        print(f"  [skip] {src.name}: {exc}")
                        continue
                ann.append({"filename": dst_name, "class_label": cls, "source": source, "license": lic,
                            "environment": env, "device": "unknown", "distance_m": ""})
                counts[cls] += 1
                added += 1
            print(f"  {cls:<25} +{added}")

    if ann and not args.dry_run:
        new = not ANN_PATH.exists()
        with open(ANN_PATH, "a", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=ANN_FIELDS)
            if new:
                w.writeheader()
            w.writerows(ann)

    print(f"\n{'(dry run) ' if args.dry_run else ''}Imported {len(ann)} clips. Counts per class (target >= 300):")
    for c in CLASSES:
        flag = "OK" if counts[c] >= 300 else f"MISSING {300 - counts[c]}"
        print(f"  {c:<25}{counts[c]:>5}   {flag}")


if __name__ == "__main__":
    main()
