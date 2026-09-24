"""
Import the datasets that are ALREADY in the project's  downloads/  folder
into audio_dataset/raw/<Class>/ and write source + licence to
audio_dataset/raw/annotations.csv.

Why this script exists: scripts/import_public_datasets.py expects the
*official* folder layouts (UrbanSound8K/metadata + audio/foldN,
ESC-50-master/meta + audio). The copies in downloads/ use the Kaggle layouts,
and the Gunshot, Scream and VSD (violence) sets are not supported there at all.

Layout that is read (missing folders are simply skipped):
    downloads/urbansound8k/UrbanSound8K.csv + fold1..fold10/   (label map: config/dataset_mapping.json)
    downloads/esc50/esc50.csv + audio/audio/*.wav              (label map: config/dataset_mapping.json)
    downloads/mimii/**/abnormal/*.wav     -> Machinery Fault
    downloads/mimii/**/normal/*.wav       -> Background Noise  ("normal machinery", SRS step 14)
    downloads/gunshot/<weapon>/*.wav      -> Gunshot
    downloads/scream/Screaming/*.wav      -> Panic Scream
    downloads/scream/NotScreaming/*.wav   -> Background Noise  (ordinary voices / shouting, SRS step 14)
    downloads/violence/VSD.xlsx + audios_VSD/audios_VSD/angry_*.wav      -> Aggression
    downloads/violence/audios_VSD/audios_VSD/noviolence_*.wav            -> Background Noise
                                                                            (random 5-s chunks: normal conversation)
    downloads/glass_extra/**              -> Glass Breaking
    downloads/extra/<Class Name>/**       -> that class (any clips you add by hand: Freesound, FSD50K, own recordings …)
    --sources lookalikes                  -> ALL ESC-50 look-alike categories (fireworks, door knock, can opening,
                                             laughing, crying baby, …) + all NotScreaming voices -> Background Noise
                                             (SRS step 14 – similar events), e.g.:
        python scripts/import_local_downloads.py --sources lookalikes

Usage (from the project folder):
    python scripts/import_local_downloads.py --dry-run
    python scripts/import_local_downloads.py --max-per-class 1200 --per-source 400 --bg-per-source 250

Selection rules
  * each source adds at most --per-source clips to a class (--bg-per-source for Background Noise),
  * inside one source the clips are taken round-robin over the original labels
    (e.g. every gunshot weapon, every ESC-50 category) so no sub-type dominates,
  * a class folder never grows beyond --max-per-class,
  * re-running is safe: existing files are skipped and annotations.csv keeps one row per file.
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

from config.settings import BASE_DIR, BACKGROUND_CLASS, CLASSES, RAW_DATASET_DIR

DL = BASE_DIR / "downloads"
MAPPING = json.loads((BASE_DIR / "config" / "dataset_mapping.json").read_text(encoding="utf-8"))
ANN_PATH = RAW_DATASET_DIR / "annotations.csv"
ANN_FIELDS = ["filename", "class_label", "source", "license", "environment", "device", "distance_m"]
AUDIO_EXT = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}


def _safe(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in name)


def cand(cls, src, dst_name, source, lic, env="unknown", cut=None, group=""):
    return {"cls": cls, "src": src, "dst": dst_name, "source": source, "license": lic,
            "env": env, "cut": cut, "group": group}


# ---------------------------------------------------------------- sources
def urbansound8k(rng):
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
                out.append(cand(cls, src, f"us8k_{src.name}", f"UrbanSound8K:{r['fsID']}:{r['class']}",
                                cfg["license"], "outdoor-urban", group=r["class"]))
                break
    return out


def esc50(rng):
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
            out.append(cand(cls, src, f"esc50_{src.name}", f"ESC-50:{r['src_file']}:{r['category']}",
                            cfg["license"], group=r["category"]))
    return out


def _mimii(kind, cls, prefix, rng):
    root = DL / "mimii"
    lic = MAPPING["mimii"]["license"]
    out = []
    for src in sorted(root.rglob(f"{kind}/*.wav")):
        tag = "_".join(src.parent.parent.relative_to(root).parts) or "mimii"
        out.append(cand(cls, src, _safe(f"{prefix}_{tag}_{src.name}"), f"MIMII:{tag}:{kind}", lic, "factory", group=tag))
    return out


def mimii(rng):
    return _mimii("abnormal", MAPPING["mimii"]["abnormal_class"], "mimii", rng)


def mimii_normal(rng):
    return _mimii("normal", BACKGROUND_CLASS, "mimiinorm", rng)


def gunshot(rng):
    out = []
    for src in sorted((DL / "gunshot").rglob("*.wav")):
        weapon = src.parent.name
        out.append(cand("Gunshot", src, _safe(f"gun_{weapon}_{src.name}"), f"Kaggle gunshot dataset:{weapon}",
                        "check Kaggle dataset licence", group=weapon))
    return out


def scream(rng):
    return [cand("Panic Scream", s, _safe(f"scream_{s.name}"), "Kaggle Human Screaming Detection:Screaming",
                 "check Kaggle dataset licence") for s in sorted((DL / "scream" / "Screaming").glob("*.wav"))]


def scream_not(rng):
    return [cand(BACKGROUND_CLASS, s, _safe(f"notscream_{s.name}"), "Kaggle Human Screaming Detection:NotScreaming",
                 "check Kaggle dataset licence") for s in sorted((DL / "scream" / "NotScreaming").glob("*.wav"))]


def _vsd_dir():
    root = DL / "violence"
    return next((p for p in (root / "audios_VSD" / "audios_VSD", root / "audios_VSD") if p.exists()), None)


def vsd(rng, min_len=1.5, max_len=6.0):
    """Aggression: one clip per annotated violence interval (from the angry_XXX segment files),
    at most `max_len` seconds, centred in the interval."""
    xlsx, audio_dir = DL / "violence" / "VSD.xlsx", _vsd_dir()
    if not xlsx.exists() or not audio_dir:
        return []
    df = pd.read_excel(xlsx, sheet_name="read_dataset")
    out = []
    for _, r in df.iterrows():
        dur = float(r["Violence_end"]) - float(r["Violence_start"])
        src = audio_dir / f"{r['File_segment_name']}.wav"
        if dur < min_len or not src.exists():
            continue
        mid = (float(r["Violence_start"]) + float(r["Violence_end"])) / 2
        half = min(dur, max_len) / 2
        start, end = max(0.0, mid - half), mid + half
        out.append(cand("Aggression", src, f"vsd_{src.stem}_{int(start*10):05d}.wav",
                        f"VSD:{src.stem}@{start:.1f}-{end:.1f}s", "check VSD dataset licence", "film/acted",
                        (start, end), group=src.stem))
    return out


def vsd_calm(rng, per_file=80, length=5.0):
    """Background Noise: random 5-s chunks of the long non-violent VSD recordings
    (normal conversation, music, street) – negatives for Aggression / Panic Scream."""
    audio_dir = _vsd_dir()
    if not audio_dir:
        return []
    out = []
    for src in sorted(audio_dir.glob("noviolence_*.wav")):
        info = sf.info(str(src))
        total = info.frames / info.samplerate
        if total < length * 2:
            continue
        starts = sorted(rng.uniform(0, total - length) for _ in range(per_file))
        for s in starts:
            out.append(cand(BACKGROUND_CLASS, src, f"vsdcalm_{src.stem}_{int(s*10):06d}.wav",
                            f"VSD:{src.stem}@{s:.1f}-{s+length:.1f}s", "check VSD dataset licence", "film",
                            (s, s + length), group=src.stem))
    return out


def _folder(root: Path, cls: str, prefix: str, source: str):
    return [cand(cls, s, _safe(f"{prefix}_{s.name}"), source, "record the licence!", group=s.parent.name)
            for s in sorted(root.rglob("*")) if s.suffix.lower() in AUDIO_EXT]


def glass_extra(rng):
    return _folder(DL / "glass_extra", "Glass Breaking", "glass", "glass_extra (manual)")


def extra(rng):
    out = []
    root = DL / "extra"
    if root.exists():
        for d in sorted(p for p in root.iterdir() if p.is_dir()):
            if d.name not in CLASSES:
                print(f"  [warn] downloads/extra/{d.name}: not a class name, skipped (use exactly: {', '.join(CLASSES)})")
                continue
            out += _folder(d, d.name, "extra", f"extra:{d.name} (manual)")
    return out


# SRS Step 14 look-alikes: real-world sounds that are easily confused with a critical class.
# They are Background Noise, but the default Background cap keeps only a few of each, so
# `--sources lookalikes` adds ALL of them (up to --lookalike-per-source per source).
LOOKALIKE_ESC = {
    "fireworks": "gunshot look-alike", "door_wood_knock": "gunshot / impact look-alike",
    "can_opening": "metal impact (glass look-alike)", "clock_tick": "metal click",
    "laughing": "loud voices (scream look-alike)", "crying_baby": "scream look-alike",
    "coughing": "ordinary vocal sound", "sneezing": "ordinary vocal sound", "clapping": "impulsive sound",
}


def lookalikes(rng):
    out = []
    root = DL / "esc50"
    csv_path = next((p for p in (root / "esc50.csv", root / "meta" / "esc50.csv") if p.exists()), None)
    audio_dir = next((p for p in (root / "audio" / "audio", root / "audio") if p.exists() and any(p.glob("*.wav"))), None)
    if csv_path and audio_dir:
        lic = MAPPING["esc50"]["license"]
        for _, r in pd.read_csv(csv_path).iterrows():
            if r["category"] in LOOKALIKE_ESC and (audio_dir / r["filename"]).exists():
                out.append(cand(BACKGROUND_CLASS, audio_dir / r["filename"], f"esc50_{r['filename']}",
                                f"ESC-50:{r['src_file']}:{r['category']}", lic, group=r["category"]))
    # ordinary voices / normal shouting (Panic Scream and Help look-alikes)
    for c in scream_not(rng):
        c["group"] = "NotScreaming"
        out.append(c)
    return out


SOURCES = {"urbansound8k": urbansound8k, "esc50": esc50, "mimii": mimii, "mimii_normal": mimii_normal,
           "gunshot": gunshot, "scream": scream, "scream_not": scream_not, "vsd": vsd, "vsd_calm": vsd_calm,
           "glass_extra": glass_extra, "extra": extra, "lookalikes": lookalikes}
DEFAULT_SOURCES = [k for k in SOURCES if k != "lookalikes"]


# ---------------------------------------------------------------- selection + copy
def round_robin(items, rng):
    groups = defaultdict(list)
    for it in items:
        groups[it["group"]].append(it)
    for g in groups.values():
        rng.shuffle(g)
    order = sorted(groups)
    rng.shuffle(order)
    while any(groups[g] for g in order):
        for g in order:
            if groups[g]:
                yield groups[g].pop()


def write_clip(c):
    dst = RAW_DATASET_DIR / c["cls"] / c["dst"]
    dst.parent.mkdir(parents=True, exist_ok=True)
    if c["cut"] is None:
        shutil.copy2(c["src"], dst)
    else:
        info = sf.info(str(c["src"]))
        a, b = int(c["cut"][0] * info.samplerate), min(int(c["cut"][1] * info.samplerate), info.frames)
        data, sr = sf.read(str(c["src"]), start=a, stop=b, dtype="float32", always_2d=True)
        sf.write(str(dst), data, sr, subtype="PCM_16")


def save_annotations(rows):
    new = pd.DataFrame(rows, columns=ANN_FIELDS)
    if ANN_PATH.exists():
        old = pd.read_csv(ANN_PATH, dtype=str)
        new = pd.concat([old, new.astype(str)], ignore_index=True)
    new = new.drop_duplicates(subset="filename", keep="last")
    new.to_csv(ANN_PATH, index=False, quoting=csv.QUOTE_MINIMAL)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", default=",".join(DEFAULT_SOURCES), help="comma list: " + ",".join(SOURCES))
    ap.add_argument("--lookalike-per-source", type=int, default=400,
                    help="cap for --sources lookalikes (may exceed --max-per-class on purpose)")
    ap.add_argument("--max-per-class", type=int, default=1200, help="total cap per class folder")
    ap.add_argument("--per-source", type=int, default=400, help="cap per class from ONE source")
    ap.add_argument("--bg-per-source", type=int, default=250, help="cap per source for Background Noise")
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
        cands = SOURCES[name](rng)
        by_cls = defaultdict(list)
        for c in cands:
            by_cls[c["cls"]].append(c)
        print(f"[{name}] found {len(cands)} candidate clips")
        for cls, items in by_cls.items():
            cap = args.bg_per_source if cls == BACKGROUND_CLASS else args.per_source
            limit = args.max_per_class
            if name == "lookalikes":
                cap, limit = args.lookalike_per_source, 10 ** 9
            added = 0
            for c in round_robin(items, rng):
                if added >= cap or counts[cls] >= limit:
                    break
                if (RAW_DATASET_DIR / cls / c["dst"]).exists():
                    continue
                if not args.dry_run:
                    try:
                        write_clip(c)
                    except Exception as exc:
                        print(f"  [skip] {Path(c['src']).name}: {exc}")
                        continue
                ann.append({"filename": c["dst"], "class_label": cls, "source": c["source"], "license": c["license"],
                            "environment": c["env"], "device": "unknown", "distance_m": ""})
                counts[cls] += 1
                added += 1
            print(f"  {cls:<25} +{added}")

    if ann and not args.dry_run:
        save_annotations(ann)

    print(f"\n{'(dry run) ' if args.dry_run else ''}Imported {len(ann)} clips. Counts per class (target >= 300):")
    for c in CLASSES:
        flag = "OK" if counts[c] >= 300 else f"MISSING {300 - counts[c]}"
        print(f"  {c:<25}{counts[c]:>5}   {flag}")


if __name__ == "__main__":
    main()
