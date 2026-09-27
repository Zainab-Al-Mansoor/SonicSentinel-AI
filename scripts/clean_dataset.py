import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import FEATURE_DIR, PROCESSED_DATASET_DIR, RAW_DATASET_DIR

KEEP_PREFIXES = ("rec_", "tts_")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--yes", action="store_true", help="really delete")
    args = ap.parse_args()

    raw = [p for p in RAW_DATASET_DIR.rglob("*") if p.is_file() and p.parent != RAW_DATASET_DIR
           and not p.name.startswith(KEEP_PREFIXES)]
    kept = [p for p in RAW_DATASET_DIR.rglob("*") if p.is_file() and p.name.startswith(KEEP_PREFIXES)]
    derived = [p for d in (PROCESSED_DATASET_DIR, FEATURE_DIR) for p in d.iterdir() if p.name != ".gitkeep"]

    print(f"raw imported clips to delete : {len(raw)}")
    print(f"own recordings / TTS kept    : {len(kept)}")
    print(f"processed + feature entries  : {len(derived)}")
    if not args.yes:
        print("\nNothing deleted. Run again with --yes to delete.")
        return
    for p in raw:
        p.unlink()
    for p in derived:
        shutil.rmtree(p) if p.is_dir() else p.unlink()
    print("Done. Now run import_local_downloads.py, build_dataset.py, augment and train_models.")


if __name__ == "__main__":
    main()
