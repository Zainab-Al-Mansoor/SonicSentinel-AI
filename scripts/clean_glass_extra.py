"""
Clean the hand-downloaded Glass Breaking clips (look-alike fix, SRS Step 14).

  python scripts/clean_glass_extra.py            # dry run – only shows what would happen
  python scripts/clean_glass_extra.py --apply    # do it

1. keeps the 43 real breaking / shattering clips in downloads/extra/Glass Breaking/
2. moves 33 ordinary glass sounds (pouring, clinking, ding, wiping, tapping …) to
   downloads/extra/Background Noise/  – they are look-alikes, not breaking glass
3. moves 11 duplicates / unsuitable clips (" (1)" copies, cinematic hit, scream + glass mix)
   to _to_delete/glass_extra/
4. moves the old imported copies audio_dataset/raw/Glass Breaking/extra_* to
   _to_delete/raw_glass_extra/ (the re-import copies the cleaned files again)

Nothing is deleted permanently – check _to_delete/ and delete that folder yourself.
A log is written to _to_delete/clean_glass_extra_log.csv.
"""
import argparse
import csv
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GLASS = ROOT / "downloads" / "extra" / "Glass Breaking"
BG = ROOT / "downloads" / "extra" / "Background Noise"
RAW_GLASS = ROOT / "audio_dataset" / "raw" / "Glass Breaking"
TRASH = ROOT / "_to_delete"

BG_KEYS = ["pour", "cutting-glass", "empty-glass-table", "cleaning-window", "filling-glass", "glass-bell",
           "bottle-clink", "glass-clink", "clinking", "cup-set-down", "glass-ding", "glass-dishes", "glass-door",
           "glass-knock", "on-table", "glass-ting", "glass-wipe", "ice-faling", "pill-bottle", "shimmer",
           "shot-glass", "spill-glass", "tapping", "wine-glass", "with-feet", "standing-on", "glass-hit",
           "bottle-glass-hit"]


def plan():
    moves = []
    for f in sorted(p for p in GLASS.iterdir() if p.is_file()):
        n = f.name
        if re.search(r" \(\d\)", n) or "lordsonny" in n or "scottishperson" in n:
            moves.append((f, TRASH / "glass_extra" / n, "delete (duplicate / unsuitable)"))
        elif any(k in n for k in BG_KEYS):
            moves.append((f, BG / n, "Background Noise (glass look-alike)"))
        else:
            moves.append((f, None, "keep (real glass breaking)"))
    if RAW_GLASS.exists():
        for f in sorted(RAW_GLASS.glob("extra_*")):
            moves.append((f, TRASH / "raw_glass_extra" / f.name, "old imported copy"))
    return moves


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()
    moves = plan()
    count = {}
    for src, dst, why in moves:
        count[why] = count.get(why, 0) + 1
    for why, n in count.items():
        print(f"  {n:3d}  {why}")
    if not a.apply:
        for src, dst, why in moves:
            if dst:
                print(f"  {why:38s} {src.name}")
        print("\nDry run only. Run again with --apply to move the files.")
        return
    TRASH.mkdir(exist_ok=True)
    log = []
    for src, dst, why in moves:
        if dst is None:
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            dst = dst.with_name(dst.stem + "_dup" + dst.suffix)
        shutil.move(str(src), str(dst))
        log.append([src, dst, why])
    with open(TRASH / "clean_glass_extra_log.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["from", "to", "reason"])
        w.writerows(log)
    print(f"\nMoved {len(log)} files. Kept in Glass Breaking: {len(list(GLASS.glob('*')))} files.")
    print(f"Background look-alikes now in: {BG}")
    print(f"Check and delete yourself: {TRASH}")


if __name__ == "__main__":
    main()
