"""
Apply the decisions exported from reports/label_audit.html.

    python scripts/apply_label_fixes.py                  # dry run
    python scripts/apply_label_fixes.py --apply          # move the files
    python scripts/apply_label_fixes.py --csv path.csv   # another decisions file

For every row with action "move:<Class>" the RAW source file
(audio_dataset/raw/<label>/<original filename>) is moved to audio_dataset/raw/<Class>/ and its row in
audio_dataset/raw/annotations.csv gets the new class_label. For "delete" the file is moved to
_to_delete/label_audit/ (nothing is deleted permanently) and its annotation row is removed.
Rows with "keep" are ignored. Afterwards rebuild: build_dataset -> augment -> train.
A log is written to _to_delete/label_audit_log.csv.
"""
import argparse
import csv
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd

from config.settings import CLASSES, DATASET_METADATA_CSV

RAW = ROOT / "audio_dataset" / "raw"
ANN = RAW / "annotations.csv"
TRASH = ROOT / "_to_delete"


def find_raw(label, name):
    p = RAW / label / name
    if p.exists():
        return p
    hits = list(RAW.glob(f"*/{name}"))
    return hits[0] if hits else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(ROOT / "reports" / "label_audit_decisions.csv"))
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()

    dec = pd.read_csv(a.csv)
    dec = dec[dec["action"].astype(str).str.strip() != "keep"]
    if dec.empty:
        print("No changes in the decisions file.")
        return
    meta = pd.read_csv(DATASET_METADATA_CSV).set_index("audio_id")
    ann = pd.read_csv(ANN, dtype=str) if ANN.exists() else None

    plan, problems = [], []
    for r in dec.itertuples():
        act = str(r.action).strip()
        if r.audio_id not in meta.index:
            problems.append(f"{r.audio_id}: not in dataset_metadata.csv"); continue
        m = meta.loc[r.audio_id]
        src = find_raw(m["class_label"], m["original_filename"])
        if src is None:
            problems.append(f"{r.audio_id}: raw file {m['original_filename']} not found"); continue
        if act == "delete":
            dst = TRASH / "label_audit" / m["class_label"] / src.name
        elif act.startswith("move:") and act[5:] in CLASSES:
            dst = RAW / act[5:] / src.name
        else:
            problems.append(f"{r.audio_id}: unknown action '{act}'"); continue
        plan.append((r.audio_id, src, dst, act))

    for aid, src, dst, act in plan:
        print(f"  {act:<32} {src.relative_to(ROOT)}")
    for p in problems:
        print("  [skip]", p)
    print(f"\n{len(plan)} changes, {len(problems)} skipped.")
    if not a.apply:
        print("Dry run only. Run again with --apply.")
        return

    log = []
    for aid, src, dst, act in plan:
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            dst = dst.with_name(dst.stem + "_dup" + dst.suffix)
        shutil.move(str(src), str(dst))
        if ann is not None:
            hit = ann["filename"] == src.name
            if act == "delete":
                ann = ann[~hit]
            elif hit.any():
                ann.loc[hit, "class_label"] = act[5:]
        log.append([aid, src, dst, act])
    if ann is not None:
        ann.to_csv(ANN, index=False, quoting=csv.QUOTE_MINIMAL)
    TRASH.mkdir(exist_ok=True)
    with open(TRASH / "label_audit_log.csv", "a", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerows(log)
    print(f"Done: {len(log)} files moved. Now run build_dataset.py -> augmentation.augment -> train_models.")


if __name__ == "__main__":
    main()
