"""
Label audit – find recordings whose label is probably wrong (step 1 of "better features").

    python scripts/label_audit.py                 # all original clips of every split
    python scripts/label_audit.py --max-rows 400  # size of the review list (default 300)

How it works
  1. Every ORIGINAL clip (augmented copies are skipped) is turned into one clip-level vector:
     mean and max of its 299 segment features (the cached features of train_models.py are reused,
     so this is fast after a training run).
  2. 5-fold cross-validation with a HistGradientBoosting classifier gives an OUT-OF-FOLD probability
     for every clip – i.e. a prediction from a model that never saw that clip.
     ("confident learning": a clip whose own label gets a very low probability while another class
     is very likely is probably mislabelled.)
  3. Filename / source keywords that do not fit the class (e.g. "pour" or "clink" in Glass Breaking,
     "firework" in Gunshot) are flagged as well.

Output
  reports/label_audit.csv    all flagged clips, most suspicious first
  reports/label_audit.html   open in Chrome: listen to every clip, choose keep / move / delete,
                             then click "Export decisions" -> reports/label_audit_decisions.csv
Apply the decisions with:  python scripts/apply_label_fixes.py --apply
Nothing is changed by this script.
"""
import argparse
import html
import json
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold

from config.settings import CLASSES, DATASET_METADATA_CSV
from python_models.train_models import clip_features

KEYWORDS = {
    "Glass Breaking": ["pour", "clink", "ding", "wipe", "tapping", "bell", "toast", "squeak", "cutting-glass", "filling"],
    "Gunshot": ["firework", "cracker", "door", "knock", "balloon", "clap"],
    "Panic Scream": ["laugh", "sing", "cheer", "notscreaming", "crying_baby"],
    "Aggression": ["noviolence", "conversation"],
    "Animal Sound": ["speech", "voice", "human"],
    "Alarm or Siren": ["car_horn", "horn"],
    "Vehicle Horn": ["siren", "alarm"],
    "Machinery Fault": [":normal", "normal_"],
    "Person Asking for Help": ["laugh", "sing", "music"],
}


def clip_vector(row):
    X, _ = clip_features(row)
    return np.concatenate([X.mean(axis=0), X.max(axis=0)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-rows", type=int, default=300)
    ap.add_argument("--folds", type=int, default=5)
    a = ap.parse_args()

    meta = pd.read_csv(DATASET_METADATA_CSV).fillna("")
    meta = meta[(meta["is_augmented"].astype(str).isin(["0", "0.0", "False"])) & meta["class_label"].isin(CLASSES)]
    meta = meta[meta["path"].map(lambda p: (ROOT / p).exists())].reset_index(drop=True)
    print(f"Original clips with audio: {len(meta)}")

    vecs = []
    for i, r in meta.iterrows():
        vecs.append(clip_vector(r))
        if (i + 1) % 250 == 0 or i + 1 == len(meta):
            print(f"  features {i + 1}/{len(meta)}", flush=True)
    X = np.vstack(vecs)
    y = meta["class_label"].map({c: i for i, c in enumerate(CLASSES)}).to_numpy()

    # out-of-fold probabilities
    oof = np.zeros((len(meta), len(CLASSES)))
    min_count = int(np.bincount(y, minlength=len(CLASSES))[np.bincount(y, minlength=len(CLASSES)) > 0].min())
    skf = StratifiedKFold(n_splits=max(2, min(a.folds, min_count)), shuffle=True, random_state=42)
    for k, (tr, te) in enumerate(skf.split(X, y)):
        clf = HistGradientBoostingClassifier(max_iter=250, learning_rate=0.08, class_weight="balanced", random_state=42)
        clf.fit(X[tr], y[tr])
        p = clf.predict_proba(X[te])
        oof[np.ix_(te, clf.classes_)] = p
        print(f"  fold {k + 1}/{skf.get_n_splits()} done", flush=True)
    acc = float((oof.argmax(1) == y).mean())
    print(f"Out-of-fold clip accuracy: {acc:.3f}")

    rows = []
    for i, r in meta.iterrows():
        p_label = oof[i, y[i]]
        top = int(oof[i].argmax())
        p_top = oof[i, top]
        text = f"{r['original_filename']} {r['source']}".lower()
        kw = [k for k in KEYWORDS.get(r["class_label"], []) if k in text]
        reason = []
        if top != y[i] and p_top >= 0.70 and p_label < 0.10:
            reason.append(f"model is sure it is {CLASSES[top]}")
        elif p_label < 0.30:
            reason.append("label gets a low probability")
        if kw:
            reason.append("filename/source: " + ", ".join(kw))
        if not reason:
            continue
        score = (1 - p_label) + (0.5 if kw else 0) + (0.3 if top != y[i] else 0)
        rows.append({"audio_id": r["audio_id"], "split": r["split"], "label": r["class_label"],
                     "suggested": CLASSES[top], "p_label": round(float(p_label), 3), "p_suggested": round(float(p_top), 3),
                     "reason": "; ".join(reason), "original_filename": r["original_filename"], "source": r["source"],
                     "path": r["path"], "score": round(score, 3)})
    df = pd.DataFrame(rows).sort_values("score", ascending=False).head(a.max_rows)
    out = ROOT / "reports"
    out.mkdir(exist_ok=True)
    df.to_csv(out / "label_audit.csv", index=False)
    write_html(df, out / "label_audit.html", acc, len(meta))
    print(f"\nFlagged {len(rows)} clips, {len(df)} written for review.")
    if len(df):
        print(df.groupby("label").size().sort_values(ascending=False).to_string())
    print(f"\nOpen in Chrome: {out / 'label_audit.html'}")


def write_html(df, path, acc, n):
    opts = ["keep", "delete"] + [f"move:{c}" for c in CLASSES]
    trs = []
    for i, r in enumerate(df.itertuples(), 1):
        sel = "".join(f'<option value="{html.escape(o)}"{" selected" if o == "keep" else ""}>'
                      f'{html.escape(o.replace("move:", "→ "))}</option>' for o in opts)
        trs.append(f"""<tr data-id="{r.audio_id}" data-label="{html.escape(r.label)}">
<td>{i}</td><td><audio controls preload="none" src="../{html.escape(r.path.replace(chr(92), '/'))}"></audio>
<div class="fn">{html.escape(r.original_filename)}</div><div class="src">{html.escape(str(r.source))} · {r.split} · {r.audio_id}</div></td>
<td><b>{html.escape(r.label)}</b><div class="p">p = {r.p_label:.2f}</div></td>
<td>{html.escape(r.suggested)}<div class="p">p = {r.p_suggested:.2f}</div></td>
<td class="why">{html.escape(r.reason)}</td>
<td><select>{sel}</select><button class="q" data-v="move:{html.escape(r.suggested)}">use suggestion</button></td></tr>""")
    page = f"""<!doctype html><html><head><meta charset="utf-8"><title>SonicSentinel – label audit</title>
<style>body{{font-family:Segoe UI,Arial,sans-serif;margin:24px;background:#FBF7F3;color:#2E2724}}
h1{{color:#4A2C21;margin:0 0 4px}} .sub{{color:#6b5e58;margin-bottom:16px}}
table{{border-collapse:collapse;width:100%;background:#fff}} th{{background:#4A2C21;color:#fff;text-align:left;padding:8px;position:sticky;top:0}}
td{{border-bottom:1px solid #E6D8CC;padding:8px;vertical-align:top;font-size:14px}} audio{{width:260px;height:34px}}
.fn{{font-family:Consolas,monospace;font-size:12px;color:#8A3F1E;margin-top:4px;word-break:break-all}} .src,.p{{font-size:12px;color:#7a6d67}}
.why{{max-width:260px;color:#5a4a42}} select{{padding:4px;font-size:13px}} button{{cursor:pointer}} .q{{display:block;margin-top:6px;font-size:12px}}
.bar{{position:sticky;top:0;background:#FBF7F3;padding:10px 0;z-index:2;display:flex;gap:12px;align-items:center}}
.bar button{{background:#A0522D;color:#fff;border:0;padding:10px 16px;border-radius:8px;font-weight:600}} tr.changed td{{background:#FFF4E5}}
</style></head><body>
<h1>Label audit</h1><div class="sub">{len(df)} clips to check (out of {n}) · out-of-fold accuracy {acc:.1%}.
Listen to each clip. Leave <b>keep</b> if the label is right, choose <b>→ class</b> to move it, or <b>delete</b> if it fits no class.
Then click <b>Export decisions</b> and save the file as <code>reports/label_audit_decisions.csv</code>.</div>
<div class="bar"><button id="exp">Export decisions</button><span id="cnt">0 changes</span></div>
<table><thead><tr><th>#</th><th>Audio</th><th>Current label</th><th>Model suggests</th><th>Why flagged</th><th>Decision</th></tr></thead>
<tbody>{''.join(trs)}</tbody></table>
<script>
const rows=[...document.querySelectorAll('tbody tr')];
function count(){{let n=0;rows.forEach(r=>{{const c=r.querySelector('select').value!=='keep';r.classList.toggle('changed',c);if(c)n++;}});document.getElementById('cnt').textContent=n+' changes';}}
rows.forEach(r=>{{r.querySelector('select').onchange=count;r.querySelector('.q').onclick=e=>{{r.querySelector('select').value=e.target.dataset.v;count();}};}});
document.getElementById('exp').onclick=()=>{{const lines=['audio_id,label,action'];rows.forEach(r=>lines.push([r.dataset.id,'"'+r.dataset.label+'"','"'+r.querySelector('select').value+'"'].join(',')));
const b=new Blob([lines.join('\\n')],{{type:'text/csv'}});const a=document.createElement('a');a.href=URL.createObjectURL(b);a.download='label_audit_decisions.csv';a.click();}};
</script></body></html>"""
    path.write_text(page, encoding="utf-8")


if __name__ == "__main__":
    main()
