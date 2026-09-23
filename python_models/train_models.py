"""
Train, tune and compare Python sound-classification models.

    python -m python_models.train_models              # all models, normal grids
    python -m python_models.train_models --fast       # small grids (quick check)
    python -m python_models.train_models --models svm,rf,xgb

Models compared : SVM (RBF), Random Forest, XGBoost (or sklearn Gradient
                  Boosting if xgboost is missing), MLP neural network.
Tuning          : GridSearchCV, GroupKFold(3) grouped by ORIGINAL Audio ID so
                  segments / augmented copies of one clip never leak across folds.
Selection       : best CLIP-level macro-F1 on the VALIDATION split
                  (critical-class recall breaks ties).
Test split      : used once, only for the final report.

Outputs
  python_models/saved/sonic_model.joblib        selected model bundle
  reports/python_model_comparison.csv           all candidates, val metrics
  reports/python_test_metrics.json              final test metrics
  reports/python_classwise_test.csv             per-class precision/recall/F1
  reports/confusion_matrix_python.png
  reports/noise_robustness.csv
"""
import argparse
import hashlib
import json
import time
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support, f1_score,
                             confusion_matrix, classification_report)
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from config.progress import progress as _progress
from config.settings import (BASE_DIR, CLASSES, CRITICAL_RECALL_CLASSES, DATASET_METADATA_CSV,
                             FEATURE_DIR, PYTHON_MODEL_PATH, TARGET_SR, BACKGROUND_CLASS,
                             DEFAULT_RUNTIME_SETTINGS)
from audio_preprocessing import load_audio, preprocess_signal, segment
from augmentation.augment import add_noise
from feature_extraction import extract_features, feature_names
from feature_extraction.features import FEATURE_VERSION
from python_models.inference import aggregate_scores

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None

REPORTS = BASE_DIR / "reports"
REPORTS.mkdir(exist_ok=True)
SEG = DEFAULT_RUNTIME_SETTINGS["segment_seconds"]
HOP = DEFAULT_RUNTIME_SETTINGS["segment_hop_seconds"]
DENOISE = DEFAULT_RUNTIME_SETTINGS["noise_reduction"]
MIN_CONF = DEFAULT_RUNTIME_SETTINGS["min_confidence"]
CACHE_TAG = hashlib.md5(f"{FEATURE_VERSION}-{SEG}-{HOP}-{DENOISE}-{TARGET_SR}-{len(feature_names())}".encode()).hexdigest()[:8]


# ---------------------------------------------------------------------------
# Feature extraction with on-disk cache (data/features/<tag>/<audio_id>.npz)
# ---------------------------------------------------------------------------
def clip_segments(path, noise_snr=None):
    audio = load_audio(path)
    y = audio.samples if audio.samples.ndim == 1 else audio.samples.mean(axis=1)
    if noise_snr is not None:
        y = add_noise(y, snr_db=noise_snr)
    clean, _ = preprocess_signal(y, audio.sample_rate, trim=True, denoise=DENOISE)
    return [s for _, _, s in segment(clean, TARGET_SR, SEG, HOP)]


def clip_features(row, noise_snr=None, use_cache=True):
    cache_dir = FEATURE_DIR / CACHE_TAG
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = row["audio_id"] + (f"_snr{noise_snr}" if noise_snr is not None else "")
    f = cache_dir / f"{key}.npz"
    if use_cache and f.exists():
        d = np.load(f)
        return d["X"], d["energy"]
    segs = clip_segments(BASE_DIR / row["path"], noise_snr)
    X = np.vstack([extract_features(s) for s in segs])
    energy = np.array([float(np.sqrt(np.mean(s ** 2))) for s in segs])
    np.savez_compressed(f, X=X, energy=energy)
    return X, energy


def build_matrix(df, drop_quiet=True, noise_snr=None):
    """Segment-level matrix. Quiet segments of event clips are dropped from TRAINING
    (a mostly-silent segment of a gunshot clip is not a gunshot)."""
    Xs, ys, groups, clip_ids = [], [], [], []
    for i, (_, r) in enumerate(df.iterrows()):
        X, e = clip_features(r, noise_snr)
        keep = np.ones(len(X), bool)
        if drop_quiet and r["class_label"] != BACKGROUND_CLASS and len(X) > 1:
            keep = e >= 0.25 * e.max()
        Xs.append(X[keep]); ys += [r["class_label"]] * int(keep.sum())
        g = r["parent_audio_id"] if isinstance(r.get("parent_audio_id"), str) and r["parent_audio_id"] else r["audio_id"]
        groups += [g] * int(keep.sum()); clip_ids += [r["audio_id"]] * int(keep.sum())
        _progress(f"  features {i+1}/{len(df)}")
    return np.vstack(Xs), np.array(ys), np.array(groups), np.array(clip_ids)


# ---------------------------------------------------------------------------
def candidates(fast: bool, wanted: set):
    c = {}
    if "svm" in wanted:
        c["svm"] = (SVC(probability=True, class_weight="balanced", random_state=42),
                    {"clf__C": [10] if fast else [1, 10, 30], "clf__gamma": ["scale"] if fast else ["scale", 0.005]})
    if "rf" in wanted:
        c["rf"] = (RandomForestClassifier(n_estimators=300, class_weight="balanced_subsample", n_jobs=-1, random_state=42),
                   {"clf__max_depth": [None] if fast else [None, 25], "clf__min_samples_leaf": [1] if fast else [1, 2]})
    if "xgb" in wanted:
        if XGBClassifier is not None:
            c["xgb"] = (XGBClassifier(n_estimators=300, tree_method="hist", eval_metric="mlogloss",
                                      n_jobs=-1, random_state=42),
                        {"clf__max_depth": [6] if fast else [4, 6], "clf__learning_rate": [0.1] if fast else [0.1, 0.05]})
        else:
            c["gboost"] = (GradientBoostingClassifier(random_state=42),
                           {"clf__n_estimators": [150], "clf__max_depth": [3]})
    if "mlp" in wanted:
        c["mlp"] = (MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=400, early_stopping=True, random_state=42),
                    {"clf__alpha": [1e-3] if fast else [1e-4, 1e-3]})
    return c


def clip_level(pipe, df, label_to_idx):
    """Clip-level predictions using the same aggregation rule as the web app."""
    y_true, y_pred, conf = [], [], []
    for _, r in df.iterrows():
        X, _ = clip_features(r)
        proba = pipe.predict_proba(X)
        seg_scores = [{CLASSES[k]: float(p[j]) for j, k in enumerate(pipe.classes_)} for p in proba]
        agg, _ = aggregate_scores(seg_scores, MIN_CONF)
        best = max(agg, key=agg.get)
        y_true.append(r["class_label"]); y_pred.append(best); conf.append(agg[best])
    return np.array(y_true), np.array(y_pred), np.array(conf)


def metrics(y_true, y_pred):
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, labels=CLASSES, zero_division=0)
    crit = {c: float(r[CLASSES.index(c)]) for c in CRITICAL_RECALL_CLASSES}
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(np.mean(p)), "recall_macro": float(np.mean(r)),
        "f1_macro": float(f1_score(y_true, y_pred, labels=CLASSES, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, labels=CLASSES, average="weighted", zero_division=0)),
        "critical_recall": crit, "critical_recall_min": float(min(crit.values())),
        "per_class": {c: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i])} for i, c in enumerate(CLASSES)},
    }


def plot_cm(y_true, y_pred, path, title):
    cm = confusion_matrix(y_true, y_pred, labels=CLASSES)
    fig, ax = plt.subplots(figsize=(9, 8), dpi=110)
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(CLASSES))); ax.set_xticklabels(CLASSES, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(CLASSES))); ax.set_yticklabels(CLASSES, fontsize=8)
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=8,
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title(title)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)
    return cm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--models", default="svm,rf,xgb,mlp")
    args = ap.parse_args()

    meta = pd.read_csv(DATASET_METADATA_CSV).fillna({"parent_audio_id": ""})
    train_df = meta[meta["split"] == "train"]
    val_df = meta[(meta["split"] == "val") & (meta["is_augmented"] == 0)]
    test_df = meta[(meta["split"] == "test") & (meta["is_augmented"] == 0)]
    print(f"Clips  train={len(train_df)} (incl. augmented)  val={len(val_df)}  test={len(test_df)}")

    label_to_idx = {c: i for i, c in enumerate(CLASSES)}
    print("Extracting training features ...")
    Xtr, ytr_s, groups, _ = build_matrix(train_df)
    ytr = np.array([label_to_idx[c] for c in ytr_s])
    print(f"Training segments: {Xtr.shape}")

    n_groups = len(set(groups))
    cv = GroupKFold(n_splits=min(3, n_groups))
    results, fitted = [], {}
    for name, (clf, grid) in candidates(args.fast, set(args.models.split(","))).items():
        t0 = time.time()
        pipe = Pipeline([("scaler", StandardScaler()), ("clf", clf)])
        gs = GridSearchCV(pipe, grid, scoring="f1_macro", cv=cv, n_jobs=1, refit=True)
        gs.fit(Xtr, ytr, groups=groups)
        best = gs.best_estimator_
        yv, pv, _ = clip_level(best, val_df, label_to_idx)
        m = metrics(yv, pv)
        results.append({"model": name, "best_params": json.dumps(gs.best_params_), "cv_f1_macro": gs.best_score_,
                        "val_accuracy": m["accuracy"], "val_f1_macro": m["f1_macro"],
                        "val_precision_macro": m["precision_macro"], "val_recall_macro": m["recall_macro"],
                        "val_critical_recall_min": m["critical_recall_min"], "train_seconds": round(time.time() - t0, 1)})
        fitted[name] = best
        print(f"[{name}] cv_f1={gs.best_score_:.3f} val_acc={m['accuracy']:.3f} val_macroF1={m['f1_macro']:.3f} "
              f"crit_recall_min={m['critical_recall_min']:.3f}  ({time.time()-t0:.0f}s) params={gs.best_params_}")

    comp = pd.DataFrame(results).sort_values(["val_f1_macro", "val_critical_recall_min"], ascending=False)
    comp.to_csv(REPORTS / "python_model_comparison.csv", index=False)
    best_name = comp.iloc[0]["model"]
    best = fitted[best_name]
    print(f"\nSelected model: {best_name}")

    # ---- final, one-time test evaluation ---------------------------------
    yt, pt, ct = clip_level(best, test_df, label_to_idx)
    test_m = metrics(yt, pt)
    plot_cm(yt, pt, REPORTS / "confusion_matrix_python.png", f"Python model ({best_name}) – test set")
    pd.DataFrame(test_m["per_class"]).T.to_csv(REPORTS / "python_classwise_test.csv")
    print(classification_report(yt, pt, labels=CLASSES, zero_division=0))

    # ---- noise robustness -------------------------------------------------
    rob = [{"condition": "clean", "accuracy": test_m["accuracy"], "f1_macro": test_m["f1_macro"]}]
    for snr in (20, 10, 5):
        yy, pp = [], []
        for _, r in test_df.iterrows():
            X, _ = clip_features(r, noise_snr=snr)
            proba = best.predict_proba(X)
            seg_scores = [{CLASSES[k]: float(p[j]) for j, k in enumerate(best.classes_)} for p in proba]
            agg, _ = aggregate_scores(seg_scores, MIN_CONF)
            yy.append(r["class_label"]); pp.append(max(agg, key=agg.get))
        mm = metrics(np.array(yy), np.array(pp))
        rob.append({"condition": f"white noise SNR {snr} dB", "accuracy": mm["accuracy"], "f1_macro": mm["f1_macro"]})
        print(f"noise SNR {snr} dB: acc={mm['accuracy']:.3f} f1={mm['f1_macro']:.3f}")
    pd.DataFrame(rob).to_csv(REPORTS / "noise_robustness.csv", index=False)

    version = f"py-{best_name}-{datetime.now():%Y%m%d-%H%M}"
    bundle = {
        "pipeline": best, "classes": CLASSES, "algorithm": best_name, "version": version,
        "feature_names": feature_names(), "trained_at": datetime.now().isoformat(),
        "settings": {"segment_seconds": SEG, "segment_hop_seconds": HOP, "noise_reduction": DENOISE,
                     "sample_rate": TARGET_SR},
        "metrics": {"validation": comp.iloc[0].to_dict(), "test": {k: v for k, v in test_m.items() if k != "per_class"},
                    "noise_robustness": rob},
    }
    joblib.dump(bundle, PYTHON_MODEL_PATH)
    (REPORTS / "python_test_metrics.json").write_text(json.dumps({"version": version, **test_m, "noise_robustness": rob}, indent=2))
    print(f"\nSaved {PYTHON_MODEL_PATH}  (version {version})")
    print(f"TEST accuracy={test_m['accuracy']:.3f}  macroF1={test_m['f1_macro']:.3f}  "
          f"critical recall={json.dumps({k: round(v, 3) for k, v in test_m['critical_recall'].items()})}")


if __name__ == "__main__":
    main()
