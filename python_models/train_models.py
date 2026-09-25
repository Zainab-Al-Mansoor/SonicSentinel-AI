"""
Train, tune and compare Python sound-classification models.

    python -m python_models.train_models              # all models, normal grids
    python -m python_models.train_models --fast       # small grids (quick check)
    python -m python_models.train_models --models svm,rf,xgb
    python -m python_models.train_models --fast --no-yamnet        # feature set v2 without TensorFlow
    python -m python_models.train_models --fast --legacy-features  # the original 299 features

Feature set v2 (default): 299 original + 94 extra hand-crafted + 53 context features
(+ 521 YAMNet AudioSet scores when TensorFlow is installed).
Class-balanced sample weights (XGBoost), per-class calibration on the VALIDATION split and a
feature-ablation table (reports/feature_ablation.csv) are part of every run.
--fast skips the cross-validated grid search: every model is fitted once with its default
parameters and compared on the validation split (much faster, same selection rule).

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
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier
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
from feature_extraction.pipeline import (segment_matrix, spec_names, spec_tag, base_names, default_spec,
                                         normalise, LEGACY_SPEC)
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
SPEC = None            # feature set used by this run (set in main(); scripts may pass their own)


def cache_tag(spec) -> str:
    spec = normalise(spec)
    if spec == LEGACY_SPEC:   # same folder as before feature set v2, so old caches are reused
        return hashlib.md5(f"{FEATURE_VERSION}-{SEG}-{HOP}-{DENOISE}-{TARGET_SR}-{len(feature_names())}".encode()).hexdigest()[:8]
    return hashlib.md5(f"{spec_tag(spec)}-{SEG}-{HOP}-{DENOISE}-{TARGET_SR}".encode()).hexdigest()[:8]


def current_spec():
    global SPEC
    if SPEC is None:
        SPEC = default_spec()
    return SPEC


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


def clip_features(row, noise_snr=None, use_cache=True, spec=None):
    spec = normalise(spec if spec is not None else current_spec())
    cache_dir = FEATURE_DIR / cache_tag(spec)
    cache_dir.mkdir(parents=True, exist_ok=True)
    # The key includes the file's size and modification time: Audio IDs are re-assigned every time
    # build_dataset.py runs, so an ID alone could return the cached features of a different clip.
    st = (BASE_DIR / row["path"]).stat()
    key = f"{row['audio_id']}_{st.st_size}_{int(st.st_mtime)}" + (f"_snr{noise_snr}" if noise_snr is not None else "")
    f = cache_dir / f"{key}.npz"
    if use_cache and f.exists():
        d = np.load(f)
        return d["X"], d["energy"]
    segs = clip_segments(BASE_DIR / row["path"], noise_snr)
    X = segment_matrix(segs, spec)
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
                                      colsample_bytree=0.6, n_jobs=-1, random_state=42),
                        {"clf__max_depth": [6] if fast else [4, 6], "clf__learning_rate": [0.1] if fast else [0.1, 0.05]})
        else:
            c["gboost"] = (GradientBoostingClassifier(random_state=42),
                           {"clf__n_estimators": [150], "clf__max_depth": [3]})
    if "mlp" in wanted:
        c["mlp"] = (MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=400, early_stopping=True, random_state=42),
                    {"clf__alpha": [1e-3] if fast else [1e-4, 1e-3]})
    return c


def seg_probs(pipe, X) -> np.ndarray:
    """Segment probabilities in CLASSES order."""
    p = pipe.predict_proba(X)
    out = np.zeros((len(p), len(CLASSES)))
    for j, k in enumerate(pipe.classes_):
        out[:, int(k)] = p[:, j]
    return out


def aggregate(P: np.ndarray, bias=None) -> dict:
    if bias is not None:
        P = P * bias
        P = P / np.clip(P.sum(axis=1, keepdims=True), 1e-12, None)
    agg, _ = aggregate_scores([dict(zip(CLASSES, row)) for row in P], MIN_CONF)
    return agg


def clip_probs(pipe, df, noise_snr=None) -> list:
    return [seg_probs(pipe, clip_features(r, noise_snr)[0]) for _, r in df.iterrows()]


def predict_clips(probs: list, bias=None) -> tuple[np.ndarray, np.ndarray]:
    preds, conf = [], []
    for P in probs:
        agg = aggregate(P, bias)
        b = max(agg, key=agg.get)
        preds.append(b); conf.append(agg[b])
    return np.array(preds), np.array(conf)


def clip_level(pipe, df, label_to_idx=None, bias=None):
    """Clip-level predictions using the same aggregation rule as the web app."""
    pred, conf = predict_clips(clip_probs(pipe, df), bias)
    return df["class_label"].to_numpy(), pred, conf


def calibrate_bias(probs: list, y_true: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Per-class factors (step 6) that maximise VALIDATION macro-F1; kept only if they help clearly."""
    def score(b):
        return f1_score(y_true, predict_clips(probs, b)[0], labels=CLASSES, average="macro", zero_division=0)
    bias = np.ones(len(CLASSES))
    base = best = score(bias)
    grid = [0.6, 0.75, 0.9, 1.0, 1.15, 1.35, 1.6, 2.0]
    for _ in range(2):
        for k in range(len(CLASSES)):
            for g in grid:
                b = bias.copy(); b[k] = g
                sc = score(b)
                if sc > best + 1e-4:
                    best, bias = sc, b
    if best - base < 0.005:
        return np.ones(len(CLASSES)), base, base
    return bias, base, best


def class_weights(y: np.ndarray) -> np.ndarray:
    """Class-balanced sample weights (step 6): rare classes count more, softened with a square root."""
    counts = np.bincount(y, minlength=len(CLASSES)).astype(float)
    w = np.sqrt(counts[counts > 0].mean() / np.clip(counts, 1, None))
    return np.clip(w, 0.5, 4.0)[y]


def ablation(Xtr, ytr, val_df, spec, max_rows=40000) -> pd.DataFrame:
    """Quick comparison of the feature groups on the VALIDATION split (same fast model for every set)."""
    names = spec_names(spec)
    nb = len(base_names(spec))
    sets = [("original 299 features", list(range(299)))]
    if nb > 299:
        sets.append(("+ extra hand-crafted", list(range(nb))))
    ctx = [i for i, n in enumerate(names) if n.startswith("ctx_")]
    if ctx:
        sets.append(("+ context", list(range(nb)) + ctx))
    if spec.get("yamnet"):
        sets.append(("+ YAMNet (all features)", list(range(len(names)))))
    rng = np.random.default_rng(0)
    rows_idx = rng.choice(len(Xtr), size=min(max_rows, len(Xtr)), replace=False)
    val_X = [clip_features(r)[0] for _, r in val_df.iterrows()]
    yv = val_df["class_label"].to_numpy()
    out = []
    for label, cols in sets:
        t0 = time.time()
        clf = HistGradientBoostingClassifier(max_iter=120, learning_rate=0.1, random_state=42)
        clf.fit(Xtr[rows_idx][:, cols], ytr[rows_idx])
        probs = []
        for X in val_X:
            p = clf.predict_proba(X[:, cols]); P = np.zeros((len(p), len(CLASSES)))
            for j, k in enumerate(clf.classes_):
                P[:, int(k)] = p[:, j]
            probs.append(P)
        pred, _ = predict_clips(probs)
        m = metrics(yv, pred)
        out.append({"features": label, "n_features": len(cols), "val_accuracy": round(m["accuracy"], 4),
                    "val_f1_macro": round(m["f1_macro"], 4), "val_critical_recall_min": round(m["critical_recall_min"], 4),
                    "seconds": round(time.time() - t0, 1)})
        print(f"  [ablation] {label:<26} {len(cols):>4} features  val macro-F1 {m['f1_macro']:.3f}  "
              f"acc {m['accuracy']:.3f}  ({time.time() - t0:.0f}s)", flush=True)
    return pd.DataFrame(out)


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
    ap.add_argument("--no-yamnet", action="store_true", help="feature set v2 without YAMNet (no TensorFlow needed)")
    ap.add_argument("--legacy-features", action="store_true", help="the original 299 features only")
    ap.add_argument("--no-ablation", action="store_true", help="skip the feature-group comparison table")
    args = ap.parse_args()

    global SPEC
    if args.legacy_features:
        SPEC = dict(LEGACY_SPEC)
    else:
        SPEC = default_spec(False if args.no_yamnet else None)
        if not args.no_yamnet and not SPEC["yamnet"]:
            from feature_extraction import embeddings
            print(f"[warn] YAMNet not available ({embeddings.unavailable_reason()[:150]}) -> training WITHOUT YAMNet. "
                  "Install: pip install tensorflow")
    print(f"Feature set: {SPEC} -> {len(spec_names(SPEC))} features per segment")

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

    if not args.no_ablation and SPEC != LEGACY_SPEC:
        print("Feature-group comparison on the validation split ...")
        abl = ablation(Xtr, ytr, val_df, SPEC)
        abl.to_csv(REPORTS / "feature_ablation.csv", index=False)

    wanted = set(args.models.split(","))
    missing = [c for c in CLASSES if c not in set(ytr_s)]
    if missing:
        print(f"[warn] no training data for: {', '.join(missing)} -> the model cannot predict these classes")
        if "xgb" in wanted:
            print("[warn] XGBoost needs every class present; skipping xgb for this run")
            wanted.discard("xgb")

    n_groups = len(set(groups))
    cv = GroupKFold(n_splits=min(3, n_groups))
    results, fitted = [], {}
    sw = class_weights(ytr)
    for name, (clf, grid) in candidates(args.fast, wanted).items():
        t0 = time.time()
        pipe = Pipeline([("scaler", StandardScaler()), ("clf", clf)])
        fit_kw = {"clf__sample_weight": sw} if name in ("xgb", "gboost") else {}
        if args.fast:
            params = {k: v[0] for k, v in grid.items()}
            pipe.set_params(**params)
            pipe.fit(Xtr, ytr, **fit_kw)
            best, cv_score = pipe, float("nan")
        else:
            gs = GridSearchCV(pipe, grid, scoring="f1_macro", cv=cv, n_jobs=1, refit=True)
            gs.fit(Xtr, ytr, groups=groups, **fit_kw)
            best, params, cv_score = gs.best_estimator_, gs.best_params_, gs.best_score_
        yv, pv, _ = clip_level(best, val_df)
        m = metrics(yv, pv)
        results.append({"model": name, "best_params": json.dumps(params), "cv_f1_macro": cv_score,
                        "val_accuracy": m["accuracy"], "val_f1_macro": m["f1_macro"],
                        "val_precision_macro": m["precision_macro"], "val_recall_macro": m["recall_macro"],
                        "val_critical_recall_min": m["critical_recall_min"], "train_seconds": round(time.time() - t0, 1)})
        fitted[name] = best
        cv_txt = "cv_f1=n/a" if np.isnan(cv_score) else f"cv_f1={cv_score:.3f}"
        print(f"[{name}] {cv_txt} val_acc={m['accuracy']:.3f} val_macroF1={m['f1_macro']:.3f} "
              f"crit_recall_min={m['critical_recall_min']:.3f}  ({time.time()-t0:.0f}s) params={params}", flush=True)

    comp = pd.DataFrame(results).sort_values(["val_f1_macro", "val_critical_recall_min"], ascending=False)
    comp.to_csv(REPORTS / "python_model_comparison.csv", index=False)
    best_name = comp.iloc[0]["model"]
    best = fitted[best_name]
    print(f"\nSelected model: {best_name}")

    # ---- per-class calibration on the VALIDATION split (step 6) ------------
    val_probs = clip_probs(best, val_df)
    bias, f1_before, f1_after = calibrate_bias(val_probs, val_df["class_label"].to_numpy())
    if f1_after > f1_before:
        print(f"Calibration: validation macro-F1 {f1_before:.3f} -> {f1_after:.3f}  factors "
              + ", ".join(f"{c}={b:.2f}" for c, b in zip(CLASSES, bias) if abs(b - 1) > 1e-9))
    else:
        print(f"Calibration: no clear gain on validation (macro-F1 {f1_before:.3f}) -> not used")

    # ---- final, one-time test evaluation ---------------------------------
    yt, pt, ct = clip_level(best, test_df, bias=bias)
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
            agg = aggregate(seg_probs(best, X), bias)
            yy.append(r["class_label"]); pp.append(max(agg, key=agg.get))
        mm = metrics(np.array(yy), np.array(pp))
        rob.append({"condition": f"white noise SNR {snr} dB", "accuracy": mm["accuracy"], "f1_macro": mm["f1_macro"]})
        print(f"noise SNR {snr} dB: acc={mm['accuracy']:.3f} f1={mm['f1_macro']:.3f}")
    pd.DataFrame(rob).to_csv(REPORTS / "noise_robustness.csv", index=False)

    version = f"py-{best_name}-{datetime.now():%Y%m%d-%H%M}"
    bundle = {
        "pipeline": best, "classes": CLASSES, "algorithm": best_name, "version": version,
        "feature_names": spec_names(SPEC), "feature_spec": SPEC,
        "class_bias": {c: float(b) for c, b in zip(CLASSES, bias)},
        "trained_at": datetime.now().isoformat(),
        "settings": {"segment_seconds": SEG, "segment_hop_seconds": HOP, "noise_reduction": DENOISE,
                     "sample_rate": TARGET_SR},
        "metrics": {"validation": comp.iloc[0].to_dict(), "test": {k: v for k, v in test_m.items() if k != "per_class"},
                    "noise_robustness": rob,
                    "calibration": {"val_f1_before": f1_before, "val_f1_after": f1_after}},
    }
    joblib.dump(bundle, PYTHON_MODEL_PATH)
    (REPORTS / "python_test_metrics.json").write_text(json.dumps({"version": version, **test_m, "noise_robustness": rob}, indent=2))
    print(f"\nSaved {PYTHON_MODEL_PATH}  (version {version})")
    print(f"TEST accuracy={test_m['accuracy']:.3f}  macroF1={test_m['f1_macro']:.3f}  "
          f"critical recall={json.dumps({k: round(v, 3) for k, v in test_m['critical_recall'].items()})}")


if __name__ == "__main__":
    main()
