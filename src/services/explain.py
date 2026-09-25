"""
"Why this prediction?" – explainable AI for the Python model.

For one event this module answers three questions:

1. WHERE is the sound?  Per-segment confidence of the predicted class (Python and GTM) plus the exact
   loud part inside the strongest segment ("event located at 1.20–1.55 s").
2. WHY this class?  Feature-group occlusion: the 299 features are grouped into 13 acoustic properties
   (onset, loudness, brightness, low/mid/high-frequency energy, timbre, pitch …). Each group is set to the
   training average (0 after the StandardScaler) and the model is asked again. The drop in confidence is
   that property's contribution. This works for every model type (XGBoost, Random Forest, MLP, SVM).
3. HOW unusual is it?  The z-score of each property (StandardScaler space) says whether it is higher or
   lower than a typical training clip.

The explanation is produced by our own Python model only – no generative AI is involved.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import librosa

from config.settings import CLASSES, BACKGROUND_CLASS, IMAGE_DIR, N_MELS, TARGET_SR
from audio_preprocessing import load_audio, preprocess_signal, segment, to_mono
from feature_extraction import extract_features, feature_names
from .settings_service import get_settings

EXPLAIN_VERSION = "x1"

# (key, label, plain-language description when HIGH, when LOW)
GROUPS = [
    ("onset", "Onset / impulsiveness", "a sharp, sudden onset (impulsive sound)", "a smooth, gradual onset"),
    ("rms", "Loudness / energy", "high energy (loud)", "low energy (quiet)"),
    ("bright", "Brightness", "bright, high-pitched spectrum", "dark, low-pitched spectrum"),
    ("bandwidth", "Frequency spread", "energy spread over a wide frequency range", "energy in a narrow frequency band"),
    ("noisy", "Noisiness", "a noise-like, rough texture", "a clean, tonal texture"),
    ("mel_low", "Low-frequency energy (< 500 Hz)", "strong low-frequency energy (rumble, bass)", "little low-frequency energy"),
    ("mel_mid", "Mid-frequency energy (0.5–2 kHz)", "strong mid-frequency energy (voice range)", "little mid-frequency energy"),
    ("mel_high", "High-frequency energy (> 2 kHz)", "strong high-frequency energy (hiss, shatter, siren)", "little high-frequency energy"),
    ("mfcc", "Timbre (MFCC)", "its characteristic timbre (spectral shape)", "its characteristic timbre (spectral shape)"),
    ("dmfcc", "Timbre change (Δ-MFCC)", "the way its timbre changes over time", "the way its timbre changes over time"),
    ("chroma", "Pitch / tonal content", "clear pitched / tonal content", "little pitched content"),
    ("contrast", "Harmonic peaks", "clear harmonic peaks", "flat, peak-less spectrum"),
    ("tempo", "Rhythm / repetition", "a fast repeating rhythm", "a slow or no rhythm"),
    # feature set v2 (only present when the model was trained with it)
    ("impulse", "Impulsiveness (crest, attack, flux)", "a very short, explosive burst", "a steady, drawn-out sound"),
    ("voice", "Voice / pitch (F0, harmonicity)", "a clear voiced, harmonic sound", "little voiced or harmonic content"),
    ("bands", "Frequency-band balance", "energy concentrated in its typical frequency bands", "an untypical frequency balance"),
    ("pcen", "Noise-robust spectrum (PCEN)", "a spectrum that stands out from the background", "a spectrum close to the background"),
    ("context", "Neighbouring segments", "similar sound in the neighbouring seconds", "different sound in the neighbouring seconds"),
    ("yamnet", "YAMNet (AudioSet knowledge)", "patterns YAMNet knows from millions of AudioSet clips", "patterns YAMNet does not associate with it"),
]


def _names(model=None) -> list[str]:
    if model is not None and getattr(model, "spec", None) is not None:
        from feature_extraction.pipeline import spec_names
        return spec_names(model.spec)
    return feature_names()


def _group_indices(names=None) -> dict[str, list[int]]:
    names = names or feature_names()
    mel_f = librosa.mel_frequencies(n_mels=N_MELS + 2, fmax=TARGET_SR / 2)[1:-1]
    idx = {g[0]: [] for g in GROUPS}
    for i, n in enumerate(names):
        if n.startswith("ctx_"):
            idx["context"].append(i)
        elif n.startswith("yamnet"):
            idx["yamnet"].append(i)
        elif n.startswith("pcen"):
            idx["pcen"].append(i)
        elif n.startswith(("crest", "attack", "decay", "flux", "onset_rate")):
            idx["impulse"].append(i)
        elif n.startswith(("f0_", "harmonic", "percussive")):
            idx["voice"].append(i)
        elif n.startswith("band") and (n[4:5].isdigit() or n.startswith("band_high")):
            idx["bands"].append(i)
        elif n.startswith(("rmsdb", "env_")):
            idx["rms"].append(i)
        elif n.startswith(("centroid_p", "spec_")):
            idx["bright"].append(i)
        elif n.startswith("onset"):
            idx["onset"].append(i)
        elif n.startswith("rms"):
            idx["rms"].append(i)
        elif n.startswith(("centroid", "rolloff")):
            idx["bright"].append(i)
        elif n.startswith("bandwidth"):
            idx["bandwidth"].append(i)
        elif n.startswith(("zcr", "flatness")):
            idx["noisy"].append(i)
        elif n.startswith("mel"):
            b = int(n[3:].split("_")[0])
            f = mel_f[b]
            idx["mel_low" if f < 500 else ("mel_mid" if f < 2000 else "mel_high")].append(i)
        elif n.startswith("dmfcc"):
            idx["dmfcc"].append(i)
        elif n.startswith("mfcc"):
            idx["mfcc"].append(i)
        elif n.startswith("chroma"):
            idx["chroma"].append(i)
        elif n.startswith("contrast"):
            idx["contrast"].append(i)
        elif n == "tempo":
            idx["tempo"].append(i)
    return idx


def _level_indices(idx: dict[str, list[int]], names=None) -> dict[str, list[int]]:
    """Features that say whether a property is HIGH or LOW (means / maxima, not std)."""
    names = names or feature_names()
    out = {}
    for k, ids in idx.items():
        sel = [i for i in ids if names[i].endswith(("_mean", "_max")) or names[i] == "tempo"]
        out[k] = sel or ids
    return out


def _proba(model, Z: np.ndarray) -> np.ndarray:
    """Class probabilities in CLASSES order from already-scaled features Z."""
    clf = model.pipeline[-1]
    p = clf.predict_proba(Z)
    raw = np.asarray(getattr(clf, "classes_", np.arange(p.shape[1])))
    names = [model.classes[int(c)] for c in raw] if np.issubdtype(raw.dtype, np.integer) else [str(c) for c in raw]
    out = np.zeros((p.shape[0], len(CLASSES)))
    for j, name in enumerate(names):
        if name in CLASSES:
            out[:, CLASSES.index(name)] = p[:, j]
    bias = getattr(model, "class_bias", None)
    if bias:
        out = out * np.array([bias.get(c, 1.0) for c in CLASSES])
        out = out / np.clip(out.sum(axis=1, keepdims=True), 1e-12, None)
    return out


def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def _contributions(model, Z: np.ndarray, t: int, idx: dict[str, list[int]]) -> tuple[dict, str]:
    """Contribution of each feature group to the target class, in log-odds.

    XGBoost: exact TreeSHAP values (booster.predict(pred_contribs=True)), summed per group.
    Other models: occlusion – group set to the training average, drop in log-odds."""
    clf = model.pipeline[-1]
    if clf.__class__.__name__.startswith("XGB"):
        import xgboost as xgb
        raw = np.asarray(clf.classes_)
        names = [model.classes[int(c)] for c in raw] if np.issubdtype(raw.dtype, np.integer) else [str(c) for c in raw]
        col = names.index(CLASSES[t])
        c = clf.get_booster().predict(xgb.DMatrix(Z), pred_contribs=True)
        c = c[0, col, :-1] if c.ndim == 3 else c[0, :-1]
        return {k: float(np.sum(c[ids])) for k, ids in idx.items()}, "TreeSHAP (exact XGBoost feature contributions)"
    base = _logit(_proba(model, Z)[0, t])
    batch = []
    for key, *_ in GROUPS:
        z = Z.copy()
        if not idx[key]:
            batch.append(z[0]); continue
        z[0, idx[key]] = 0.0                         # training average
        batch.append(z[0])
    occl = _logit(_proba(model, np.vstack(batch))[:, t])
    return {key: float(base - o) for (key, *_), o in zip(GROUPS, occl)}, "feature-group occlusion (log-odds)"


def explain_vector(model, x: np.ndarray, target: str | None = None) -> dict:
    """Explanation for one 299-value feature vector: how much each acoustic property pushed the model
    towards (positive) or away from (negative) the target class."""
    Z = model.pipeline[:-1].transform(x.reshape(1, -1))
    base = _proba(model, Z)[0]
    t = CLASSES.index(target) if target in CLASSES else int(np.argmax(base))
    names = _names(model)
    if len(names) != Z.shape[1]:
        names = feature_names() if Z.shape[1] == len(feature_names()) else [f"f{i}" for i in range(Z.shape[1])]
    idx = _group_indices(names)
    lvl = _level_indices(idx, names)
    contrib, method = _contributions(model, Z, t, idx)
    total = sum(abs(v) for v in contrib.values()) or 1.0
    rows = []
    for key, label, hi, lo in GROUPS:
        if not idx[key]:
            continue
        zscore = float(np.mean(Z[0, lvl[key]])) if lvl[key] else 0.0
        v = contrib[key]
        rows.append({"key": key, "label": label, "impact": round(v, 4), "share": round(v / total, 4),
                     "z": round(zscore, 2), "level": "high" if zscore >= 0 else "low",
                     "description": hi if zscore >= 0 else lo, "n_features": len(idx[key])})
    rows.sort(key=lambda r: -r["impact"])
    return {"target": CLASSES[t], "confidence": round(float(base[t]), 4), "groups": rows, "method": method}


def _event_span(y: np.ndarray, sr: int, start: float, end: float) -> tuple[float, float]:
    """Loud part of [start, end]: frames within 10 dB of the segment's peak, around the peak."""
    a, b = int(start * sr), int(end * sr)
    seg = y[a:b]
    if len(seg) < sr // 20:
        return start, end
    hop = max(1, sr // 100)                                          # 10 ms
    rms = librosa.feature.rms(y=seg, frame_length=hop * 4, hop_length=hop)[0]
    db = 20 * np.log10(rms + 1e-9)
    pk = int(np.argmax(db))
    on = db >= db[pk] - 10
    i, j = pk, pk
    while i > 0 and on[i - 1]:
        i -= 1
    while j < len(on) - 1 and on[j + 1]:
        j += 1
    s = start + i * hop / sr
    e = min(end, start + (j + 1) * hop / sr + 0.04)
    return round(s, 2), round(max(e, s + 0.05), 2)


def _paths(ev):
    base = Path(IMAGE_DIR) / f"{ev.audio_id}_explain"
    return base.with_suffix(".json"), base.with_suffix(".png")


def image_path(ev) -> str | None:
    p = _paths(ev)[1]
    return str(p) if p.exists() else None


def _draw(y, sr, segs, target, span, png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import librosa.display
    from feature_extraction.visuals import BG, FG, INK, _style
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 4.6), dpi=110, sharex=True,
                                   gridspec_kw={"height_ratios": [3, 1.4]})
    fig.patch.set_facecolor(BG)
    S = librosa.power_to_db(librosa.feature.melspectrogram(y=y, sr=sr, n_mels=96, n_fft=2048, hop_length=512), ref=np.max)
    librosa.display.specshow(S, sr=sr, hop_length=512, x_axis="time", y_axis="mel", ax=ax1, cmap="magma")
    for s in segs:
        if s["python"] is not None and s["python"] >= s["threshold"]:
            ax1.axvspan(s["start"], s["end"], color="#FFE0B2", alpha=0.12)
    if span:
        ax1.axvspan(span[0], span[1], facecolor="none", edgecolor="#FFE0B2", linewidth=2.2)
        ax1.text(span[0], ax1.get_ylim()[1] * 0.92, f"  {target}", color="#FFF2DF", fontsize=9, fontweight="bold", va="top")
    ax1.set_xlabel("")
    ax1.set_ylabel("Hz (Mel)", color=FG, fontsize=8)
    _style(ax1, f"Where the model heard “{target}”")
    xs = [(s["start"] + s["end"]) / 2 for s in segs]
    ax2.plot(xs, [s["python"] or 0 for s in segs], marker="o", ms=3, color="#5B3A31", label="Python")
    if any(s["gtm"] is not None for s in segs):
        ax2.plot(xs, [s["gtm"] or 0 for s in segs], marker="s", ms=3, color="#C0843F", label="GTM")
    if segs:
        ax2.axhline(segs[0]["threshold"], color="#B7A08A", linestyle=":", linewidth=1)
    ax2.set_ylim(0, 1.02)
    ax2.set_xlim(0, max(len(y) / sr, 1e-3))
    ax2.set_ylabel("Confidence", color=FG, fontsize=8)
    ax2.set_xlabel("Time (s)", color=FG, fontsize=8)
    ax2.legend(loc="upper right", fontsize=7, frameon=False, labelcolor=INK)
    _style(ax2, f"Confidence for “{target}” per 2-s segment")
    fig.tight_layout()
    fig.savefig(png, facecolor=BG)
    plt.close(fig)


def explain_event(ev, model) -> dict:
    """Full explanation for an AudioEvent (cached per model version)."""
    js, png = _paths(ev)
    key = f"{EXPLAIN_VERSION}|{model.version}|{ev.gtm_model_version}|{ev.gtm_status}|{ev.final_category}"
    if js.exists() and png.exists():
        try:
            cached = json.loads(js.read_text(encoding="utf-8"))
            if cached.get("cache_key") == key:
                return cached
        except Exception:
            pass
    if not ev.stored_path or not Path(ev.stored_path).exists():
        raise FileNotFoundError("The audio of this event was removed by the data-retention policy.")

    s = get_settings()
    target = ev.python_prediction or ev.final_category
    if target not in CLASSES:
        target = None
    a = load_audio(ev.stored_path)
    raw = to_mono(a.samples).astype(np.float32)
    clean, offset = preprocess_signal(a.samples, a.sample_rate, trim=True, denoise=s["noise_reduction"])
    segs_audio = segment(clean, TARGET_SR, s["segment_seconds"], s["segment_hop_seconds"])
    if not segs_audio:
        raise ValueError("No audio segments to explain.")

    # which segment to explain: the one the clip decision used, else the strongest for the target
    stored = sorted(ev.segments, key=lambda x: x.index)
    pos = ev.python_segment_index if ev.python_segment_index is not None and ev.python_segment_index >= 0 else None
    if pos is None or pos >= len(segs_audio):
        tgt = target or CLASSES[0]
        vals = [(sg.python_scores or {}).get(tgt, 0.0) for sg in stored] or [0.0]
        pos = int(np.argmax(vals)) if len(vals) == len(segs_audio) else 0
    st, en, seg = segs_audio[pos]
    if hasattr(model, "featurize"):
        x = model.featurize([sg for _, _, sg in segs_audio])[pos]
    else:
        x = extract_features(seg)
    res = explain_vector(model, x, target)
    target = res["target"]

    a0, b0 = st + offset, en + offset
    span = _event_span(raw, a.sample_rate, a0, min(b0, len(raw) / a.sample_rate))
    th = float(s["min_confidence"])
    seg_rows = [{"index": sg.index, "start": round(sg.start_s, 2), "end": round(sg.end_s, 2),
                 "python": (sg.python_scores or {}).get(target), "gtm": (sg.gtm_scores or {}).get(target) if sg.gtm_scores else None,
                 "threshold": th} for sg in stored]
    strong = [r for r in seg_rows if r["python"] is not None and r["python"] >= th]

    pro = [g for g in res["groups"] if g["share"] > 0.03][:3]
    con = sorted([g for g in res["groups"] if g["share"] < -0.03], key=lambda g: g["impact"])[:2]
    if target == BACKGROUND_CLASS:
        headline = "No distinct event: the sound looks like ordinary background noise"
    else:
        headline = f"{target} located at {span[0]:.2f}–{span[1]:.2f} s"
    because = ", ".join(g["description"] for g in pro) or "the overall combination of features (no single property dominates)"
    summary = f"{target} ({res['confidence'] * 100:.0f}%): mainly because of {because}."
    against = ("Pulling the other way: " + ", ".join(g["description"] for g in con) + ".") if con else ""

    _draw(raw, a.sample_rate, seg_rows, target, span if target != BACKGROUND_CLASS else None, png)
    out = {
        "cache_key": key, "audio_id": ev.audio_id, "model_version": model.version,
        "event_model_version": ev.python_model_version, "target": target,
        "confidence": res["confidence"], "headline": headline, "summary": summary, "against": against,
        "span": {"start": span[0], "end": span[1]}, "segment_used": {"index": pos, "start": round(a0, 2), "end": round(b0, 2)},
        "segments_over_threshold": len(strong), "segments": seg_rows, "groups": res["groups"],
        "method": res["method"],
    }
    js.write_text(json.dumps(out), encoding="utf-8")
    return out
