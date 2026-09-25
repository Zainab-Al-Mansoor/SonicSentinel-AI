"""
One place that turns the segments of ONE recording into the model's input matrix.

The feature set is described by a small "spec" that is saved inside the model bundle, so the web app always computes
exactly the features the loaded model was trained with:

    {"version": "v1", "context": False, "yamnet": False}   299 features  (models trained before feature set v2)
    {"version": "v2", "context": True,  "yamnet": True}    299 + 94 extra + 53 context + 521 YAMNet = 967 features

  v2       = v1 + extra hand-crafted features (feature_extraction/extra.py)
  context  = average of the previous and next segment for a compact subset of features (step 4)
  yamnet   = pretrained AudioSet knowledge (feature_extraction/embeddings.py, step 2)
"""
import hashlib
import json

import numpy as np

from config.settings import TARGET_SR
from .features import extract_features, feature_names, FEATURE_VERSION
from .extra import extract_extra, extra_names

LEGACY_SPEC = {"version": "v1", "context": False, "yamnet": False}


def default_spec(yamnet: bool | None = None) -> dict:
    if yamnet is None:
        from . import embeddings
        yamnet = embeddings.available()
    return {"version": "v2", "context": True, "yamnet": bool(yamnet)}


def normalise(spec: dict | None) -> dict:
    s = dict(LEGACY_SPEC)
    s.update(spec or {})
    return s


def base_names(spec) -> list[str]:
    spec = normalise(spec)
    return feature_names() + (extra_names() if spec["version"] == "v2" else [])


def _context_cols(spec) -> list[int]:
    names = base_names(spec)
    want = [f"mfcc{i}_mean" for i in range(40)] + ["rms_mean", "centroid_mean", "onset_mean"]
    if spec["version"] == "v2":
        want += ["flux_mean", "crest_log", "harmonic_share"] + [f"band{i}_share" for i in range(7)]
    return [names.index(w) for w in want if w in names]


def spec_names(spec) -> list[str]:
    spec = normalise(spec)
    names = base_names(spec)
    if spec["context"]:
        names = names + ["ctx_" + names[i] for i in _context_cols(spec)]
    if spec["yamnet"]:
        from .embeddings import yamnet_names
        names = names + yamnet_names()
    return names


def spec_tag(spec) -> str:
    spec = normalise(spec)
    raw = json.dumps(spec, sort_keys=True) + f"|{FEATURE_VERSION}|{len(spec_names(spec))}"
    return hashlib.md5(raw.encode()).hexdigest()[:6]


def segment_matrix(segments: list[np.ndarray], spec=None, sr: int = TARGET_SR) -> np.ndarray:
    """(n_segments x n_features) for the segments of one recording, in time order."""
    spec = normalise(spec)
    if not segments:
        return np.zeros((0, len(spec_names(spec))), dtype=np.float32)
    rows = []
    for s in segments:
        v = extract_features(s, sr)
        if spec["version"] == "v2":
            v = np.concatenate([v, extract_extra(s, sr)])
        rows.append(v)
    X = np.vstack(rows).astype(np.float32)
    parts = [X]
    if spec["context"]:
        cols = _context_cols(spec)
        B = X[:, cols]
        prev = np.vstack([B[:1], B[:-1]])
        nxt = np.vstack([B[1:], B[-1:]])
        parts.append((prev + nxt) / 2)
    if spec["yamnet"]:
        from .embeddings import yamnet_scores
        parts.append(yamnet_scores(segments, sr))
    return np.hstack(parts).astype(np.float32)
