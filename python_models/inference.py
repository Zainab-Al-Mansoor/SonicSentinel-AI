"""
Loading the saved Python model and turning segment features into confidence
scores for ALL classes. Also holds the clip-level aggregation rule that is
used identically for the Python model and the GTM model.
"""
from pathlib import Path

import joblib
import numpy as np

from config.settings import CLASSES, BACKGROUND_CLASS, PYTHON_MODEL_PATH


class ModelNotAvailable(Exception):
    pass


class PythonSoundModel:
    """Wrapper around the joblib bundle written by python_models/train_models.py"""

    def __init__(self, path: Path = PYTHON_MODEL_PATH):
        if not Path(path).exists():
            raise ModelNotAvailable(
                "No trained Python model found. Run: python -m python_models.train_models")
        bundle = joblib.load(path)
        self.pipeline = bundle["pipeline"]
        self.classes = list(bundle["classes"])
        self.version = bundle["version"]
        self.algorithm = bundle["algorithm"]
        self.metrics = bundle.get("metrics", {})
        self.feature_names = bundle.get("feature_names")
        self.settings = bundle.get("settings", {})
        # feature set the model was trained with (older bundles: the original 299 features)
        from feature_extraction.pipeline import normalise
        self.spec = normalise(bundle.get("feature_spec"))
        # per-class calibration factors found on the VALIDATION split (1.0 = unchanged)
        bias = bundle.get("class_bias") or {}
        self.class_bias = {c: float(bias.get(c, 1.0)) for c in CLASSES}
        if self.spec.get("yamnet"):
            from feature_extraction import embeddings
            if not embeddings.available():
                raise ModelNotAvailable(
                    "This model uses YAMNet features, but TensorFlow could not load YAMNet "
                    f"({embeddings.unavailable_reason()}). Install it with: pip install tensorflow")

    def featurize(self, segments: list) -> np.ndarray:
        """Model input matrix for the segments of ONE recording (same features as in training)."""
        from feature_extraction.pipeline import segment_matrix
        return segment_matrix(segments, self.spec)

    def segment_scores(self, segments: list) -> list[dict]:
        """Scores for every segment of one recording (features + prediction in one call)."""
        return self.predict_proba(self.featurize(segments))

    def predict_proba(self, X: np.ndarray) -> list[dict]:
        """Returns one {class: probability} dict per row, covering every class in CLASSES."""
        proba = self.pipeline.predict_proba(X)
        model_classes = [self.classes[i] for i in self.pipeline.classes_] \
            if np.issubdtype(np.asarray(self.pipeline.classes_).dtype, np.integer) else list(self.pipeline.classes_)
        out = []
        for row in proba:
            d = {c: 0.0 for c in CLASSES}
            for c, p in zip(model_classes, row):
                d[c] = float(p) * self.class_bias.get(c, 1.0)
            tot = sum(d.values()) or 1.0
            out.append({c: v / tot for c, v in d.items()})
        return out


def top_k(scores: dict, k: int = 3) -> list[tuple[str, float]]:
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:k]


def top_margin(scores: dict) -> float:
    t = top_k(scores, 2)
    return t[0][1] - (t[1][1] if len(t) > 1 else 0.0)


def aggregate_scores(segment_scores: list[dict], min_conf: float) -> tuple[dict, int]:
    """
    Clip-level scores from segment-level scores (same rule for both models):
      * if any segment has a NON-background class with confidence >= min_conf,
        the clip takes the scores of the strongest such segment (a 0.3 s gunshot
        in a 20 s clip must not be averaged away);
      * otherwise the clip takes the mean over all segments.
    Returns (scores, index_of_segment_used or -1 for mean).
    """
    if not segment_scores:
        return {c: 0.0 for c in CLASSES}, -1
    best_i, best_v = -1, -1.0
    for i, s in enumerate(segment_scores):
        cls, v = top_k(s, 1)[0]
        if cls != BACKGROUND_CLASS and v >= min_conf and v > best_v:
            best_i, best_v = i, v
    if best_i >= 0:
        return dict(segment_scores[best_i]), best_i
    keys = segment_scores[0].keys()
    mean = {k: float(np.mean([s.get(k, 0.0) for s in segment_scores])) for k in keys}
    return mean, -1
