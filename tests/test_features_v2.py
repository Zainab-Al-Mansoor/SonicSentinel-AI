"""Feature set v2: extra hand-crafted features, context features, spec handling and class calibration."""
import numpy as np

from config.settings import CLASSES
from conftest import class_signal, SR


def _segments():
    from audio_preprocessing import preprocess_signal, segment
    clean, _ = preprocess_signal(class_signal(CLASSES.index("Gunshot"), 5.0), SR)
    return [s for _, _, s in segment(clean, 22050, 2.0, 1.0)]


def test_extra_features_are_finite_and_named():
    from feature_extraction.extra import extract_extra, extra_names
    for y in (np.zeros(44100, np.float32), (0.3 * np.random.default_rng(0).standard_normal(44100)).astype(np.float32)):
        v = extract_extra(y)
        assert v.shape == (len(extra_names()),) and np.isfinite(v).all()


def test_segment_matrix_matches_spec_names():
    from feature_extraction.pipeline import segment_matrix, spec_names, LEGACY_SPEC
    segs = _segments()
    for spec in (LEGACY_SPEC, {"version": "v2", "context": False}, {"version": "v2", "context": True}):
        X = segment_matrix(segs, spec)
        assert X.shape == (len(segs), len(spec_names(spec))) and np.isfinite(X).all()
    assert segment_matrix(segs, LEGACY_SPEC).shape[1] == 299


def test_context_is_average_of_neighbours():
    from feature_extraction.pipeline import segment_matrix, _context_cols, base_names
    spec = {"version": "v2", "context": True}
    segs = _segments()
    X = segment_matrix(segs, spec)
    n, cols = len(base_names(spec)), _context_cols(spec)
    assert np.allclose(X[1, n:], (X[0, cols] + X[2, cols]) / 2, atol=1e-5)


def test_class_bias_rescales_and_renormalises(app):
    from src.services.analysis import python_model
    with app.app_context():
        m = python_model()
        X = m.featurize(_segments())
        before = m.predict_proba(X)[0]
        old = dict(m.class_bias)
        try:
            m.class_bias = {c: (3.0 if c == "Glass Breaking" else 1.0) for c in CLASSES}
            after = m.predict_proba(X)[0]
        finally:
            m.class_bias = old
    assert abs(sum(after.values()) - 1) < 1e-6
    if 0 < before["Glass Breaking"] < 1:
        assert after["Glass Breaking"] > before["Glass Breaking"]


def test_legacy_bundle_uses_v1_features(app):
    from src.services.analysis import python_model
    with app.app_context():
        m = python_model()
    assert m.spec["version"] == "v1" and not m.spec["yamnet"]
    assert m.featurize(_segments()).shape[1] == 299
