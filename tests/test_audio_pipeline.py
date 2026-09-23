"""Unit tests: validation, pre-processing, quality, features, fingerprint, augmentation."""
import shutil

import numpy as np
import pytest

from audio_preprocessing import (validate_file, analyze_quality, preprocess_signal, segment, to_mono,
                                 resample, peak_normalize, pad_or_truncate, load_audio)
from augmentation.augment import add_noise, time_shift, pitch_shift, time_stretch, volume, reverb, distance, device
from feature_extraction import extract_features, feature_names, compute_fingerprint, similarity
from config.settings import TARGET_SR
from conftest import tone, SR


# ---------------- validation / audio-format tests ----------------
def test_valid_wav(wav_file):
    v = validate_file(wav_file(tone()))
    assert v.ok and v.audio.sample_rate == SR and v.audio.channels == 1


@pytest.mark.parametrize("ext,subtype", [("flac", "PCM_16"), ("ogg", "VORBIS")])
def test_other_formats(wav_file, ext, subtype):
    assert validate_file(wav_file(tone(), f"a.{ext}", subtype=subtype)).ok


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="FFmpeg not installed")
def test_mp3_via_ffmpeg(wav_file, tmp_path):
    import subprocess
    src = wav_file(tone())
    mp3 = tmp_path / "a.mp3"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(src), str(mp3)], check=True)
    v = validate_file(mp3)
    assert v.ok and abs(v.audio.duration - 2.0) < 0.1


def test_unsupported_format(tmp_path):
    p = tmp_path / "notes.txt"; p.write_text("hello")
    v = validate_file(p)
    assert not v.ok and "Unsupported format" in v.errors[0]


def test_empty_and_corrupt(tmp_path):
    e = tmp_path / "empty.wav"; e.write_bytes(b"")
    assert "empty" in validate_file(e).errors[0]
    c = tmp_path / "bad.wav"; c.write_bytes(b"RIFF\x00\x00garbage" * 50)
    assert not validate_file(c).ok


def test_silent_rejected(wav_file):
    v = validate_file(wav_file(np.zeros(SR * 2, np.float32)))
    assert not v.ok and any("silent" in e for e in v.errors)


def test_too_short_rejected(wav_file):
    v = validate_file(wav_file(tone(seconds=0.2)))
    assert not v.ok and any("too short" in e for e in v.errors)


def test_stereo_metadata(wav_file):
    st = np.stack([tone(), tone(660)], axis=1)
    a = load_audio(wav_file(st))
    assert a.channels == 2 and to_mono(a.samples).ndim == 1


# ---------------- pre-processing ----------------
def test_resample_and_normalize():
    y = resample(tone(), SR, TARGET_SR)
    assert abs(len(y) - 2 * TARGET_SR) <= 2
    assert np.isclose(np.max(np.abs(peak_normalize(y))), 0.95, atol=1e-3)


def test_segmentation_and_padding():
    y = np.ones(int(5.3 * TARGET_SR), np.float32)
    segs = segment(y, TARGET_SR, 2.0, 1.0)
    assert all(len(s) == 2 * TARGET_SR for _, _, s in segs)
    assert segs[0][0] == 0 and segs[1][0] == 1.0
    short = segment(np.ones(TARGET_SR // 2, np.float32), TARGET_SR, 2.0, 1.0)
    assert len(short) == 1 and len(short[0][2]) == 2 * TARGET_SR
    assert len(pad_or_truncate(np.ones(10), 4)) == 4


def test_preprocess_trims_silence():
    y = np.concatenate([np.zeros(SR), tone(seconds=1.0), np.zeros(SR)])
    clean, offset = preprocess_signal(y, SR, denoise=False)
    assert 0.8 < offset < 1.1 and len(clean) < 2 * TARGET_SR


# ---------------- quality ----------------
def test_quality_good_tone():
    assert analyze_quality(tone(), SR)["label"] == "Good"


def test_quality_silence_unusable():
    assert analyze_quality(np.zeros(SR * 2, np.float32), SR)["label"] == "Unusable"


def test_quality_clipping():
    q = analyze_quality(np.clip(tone(amp=3.0), -1, 1), SR)
    assert q["label"] in ("Poor", "Unusable") and any("clipping" in i.lower() for i in q["issues"])


def test_quality_low_signal():
    q = analyze_quality(tone(amp=0.002, noise=0.0), SR)
    assert any("signal strength" in i for i in q["issues"])


def test_quality_noise():
    q = analyze_quality((0.2 * np.random.default_rng(1).standard_normal(SR * 2)).astype(np.float32), SR)
    assert any("noise" in i.lower() for i in q["issues"])


def test_quality_missing_frames():
    y = tone(seconds=3.0)
    y[SR: SR + 4000] = 0.0
    assert analyze_quality(y, SR)["dropouts"] >= 1


# ---------------- features ----------------
def test_feature_vector():
    f = extract_features(resample(tone(), SR, TARGET_SR))
    assert f.shape == (len(feature_names()),) and np.all(np.isfinite(f))
    assert np.allclose(f, extract_features(resample(tone(), SR, TARGET_SR)))   # deterministic


def test_features_on_silence_are_finite():
    assert np.all(np.isfinite(extract_features(np.zeros(2 * TARGET_SR, np.float32))))


# ---------------- augmentation ----------------
def test_augmentations_keep_length_and_finite():
    y = resample(tone(), SR, TARGET_SR)
    for out in (add_noise(y, 10), time_shift(y), volume(y, -6), reverb(y, TARGET_SR), distance(y, TARGET_SR),
                device(y, TARGET_SR), pitch_shift(y, TARGET_SR, 1)):
        assert len(out) == len(y) and np.all(np.isfinite(out))
    assert np.all(np.isfinite(time_stretch(y, 1.1)))


# ---------------- duplicate audio ----------------
def test_near_duplicate_trimmed_and_quieter():
    rng = np.random.default_rng(3)
    t = np.arange(SR * 5) / SR
    env = np.zeros_like(t)
    for s in (0.5, 2.1, 3.7):
        env[(t > s) & (t < s + 0.3)] = 1
    y = (rng.standard_normal(len(t)) * env * 0.5 + 0.01 * rng.standard_normal(len(t))).astype(np.float32)
    a, b = compute_fingerprint(y, SR), compute_fingerprint(0.5 * y[SR // 2:], SR)
    assert similarity(a, b) >= 0.9
    other = compute_fingerprint(tone(1500, 5.0), SR)
    assert similarity(a, other) < 0.9
