"""Teachable Machine upload archives (gtm_model/prepare_gtm_samples.py --zip)."""
import json
import zipfile

import numpy as np

from gtm_model.prepare_gtm_samples import tm_frequency_frames, write_tm_zip, TM_FRAMES, TM_COLS


def test_frequency_frames_shape_and_peak():
    sr = 44100
    t = np.arange(int(sr * 1.1)) / sr
    y = (0.5 * np.sin(2 * np.pi * 1000 * t)).astype(np.float32)
    f = tm_frequency_frames(y)
    assert f.shape == (TM_FRAMES, TM_COLS)
    assert np.isfinite(f).all()
    # 1 kHz tone -> strongest bin ≈ 1000 / (44100 / 2048) ≈ 46
    assert abs(int(np.argmax(f[10])) - 46) <= 1


def test_tm_zip_has_teachable_machine_format(tmp_path):
    sr = 44100
    y = (0.3 * np.random.default_rng(0).standard_normal(sr * 2)).astype(np.float32)
    path = tmp_path / "Gunshot.zip"
    assert write_tm_zip(path, [(y, 0), (y, sr // 2)]) == 2
    with zipfile.ZipFile(path) as z:
        names = set(z.namelist())
        assert {"samples.json", "sample-1.webm"} <= names
        samples = json.loads(z.read("samples.json"))
    assert len(samples) == 2
    for s in samples:
        assert set(s) == {"frequencyFrames", "blob", "blobFilePath", "startTime", "endTime", "recordingDuration"}
        assert s["blobFilePath"] in names
        assert len(s["frequencyFrames"]) == TM_FRAMES and len(s["frequencyFrames"][0]) == TM_COLS
        assert s["endTime"] > s["startTime"]
