# Testing strategy

Run all automated tests with `python -m pytest -q`. The tests use a temporary database, a temporary data folder and a tiny test-only model, so they never touch real data.

| Category (SRS) | Where |
|---|---|
| Functional / integration | `tests/test_app.py` (upload → Python → GTM → decision, review, alerts, live) |
| Boundary | too-short clip, padding of the last segment, thresholds exactly at limits (`test_decision_rules.py`) |
| Negative | unsupported format, empty file, corrupt file, silent file, wrong password, bad CSRF token |
| Security | CSRF, role-based access (403), login lock-out, protected media routes |
| Database | table creation, audit records, preserved model outputs after override |
| Audio format | WAV, FLAC, OGG, MP3 (via FFmpeg), stereo |
| Microphone / live | live window API, silent window, repeated-detection confirmation, alert acknowledgement |
| Silence / clipping / noise | `test_audio_pipeline.py` quality tests |
| Pre-processing / features | resampling, normalisation, trimming, segmentation, feature vector shape and finiteness |
| Python model | tiny model fixture + clip aggregation |
| GTM model | GTM score submission API; browser inference tested manually (see below) |
| Comparison / alert rules | `test_decision_rules.py` |
| Duplicate audio | exact duplicate (SHA-256) and near-duplicate (trimmed + quieter copy) |
| Low confidence / overlap | `test_low_confidence_and_unknown`, `test_overlapping_sounds` |

## Manual browser checks (record the results in DEVELOPMENT_LOG.md)

1. Upload one clip per class and check the waveform, spectrogram, both models' scores and the decision.
2. Upload every file in `sample_audio/test_cases/` and check the expected messages (see sample_audio/README.md).
3. Live: allow the mic, play a siren or scream from a phone, and check that the alert toast appears. Test Pause/Stop, deny the permission, and unplug a USB mic.
4. Log in as each role and check the menu items and 403 pages.
5. Admin → Model comparison → run the test-set evaluation (≥ 10 per class) → export Excel.

## Hidden-test readiness checklist

- [ ] Trained with background noise / echo / distance / device augmentation
- [ ] Tested with quiet and distant recordings (quality warnings, not crashes)
- [ ] Tested with re-encoded files (MP3 ↔ WAV) → near-duplicate notice and a correct class
- [ ] Tested with partial events (a clip that starts mid-event)
- [ ] Tested overlapping sounds (e.g. siren + horn) → overlap + manual review
- [ ] Tested the confusable pairs from the SRS (gunshot vs fireworks, scream vs shouting, alarm vs horn …)
- [ ] Thresholds tuned on the **validation** split, never on the test split
- [ ] Noise-robustness table in `reports/noise_robustness.csv` reviewed
