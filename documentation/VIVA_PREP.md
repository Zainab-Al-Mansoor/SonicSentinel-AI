# Evaluation-day preparation (SRS 1.8 – competition integrity)

SRS 1.8 says evaluators may ask **any member** to explain a module or function, make a **surprise modification**,
**fix a deliberate defect**, and test with **unseen recordings**. This guide prepares each member for that.

## 1. Who explains what (fill in names)

| Member | Modules | Must be able to explain |
|---|---|---|
| _name_ | `audio_preprocessing/` | `load_audio` (soundfile, FFmpeg fallback), `validate_file` / `validate_audio`, `analyze_quality` (RMS, clipping, SNR, silence → Good/Acceptable/Poor/Unusable), `preprocess_signal` (mono → 22,050 Hz → normalise → trim → noise reduction), `segment` (2-s windows, 1-s hop, padding) |
| _name_ | `feature_extraction/` | `extract_features` (the 299 values: MFCC+Δ, Mel, chroma, contrast, ZCR/RMS/centroid/bandwidth/roll-off/onset, flatness, tempo), `save_waveform`, `save_spectrogram`, `compute_fingerprint` + `similarity` (near-duplicates) |
| _name_ | `python_models/`, `augmentation/`, `scripts/build_dataset.py` | 70/15/15 split by original clip, GroupKFold by Audio ID (no leakage), `train_models.main` (grids, selection by validation macro-F1), `aggregate_scores` (strongest segment vs mean), augmentation operations |
| _name_ | `gtm_model/`, `static/js/gtm.js` | Teachable Machine training (`--zip` upload archives), `spectrogram()` in gtm.js (43 × 232, FFT 2048, Blackman, z-norm), why GTM runs in the browser and never sees the Python result |
| _name_ | `src/services/decision.py`, `rules.py`, `analysis.py`, `src/routes/`, `tests/` | `compare_models` (Acceptable Match / Weak Match / Model Disagreement / Uncertain Result), `combine`, `decide` (unknown, overlap, rules, severity, manual review), `evaluate_rule`, `analyze_upload`, `submit_gtm`, `finalize`, `analyze_live_window`, roles and CSRF |

**One-minute pipeline answer:** upload → `validate_file` → `load_audio` → `analyze_quality` → SHA-256 + fingerprint
(duplicates) → `preprocess_signal` → `segment` → `extract_features` → Python model per segment → `aggregate_scores`
→ browser runs GTM on the same segments → `submit_gtm` → `compare_models` → `decide` with `alert_rules.json` →
severity, alert, manual review → stored in `audio_events` / `segments` / `alerts` → dashboards.

## 2. Surprise modifications – where to change what

| Request | Change | Then |
|---|---|---|
| Change the confidence threshold | Admin → Settings → *min confidence* (runtime), or `DEFAULT_RUNTIME_SETTINGS["min_confidence"]` in `config/settings.py` | no restart needed for Admin changes |
| Per-class threshold / severity / action | Admin → Alert rules (writes `alert_rules/alert_rules.json`) | validated by `rules.validate_rule` |
| Number of repeated detections | Alert rules → *required consecutive*; time window: Settings → *repeat window seconds* | |
| Add an audio format (e.g. `aac`) | add to `SUPPORTED_FORMATS` in `config/settings.py`; FFmpeg decodes it in `loader._decode_with_ffmpeg` | add a test in `tests/test_audio_pipeline.py` |
| Change maximum upload size | `MAX_UPLOAD_MB` in `config/settings.py` | |
| Change segment length | Admin → Settings → *segment seconds* / *hop* / *live window seconds* | retrain the Python model with the same length |
| New history filter | `filtered_events()` in `src/routes/main.py` (add `if args.get("x"): q = q.filter(...)`) + a field in `templates/history.html` | |
| New sound class | add to `CLASSES` in `config/settings.py` → data in `audio_dataset/raw/<Class>/` → `build_dataset.py` → `augment` → `train_models` → GTM class with the same name → rule in Admin | |
| New alert action / role | `ROLES` in `config/settings.py`, `roles_required` decorators in `src/routes/` | |
| Change retention period | Admin → Settings → audio / record retention days (`services/retention.py`) | |

## 3. Fixing a deliberate defect – checklist

1. Reproduce it and read the traceback (`python run.py` terminal, or F12 → Console for GTM).
2. Run `python -m pytest -q`; the failing test names point to the module.
3. Common places a defect is planted:
   * wrong threshold comparison (`>` vs `>=`) in `rules.evaluate_rule` or `decision.compare_models`;
   * class order / label names (`CLASSES`, GTM `metadata.json`);
   * sample rate (`TARGET_SR` 22,050 for Python, 44,100 for GTM);
   * segment hop, padding in `segment()`;
   * missing `csrf_token` in a form, or a missing `@login_required` / `roles_required`;
   * split leakage in `build_dataset.py` (split must be by original clip);
   * wrong column in `comparison_dataframe` / exports.
4. Fix, re-run the tests, and add a test that would have caught it.

## 4. Unseen recordings (hidden test)

* Measured: `python scripts/hidden_test_robustness.py` → `reports/hidden_test_robustness.md` (noise, real background,
  echo, low volume, other device, distance, partial event, MP3 re-encoding, overlap, confusable pairs).
* The app never crashes on bad input (validation + tests); hard cases become *Uncertain* or go to manual review.
* Before the demo: record 30 s of the demo room as Background Noise, check Admin → Settings thresholds.

## 5. AI usage questions

Every AI-assisted file is listed in `AI_USAGE.md`. Each member should read their rows, be able to explain the code in
their own words, and fill in *Student modifications* and *Verified by*. Classification never uses a generative-AI API.
