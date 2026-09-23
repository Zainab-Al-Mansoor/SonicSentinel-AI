# SonicSentinel AI

**AI-powered, web-based, real-time sound-event detection** (Theme: AcousticX Intelligence · Category: NextWave AI and ML)

SonicSentinel AI analyses **uploaded audio clips** and **live microphone input**. Two **independently trained** models classify every audio segment:

| | Python model | Google Teachable Machine (GTM) model |
|---|---|---|
| Runs on | Flask server | the user's browser (TensorFlow.js – the official GTM export) |
| Input | acoustic features (MFCC, Mel, chroma, spectral …) | 1-second spectrograms of the same audio segment |
| Trained on | `audio_dataset` TRAIN split | 1-second samples cut from the **same** TRAIN recordings |

The app compares the two predictions and their confidence scores. It also checks audio quality, detects uncertain and overlapping sounds, applies configurable alert rules and assigns a severity. Doubtful results go to a manual-review queue. Everything is stored for event history, dashboards and reports.

> ⚠️ This is a competition prototype. It is **not** a certified emergency-response or law-enforcement system.

---

## 1. Features (SRS mapping)

| Area | What is implemented |
|---|---|
| Users | Registration, login/logout, lock-out after 5 failed logins, profile, unique User ID (`USR-00001`), 5 roles (user, reviewer, security, maintenance, admin) |
| Input | Single upload, batch upload, live microphone (start / pause / stop, status: Available · Active · Paused · Disconnected · Permission denied), privacy banner while the mic is on |
| Validation | Format (WAV/MP3/FLAC/OGG/M4A), size, duration, sample rate, channels, integrity, silence |
| Pre-processing | Resampling, mono, normalisation, silence trimming, noise reduction, fixed-duration segmentation with timestamps, padding/truncation |
| Quality | Silence, clipping, noise/SNR, low level, duration, encoding problems, missing frames → Good / Acceptable / Poor / Unusable |
| Visuals | Waveform (with segments and the deciding segment highlighted) and Mel spectrogram |
| Python ML | 299 acoustic features; SVM, Random Forest, XGBoost and MLP compared with tuning; confusion matrix, macro-F1, critical recall, noise robustness |
| GTM | Browser-side TF.js inference on the **same** segments; the Python result is never given to GTM |
| Comparison | Class match, \|Python top − GTM top\| confidence difference, top-2 margins, top-3 of each model, status: Acceptable Match / Weak Match / Model Disagreement / Uncertain Result |
| Decision | Combined scores, Unknown handling, overlap detection, repeated-detection confirmation, rule engine, severity (Informational → Critical), recommended action, trace of every step |
| Alerts | Real-time alerts, toasts, acknowledge / dismiss / escalate, alert history, alerts routed to the right role |
| Review | Queue, segment playback, confirm/correct, comments, override (original model outputs kept) |
| Dashboards | User, live, admin (trends, categories, confidence, quality, FP/FN, disagreements, alert response) |
| History | Search and filter (ID, filename, class, dates, confidence, severity, quality, review status, user), timeline |
| Reports | Downloadable per-event report (HTML → print to PDF), CSV/Excel exports, model-comparison report |
| Admin | Thresholds and settings, alert-rule editor (JSON), users and roles, audit trail, anomaly notifications, model versions, data retention |
| Integrity | SHA-256 duplicate detection, near-duplicate detection (re-encoded / trimmed / volume-changed copies), model version stored with every prediction |

---

## 2. Requirements

* Windows 10/11 (also works on Linux and macOS)
* **Python 3.10 – 3.12** (Anaconda is fine)
* **FFmpeg** (needed for MP3/M4A). On Windows run `winget install Gyan.FFmpeg`, or download it from ffmpeg.org and add the `bin` folder to PATH.
* Chrome or Edge (live microphone + GTM run in the browser)
* 8 GB RAM minimum (16 GB recommended for training)

## 3. Installation

```bash
cd "SonicSentinel AI"
python -m venv .venv
.venv\Scripts\activate            # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
python -m database.init_db        # creates data/sonicsentinel.db + default accounts
```

Default accounts (**change the passwords** after the first login):

| Username | Password | Role |
|---|---|---|
| admin | Admin@12345 | Administrator |
| reviewer | Review@12345 | Audio reviewer |
| security | Secure@12345 | Security operator |
| maintenance | Maint@12345 | Maintenance operator |
| evaluator | Eval@12345 | Normal user (for evaluators) |

Optional environment variables: `SONIC_SECRET_KEY` (always set this in production), `SONIC_DB_PATH`, `SONIC_DATA_DIR`, `PORT`, `HOST`.

## 4. Building the dataset (at least 3,000 original clips, 300 per class)

See **documentation/DATASET_GUIDE.md** for sources, licences and ethics. Short version:

```bash
# 1. put clips in audio_dataset/raw/<Class Name>/  – from any of:
python scripts/import_public_datasets.py --urbansound8k D:/data/UrbanSound8K --esc50 D:/data/ESC-50-master
python scripts/import_public_datasets.py --mimii D:/data/mimii --fsd50k D:/data/FSD50K
python scripts/record_samples.py --class "Person Asking for Help" --speaker S01 --count 20   # voluntary
python scripts/generate_help_phrases_tts.py --per-phrase 10                              # synthetic

# 2. validate, de-duplicate, assign Audio IDs, 70/15/15 stratified split
python scripts/build_dataset.py

# 3. augment TRAINING clips only (they stay in the train split)
python -m augmentation.augment --per-clip 2
```

Outputs: `data/dataset_metadata.csv`, `data/dataset_statistics.json`, `data/dataset_quality_report.csv`, `data/dataset_rejected.csv`, and `audio_dataset/processed/<split>/<class>/AUD-xxxxxx.wav`.

## 5. Training the Python model

```bash
python -m python_models.train_models            # full grids (slower)
python -m python_models.train_models --fast     # quick check
```

This saves `python_models/saved/sonic_model.joblib` and writes reports to `reports/`: model comparison, test metrics, class-wise scores, confusion matrix and noise robustness. The admin **Models** page shows them.

## 6. Training and installing the GTM model

Full steps are in **gtm_model/README.md**:

1. Run `python -m gtm_model.prepare_gtm_samples --playlist`. It creates 1-second samples from the **TRAIN** split only.
2. Go to https://teachablemachine.withgoogle.com → *Audio Project*. Create classes with **exactly** the names in `config/settings.py` (`Background Noise` already exists), add the samples, then train.
3. Choose *Export Model* → *TensorFlow.js* → *Download*. Unzip `model.json`, `metadata.json` and `weights.bin` into `gtm_model/model/`.
4. Restart the app. The admin dashboard shows the GTM version and warns if any class is missing.

## 7. Running

```bash
python run.py                       # open http://127.0.0.1:5000
python run.py --host 0.0.0.0        # LAN access (the microphone needs https or localhost)
```

Production on Windows: `waitress-serve --port 5000 run:app`. On Linux (Render/Railway): `gunicorn -w 2 -b 0.0.0.0:$PORT run:app`.

## 8. How to use

| Task | Where |
|---|---|
| Register / log in | `/register`, `/login` |
| Upload and preview audio | **Upload** (drag and drop, preview with play/pause/seek/volume) |
| Metadata, waveform, spectrogram | event page after the upload |
| Python and GTM predictions | event page → *Model prediction & confidence comparison* (top-3, all scores, \|Δconf\|, margins) |
| Confidence | High ≥ 85 %, Medium ≥ minimum confidence (60 % default), Low below that |
| Audio quality | event page → *Audio quality* (issues are listed) |
| Live monitoring | **Live** → Start. Chrome asks for mic permission, and a red banner stays on while the mic is active |
| Alerts | **Alerts** (security/maintenance/admin) → Acknowledge / Escalate / Dismiss; toasts appear on every page |
| Manual review | **Review** (reviewer/admin) → listen, confirm or correct, comment, close |
| Dashboards | **Dashboard** (all users), **Live**, **Admin** |
| Search history | **History** filters and **Timeline** |
| Reports | event page → *Download report*; Admin → exports (CSV/Excel) |
| Model comparison report | Admin → *Model comparison* → *Run test-set evaluation* (≥ 10 unseen test clips per class) → Export Excel |
| Thresholds / rules | Admin → *Settings*, Admin → *Alert rules* |

## 9. Tests

```bash
python -m pytest -q
```

The tests cover functional, integration, boundary, negative, security (CSRF, roles, lock-out), database, audio-format, silence, clipping, noise, pre-processing, feature, model, comparison, alert-rule, duplicate, low-confidence, overlap and live-window cases. They run on a temporary database with a tiny test-only model, so they never touch your data.

## 10. Common surprise modifications

| Request | What to change |
|---|---|
| New sound category | add it to `CLASSES` in `config/settings.py`, add a folder in `audio_dataset/raw/`, rebuild the dataset, retrain both models (same name in GTM), add a rule in Admin → Alert rules |
| Change confidence threshold | Admin → Settings → *min confidence* (or the per-class value in Alert rules) |
| New audio format | add the extension to `SUPPORTED_FORMATS` (FFmpeg decodes almost anything) |
| Segment duration | Admin → Settings → *segment seconds* / *live window seconds*, then retrain the Python model |
| Add an alert rule / change repeated detection | Admin → Alert rules (`required_consecutive`, `strong_confidence`, `escalate_after_repeats`) |
| Add a dashboard filter | `filtered_events()` in `src/routes/main.py` and the form in `templates/history.html` |

## 11. Project structure

```
config/               settings, class list, dataset label mapping
audio_preprocessing/  loader, validation, pre-processing, quality analysis
feature_extraction/   acoustic features, waveform/spectrogram images, fingerprints
augmentation/         training-only augmentation
python_models/        training/tuning/evaluation script, inference wrapper, saved model
gtm_model/            GTM sample preparation, exported model goes in gtm_model/model/
alert_rules/          alert_rules.json (editable in the admin UI)
database/             init_db.py, schema.sql
src/                  Flask app: models, routes, services (analysis, decision, rules, audit …)
templates/, static/   Tailwind UI, gtm.js (browser GTM), live.js, batch.js
scripts/              dataset import / build / recording / TTS
tests/                pytest suite
notebooks/            exploration notebook
documentation/        dataset guide, data dictionary, architecture, testing
reports/              generated model reports
sample_audio/         small test clips (silent, clipped, short, tone)
```

## 12. Assumptions and limitations

* GTM runs in the browser, so uploads need a browser tab open to get GTM scores. Without GTM the result is marked *Uncertain Result* and uses the Python model only.
* The Python model classifies fixed segments. Very short events in long noisy recordings may still be missed.
* Help-phrase detection works only for the defined safety phrases in the training data. It is not speech recognition.
* Accuracy targets (≥ 85 % accuracy, macro-F1 ≥ 0.80, critical recall ≥ 85 %) depend on the size and quality of the collected dataset.
* SQLite suits a single server. Use PostgreSQL (`SQLALCHEMY_DATABASE_URI`) for many concurrent users.

AI tools used during development are declared in **AI_USAGE.md**.
