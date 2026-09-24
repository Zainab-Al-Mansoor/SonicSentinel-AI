# SRS compliance checklist – SonicSentinel AI

Checked against *SonicSentinel AI – NextWave AI and ML SRS v1.0* on 2026-09-24, using the code, data, reports and models in this repository.

**Scoring:** ✅ Complete = 1 · ⚠️ Partial = 0.5 · ❌ Missing = 0 · ➖ Not checkable from the code (team / evaluation-day items, not scored)

## Summary

| Section of the SRS | Items | Score | % |
|---|---|---|---|
| 1.2 Development phase (Steps 1–20) | 20 | 17.5 | **88 %** |
| Hint – dataset requirements | 8 | 6 | **75 %** |
| 1.6 Functional requirements (i – lxxx + responsive UI) | 81 | 78 | **96 %** |
| 1.7 Non-functional requirements | 5 | 2.5 | **50 %** |
| 1.8 Competition integrity (checkable items) | 8 | 6.5 | **81 %** |
| 1.9 Interface requirements | 2 | 2 | **100 %** |
| 1.10 Project deliverables | 16 | 8 | **50 %** |
| **Overall** | **140** | **120.5** | **≈ 86 %** |

* **Application features (Steps + FR + interface): ≈ 95 %.** The web application is almost complete.
* **Data, model targets and submission deliverables: ≈ 55 %.** This is where most of the remaining work is.

---

## 1. Development phase (SRS 1.2, Steps 1–20) – 88 %

| Step | Requirement | Status | Evidence / what is missing |
|---|---|---|---|
| 1 | Audio data collection, unique Audio ID, metadata (file, class, duration, SR, channels, environment, device, distance, original/augmented, split) | ⚠️ | Audio IDs and all metadata columns exist (`data/dataset_metadata.csv`). **Person Asking for Help has 0 clips**; environment/device/distance are mostly "unknown" for public clips |
| 2 | Upload and live-microphone input modes | ✅ | Upload, Batch, Live pages |
| 3 | Validation: format, size, duration, SR, channels, integrity, signal, mic availability, permission | ✅ | `audio_preprocessing/validation.py`, `static/js/live.js` |
| 4 | Pre-processing: resample, mono, normalise, trim, noise reduction, segmentation, pad/truncate, conversion | ✅ | `audio_preprocessing/preprocess.py` |
| 5 | One common dataset for both models; GTM samples from the same recordings; Audio ID kept | ✅ | `scripts/build_dataset.py`, `gtm_model/prepare_gtm_samples.py` (`gtm_samples_metadata.csv`) |
| 6 | Features: MFCC, Mel, chroma, ZCR, RMS, centroid, bandwidth, roll-off, onset, tempo | ✅ | 299 features, `feature_extraction/features.py` |
| 7 | Train and compare **at least three** models; select by accuracy, precision, recall, F1, macro-F1, CM, class-wise, critical recall, noise robustness | ⚠️ | Code supports SVM, RF, XGBoost, MLP, but the **installed run compared only 2 (RF, MLP)**. SVM was too slow; XGBoost is skipped while Help has no data |
| 8 | Python model gives scores for all ten classes | ⚠️ | Scores for all 10, but Help is always 0 (no training data) |
| 9 | GTM audio project with the same 10 class names, trained on the same TRAIN recordings | ⚠️ | 9 classes installed – **Person Asking for Help missing** |
| 10 | GTM predicts independently; Python result not given to GTM | ✅ | `static/js/gtm.js`, `/api/events/<id>/gtm` |
| 11 | Compare categories, all-class confidences, match status, \|top diff\|, top-2 diff | ✅ | event page comparison panel |
| 12 | Continuous live windows (1–3 s): validate → preprocess → Python → GTM → compare → rules → live dashboard | ✅ | 2-s live windows |
| 13 | Quality: silence, clipping, noise, low level, duration, encoding, missing frames → Good/Acceptable/Poor/Unusable | ✅ | `audio_preprocessing/quality.py` |
| 14 | Critical-event recognition + distinguish similar pairs (gunshot vs fireworks/backfire, scream vs shouting, aggression vs conversation, glass vs metal, alarm vs horn, fault vs normal machinery, help vs speech) | ⚠️ | Critical classes detected, but **no training data for the look-alike sounds** (fireworks, backfire, metal impact, normal machinery, normal speech) |
| 15 | Repeated-detection confirmation | ✅ | `required_consecutive`, `strong_confidence`, repeat window |
| 16 | Alerts with 5 severity levels and the example mapping | ✅ | `alert_rules/alert_rules.json` |
| 17 | Manual-review routing (disagreement, low confidence, poor quality, close top-2, overlap, unsupported, critical without agreement, false alarm) | ✅ | `src/services/decision.py` |
| 18 | Dashboard and history items | ✅ | Dashboard, Live, History, Timeline |
| 19 | Final decision inputs and outputs | ✅ | decision trace per event |
| 20 | Event storage (IDs, metadata, both models' scores, quality, severity, alert, review, model versions, timestamp) | ✅ | `audio_events`, `segments`, `reviews` tables |

## 2. Dataset requirements (SRS "Hint") – 75 %

| Requirement | Status | Current state |
|---|---|---|
| ≥ 3,000 unique original clips | ✅ | **5,622** original clips |
| ≈ 300 clips per class, roughly balanced | ⚠️ | Help **0**, Glass Breaking **40**, Machinery Fault **138**; others 298 – 1,040 (imbalance 26 : 1) |
| Variation: device, distance, indoor/outdoor, intensity, interference, echo, duration, clean/noisy, overlapping, near/far, speakers | ⚠️ | Comes from many public datasets + augmentation (noise, reverb, distance, device). Only 40 own recordings; overlapping events and speaker variety not collected |
| Help class limited to safety phrases, recorded voluntarily / TTS / licensed | ❌ | No clips yet (`record_samples.py`, `generate_help_phrases_tts.py` are ready) |
| Stratified 70 / 15 / 15 split | ✅ | 3,935 / 843 / 844 |
| Same split for both models; GTM only from TRAIN; val/test never used for training | ✅ | build + GTM sample scripts |
| Segments and augmented copies stay in the parent's split; augmented not counted as originals | ✅ | `parent_audio_id`, GroupKFold |
| Augmentation: noise, shift, pitch, stretch, volume, reverb, distance, device | ✅ | `augmentation/augment.py` |

## 3. Functional requirements (SRS 1.6) – 96 %

| # | Requirement | Status | Note |
|---|---|---|---|
| i | Registration and secure login | ✅ | |
| ii | Role-based access (5 roles) | ✅ | |
| iii | Profile management + unique User ID | ✅ | `USR-00001` |
| iv | Upload WAV/MP3/FLAC/OGG/M4A | ✅ | |
| v | Batch upload | ✅ | |
| vi | Start/stop live mic after permission | ✅ | |
| vii | Mic status: Available/Active/Paused/Disconnected/Permission denied | ✅ | |
| viii | File validation | ✅ | |
| ix | Audio preview (play, pause, replay, seek, volume) | ✅ | |
| x | Metadata extraction incl. bit depth, size, upload time | ✅ | |
| xi | Pre-processing | ✅ | |
| xii | Silence detection | ✅ | |
| xiii | Clipping detection + warning | ✅ | |
| xiv | Background-noise estimation | ✅ | |
| xv | Segmentation with start/end timestamps | ✅ | |
| xvi | Common dataset with **all ten** classes | ⚠️ | Help missing |
| xvii | Dataset metadata management | ✅ | |
| xviii | Dataset balance checking | ✅ | `build_dataset.py` warns, `dataset_statistics.json` |
| xix | Augmentation | ✅ | |
| xx | Acoustic feature extraction | ✅ | |
| xxi | Waveform generation | ✅ | |
| xxii | (Mel) spectrogram generation | ✅ | |
| xxiii | Train and compare ≥ 3 models | ✅ | implemented in code (see Step 7 for the actual run) |
| xxiv | Hyperparameter tuning | ✅ | GridSearchCV + GroupKFold |
| xxv | Python model predicts one of the **ten** classes | ⚠️ | can never predict Help |
| xxvi | Python confidence for all classes | ✅ | |
| xxvii | Separately trained GTM audio model | ⚠️ | 9 of 10 classes |
| xxviii | GTM integrated into the web app | ✅ | TF.js export in `gtm_model/model/` |
| xxix | GTM classifies the same segment independently | ✅ | |
| xxx | GTM confidence for **all** classes | ⚠️ | Help missing |
| xxxi | Prediction comparison | ✅ | |
| xxxii | Absolute top-class confidence difference | ✅ | |
| xxxiii | Consistency status (4 values) | ✅ | |
| xxxiv | Top-3 predictions per model | ✅ | |
| xxxv | Configurable confidence threshold | ✅ | Admin → Settings |
| xxxvi | Configurable top-2 margin | ✅ | |
| xxxvii | Quality classification | ✅ | |
| xxxviii | Uncertain-sound detection | ✅ | |
| xxxix | Overlapping-sound detection | ✅ | |
| xl | Repeated event confirmation | ✅ | |
| xli | Machinery Fault → maintenance alert + inspection action | ✅ | |
| xlii | Glass Breaking → high security alert | ✅ | model recall only 0.67 |
| xliii | Alarm/Siren → high attention alert | ✅ | |
| xliv | Vehicle Horn → stored as traffic event | ✅ | |
| xlv | Animal Sound → non-critical event | ✅ | |
| xlvi | Gunshot → critical alert after confirmation | ✅ | |
| xlvii | Panic Scream → critical safety alert | ✅ | |
| xlviii | Aggression → high/critical alert | ✅ | |
| xlix | **Person Asking for Help detection → critical alert** | ❌ | rule exists, **no data / not in either model** |
| l | Background noise non-critical unless above limit | ✅ | `background_noise_limit_db` |
| li | Unknown-sound handling | ✅ | |
| lii | Severity assignment | ✅ | |
| liii | Configurable alert rules | ✅ | Admin → Alert rules (JSON) |
| liv | Real-time visible alerts | ✅ | toasts |
| lv | Acknowledge / dismiss / escalate | ✅ | |
| lvi | Alert history with user actions | ✅ | |
| lvii | Manual-review queue | ✅ | |
| lviii | Reviewer audio playback | ✅ | |
| lix | Reviewer confirm/correct | ✅ | |
| lx | Reviewer comments + actions | ✅ | |
| lxi | Override keeping original outputs | ✅ | |
| lxii | Event statuses (7) | ✅ | |
| lxiii | User dashboard | ✅ | |
| lxiv | Live dashboard | ✅ | |
| lxv | Admin dashboard | ✅ | |
| lxvi | Event timeline | ✅ | |
| lxvii | Search and filtering (9 filters) | ✅ | |
| lxviii | Analytics (frequency, confidence, FP, FN, disagreement, quality, alert response) | ✅ | |
| lxix | Downloadable analysis report | ✅ | HTML → PDF |
| lxx | CSV / Excel export | ✅ | |
| lxxi | Secure audio storage | ✅ | protected media routes |
| lxxii | Database storage | ✅ | 11 tables |
| lxxiii | SHA-256 duplicate detection | ✅ | |
| lxxiv | Near-duplicate detection | ✅ | fingerprint similarity |
| lxxv | Model version per prediction | ✅ | |
| lxxvi | Audit trail (logins, uploads, mic sessions, predictions, alerts, reviews, overrides, exports, model updates) | ✅ | |
| lxxvii | Understandable errors (file, format, decoding, model, database, report) | ✅ | error pages + notifications |
| lxxviii | Admin anomaly alerts (failed uploads, model failures, low-confidence spikes, excessive critical alerts, duplicates, failed logins) | ✅ | `src/services/audit.py` |
| lxxix | Privacy: mic-active indicator, no secret recording | ✅ | red banner |
| lxxx | Configurable data retention | ✅ | |
| – | Responsive UI (desktop, tablet, mobile browsers) | ✅ | Tailwind + viewport meta (not formally tested on tablet/mobile) |

**Result:** 76 ✅ · 4 ⚠️ · 1 ❌ → 78 / 81 = **96 %**

## 4. Non-functional requirements (SRS 1.7) – 50 %

| # | Requirement | Status | Current state |
|---|---|---|---|
| 1 | Performance: 30-s clip in ≤ 8 s; live window prediction ≤ 3 s | ⚠️ | Design is suitable (2-s windows, browser-side GTM) but **not measured yet** |
| 2 | Scalability: ≥ 20,000 event records, multiple concurrent users | ⚠️ | SQLite + threaded server; **no load test**; PostgreSQL supported |
| 3 | Usability | ✅ | role-specific dashboards, drag-and-drop, badges, toasts |
| 4 | Accuracy for **both** models: ≥ 85 % accuracy, macro-F1 ≥ 0.80, critical recall ≥ 85 % | ⚠️ | Python: accuracy **0.931 ✅**, macro-F1 **0.819 ✅**, critical recall Gunshot 0.95 ✅, Panic Scream 0.92 ✅, Aggression 0.80 ❌, Glass Breaking 0.67 ❌, Help 0 ❌. **GTM accuracy not measured yet** |
| 5 | Availability ≥ 99 % during evaluation hours | ❌ | not deployed yet |

## 5. Competition integrity (SRS 1.8) – 81 %

| # | Requirement | Status | Note |
|---|---|---|---|
| 1 | Each member explains assigned modules | ➖ | team – split suggested in `AI_USAGE.md` |
| 2 | Meaningful GitHub commits across all five days (all members) | ⚠️ | commits exist; continue daily, from every member |
| 3 | Development log (work, problems, dataset, failures, code, tests) | ✅ | `DEVELOPMENT_LOG.md` Day 1–2; keep adding days 3–5 |
| 4 | Explain any function on request | ➖ | team |
| 5 | Surprise modification | ➖ | supported: new class, threshold, format, segment length, rule, repeats, filter |
| 6 | Fix a deliberate defect | ➖ | team |
| 7 | No hard-coded predictions / invented confidences / hidden APIs | ✅ | |
| 8 | Tested with unseen recordings | ➖ | evaluation day |
| 9 | Hidden-test readiness (noise, echo, low volume, devices, partial, overlap, similar classes, re-encoded, distant) | ⚠️ | re-encoded/duplicate handling ✅; accuracy drops to 0.69 at 5 dB SNR; no look-alike or overlap training data |
| 10 | AI tools declared in `AI_USAGE.md` | ⚠️ | file exists but "Student modifications" and "Verified by" are still placeholders; later AI help (training, dataset import, README) not yet declared |
| 11 | AI output does not replace understanding | ➖ | team |
| 12 | No external generative-AI API for classification | ✅ | |
| 13 | Show that both models predict independently | ✅ | |
| 14 | Python output never given to GTM | ✅ | |

## 6. Interface requirements (SRS 1.9) – 100 %

| Requirement | Status | Used |
|---|---|---|
| Hardware (i5/i7, 8–16 GB RAM …) | ✅ | runs on the team laptop |
| Software stack | ✅ | HTML/CSS/JS (Tailwind), Flask, SQLite, Python/Anaconda/VS Code, scikit-learn, XGBoost, GTM, librosa, soundfile, SciPy, NumPy, Pandas, Matplotlib, Joblib, pytest, Git/GitHub |

## 7. Project deliverables (SRS 1.10) – 50 %

| # | Deliverable | Status | What exists / what is missing |
|---|---|---|---|
| 1 | Project report (problem … future enhancements, incl. **DFD, Use Case, Activity, Sequence, Decision Flow diagrams**, FP/FN analysis) | ❌ | Much of the text exists in `README.md` and `documentation/`, but no report document and no UML diagrams |
| 2 | Source code with the required folders and files (saved Python model, GTM export, label files, waveform/spectrogram code, rule files, tests, sample audio) | ✅ | all folders present |
| 3 | Dataset (train/val/test audio, metadata, IDs, labels, source, split, statistics, data dictionary, augmentation scripts, quality report) | ⚠️ | metadata, statistics, quality report, scripts ✅; audio not in the repo (too large) – **provide a shared download link** |
| 4 | Python model evidence (models tested, hyperparameters, train/val/test results, CM, metrics, class-wise, critical recall, saved model, version, sample predictions) | ⚠️ | all present in `reports/` except **≥ 3 models compared** and **sample predictions** |
| 5 | GTM evidence (project link, screenshots of all classes, sample counts, Background Noise samples, config, observations, wrong classifications, retraining, export, labels, version, test screenshots, integration) | ⚠️ | export, `metadata.json` and integration ✅; **screenshots, project link, sample counts, training settings and observations not saved** |
| 6 | Model prediction & confidence comparison report (≥ 100 unseen test clips, ≥ 10 per class, all listed columns, summary) | ❌ | feature exists (Admin → Model comparison → Excel) but **not generated**; Glass Breaking has only 6 test clips and Help 0, so "≥ 10 per class" is impossible until data is added |
| 7 | Alert-rule file with all required fields | ✅ | `alert_rules/alert_rules.json` |
| 8 | Test cases (all 20 categories + hidden-test checklist) | ✅ | 55 pytest tests + `documentation/TESTING.md` (demonstrating "every mandatory class" needs Help data) |
| 9 | Installation instructions (incl. troubleshooting, admin setup, env vars, model placement) | ⚠️ | README + `RUN_AND_TRAIN_GUIDE.md` cover almost everything; **Troubleshooting section missing** |
| 10 | Execution instructions in README | ✅ | README §15–17 |
| 11 | GitHub repository (public, all members' commits, screenshots, test results, report/blog/video links, credentials, assumptions, limitations) | ⚠️ | credentials, limitations, test results ✅; **`screenshots/` is empty**, no report/blog/video links, commits from all members still needed |
| 12 | Deployed application (URL, credentials, sample audio, test instructions) | ⚠️ | not deployed; complete local execution instructions exist (allowed as fallback) |
| 13 | Demonstration video (.mp4, 29 required scenes) | ❌ | not recorded |
| 14 | Technical blog (≥ 2,000 words) | ❌ | not written |
| 15 | `AI_USAGE.md` with all fields | ⚠️ | placeholders to fill in |
| 16 | Final submission checklist | ⚠️ | depends on items 1, 6, 12–14 |

---

## Priority to-do list

| Priority | Task | Fixes |
|---|---|---|
| 1 | Collect **Person Asking for Help** (≈ 150 real voluntary recordings + 150 TTS) | Step 1/8/9, dataset, FR xvi/xxv/xxvii/xxx/xlix, critical recall |
| 2 | Add **Glass Breaking** (to ≈ 300, e.g. FSD50K / TUT Rare Sound Events) and **Machinery Fault** (more MIMII zips) | balance, critical recall, ≥ 10 test clips per class |
| 3 | Rebuild + retrain Python with **≥ 3 models** (`--fast --models rf,mlp,xgb` once every class has data) | Step 7, deliverable 4 |
| 4 | Retrain GTM with **all 10 classes**; save screenshots, sample counts, epochs/batch/LR, observations | Step 9, FR xxvii/xxx, deliverable 5 |
| 5 | Admin → Model comparison → run test-set evaluation (≥ 100 clips, ≥ 10 per class) → export Excel; write the disagreement explanation | deliverable 6, NFR 4 (GTM accuracy) |
| 6 | Time a 30-s upload and a live window; add a record-count/load test | NFR 1–2 |
| 7 | Deploy (Render / Railway / PythonAnywhere) | NFR 5, deliverable 12 |
| 8 | Project report with UML diagrams, 2,000-word blog, demo video, screenshots, Troubleshooting section, fill `AI_USAGE.md`, daily commits by every member | deliverables 1, 9, 11, 13–15 |
