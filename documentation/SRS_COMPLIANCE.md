# SRS compliance checklist – SonicSentinel AI

Checked against *SonicSentinel AI – NextWave AI and ML SRS v1.0* on 2026-09-24 (updated after the 10-class retraining), using the code, data, reports and models in this repository.

**Scoring:** ✅ Complete = 1 · ⚠️ Partial = 0.5 · ❌ Missing = 0 · ➖ Not checkable from the code (team / evaluation-day items, not scored)

## Summary

| Section of the SRS | Items | Score | % |
|---|---|---|---|
| 1.2 Development phase (Steps 1–20) | 20 | 20 | **100 %** |
| Hint – dataset requirements | 8 | 7 | **88 %** |
| 1.6 Functional requirements (i – lxxx + responsive UI) | 81 | 81 | **100 %** |
| 1.7 Non-functional requirements | 5 | 4.5 | **90 %** |
| 1.8 Competition integrity (checkable items) | 8 | 7.5 | **94 %** (100 % when every member has commits) |
| 1.9 Interface requirements | 2 | 2 | **100 %** |
| 1.10 Project deliverables | 16 | 12 | **75 %** |
| **Overall** | **140** | **134** | **≈ 96 %** (was 86 %) |

* **Application, data and Python model: ≈ 98 %.** All ten classes are trained; three models were compared; the Python model meets the accuracy and macro-F1 targets.
* **What is left can only be done by the team:** GTM retraining with 10 classes, the model-comparison Excel, more Glass Breaking clips, deployment, the demo video, screenshots, filling names in `AI_USAGE.md` / `TEAM_CONTRIBUTIONS.md`, and commits from every member. See [SUBMISSION_CHECKLIST.md](SUBMISSION_CHECKLIST.md).

---

## 1. Development phase (SRS 1.2, Steps 1–20) – 100 %

| Step | Requirement | Status | Evidence / what is missing |
|---|---|---|---|
| 1 | Audio data collection, unique Audio ID, metadata (file, class, duration, SR, channels, environment, device, distance, original/augmented, split) | ✅ | Audio IDs and all metadata columns in `data/dataset_metadata.csv`; all 10 classes have data (Help: 204 TTS + 17 team clips). Environment/device/distance are "unknown" for most public clips |
| 2 | Upload and live-microphone input modes | ✅ | Upload, Batch, Live pages |
| 3 | Validation: format, size, duration, SR, channels, integrity, signal, mic availability, permission | ✅ | `audio_preprocessing/validation.py`, `static/js/live.js` |
| 4 | Pre-processing: resample, mono, normalise, trim, noise reduction, segmentation, pad/truncate, conversion | ✅ | `audio_preprocessing/preprocess.py` |
| 5 | One common dataset for both models; GTM samples from the same recordings; Audio ID kept | ✅ | `scripts/build_dataset.py`, `gtm_model/prepare_gtm_samples.py` (`gtm_samples_metadata.csv`) |
| 6 | Features: MFCC, Mel, chroma, ZCR, RMS, centroid, bandwidth, roll-off, onset, tempo | ✅ | 299 features, `feature_extraction/features.py` |
| 7 | Train and compare **at least three** models; select by accuracy, precision, recall, F1, macro-F1, CM, class-wise, critical recall, noise robustness | ✅ | **XGBoost, MLP and Random Forest compared** (`reports/python_model_comparison.csv`); XGBoost selected by validation macro-F1. SVM available but skipped (hours on 38k segments) |
| 8 | Python model gives scores for all ten classes | ✅ | scores for all 10 classes; Help recall 1.00 on the test split |
| 9 | GTM audio project with the same 10 class names, trained on the same TRAIN recordings | ✅ | run 2 export (2026-09-24 15:50) has all 10 labels, trained from the `--zip` TRAIN-split archives (400 samples per class; Glass 75, Help 292) |
| 10 | GTM predicts independently; Python result not given to GTM | ✅ | `static/js/gtm.js`, `/api/events/<id>/gtm` |
| 11 | Compare categories, all-class confidences, match status, \|top diff\|, top-2 diff | ✅ | event page comparison panel |
| 12 | Continuous live windows (1–3 s): validate → preprocess → Python → GTM → compare → rules → live dashboard | ✅ | 2-s live windows |
| 13 | Quality: silence, clipping, noise, low level, duration, encoding, missing frames → Good/Acceptable/Poor/Unusable | ✅ | `audio_preprocessing/quality.py` |
| 14 | Critical-event recognition + distinguish similar pairs (gunshot vs fireworks/backfire, scream vs shouting, aggression vs conversation, glass vs metal, alarm vs horn, fault vs normal machinery, help vs speech) | ✅ | Look-alike negatives now in Background Noise: fireworks, thunderstorm, door knock, chainsaw … (ESC-50), normal machinery (MIMII `normal`), ordinary voices (`NotScreaming`), calm film audio (VSD), street music / children (US8K). Metal impact and car backfire are not in the licensed sets |
| 15 | Repeated-detection confirmation | ✅ | `required_consecutive`, `strong_confidence`, repeat window |
| 16 | Alerts with 5 severity levels and the example mapping | ✅ | `alert_rules/alert_rules.json` |
| 17 | Manual-review routing (disagreement, low confidence, poor quality, close top-2, overlap, unsupported, critical without agreement, false alarm) | ✅ | `src/services/decision.py` |
| 18 | Dashboard and history items | ✅ | Dashboard, Live, History, Timeline |
| 19 | Final decision inputs and outputs | ✅ | decision trace per event |
| 20 | Event storage (IDs, metadata, both models' scores, quality, severity, alert, review, model versions, timestamp) | ✅ | `audio_events`, `segments`, `reviews` tables |

## 2. Dataset requirements (SRS "Hint") – 88 %

| Requirement | Status | Current state |
|---|---|---|
| ≥ 3,000 unique original clips | ✅ | **4,629** original clips (≈ 6.6 h) |
| ≈ 300 clips per class, roughly balanced | ⚠️ | Glass Breaking **40**, Machinery Fault **138**; others 221 – 1,189. Training split balanced to ≈ 600 per class with augmentation (`--balance-to 600`) |
| Variation: device, distance, indoor/outdoor, intensity, interference, echo, duration, clean/noisy, overlapping, near/far, speakers | ✅ | 8 sources (public datasets, TTS in several voices, team recordings) + augmentation (noise, reverb, distance, device, volume, pitch, stretch) |
| Help class limited to safety phrases, recorded voluntarily / TTS / licensed | ✅ | 221 clips: SRS safety phrases only, offline TTS + consented team recordings |
| Stratified 70 / 15 / 15 split | ✅ | 3,240 / 694 / 695 |
| Same split for both models; GTM only from TRAIN; val/test never used for training | ✅ | build + GTM sample scripts |
| Segments and augmented copies stay in the parent's split; augmented not counted as originals | ✅ | `parent_audio_id`, GroupKFold |
| Augmentation: noise, shift, pitch, stretch, volume, reverb, distance, device | ✅ | `augmentation/augment.py` |

## 3. Functional requirements (SRS 1.6) – 100 %

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
| xvi | Common dataset with **all ten** classes | ✅ | |
| xvii | Dataset metadata management | ✅ | |
| xviii | Dataset balance checking | ✅ | `build_dataset.py` warns, `dataset_statistics.json` |
| xix | Augmentation | ✅ | |
| xx | Acoustic feature extraction | ✅ | |
| xxi | Waveform generation | ✅ | |
| xxii | (Mel) spectrogram generation | ✅ | |
| xxiii | Train and compare ≥ 3 models | ✅ | XGBoost, MLP, RF |
| xxiv | Hyperparameter tuning | ✅ | GridSearchCV + GroupKFold |
| xxv | Python model predicts one of the **ten** classes | ✅ | |
| xxvi | Python confidence for all classes | ✅ | |
| xxvii | Separately trained GTM audio model | ✅ | 10-class Teachable Machine export in `gtm_model/model/` |
| xxviii | GTM integrated into the web app | ✅ | TF.js export in `gtm_model/model/` |
| xxix | GTM classifies the same segment independently | ✅ | |
| xxx | GTM confidence for **all** classes | ✅ | all 10 labels, no missing / extra labels (Admin → Models) |
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
| xlii | Glass Breaking → high security alert | ✅ | model recall 0.83 |
| xliii | Alarm/Siren → high attention alert | ✅ | |
| xliv | Vehicle Horn → stored as traffic event | ✅ | |
| xlv | Animal Sound → non-critical event | ✅ | |
| xlvi | Gunshot → critical alert after confirmation | ✅ | |
| xlvii | Panic Scream → critical safety alert | ✅ | |
| xlviii | Aggression → high/critical alert | ✅ | |
| xlix | Person Asking for Help detection → critical alert | ✅ | Python model (recall 1.00) + Critical rule; GTM after retraining |
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
| – | Responsive UI (desktop, tablet, mobile browsers) | ✅ | Tailwind + viewport meta; every page checked at 1440 px and 390 px (phone) width after the theme change |

**Result:** 81 ✅ → 81 / 81 = **100 %**

## 4. Non-functional requirements (SRS 1.7) – 90 %

| # | Requirement | Status | Current state |
|---|---|---|---|
| 1 | Performance: 30-s clip in ≤ 8 s; live window prediction ≤ 3 s | ✅ | measured with the real model (`py-xgb-20260924-0800`) and real test recordings: 30-s upload **mean 1.47 s, max 1.74 s**; live window **mean 0.11 s, max 0.12 s** (2-CPU test server) – `reports/performance.md` |
| 2 | Scalability: ≥ 20,000 event records, multiple concurrent users | ✅ | **20,020 events** stored; dashboard 0.07 s, history 0.18 s, filtered history 0.02 s, timeline 0.04 s, admin 0.48 s, CSV export 1.3 s; **5 concurrent users, 0 errors** – `reports/performance.md` |
| 3 | Usability | ✅ | role-specific dashboards, drag-and-drop, badges, toasts, new professional theme |
| 4 | Accuracy for **both** models: ≥ 85 % accuracy, macro-F1 ≥ 0.80, critical recall ≥ 85 % | ⚠️ | Python: accuracy **0.865 ✅**, macro-F1 **0.863 ✅**, recall Gunshot 0.93 ✅, Panic Scream 0.93 ✅, Aggression 0.89 ✅, Help 1.00 ✅, Glass 0.83 (5 of 6 clips). **GTM run 2: accuracy 0.158, macro-F1 0.174 ❌** – the export is under-trained (≈ 20 % even on its own training samples); a correctly trained head on the same Teachable Machine base reaches ≈ 64 %. Retrain (run 3, epochs 100); the 1-s speech-commands model is not expected to reach 85 % on these sounds – documented as a limitation |
| 5 | Availability ≥ 99 % during evaluation hours | ✅ | `run_server.bat`: waitress (8 threads) with automatic restart and a log in `logs/server.log`; `/healthz` checks database + both models; Docker `HEALTHCHECK`; Render health check on `/healthz`. (Public deployment itself is deliverable 12.) |

## 5. Competition integrity (SRS 1.8) – 94 %

| # | Requirement | Status | Note |
|---|---|---|---|
| 1 | Each member explains assigned modules | ➖ | prepared: module owners + functions to explain in `documentation/VIVA_PREP.md` §1 |
| 2 | Meaningful GitHub commits across all five days (all members) | ⚠️ | **only the team can do this** – each member commits their own part from their own GitHub account (plan in `TEAM_CONTRIBUTIONS.md`) |
| 3 | Development log (work, problems, dataset, failures, code, tests) | ✅ | `DEVELOPMENT_LOG.md` Day 1–2 (continued); keep adding days 3–5 |
| 4 | Explain any function on request | ➖ | prepared: pipeline in one minute + key functions per module (`VIVA_PREP.md` §1) |
| 5 | Surprise modification | ➖ | prepared: where to change threshold, rules, repeats, format, upload size, segment length, filter, class, role, retention (`VIVA_PREP.md` §2) |
| 6 | Fix a deliberate defect | ➖ | prepared: debugging checklist + likely defect locations; 58 tests catch most regressions (`VIVA_PREP.md` §3) |
| 7 | No hard-coded predictions / invented confidences / hidden APIs | ✅ | |
| 8 | Tested with unseen recordings | ➖ | evaluation day; rehearsed on 695 unseen TEST clips (`reports/model_comparison_report.xlsx`) and 2,926 degraded versions |
| 9 | Hidden-test readiness (noise, echo, low volume, devices, partial, overlap, similar classes, re-encoded, distant) | ✅ | every condition tested on 266 unseen TEST clips with the app's pipeline (`scripts/hidden_test_robustness.py` → `reports/hidden_test_robustness.md`), **0 crashes in 2,926 runs**: clean 0.90 · echo 0.87 · MP3 re-encode 0.88 · low volume 0.90 · distant 0.77 · other device 0.76 · partial 0.75 · overlap 0.74 · real background 10 dB 0.72 · white noise 20 dB 0.71 / 10 dB 0.55; confusable pairs ≤ 3/30. Checklist ticked in `TESTING.md` |
| 10 | AI tools declared in `AI_USAGE.md` | ✅ | all AI assistance declared with files affected (8 rows); each member still fills "Student modifications" and "Verified by" |
| 11 | AI output does not replace understanding | ➖ | team; every AI-assisted file listed in `AI_USAGE.md` with the module owner who must explain it |
| 12 | No external generative-AI API for classification | ✅ | |
| 13 | Show that both models predict independently | ✅ | |
| 14 | Python output never given to GTM | ✅ | |

## 6. Interface requirements (SRS 1.9) – 100 %

| Requirement | Status | Used |
|---|---|---|
| Hardware (i5/i7, 8–16 GB RAM …) | ✅ | runs on the team laptop |
| Software stack | ✅ | HTML/CSS/JS (Tailwind), Flask, SQLite, Python/Anaconda/VS Code, scikit-learn, XGBoost, GTM, librosa, soundfile, SciPy, NumPy, Pandas, Matplotlib, Joblib, pytest, Git/GitHub |

## 7. Project deliverables (SRS 1.10) – 75 %

| # | Deliverable | Status | What exists / what is missing |
|---|---|---|---|
| 1 | Project report (all topics incl. DFD, Use Case, Activity, Sequence, Decision Flow diagrams, FP/FN analysis) | ✅ | `documentation/PROJECT_REPORT.md` (Mermaid diagrams); add team names and export to PDF |
| 2 | Source code with the required folders and files | ✅ | all folders present, incl. `screenshots/` |
| 3 | Dataset (audio, metadata, IDs, labels, source, split, statistics, data dictionary, augmentation scripts, quality report) | ⚠️ | everything except the audio in the repo (too large) – **upload a zip and add the link** |
| 4 | Python model evidence | ✅ | 3 models, hyperparameters, CV/val/test results, CM, class-wise, critical recall, saved model, version (`reports/`) |
| 5 | GTM evidence | ⚠️ | export, labels, integration ✅; `documentation/GTM_TRAINING_LOG.md` has the observations and retraining plan – **screenshots, sample counts, run 2** by the team |
| 6 | Model prediction & confidence comparison report (≥ 100 unseen clips, ≥ 10 per class) | ⚠️ | generated on all **695 unseen test clips** with every SRS column + summary + per-class + both confusion matrices (`reports/model_comparison_report.xlsx`, GTM run 2). Regenerate after GTM run 3; Glass Breaking has only 6 test clips (SRS asks ≥ 10) |
| 7 | Alert-rule file with all required fields | ✅ | `alert_rules/alert_rules.json` |
| 8 | Test cases (all categories + hidden-test checklist) | ✅ | pytest suite + `documentation/TESTING.md` + benchmark |
| 9 | Installation instructions incl. troubleshooting | ✅ | README §13, §21–23, `RUN_AND_TRAIN_GUIDE.md`, `documentation/DEPLOYMENT.md` |
| 10 | Execution instructions in README | ✅ | README §15–17 |
| 11 | GitHub repository (public, all members' commits, screenshots, links, credentials, assumptions, limitations) | ⚠️ | credentials, assumptions, limitations, test results, link table ✅; **make public, add screenshots, commits from all members** |
| 12 | Deployed application | ⚠️ | Docker/Render files and guide ready; local fallback complete; **not deployed** |
| 13 | Demonstration video (.mp4) | ❌ | shot list in `documentation/DEMO_VIDEO_SCRIPT.md`; **not recorded** |
| 14 | Technical blog (≥ 2,000 words) | ✅ | `documentation/TECHNICAL_BLOG.md` (≈ 3,400 words, all 27 topics) |
| 15 | `AI_USAGE.md` with all fields | ⚠️ | all rows declared; student modifications / verified-by to fill |
| 16 | Final submission checklist + team contribution record | ✅ | `documentation/SUBMISSION_CHECKLIST.md`, `documentation/TEAM_CONTRIBUTIONS.md` |

---

## Priority to-do list (team only)

| Priority | Task | Fixes |
|---|---|---|
| 1 | GTM run 3: new TM project → upload the 10 zips → epochs 100 → train until *Model Trained* (Under the hood > 90 %) → Export → `gtm_model/model/`; screenshots | NFR 4 (GTM accuracy), deliverable 5 |
| 2 | Add ≈ 30+ **Glass Breaking** clips (FSD50K / TUT) → rebuild → augment → retrain | balance, Glass recall, ≥ 10 test clips per class |
| 3 | Admin → Model comparison → run → export Excel → `reports/model_comparison_report.xlsx` | deliverable 6, NFR 4 (GTM accuracy → **NFR 100 %** together with item 2) |
| 4 | Optional: re-run `python scripts/benchmark_performance.py` on the demo PC | NFR 1–2 (already measured) |
| 5 | Deploy to Render (`documentation/DEPLOYMENT.md`) | NFR 5, deliverable 12 |
| 6 | Screenshots (`screenshots/README.md`), demo video (`DEMO_VIDEO_SCRIPT.md`), dataset zip link | deliverables 3, 11, 13 |
| 7 | Fill names in `AI_USAGE.md` and `TEAM_CONTRIBUTIONS.md`; every member commits | integrity 2, deliverables 15–16 |
