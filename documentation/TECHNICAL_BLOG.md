# Teaching a Web App to Listen: Building SonicSentinel AI

*Technical blog · NextWave AI and ML · Theme: AcousticX Intelligence*

*Authors: _team member names_ · Repository: _GitHub URL_ · Live demo: _deployment URL_ · Video: _video link_*

---

## 1. The business problem

Cameras have become the default tool for safety monitoring, but a camera only sees what is in front of it. A gunshot around the corner, a scream from the next corridor, a window breaking in an empty office at night, or a pump that starts grinding before it fails: all of these are *heard* long before, or instead of, being seen. Most buildings still depend on a person hearing the sound, recognising what it is, and deciding whether to act. Security staff get tired, machine rooms are loud, and many places have nobody listening at all.

The cost of that gap is easy to see. A late response to aggression or a cry for help can put people at risk. A missed machinery fault means unplanned downtime. A false alarm has a cost too: every siren that turns out to be a car radio teaches the operator to ignore the next one.

SonicSentinel AI is our answer to the competition brief: a **web application that listens to uploaded audio and a live microphone, recognises ten safety-critical and environmental sound classes, and turns those recognitions into explainable, reviewable alerts**. The classes are Machinery Fault, Glass Breaking, Alarm or Siren, Vehicle Horn, Animal Sound, Gunshot, Panic Scream, Aggression, Person Asking for Help and Background Noise.

## 2. Background

Environmental sound classification (ESC) has been researched for more than ten years. Public benchmarks such as ESC-50 and UrbanSound8K made the task measurable, and the classic recipe still works well: turn audio into a time–frequency representation, summarise it with features such as MFCCs, and train a classifier. Deep models trained on large corpora (AudioSet) are stronger, but they are heavy and hard to explain.

The competition asked for something different from a leaderboard entry. It asked for **two independently trained models**: one we build and train in Python, and one trained in Google Teachable Machine (GTM). The application must compare the two, show how confident each one is, and never let a generative-AI API make the final decision. That constraint shaped the whole design. The interesting question is not only "which class is this?" but also "how sure are we, do our two models agree, and what should a human do about it?"

## 3. Proposed solution

For every piece of audio, SonicSentinel AI:

1. **validates** the file (format, size, duration, sample rate, integrity, silence) and rejects bad input with a clear message;
2. **measures quality** (level, clipping, estimated SNR, silence ratio, missing frames) and labels it Good, Acceptable, Poor or Unusable;
3. **pre-processes** it (mono, resample, normalise, trim silence, spectral-gating noise reduction) and cuts it into 2-second segments;
4. classifies every segment with the **Python model** on the server and the **GTM model** in the browser;
5. **compares** the two predictions: class match, confidence difference and top-two margins;
6. runs the result through **configurable alert rules** (minimum confidence, margin, consecutive detections, model agreement, quality) that assign a severity from Informational to Critical;
7. stores everything for **history, dashboards, reports and audit**, and sends doubtful results to a **manual-review queue**.

Users see the waveform, spectrogram, both models' confidence bars and the reasoning behind every alert on the same page.

## 4. Architecture

The system is a Flask application with a server-rendered UI (Jinja templates with Tailwind CSS) and a small amount of JavaScript.

* **Browser.** Upload and batch pages, the live monitor (Web Audio API recording 2-second windows), Chart.js dashboards, and `gtm.js`, which loads the exported Teachable Machine model with TensorFlow.js and classifies audio locally.
* **Flask server.** Routes for auth, upload, live sessions, events, alerts, review, history, reports and administration. The service layer (`analysis.py`, `decision.py`, `rules.py`) holds the audio pipeline, model inference, comparison and rule engine.
* **Storage.** SQLite through SQLAlchemy: users, audio events, segments, alerts, alert actions, reviews, audit logs, notifications, settings and model versions. Uploaded audio and the generated images are stored in `data/`.
* **Models.** `python_models/saved/sonic_model.joblib` (a scikit-learn pipeline with its class list, version and feature names) and `gtm_model/model/` (`model.json`, `metadata.json`, `weights.bin`).

Because GTM runs in the browser, it adds no load to the server, and it keeps the two models independent: the browser never sees the Python prediction before it posts its own scores.

## 5. Dataset collection and ethical sourcing

The SRS allows public, licensed datasets plus our own consented recordings. We did not want any audio we did not have the right to use, so every clip carries its source and licence in `audio_dataset/raw/annotations.csv` and in the dataset metadata.

| Source | What we took |
|---|---|
| **UrbanSound8K** (CC BY-NC 3.0) | sirens, car horns, dog barks, gunshots; air conditioner, engine idling, street music, children playing, drilling and jackhammer as Background |
| **ESC-50** (CC BY-NC 3.0) | glass breaking, siren, clock alarm, car horn, 11 animal classes; 35 everyday "look-alike" categories (fireworks, thunderstorm, door knock, crying baby, laughing, coughing, chainsaw …) as Background |
| **MIMII** (CC BY-SA 4.0) | `abnormal` machine recordings as Machinery Fault; `normal` recordings as Background |
| **Kaggle gunshot audio** | nine weapon types, picked round-robin so no weapon dominates |
| **Kaggle Human Screaming Detection** | `Screaming` as Panic Scream; `NotScreaming` voices as Background |
| **VSD – Violence Sound Dataset** | one clip (up to 6 s) per annotated violence interval as Aggression; calm film audio as Background |
| **Team recordings and offline TTS** | "Help me", "Somebody help", "Call the police" and other SRS phrases; our own room noise |

Two ethical rules guided us. First, **we never staged or recorded a real emergency**: screams and gunshots come from public datasets, and our own voices only say help phrases in a calm setting. Second, **every recorded person gave consent** and appears only as an anonymous speaker code such as `S01`. Raw audio is not redistributed in the repository; the import scripts rebuild the dataset from the original downloads.

After validation and de-duplication the dataset has **4,629 original clips (≈ 6.6 hours)**, split 70/15/15 by original clip into 3,240 training, 694 validation and 695 test clips. Every clip is converted to 44.1 kHz mono 16-bit WAV and gets a unique Audio ID.

## 6. Dataset challenges

Getting good data was harder than building the models.

* **Imbalance.** Background Noise has 1,189 clips; Glass Breaking has only 40. A model trained on that would almost never predict Glass Breaking. We capped each source per class, picked clips round-robin over original labels, and **balanced the training split with augmentation** (up to ten augmented copies per clip for small classes, about 600 training clips per class).
* **No public "help" dataset.** Person Asking for Help is the class the competition cares about most, and no licensed corpus of people calling for help exists. We generated 204 phrases with Windows' offline text-to-speech in several voices and speeds, and recorded team members. Our first batch of team recordings was **near-silent (−64 to −75 dBFS)** because the wrong microphone input was selected. The quality check rejected them automatically, which is why `record_samples.py` now lists devices and refuses to save clips below −45 dBFS.
* **Negatives matter as much as positives.** Our first live tests classified keyboard typing and door knocks as Animal Sound, because the model had never seen those sounds. Adding 35 everyday ESC-50 categories and ordinary voices to Background Noise cut those false alarms sharply.
* **Duplicates and leakage.** Several datasets contain the same recording in different formats or lengths. Exact duplicates are removed by SHA-256 and near-duplicates by an audio fingerprint. The split is done by original clip, so segments and augmented copies never cross from train into test.

## 7. Audio preprocessing

Every file, whether a training clip, an upload or a live window, goes through the same pipeline so the model sees the same kind of input it was trained on:

1. **Load** with soundfile (WAV/FLAC/OGG) or FFmpeg (MP3/M4A).
2. **Validate** and **analyse quality**.
3. Convert to **mono** and **resample to 22,050 Hz**.
4. **Peak-normalise**, **trim leading and trailing silence**, and apply **spectral-gating noise reduction**.
5. **Segment** into 2.0-second windows with a 1.0-second hop; the last window is padded.

Short events are the tricky case. A gunshot lasts a fraction of a second, so averaging over a 30-second clip would wash it out. Our clip-level rule takes the **strongest non-background segment** if its confidence passes the threshold, and only averages when nothing stands out.

## 8. Waveforms and spectrograms

For each event the app draws a **waveform** and a **log-Mel spectrogram** (`feature_extraction/visuals.py`) and shows them next to the predictions. They are more than decoration. A reviewer can see at once that a "gunshot" was a single sharp transient, that a "scream" has a strong harmonic band in the 1–3 kHz range, or that the whole clip is clipped at full scale. In our testing, the spectrogram was often the fastest way to explain *why* the model was wrong: a siren and a crying baby both draw rising and falling tonal lines.

## 9. Feature extraction

Each 2-second segment becomes a vector of **299 features**:

* 40 MFCCs (mean and standard deviation) and 40 delta-MFCCs (mean): the spectral envelope and how it changes;
* 64 log-Mel bands (mean and standard deviation): overall energy distribution;
* 12 chroma bins: tonal content (sirens, horns);
* 7 spectral-contrast bands: the gap between peaks and valleys (harmonic vs noisy);
* zero-crossing rate, RMS, spectral centroid, bandwidth, roll-off and onset strength (mean, standard deviation and maximum): brightness, loudness and impulsiveness;
* spectral flatness and tempo.

Features are cached per clip (keyed by Audio ID, file size and modification time), so retraining after a small dataset change only extracts new or changed clips.

## 10. Python model development and the models compared

We trained each candidate inside a `StandardScaler` pipeline and tuned it with `GridSearchCV` using **GroupKFold grouped by Audio ID**, scored by macro-F1. Grouping matters: without it, segments of the same recording end up in both the training and validation folds and the scores look much better than they really are.

The training set had 8,270 clips (3,240 originals + 5,030 augmented), which gave **38,210 segments**. Silent segments inside event clips are dropped, because a silent second of a gunshot clip is not a gunshot.

| Model | CV macro-F1 | Val accuracy | Val macro-F1 | Training time |
|---|---|---|---|---|
| **XGBoost** (depth 6, lr 0.1) ✅ | 0.790 | **0.844** | **0.847** | 1,462 s |
| MLP (256-128, α 0.001) | 0.779 | 0.805 | 0.818 | 292 s |
| Random Forest (300 trees) | 0.742 | 0.769 | 0.802 | 953 s |

An RBF-kernel SVM is implemented too, but it needed several hours on 38k segments on a laptop, so we did not include it in the final run. XGBoost won on validation macro-F1 and was selected. On the **unseen test split (695 clips)** it reached **86.5 % accuracy and 0.863 macro-F1**, above the SRS targets of 85 % and 0.80. Recall on the critical classes was Gunshot 0.93, Panic Scream 0.93, Aggression 0.89 and Person Asking for Help 1.00. Glass Breaking reached 0.83, but that is 5 of only 6 test clips, which is too few to be confident about.

One lesson: an earlier 9-class MLP had a higher *accuracy* (93 %) but a lower macro-F1 (0.82), because it was very good at the large classes and could not predict Help at all. Macro-F1 is the honest metric when classes are imbalanced.

## 11. GTM training

Teachable Machine's audio project uses transfer learning on the TensorFlow.js *speech-commands* model: 1-second windows, 44.1 kHz, spectrograms computed by the browser's FFT. We cut 1-second samples from the **same training recordings** as the Python model (`prepare_gtm_samples.py`), so neither model sees test data.

The hard part was getting audio *into* GTM. Its audio "Upload" button only accepts GTM's own sample archives, not WAV files. For the first model we built per-class playlists and played them through the speakers into the GTM recorder. That works, but the speaker-to-microphone path colours the sound and limits the number of samples, so the first GTM model was clearly weaker than the Python model. We then studied the archive format Teachable Machine itself downloads (a `samples.json` with the 43 × 232 spectrogram of every sample plus a `.webm` recording) and wrote `prepare_gtm_samples.py --zip`, which produces exactly that format from our WAV files, with the spectrogram computed the same way as Teachable Machine's recorder. Every class, including the tenth (Person Asking for Help), can now be uploaded digitally with up to 400 samples. The details are in `GTM_TRAINING_LOG.md`.

In the app, `gtm.js` reproduces the exact spectrogram the Web Audio `AnalyserNode` would produce (FFT 2048, 43 frames × 232 bins), so uploaded files and live windows are classified the same way GTM saw its training samples.

## 12. Confidence comparison and model disagreement

For every event we store both full probability vectors and compute:

* **class match**: do the top classes agree?
* **top-class confidence difference**: |Python top − GTM top|;
* **top-two margin** for each model: how far the winner is ahead of the runner-up.

These give a **consistency status**: *Acceptable Match* (same class, both confident, confidences within 0.20), *Weak Match* (same class but a larger gap or low confidence), *Model Disagreement*, or *Uncertain Result* (both below the minimum confidence, or GTM unavailable). Disagreement is not treated as an error to hide. It is a signal. When a rule requires model agreement (Gunshot, for example) and the models disagree, the rule engine does not raise the alert automatically; it sends the event to manual review with both opinions visible. When GTM is not available (for example, the browser tab was closed), the event is marked *Uncertain Result* and decided by the Python model alone.

Most disagreements we saw fell into three groups: (1) short events where GTM's 1-second window missed the transient; (2) tonal sounds (siren vs alarm vs horn) where GTM confused neighbours; and (3) noisy live audio where GTM said Background and Python said something weak. The Admin → Model comparison report runs both models on the unseen test clips and exports the full SRS column set to Excel, including an explanation for each major disagreement.

## 13. Alert rules

Rules live in `alert_rules/alert_rules.json` and can be edited in the admin UI. Each rule has a sound category, minimum confidence, minimum top-two margin, required consecutive detections, a model-agreement requirement, a minimum audio quality, a severity, a recommended action, a manual-review condition and an escalation condition. For example, Gunshot is **Critical**: it needs ≥ 0.70 confidence, a top-two margin of 0.20, two consecutive detections, model agreement and at least Acceptable quality, otherwise it goes to manual review. Glass Breaking is **High** and escalates to Critical after three repeats. Vehicle Horn is **Low**, only logged, and escalates to Medium if it repeats six times. Every decision stores a **rule trace**, so the event page can say exactly why an alert was or was not raised.

## 14. Real-time monitoring

The live page records 2-second windows with the Web Audio API and posts each window to the server. The browser classifies the same window with GTM in parallel. The server keeps a short history per session, so a rule that needs three consecutive detections can fire on the third window. A red privacy banner stays on while the microphone is active. `scripts/benchmark_performance.py` measures this with the real model: a live window takes about 0.11 s on the server and a 30-second upload about 1.5 s, far inside the SRS targets of 3 s and 8 s. With 20,000 stored events every page still loads in under half a second.

## 15. Difficulties

* **Long training runs on a laptop.** Grid search over 38k segments took more than 45 minutes. Windows' console QuickEdit mode paused the process whenever someone clicked in the window, and Ctrl+C did not stop scikit-learn. We added feature caching, a `--fast` grid and `--models` selection.
* **Stale caches.** After rebuilding the dataset, Audio IDs were reassigned and cached features belonged to the wrong clips. The cache key now includes the file size and modification time.
* **Duplicate annotations** after re-running TTS generation caused the same file to appear twice. `build_dataset.py` now keeps the last annotation per filename.
* **Getting audio into GTM**, described above.

## 16. False positives

The largest false-positive source on the test split was **Panic Scream (32 false positives)**: loud, high-pitched voices, laughing children and some animal calls look similar in the spectrum. Background Noise produced 26, and Animal Sound 12. In live use the main problem was everyday room sound (typing, knocking, chairs) being called Animal Sound before we added those categories as negatives. Our mitigations are more varied negative data, per-class minimum confidence and margin in the rules, consecutive-detection requirements for non-critical classes, and manual review for single-model detections.

## 17. False negatives

For a safety system a missed event is worse than a false alarm. On the test split the largest false-negative counts were Background Noise (39, mostly predicted as Scream or Animal), Animal Sound (24) and Gunshot (8; very distant or very short shots). **Glass Breaking missed 1 of 6** test clips, which shows the real weakness: 40 clips is not enough. The segment-maximum aggregation, which lets a single strong segment decide the clip, was added specifically to reduce false negatives on short events.

## 18. Noise robustness

We add white noise to every test clip at 20, 10 and 5 dB SNR and classify again. With the final XGBoost model, accuracy goes from 0.86 (clean) to 0.81 at 20 dB, 0.71 at 10 dB and 0.66 at 5 dB. Moderate noise is handled; very strong noise is not. Training with background-noise augmentation at random SNR helped, and the quality score warns the user when the SNR is too low for a reliable decision; Poor-quality detections of critical classes go to manual review instead of raising an automatic alert.

## 19. Security

Passwords are hashed with Werkzeug and must follow a policy. Accounts lock after repeated failed logins. Every POST carries a CSRF token, and session cookies are HttpOnly and SameSite. Access is role-based with five roles (Normal User, Audio Reviewer, Security Operator, Maintenance Operator, Administrator), each with its own route permissions. Media files are served only to users allowed to see the event. Admin actions are written to an audit log. Uploads are checked by content, not just extension, and size limits prevent resource exhaustion.

## 20. Privacy

Audio of real people is sensitive. The live monitor shows a clear red banner while recording. Audio is kept for 90 days and event records for 365 days by default (configurable). The team recordings use anonymous speaker codes and consent. No audio is sent to any third-party AI service: GTM runs inside the user's own browser, and the Python model runs on our server.

## 21. Limitations

* Glass Breaking (40 clips) and Machinery Fault (138 clips, one recording set) are still small.
* The Help class is based largely on synthetic TTS voices and a few team speakers; real, varied voices would make it more robust.
* The GTM model is weaker than the Python model because of how its samples had to be recorded.
* Very strong noise (5 dB SNR) reduces accuracy a lot.
* Help detection is sound classification, not speech recognition; it knows only the trained phrases.
* SQLite fits a single server; a real deployment needs PostgreSQL and a message queue.
* This is a competition prototype, not a certified emergency system.

## 22. Lessons learned

1. **Data work dominates.** Most of our time went into sourcing, mapping, cleaning, balancing and checking data, not into model code.
2. **Measure the right thing.** Macro-F1 and per-class recall told the truth when accuracy did not.
3. **Negatives are features.** A model is only as good as its idea of "nothing important is happening".
4. **Automate quality checks.** The level check caught 73 silent recordings that would otherwise have poisoned the Help class.
5. **Disagreement is information.** Two independent models are most useful when they disagree, because that is where a human should look.
6. **Make everything explainable.** Storing rule traces, both models' scores and the model version made debugging and review far easier.

## 23. Future enhancements

* Larger and more varied Glass Breaking and Machinery Fault data (FSD50K, TUT Rare Sound Events, more MIMII machine types).
* Real, consented help recordings in several languages, and optional keyword spotting.
* A pretrained audio embedding (YAMNet or PANNs) as an extra Python candidate.
* Sound-source direction with a microphone array, and multi-device monitoring of several rooms.
* SMS, e-mail or push notifications for Critical alerts, and PostgreSQL with a worker queue for scale.
* Active learning: reviewer corrections flow back into the training set automatically.

---

*SonicSentinel AI shows that a small team, working only with licensed public data and careful engineering, can build a listening system that is useful and honest about its uncertainty.*
