# Development log

The SRS requires a log of work completed, problems, dataset changes, model failures, code changes and tests performed. Add an entry at least once per competition day and commit it together with the code.

## Suggested 5-day plan

| Day | Goals | Owner(s) |
|---|---|---|
| 1 | Install and run the app, collect/import the dataset, start voluntary recordings | all |
| 2 | Reach 300 clips per class, run `build_dataset.py`, augment, first `train_models --fast` | data + ML |
| 3 | Full Python training, GTM training/export/installation, tune thresholds on the validation split | ML + GTM |
| 4 | Test-set evaluation (comparison report), live-mic testing in the demo room, fix false alarms, write the report/blog | all |
| 5 | Deploy, record the demo video, final tests, screenshots, final submission checklist | all |

## Log

### Day 1 – 2026-09-23
* **Work completed:**
  * Environment set up (Anaconda, Python, FFmpeg), database initialised, test suite passing (55 tests).
  * Public datasets downloaded into `downloads/` (UrbanSound8K, ESC-50, MIMII, Kaggle gunshot, Kaggle screaming, VSD violence).
  * Dataset imported with the new `scripts/import_local_downloads.py`, validated and split with `scripts/build_dataset.py`,
    training split augmented (`--per-clip 1`).
  * Python model trained (`train_models --fast --models rf,mlp`); selected model **MLP** (`py-mlp-20260923-1457`).
* **Problems encountered:**
  * `scripts/import_public_datasets.py` expects the official dataset folder layouts; the downloaded copies use the Kaggle
    layouts and the gunshot / scream / VSD sets were not supported → wrote `scripts/import_local_downloads.py`.
  * `train_models.py` crashed in XGBoost when a class has no training data (Person Asking for Help) → fixed (see code changes).
  * SVM training on ~31k segments was too slow (hours) and could not be stopped with Ctrl+C on Windows → trained RF + MLP only.
  * Anaconda Prompt "QuickEdit/Select" mode paused the training process when the window was clicked.
* **Dataset changes:** 5,582 original clips after validation (225 rejected), 70/15/15 split (train 3,907 · val 837 · test 838).

  | Class | Clips | Source |
  |---|---|---|
  | Machinery Fault | 138 | MIMII (abnormal only) |
  | Glass Breaking | 40 | ESC-50 |
  | Alarm or Siren | 983 | UrbanSound8K, ESC-50 |
  | Vehicle Horn | 390 | UrbanSound8K, ESC-50 |
  | Animal Sound | 950 | UrbanSound8K, ESC-50 |
  | Gunshot | 921 | UrbanSound8K, Kaggle gunshot (9 weapons) |
  | Panic Scream | 862 | Kaggle Human Screaming Detection |
  | Aggression | 298 | VSD (one clip per annotated violence interval) |
  | Person Asking for Help | 0 | still to be recorded / TTS |
  | Background Noise | 1,000 | UrbanSound8K, ESC-50 |

* **Model failures / results:**
  * Validation: MLP accuracy 0.931, macro-F1 0.797 · RF accuracy 0.870, macro-F1 0.690.
  * Test (MLP): **accuracy 0.931, macro-F1 0.819** (≈0.91 over the 9 classes that have data).
  * Weak classes: Glass Breaking (F1 0.73, only 40 clips), Person Asking for Help (no data → F1 0).
  * Noise robustness: accuracy 0.86 at 20 dB SNR, 0.77 at 10 dB SNR.
  * Live microphone: room noise sometimes classified as Animal Sound (no recordings of the demo room in the training data yet).
* **Code changes:**
  * `python_models/train_models.py`: warns about classes without training data and skips XGBoost instead of crashing.
  * New `scripts/import_local_downloads.py` (imports the local `downloads/` layouts, caps clips per class and per source).
  * New `RUN_AND_TRAIN_GUIDE.md` (setup → dataset → training → GTM → run).
  * `.gitignore`: `downloads/`, `_training_pack/` and editor swap files are no longer tracked.
* **Tests performed:** `python -m pytest -q` (55 passed), test-set evaluation and noise-robustness run by `train_models`.

### Day 2 – 2026-09-24
* **Work completed:**
  * GTM model trained in Teachable Machine (9 classes: Aggression, Alarm or Siren, Animal Sound, Background Noise,
    Glass Breaking, Gunshot, Machinery Fault, Panic Scream, Vehicle Horn) and exported to `gtm_model/model/`.
  * App running with both models (`python run.py`).
* **Problems encountered:**
  * Teachable Machine's audio "Upload" only accepts its own sample zips → samples recorded through the microphone
    while playing the `_playlists/<Class>.wav` files.
  * Many GTM predictions are wrong: speaker→microphone recording distorts the sound, few samples per class.
* **Next steps:**
  * Record Person Asking for Help (voluntary speakers + TTS) and demo-room Background Noise, then retrain both models.
  * Add more Glass Breaking clips (FSD50K / TUT Rare Sound Events) and more MIMII abnormal clips.
  * Map extra everyday ESC-50 categories (typing, knocking, coughing …) to Background Noise to reduce false alarms.
  * Raise `min_confidence` / `unknown_threshold` in Admin → Settings after testing in the demo room.
* **Tests performed:** manual upload and live-mic tests from the dashboard.

### Day 2 (continued) – 2026-09-24
* **Work completed:**
  * Dataset rebuilt from scratch (`scripts/clean_dataset.py --yes`, re-import with per-source caps, `build_dataset.py`).
  * **Person Asking for Help** added: 204 offline-TTS phrases + 17 team recordings.
  * More negatives mapped to Background Noise: 35 everyday ESC-50 categories, US8K street music / children playing /
    drilling / jackhammer, MIMII `normal`, Kaggle `NotScreaming`, calm VSD film audio.
  * Training split balanced with augmentation (`--per-clip 1 --balance-to 600`): 5,030 augmented clips.
  * Python model retrained on 10 classes: `train_models --fast --models rf,mlp,xgb`; **XGBoost selected**.
  * New Mocha colour theme with glass effect across all pages.
  * Added: performance benchmark script, Dockerfile + Render blueprint + deployment guide, microphone
    device selection and level check in `record_samples.py`, project report, technical blog, GTM training log,
    demo-video script, submission checklist, team contribution record.
* **Problems encountered:**
  * 73 Help recordings and 10 Background recordings were rejected as near-silent (−64 to −75 dBFS): the wrong
    microphone input was selected. → `record_samples.py --list-devices` / `--input N`, and clips under −45 dBFS are no longer saved.
  * Re-running TTS created duplicate annotation rows → `build_dataset.py` keeps the last row per filename.
  * Cached features belonged to the wrong clips after the rebuild → cache key now includes file size and mtime.
  * XGBoost training took ~25 minutes; SVM still skipped (hours on 38k segments).
* **Dataset changes:** 4,629 original clips (241 rejected), ≈ 6.6 h; split train 3,240 · val 694 · test 695;
  38,210 training segments.
* **Model results (unseen test split, 695 clips):** accuracy **0.865**, macro precision 0.847, macro recall 0.895,
  macro-F1 **0.863**. Critical recall: Gunshot 0.93, Panic Scream 0.93, Aggression 0.89, Help 1.00, Glass Breaking 0.83 (5/6).
  Validation: XGB 0.844 / 0.847 F1 · MLP 0.805 / 0.818 · RF 0.769 / 0.802.
  Most false positives: Panic Scream (32), Background (26), Animal (12).
* **Tests performed:** `python -m pytest -q`; test-set evaluation and noise-robustness from `train_models`;
  benchmark script tested on a copy of the project (all NFR targets met); every page checked after the theme change.
* **Later the same day:**
  * Training finished: model `py-xgb-20260924-0800`; noise robustness 0.81 (20 dB) · 0.71 (10 dB) · 0.66 (5 dB).
  * `prepare_gtm_samples.py --zip`: writes Teachable Machine's own sample-archive format (samples.json + webm) for every
    class, so GTM samples are uploaded digitally instead of being played into the microphone. Verified in the TM web app.
  * `/healthz` health check, `run_server.bat` (waitress + automatic restart), Docker HEALTHCHECK, Render health check.
  * Benchmark with the real model: 30-s upload 1.5 s, live window 0.11 s, 20,020 events, 5 concurrent users, 0 errors.
  * Tests: 58 passed.
* **Next steps:** GTM run 2 with the 10 upload archives, model-comparison Excel, more Glass Breaking (TUT Rare Sound
  Events), screenshots, deploy, demo video.
