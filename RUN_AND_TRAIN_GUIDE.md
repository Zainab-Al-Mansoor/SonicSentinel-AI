# SonicSentinel AI – run & train guide (from zero)

All commands run in **Command Prompt (cmd)** from the project folder on Windows.

---

## 0. One-time checks

```bat
cd "C:\Users\CZ\Desktop\SonicSentinel AI"
python --version            REM must be 3.10, 3.11 or 3.12
ffmpeg -version             REM if missing:  winget install Gyan.FFmpeg   (then open a NEW cmd window)
```

**Two fixes to do once:**

```bat
REM a) a stale git lock file exists – git will refuse to commit until it is removed
del ".git\index.lock"

REM b) downloads\ holds ~6 GB of audio and is NOT in .gitignore – stop git from committing it
echo downloads/>> .gitignore
```

## 1. Create the environment and the database

```bat
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install sounddevice pyttsx3          REM for recording + TTS help phrases
python -m database.init_db               REM creates data\sonicsentinel.db + default users
```

(PowerShell instead of cmd? Use `.venv\Scripts\Activate.ps1`; if blocked, run
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once.)

Quick check that the app starts (models not trained yet – that is fine):

```bat
python -m pytest -q
python run.py                            REM open http://127.0.0.1:5000  (login: admin / Admin@12345)
```

Stop it with **Ctrl+C**.

---

## 2. Build the dataset

### 2a. Import what is already in `downloads\`

The project's `scripts\import_public_datasets.py` expects the **official** folder layouts,
but your `downloads\` folder has the **Kaggle** layouts (e.g. `urbansound8k\fold1` instead of
`UrbanSound8K\audio\fold1`, `esc50\audio\audio\`), and it does not know the Gunshot, Scream
and VSD sets at all. Use the new script **`scripts\import_local_downloads.py`** instead:

```bat
python scripts\import_local_downloads.py --dry-run     REM preview only
python scripts\import_local_downloads.py               REM copy into audio_dataset\raw\<Class>\
```

Expected result (defaults: max 500 per class, max 300 from any one source):

| Class | Source in downloads\ | ≈ clips | Status |
|---|---|---|---|
| Machinery Fault | MIMII `abnormal` (138) | 138 | **short ~160** |
| Glass Breaking | ESC-50 `glass_breaking` (40) | 40 | **short ~260** |
| Alarm or Siren | US8K siren + ESC-50 siren/clock_alarm | ~380 | OK |
| Vehicle Horn | US8K car_horn + ESC-50 car_horn | ~340 | OK |
| Animal Sound | US8K dog_bark + ESC-50 animals | 500 | OK |
| Gunshot | US8K gun_shot + Kaggle gunshot (9 weapons) | 500 | OK |
| Panic Scream | Kaggle `Screaming` | 300 | OK |
| Aggression | VSD `angry_*` violence intervals (one ≤6 s clip each) | ~298 | OK (borderline) |
| Person Asking for Help | – | **0** | **short 300** |
| Background Noise | US8K air_conditioner/engine_idling + ESC-50 rain/wind/… | 500 | OK |

### 2b. Fill the missing classes (see section 5 for downloads)

```bat
REM Person Asking for Help – real voices (each team member, anonymous codes)
python scripts\record_samples.py --class "Person Asking for Help" --speaker S01 --count 30 --environment indoor --device laptop --distance 1
python scripts\record_samples.py --class "Person Asking for Help" --speaker S02 --count 30 --environment outdoor --device phone --distance 3
REM ...repeat for S03, S04... until ~150 real clips

REM Person Asking for Help – synthetic (10 phrases x 15 = 150 clips)
python scripts\generate_help_phrases_tts.py --per-phrase 15

REM Machinery Fault – after downloading more MIMII zips, unzip them INTO downloads\mimii\ and re-run:
python scripts\import_local_downloads.py --sources mimii

REM Glass Breaking – put extra glass clips in downloads\glass_extra\ and re-run:
python scripts\import_local_downloads.py --sources glass_extra
REM   or, if you download FSD50K (official layout), use the original importer:
python scripts\import_public_datasets.py --fsd50k "D:\data\FSD50K"
```

### 2c. Validate, split, augment

```bat
python scripts\build_dataset.py --min-per-class 300      REM validate, dedupe, Audio IDs, 70/15/15 split
python -m augmentation.augment --per-clip 2              REM augments TRAIN split only
```

Check `data\dataset_statistics.json` and `data\dataset_rejected.csv` before training.

---

## 3. Train the Python model

```bat
python -m python_models.train_models --fast              REM quick sanity run (minutes)
python -m python_models.train_models                     REM full grid search SVM/RF/XGB/MLP (slow, 16 GB RAM recommended)
REM only some models:  python -m python_models.train_models --models rf,xgb
```

Output: `python_models\saved\sonic_model.joblib` + comparison/confusion-matrix reports in `reports\`.

## 4. Train the Google Teachable Machine model

```bat
python -m gtm_model.prepare_gtm_samples --max-per-class 400 --playlist
```

1. Open https://teachablemachine.withgoogle.com/train/audio
2. Create 10 classes with **exactly** these names: Machinery Fault, Glass Breaking, Alarm or Siren,
   Vehicle Horn, Animal Sound, Gunshot, Panic Scream, Aggression, Person Asking for Help
   (+ the existing Background Noise).
3. Add samples from `gtm_model\training_samples\<Class>\` (or play `_playlists\<Class>.wav` through
   VB-Audio Virtual Cable while GTM records). Train Model.
4. Export Model → TensorFlow.js → **Download** → unzip `model.json`, `metadata.json`, `weights.bin`
   into `gtm_model\model\`.

## 5. Run the app

```bat
.venv\Scripts\activate
python run.py                                    REM http://127.0.0.1:5000  (or double-click run_windows.bat)
python run.py --host 0.0.0.0                     REM LAN access
waitress-serve --port 5000 run:app               REM production-style on Windows
```

Then: Admin → Models (check Python + GTM loaded) → Admin → Model comparison → *Run test-set evaluation*.

---

## 6. Datasets still missing

| Class | Gap | Download | What to do |
|---|---|---|---|
| **Person Asking for Help** | 300 | No public dataset fits – **record** (voluntary, consented) + TTS | Commands in 2b. Aim ≥ 50 % real voices, several speakers, distances, rooms. |
| **Glass Breaking** | ~260 | **FSD50K** – https://zenodo.org/records/4060432 (labels `Shatter`, `Glass`; importer already supports it). Lighter options: **FSDKaggle2019** https://zenodo.org/records/3612637 (label `Shatter`), **TUT Rare Sound Events 2017** https://zenodo.org/records/401395 (isolated `glassbreak` events) | Clips from FSDKaggle2019/TUT go into `downloads\glass_extra\`, then `--sources glass_extra`. Safe own recordings (bottles in a bin) help a lot. |
| **Machinery Fault** | ~160 | **MIMII** – https://zenodo.org/records/3384388 – download one or two more zips, e.g. `6_dB_pump.zip`, `6_dB_valve.zip` (or `0_dB_…`) | Unzip into `downloads\mimii\` (keep the `pump\id_00\abnormal\…` folders) → `--sources mimii`. Only `abnormal` files are used. |
| Aggression (optional) | borderline | FSD50K `Shout` / `Yell` | Adds variety beyond film audio. |
| Background Noise (recommended) | – | **Your own demo room / corridor** recordings | `python scripts\record_samples.py --class "Background Noise" --speaker S00 --count 30 --seconds 5` |

**Empty folder:** `downloads\glass_extra\` exists but has no files yet.

**Not used on purpose:** `mimii\normal` (normal machine sound is not a fault), `scream\NotScreaming`,
the long VSD `angry_01…angry_21` master files and `noviolence_*` (hours long, and the short
`angry_XXX` files already cover them).

**Licences:** UrbanSound8K / ESC-50 are CC BY-NC; the Kaggle gunshot and scream sets and VSD
have their own terms – check them and keep the `annotations.csv` licence column accurate.
