# Google Teachable Machine – training log and evidence

SRS deliverable 5 ("Google Teachable Machine Evidence"). Fields marked _fill in_ can only be completed by the team,
from the Teachable Machine project and the screenshots in `screenshots/gtm/`.

## 1. Project

| Item | Value |
|---|---|
| Tool | Google Teachable Machine → **Audio Project** (https://teachablemachine.withgoogle.com/train/audio) |
| Base model | TensorFlow.js `speech-commands` (BROWSER_FFT), transfer learning; exported as TMv2 |
| Input | 1-second windows, 44.1 kHz, 43 frames × 232 FFT bins (browser FFT 2048) |
| Project link / saved project file | _fill in: shareable link, or `gtm_model/gtm_project.tm` saved from "Save project to Drive/file"_ |
| Exported model | `gtm_model/model/model.json`, `weights.bin`, `metadata.json` (labels + version) |
| Model version shown in the app | read from `metadata.json` (`tfjsSpeechCommandsVersion`, `modelName`, `timeStamp`) |

## 2. Training data

Samples are cut **only from the TRAIN split**, the same recordings the Python model trains on, so GTM never sees
validation or test audio:

```powershell
python -m gtm_model.prepare_gtm_samples --max-per-class 400 --playlist
```

This writes 1-second WAV samples to `gtm_model/training_samples/<Class>/` and one playlist per class to
`gtm_model/training_samples/_playlists/<Class>.wav`.

**How the samples entered GTM.** GTM's audio *Upload* button only accepts sample zips that GTM exported itself, not
WAV files. So each class playlist was played through the speakers while GTM's **Microphone** recorder captured
it (Extract samples). Background Noise was recorded directly in the room where the demo happens (GTM requires
at least 20 background samples).

### Sample counts

| Class | Run 1 (9 classes) | Run 2 (10 classes) |
|---|---|---|
| Background Noise | _fill in_ | from `gtm_sample_counts.csv` |
| Machinery Fault | _fill in_ | _fill in_ |
| Glass Breaking | _fill in_ | _fill in_ |
| Alarm or Siren | _fill in_ | _fill in_ |
| Vehicle Horn | _fill in_ | _fill in_ |
| Animal Sound | _fill in_ | _fill in_ |
| Gunshot | _fill in_ | _fill in_ |
| Panic Scream | _fill in_ | _fill in_ |
| Aggression | _fill in_ | _fill in_ |
| Person Asking for Help | – (no data yet) | _fill in_ |

(Read the counts from each class card in GTM, e.g. "42 Audio Samples", and take a screenshot of every card.)

## 3. Training configuration

| Setting | Run 1 | Run 2 |
|---|---|---|
| Epochs | 50 (default) | _fill in, e.g. 50–80_ |
| Batch size | default – _fill in from Advanced_ | _fill in_ |
| Learning rate | default – _fill in from Advanced_ | _fill in_ |
| Date | 2026-09-24 | _fill in_ |
| Browser / machine | _fill in_ | _fill in_ |

Screenshot the **Advanced** panel and the **Under the hood** accuracy/loss charts for each run.

## 4. Training observations

**Run 1 (2026-09-24, 9 classes):**

* Trained and exported without errors, integrated in the app (`static/js/gtm.js`).
* Many predictions in the app were wrong or disagreed with the Python model. Causes found:
  * **Speaker → microphone path.** Playback colours the sound: laptop speakers cut the low frequencies (gunshot body,
    machinery hum) and add room echo, so GTM learned "speaker sound" as much as the event.
  * **Few samples per class**, 20–40 seconds of audio for some classes, which is too little for 9 classes.
  * **Background recorded at a different level** from the playlists, so loudness alone separated Background from events.
  * **Person Asking for Help** was missing, so help phrases were forced into Panic Scream or Aggression.
* Most confused pairs: Alarm or Siren ↔ Vehicle Horn, Panic Scream ↔ Animal Sound, Gunshot ↔ Glass Breaking
  (short transients), everything quiet → Background Noise.

**Run 2 (retraining, 10 classes) – digital upload instead of microphone playback:**

1. `python -m gtm_model.prepare_gtm_samples --max-per-class 400 --zip` → one Teachable Machine archive per class in
   `gtm_model/training_samples/_tm_upload/` (+ `gtm_sample_counts.csv` with the exact counts for the table above).
2. The archives use Teachable Machine's own sample format (`samples.json` + `.webm`) with spectrograms computed exactly
   like its recorder, so there is **no speaker → microphone distortion** and every class gets up to 400 samples.
   The format was verified in the Teachable Machine web app: the archives were accepted and passed its data validation.
3. New class **Person Asking for Help** (TTS + team recordings from the TRAIN split).
4. Background Noise comes from the same TRAIN split (room noise, everyday sounds, normal machinery, ordinary voices).
5. Train in Chrome with the tab visible; epochs 50–80; check *Under the hood*.
6. Export → TensorFlow.js → Download → replace the three files in `gtm_model/model/`; the app picks the new
   version up after a restart (Admin → Models shows labels and version).
7. Measure accuracy: Admin → **Model comparison** → *Run test-set evaluation* (unseen TEST split) → *Export Excel*.

_fill in after run 2: observations, per-class accuracy from "Under the hood", remaining problems._

## 5. Incorrect classifications (test evidence)

Upload the test clips from `sample_audio/classes/<Class>/` (or any test-split clips) and record wrong GTM results:

| # | Audio ID / file | Actual class | GTM prediction (confidence) | Python prediction (confidence) | Likely reason |
|---|---|---|---|---|---|
| 1 | _fill in_ | | | | |
| 2 | | | | | |
| 3 | | | | | |
| 4 | | | | | |
| 5 | | | | | |

The complete comparison is produced by Admin → **Model comparison** → *Run test-set evaluation* → *Export Excel*
(SRS deliverable 6).

## 6. Retraining details

| Run | Date | Classes | Change | Result |
|---|---|---|---|---|
| 1 | 2026-09-24 | 9 | first model, playlist playback, default settings | works end-to-end, low accuracy, often disagrees with Python |
| 2 | _fill in_ | 10 | + Help, digital upload with `--zip` (up to 400 samples per class) | _fill in_ |

## 7. Integration evidence

* `gtm_model/model/metadata.json` labels must be spelled exactly like `config/settings.py → CLASSES`
  (the app maps them by name; unknown labels are ignored).
* `static/js/gtm.js` loads the model with `speechCommands.create("BROWSER_FFT", undefined, model.json, metadata.json)`,
  computes the same spectrogram as the Web Audio `AnalyserNode`, and posts the scores to `/api/events/<id>/gtm`.
* Screenshots to include in `screenshots/gtm/`:
  1. every class card with its sample count;
  2. the training settings and "Model Trained" state;
  3. the *Under the hood* charts;
  4. the preview panel predicting at least three classes correctly;
  5. the export dialog (TensorFlow.js);
  6. an event page in SonicSentinel showing the GTM prediction and confidence bars next to the Python model.
