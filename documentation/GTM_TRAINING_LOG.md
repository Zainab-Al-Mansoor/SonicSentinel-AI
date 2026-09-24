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
| Background Noise | _fill in_ | 400 |
| Machinery Fault | _fill in_ | 400 |
| Glass Breaking | _fill in_ | 75 |
| Alarm or Siren | _fill in_ | 400 |
| Vehicle Horn | _fill in_ | 400 |
| Animal Sound | _fill in_ | 400 |
| Gunshot | _fill in_ | 400 |
| Panic Scream | _fill in_ | 400 |
| Aggression | _fill in_ | 400 |
| Person Asking for Help | – (no data yet) | 292 |

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

**Run 2 result (export 2026-09-24T15:50, 10 classes) – checked on the unseen TEST split (695 clips) with the app's own
segment / aggregation / comparison logic:**

| Model | Accuracy | Macro-F1 |
|---|---|---|
| Python (XGBoost) | 0.865 | 0.863 |
| GTM run 2 export | **0.158** | 0.174 |
| Final combined decision | 0.845 | 0.846 |

* The exported GTM model also scores only ≈ 20 % on **its own training samples** (from the `_tm_upload` zips), with low
  confidence (top score ≈ 0.5). A correctly trained head reaches ≈ 99 % on training samples, so this export is
  **under-trained** (training stopped early, too few epochs / very low learning rate, or exported before training finished).
* Diagnostic: a new classification head trained on the same Teachable Machine base model and the same upload samples
  (SGD, learning rate 0.01, 50 epochs) reaches **≈ 64 % test accuracy**. That is roughly the ceiling of GTM's 1-second
  speech-commands transfer model on these environmental sounds; it will not reach the 85 % that the Python model reaches.
* **Run 3 (done, export 2026-09-24T20:51):** test accuracy **0.606** (was 0.158), macro-F1 0.607; ≈ 80 % on its own training samples. Best classes: Alarm or Siren (F1 0.82), Help (0.73), Vehicle Horn (0.72); weakest: Panic Scream (0.41), Machinery Fault precision (0.29). Combining both models with `python_weight` 0.7 gives 0.868 accuracy – better than either model alone.
* **Run 3 steps that were used:** new Teachable Machine project → upload the 10 zips → *Advanced*: epochs 100 → Train with the tab
  visible → wait for "Model Trained" → check *Under the hood* (training accuracy should be > 90 %) → test 2–3 classes in
  the preview → Export → replace `gtm_model/model/`.

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
| 2 | 2026-09-24 | 10 | + Help, digital upload with `--zip` (up to 400 samples per class) | all 10 labels ✅; under-trained export: 15.8 % test accuracy |
| 3 | 2026-09-24 | 10 | same `--zip` uploads, training fix, epochs 50, trained to *Model Trained* | **test accuracy 0.606, macro-F1 0.607** (695 unseen clips); combined with Python (weight 0.7): **0.868 / 0.866** |

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
