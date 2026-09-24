# Google Teachable Machine (GTM) model

The GTM model is trained **separately** from the Python model, but on the **same underlying TRAIN recordings**. It runs in the browser with TensorFlow.js. The web app never sends the Python prediction to it.

## 1. Prepare samples (train split only)

```bash
python -m gtm_model.prepare_gtm_samples --max-per-class 400 --zip
```

This creates:

* `gtm_model/training_samples/_tm_upload/<Class Name>.zip`: **Teachable Machine upload files**, one per class
* `gtm_model/training_samples/<Class Name>/*.wav`: the same 1-second, 44.1 kHz samples as WAV (for listening)
* `gtm_samples_metadata.csv`: every sample traced back to its Audio ID; `gtm_sample_counts.csv`: samples per class

Teachable Machine's audio **Upload** button only accepts archives in its own format (`samples.json` with the
spectrogram of every sample + a `.webm` audio file). The `--zip` files use exactly that format (checked against the
Teachable Machine web app: the archives are accepted and pass its training-data validation). The spectrograms are
computed exactly like Teachable Machine's own recorder (44.1 kHz, FFT 2048, Blackman window, no smoothing, 1024-sample hop,
43 frames × 232 bins), so the audio goes in **digitally** – no speaker → microphone recording, and up to 400 samples per class.

## 2. Train in Teachable Machine

1. Open https://teachablemachine.withgoogle.com/train/audio in **Chrome** (keep the tab visible while training).
2. Create 10 classes with **exactly** these names: `Background Noise` (already there),
   `Machinery Fault, Glass Breaking, Alarm or Siren, Vehicle Horn, Animal Sound, Gunshot, Panic Scream, Aggression, Person Asking for Help`.
3. For every class: **Upload** → *Choose files from your computer* → select `_tm_upload/<same class name>.zip`.
   The class card then shows the number of samples (e.g. "400 Audio Samples").
4. **Advanced**: Epochs 50 (default) – 80, keep the default batch size and learning rate (write them down). Click **Train Model** (takes a few minutes).
5. Check *Advanced → Under the hood* (accuracy per class, confusion matrix) and take screenshots of every class card,
   the settings and these charts for `documentation/GTM_TRAINING_LOG.md`.

Old method (still possible): `--playlist` writes `_playlists/<Class>.wav`; play it into GTM's microphone recorder
(best through a virtual audio cable such as VB-Audio Virtual Cable).

## 3. Export and install

1. Click **Export Model** → **TensorFlow.js** → **Download** (not "Upload"). This gives you `tm-my-audio-model.zip`.
2. Unzip it into this folder so you have:
   ```
   gtm_model/model/model.json
   gtm_model/model/metadata.json
   gtm_model/model/weights.bin
   ```
3. Restart the app. The Admin → Models page shows the GTM version and labels, and warns about missing or extra labels.
4. Optionally, keep the shareable GTM project link in `gtm_model/GTM_PROJECT_LINK.txt`.

## How the app uses it (explain this to evaluators)

* `static/js/gtm.js` loads the model with `speechCommands.create("BROWSER_FFT", …)`. This is the same library the GTM export snippet uses (tfjs 1.3.1, speech-commands 0.4.0).
* GTM normally listens only to its own microphone stream. To classify **uploaded files** and **our live windows**, `gtm.js` computes the same spectrogram the Web Audio `AnalyserNode` would produce: 44.1 kHz audio, FFT 2048, Blackman window, dB magnitude, 43 frames × 232 bins with a 1024-sample hop, then z-normalisation. It passes that spectrogram to `recognizer.recognize()`.
* Segments longer than 1 s are classified in 1-s windows with a 0.5-s hop, and the scores are averaged.
* Scores go to `/api/events/<id>/gtm` (uploads) or come with each live window. The server stores them and only then combines them with the Python result.
