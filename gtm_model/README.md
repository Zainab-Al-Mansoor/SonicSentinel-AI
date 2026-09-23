# Google Teachable Machine (GTM) model

The GTM model is trained **separately** from the Python model, but on the **same underlying TRAIN recordings**. It runs in the browser with TensorFlow.js. The web app never sends the Python prediction to it.

## 1. Prepare samples (train split only)

```bash
python -m gtm_model.prepare_gtm_samples --max-per-class 400 --playlist
```

This creates:

* `gtm_model/training_samples/<Class Name>/*.wav`: 1-second, 44.1 kHz clips
* `gtm_model/training_samples/_playlists/<Class Name>.wav`: all samples of one class joined into one file
* `gtm_samples_metadata.csv`: every sample traced back to its Audio ID

## 2. Train in Teachable Machine

1. Open https://teachablemachine.withgoogle.com/train/audio
2. Rename/add classes so the names match **exactly**:
   `Machinery Fault, Glass Breaking, Alarm or Siren, Vehicle Horn, Animal Sound, Gunshot, Panic Scream, Aggression, Person Asking for Help` (the `Background Noise` class already exists).
3. Add samples to each class:
   * **If your GTM version offers a file-upload option for audio samples**, upload the WAV files from the matching `training_samples/<Class>` folder.
   * **Otherwise**, record through the microphone input. Play `_playlists/<Class>.wav` while GTM records. A free virtual audio cable (e.g. VB-Audio Virtual Cable) lets you route the audio straight in without a speaker and mic. Record at least 20 seconds of Background Noise the same way.
4. Click **Train Model**. Note the settings you used (epochs, batch size, learning rate) in `documentation/GTM_TRAINING_LOG.md`.
5. Test with a few **validation** clips, then take screenshots of every class, the sample counts and the test results. These are needed for the SRS evidence.

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
