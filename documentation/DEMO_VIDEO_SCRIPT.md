# Demonstration video – script and shot list

SRS deliverable 13: one **.mp4** that shows every item below. Target length **8–12 minutes**, 1080p screen recording
with narration (OBS Studio, or Windows `Win + Alt + R` / Xbox Game Bar). Every team member should present at least one part.

**Before recording**

* Start the app (`python run.py`), log out, clear old alerts (or use a fresh database: `python -m database.init_db`).
* Prepare a folder `demo_clips/` with one test clip per class (from `audio_dataset/processed/test/<Class>/`) and:
  `sample_audio/test_cases/silent_3s.wav`, `too_short_0.2s.wav`, `corrupted.wav`, `not_audio.txt`, `very_low_level.wav`, `clipped_tone.wav`,
  `noisy_white_noise.wav` or a class clip mixed with loud noise, an overlap clip (e.g. siren + dog bark), and a copy of one clip for the duplicate test.
* For the live part, play a gunshot / scream clip from a phone near the laptop microphone (never stage a real emergency).
* Open Chrome at 100 % zoom and close notifications.

| # | Time | SRS item | What to show | What to say |
|---|---|---|---|---|
| 1 | 0:00 | Intro | title slide / home page | team, problem, two models, 10 classes |
| 2 | 0:30 | **Login** | log in as `reviewer` or `admin`; show that a wrong password is refused | roles, password hashing, lock-out |
| 3 | 1:00 | **Audio upload** | Upload page, drag a Gunshot clip, show the preview player | supported formats and limits |
| 4 | 1:15 | **File validation** | upload `not_audio.txt`, `corrupted.wav`, `silent_3s.wav`, `too_short_0.2s.wav` → clear error messages | nothing crashes, bad input is explained |
| 5 | 1:45 | **Metadata** | event page → Audio ID, format, duration, sample rate, channels, size, SHA-256 | |
| 6 | 2:00 | **Preprocessing** | pre-processing panel (mono, 22,050 Hz, normalise, trim, noise reduction, segments) | same pipeline as training |
| 7 | 2:15 | **Waveform** | waveform image | where the event is |
| 8 | 2:25 | **Spectrogram** | log-Mel spectrogram | what the model "sees" |
| 9 | 2:35 | **Audio quality** | quality badge + level, clipping, SNR, silence ratio | Good / Acceptable / Poor / Unusable |
| 10 | 2:50 | **Python prediction** | Python class | XGBoost on 299 features |
| 11 | 3:00 | **Python confidence** | confidence bars for all classes | |
| 12 | 3:10 | **GTM prediction** | GTM class (browser) | TensorFlow.js model from Teachable Machine |
| 13 | 3:20 | **GTM confidence** | GTM bars for all classes | |
| 14 | 3:30 | **Model comparison** | class-match status, consistency status | Acceptable Match / Weak Match / Disagreement / Uncertain |
| 15 | 3:45 | **Confidence difference** | \|Python − GTM\| and both top-2 margins | |
| 16 | 4:00 | **Every sound class** | Batch page: upload the 10 class clips at once, then show the history list with 10 results | one clip each: Machinery, Glass, Alarm, Horn, Animal, Gunshot, Scream, Aggression, Help, Background |
| 17 | 5:00 | **Low-confidence case** | `very_low_level.wav` or an ambiguous clip → *Uncertain* / *Unknown* | no automatic alert when confidence is low |
| 18 | 5:20 | **Model-disagreement case** | a clip where Python and GTM disagree → *Model Disagreement* → sent to review | |
| 19 | 5:40 | **Noisy case** | noisy clip → quality Poor, lower confidence | noise robustness numbers |
| 20 | 6:00 | **Overlapping-sound case** | siren + dog clip → *Overlapping sounds* flag with both classes | |
| 21 | 6:15 | Duplicate audio | re-upload the same file → duplicate warning (SHA-256 / fingerprint) | |
| 22 | 6:30 | **Live microphone monitoring** | Live → Start; red privacy banner; windows updating every 2 s; Python + GTM per window; latency | |
| 23 | 7:00 | **Repeated detection** | play a siren for ~6 s → counter reaches the required consecutive detections | |
| 24 | 7:20 | **Real-time critical detection** | play a gunshot / scream → Critical alert toast within a few seconds | |
| 25 | 7:40 | **Critical alert** | Alerts page: severity, rule trace, recommended action | |
| 26 | 7:55 | **Alert acknowledgement** | Acknowledge (and show Escalate / Dismiss buttons) | action is logged with user and time |
| 27 | 8:10 | **Manual review** | Review queue → open an event, listen, see both models | |
| 28 | 8:30 | **Reviewer override** | correct the label, add a comment, close → final label changes, audit entry | |
| 29 | 8:50 | **Dashboard** | user dashboard + admin dashboard charts (class counts, severity, trends) | |
| 30 | 9:10 | **Event history** | History filters (class, severity, date, confidence) + Timeline | |
| 31 | 9:30 | **Report generation** | event → Download report; Admin → CSV/Excel exports; Model comparison Excel | |
| 32 | 9:50 | Admin | Settings (thresholds), Alert rules editor, Models page (metrics, confusion matrix) | configurable without code |
| 32a | 10:15 | ⭐ Explainable AI | event page → *Why this prediction?*: highlighted event span, property bars | "our system says not only *what* but *why* and *where*" |
| 32b | 10:35 | ⭐ Robustness Lab | Lab → random Gunshot test clip → presets Busy street / Worst case → stress test chart | hidden-test readiness live: when each model breaks |
| 33 | 11:15 | Close | limitations + future work, repository and deployment URLs | |

All 29 items required by SRS §1.10-13 are **bold** in the table.

**Tips**

* Record in short parts (one row group per file) and join them in Clipchamp / Shotcut; re-record only the parts that go wrong.
* Zoom the browser to 110–125 % for the event page so numbers are readable.
* Export as **MP4 (H.264), 1080p**. Upload to YouTube (unlisted) or Google Drive and put the link in README → *Deliverables*.
