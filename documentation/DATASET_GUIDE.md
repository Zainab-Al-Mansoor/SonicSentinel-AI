# Dataset guide

Target: **≥ 3,000 unique original clips, about 300 per class**. Augmented copies do **not** count as originals.

## Suggested free sources (check each licence and record it)

| Class | Good sources | Notes |
|---|---|---|
| Machinery Fault | **MIMII** (abnormal fan/pump/valve/slider, CC BY-SA 4.0), MIMII DUE / DCASE Task 2 | use only `abnormal` files |
| Glass Breaking | ESC-50 `glass_breaking` (40), FSD50K `Shatter`/`Glass`, own recordings (bottles in a bin, safely) | |
| Alarm or Siren | UrbanSound8K `siren`, ESC-50 `siren` + `clock_alarm`, FSD50K `Siren`/`Alarm` | |
| Vehicle Horn | UrbanSound8K `car_horn`, ESC-50 `car_horn`, FSD50K | |
| Animal Sound | UrbanSound8K `dog_bark`, ESC-50 animals, FSD50K | |
| Gunshot | UrbanSound8K `gun_shot` (374), FSD50K `Gunshot_and_gunfire` | the hardest class to source ethically, so prefer licensed sets |
| Panic Scream | FSD50K `Screaming`, **voluntary team recordings** | never record anyone who is actually in distress |
| Aggression | FSD50K `Shout`/`Yell`, voluntary acted recordings (angry shouting, arguments) | |
| Person Asking for Help | **voluntary recordings** of the defined safety phrases + TTS (`scripts/generate_help_phrases_tts.py`) | phrases: "Help me", "Somebody help", "Please help", "Call for help", "Emergency" |
| Background Noise | UrbanSound8K `air_conditioner`/`engine_idling`, ESC-50 rain/wind/waves, **your own rooms, corridors, streets** | record the real place where you will demo |

Downloads:

* UrbanSound8K: https://urbansounddataset.weebly.com/urbansound8k.html
* ESC-50: https://github.com/karolpiczak/ESC-50
* FSD50K: https://zenodo.org/records/4060432
* MIMII: https://zenodo.org/records/3384388

## Variation checklist (required by the SRS)

Record or collect across: different devices (phone, laptop, USB mic), distances (near, 3 m, 10 m+), indoor and outdoor, quiet and noisy, echo/reverberation, loud and soft, single and overlapping events, and several speakers for speech classes. Put these values in `audio_dataset/raw/annotations.csv` (`environment`, `device`, `distance_m`). They end up in the dataset metadata.

## Ethics

* Record people only with consent, and use anonymous speaker codes (`S01`).
* Never record real emergencies or people who have not agreed.
* Keep the licence of every imported clip in `annotations.csv`. Do not publish clips whose licence forbids redistribution. Share them through a private link and give download instructions instead.

## Pipeline

1. `scripts/import_public_datasets.py`, `scripts/record_samples.py` and `scripts/generate_help_phrases_tts.py` put clips in `audio_dataset/raw/<Class>/`.
2. `scripts/build_dataset.py` validates each clip, removes exact duplicates, assigns Audio IDs, converts to 44.1 kHz mono WAV, makes the stratified 70/15/15 split and writes the metadata and statistics.
3. `python -m augmentation.augment` augments TRAIN clips only. Augmented copies keep the parent Audio ID in `parent_audio_id`.
4. The Python model uses the TRAIN split (with segments and augmentation). Validation is used for model selection and test for the final report.
5. `python -m gtm_model.prepare_gtm_samples` builds GTM samples from the same TRAIN recordings.

Segments and augmented copies always stay in the split of their original clip, and cross-validation is grouped by original Audio ID. This stops "leakage", where the model has already heard a test clip during training.
