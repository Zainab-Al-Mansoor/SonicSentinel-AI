# Sample audio

`test_cases/` contains small **synthetic** files for demonstrating validation and quality checks (safe to publish):

| File | Expected result |
|---|---|
| silent_3s.wav | rejected – no usable audio signal |
| too_short_0.2s.wav | rejected – shorter than 0.5 s |
| corrupted.wav | rejected – file integrity check failed |
| not_audio.txt | rejected – unsupported format |
| clipped_tone.wav | quality **Poor** – severe clipping |
| very_low_level.wav | quality warning – very low signal strength |
| noisy_white_noise.wav | noise-dominated recording |
| missing_frames.wav | missing-frame gaps detected |
| stereo_tone.flac | accepted – stereo FLAC converted to mono |

Add 1–2 **licensed or self-recorded** example clips per sound class to `sample_audio/classes/<Class Name>/` for evaluators to try (for example, test-split clips you are allowed to share).
