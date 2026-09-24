# Hidden-test readiness – robustness results

Python model `py-xgb-20260924-0800` · 266 unseen TEST clips (30 per class max) · same pipeline as the web app · crashes: **0**

| Condition | Clips | Accuracy | Rated Poor/Unusable (→ manual review) | Overlap flagged |
|---|---|---|---|---|
| clean | 266 | 0.898 | 4% | 0% |
| white noise 20 dB | 266 | 0.707 | 4% | 3% |
| white noise 10 dB | 266 | 0.549 | 5% | 3% |
| real background 10 dB | 266 | 0.718 | 5% | 2% |
| echo / reverb (RT60 0.6 s) | 266 | 0.868 | 3% | 2% |
| low volume (-30 dB) | 266 | 0.898 | 63% | 0% |
| other device (band-pass 300 Hz-4 kHz) | 266 | 0.763 | 4% | 3% |
| distant source (15 m) | 266 | 0.767 | 13% | 3% |
| partial event (first 40 % cut) | 266 | 0.752 | 9% | 0% |
| re-encoded MP3 64 kbps | 266 | 0.876 | 3% | 2% |
| overlap with another class (0 dB) | 266 | 0.744 | 4% | 2% |

For the overlap condition a prediction counts as correct if it is either of the two mixed classes.

## Confusable pairs from the SRS (clean test clips)

| Pair | A predicted as B | B predicted as A |
|---|---|---|
| gunshot vs fireworks / door slam (Background) | 1/30 | 1/30 |
| scream vs normal shouting / voices (Background) | 0/30 | 3/30 |
| aggression vs normal conversation (Background) | 0/30 | 0/30 |
| glass vs other impacts (Background) | 0/6 | 1/30 |
| alarm vs horn | 0/30 | 1/30 |
| faulty vs normal machinery (Background) | 0/20 | 1/30 |
| help phrase vs normal speech (Background) | 0/30 | 0/30 |

How the application handles hard cases: low confidence → *Uncertain*; Poor/Unusable quality, close top-2 scores, overlapping sounds, model disagreement or an unconfirmed critical class → manual review; re-encoded copies → near-duplicate notice (SHA-256 + fingerprint).