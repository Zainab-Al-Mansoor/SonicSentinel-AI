# Hidden-test readiness – robustness results

Python model `py-xgb-20260925-0316` · 287 unseen TEST clips (30 per class max) · same pipeline as the web app · crashes: **0**

| Condition | Clips | Accuracy | Rated Poor/Unusable (→ manual review) | Overlap flagged |
|---|---|---|---|---|
| clean | 287 | 0.909 | 5% | 2% |
| white noise 20 dB | 287 | 0.749 | 5% | 2% |
| white noise 10 dB | 287 | 0.505 | 6% | 3% |
| real background 10 dB | 287 | 0.714 | 6% | 4% |
| echo / reverb (RT60 0.6 s) | 287 | 0.843 | 6% | 3% |
| low volume (-30 dB) | 287 | 0.909 | 65% | 2% |
| other device (band-pass 300 Hz-4 kHz) | 287 | 0.833 | 6% | 3% |
| distant source (15 m) | 287 | 0.826 | 14% | 3% |
| partial event (first 40 % cut) | 287 | 0.763 | 10% | 3% |
| re-encoded MP3 64 kbps | 287 | 0.885 | 4% | 3% |
| overlap with another class (0 dB) | 287 | 0.669 | 7% | 4% |

For the overlap condition a prediction counts as correct if it is either of the two mixed classes.

## Confusable pairs from the SRS (clean test clips)

| Pair | A predicted as B | B predicted as A |
|---|---|---|
| gunshot vs fireworks / door slam (Background) | 0/30 | 0/30 |
| scream vs normal shouting / voices (Background) | 0/30 | 1/30 |
| aggression vs normal conversation (Background) | 2/30 | 0/30 |
| glass vs other impacts (Background) | 3/17 | 2/30 |
| alarm vs horn | 0/30 | 0/30 |
| faulty vs normal machinery (Background) | 1/30 | 0/30 |
| help phrase vs normal speech (Background) | 0/30 | 0/30 |

How the application handles hard cases: low confidence → *Uncertain*; Poor/Unusable quality, close top-2 scores, overlapping sounds, model disagreement or an unconfirmed critical class → manual review; re-encoded copies → near-duplicate notice (SHA-256 + fingerprint).