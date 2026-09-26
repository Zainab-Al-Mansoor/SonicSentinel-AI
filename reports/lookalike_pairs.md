# Similar-event evaluation (SRS Step 14)

Python model `py-xgb-20260926-0308` · validation + test clips only (never used for training) · look-alike gap threshold 0.20

| Pair | Event clips | Event recognised | Look-alike clips (train) | Look-alike mistaken for the event | Look-alike recognised as itself |
|---|---|---|---|---|---|
| Gunshot vs fireworks | 320 | 99% | 10 (29) | 0% | 100% |
| Gunshot vs vehicle backfire | 320 | 99% | **no data yet** | – | – |
| Gunshot vs door slam / knock | 320 | 99% | 16 (23) | 0% | 100% |
| Panic scream vs normal shouting / voices | 237 | 94% | 98 (236) | 21% | 66% |
| Aggression vs normal conversation | 86 | 86% | 46 (104) | 2% | 96% |
| Glass breaking vs metal impact | 25 | 96% | 16 (33) | 0% | 88% |
| Alarm vs vehicle horn | 257 | 99% | 112 (257) | 0% | 98% |
| Vehicle horn vs alarm | 115 | 98% | 252 (593) | 0% | 99% |
| Machinery fault vs normal machinery | 161 | 98% | 67 (181) | 0% | 100% |
| Help request vs ordinary speech | 66 | 95% | 91 (213) | 1% | 67% |

In the application every decision for these classes also runs the **look-alike check**: if the gap between the event and its look-alike is below the threshold (Admin → Settings → `lookalike_margin`), the event is sent to manual review with the reason “Possible look-alike: could be …”; a Background decision with a strong critical-class score is flagged as a possible hidden event.

Pairs marked *no data yet* need licensed recordings of the look-alike (e.g. Freesound CC0 “car backfire”, “metal hit”) in `downloads/extra/Background Noise/` followed by import → build → augment → train.