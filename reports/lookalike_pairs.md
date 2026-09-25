# Similar-event evaluation (SRS Step 14)

Python model `py-xgb-20260925-0316` · validation + test clips only (never used for training) · look-alike gap threshold 0.20

| Pair | Event clips | Event recognised | Look-alike clips (train) | Look-alike mistaken for the event | Look-alike recognised as itself |
|---|---|---|---|---|---|
| Gunshot vs fireworks | 320 | 95% | 13 (27) | 8% | 77% |
| Gunshot vs vehicle backfire | 320 | 95% | **no data yet** | – | – |
| Gunshot vs door slam / knock | 320 | 95% | 9 (31) | 0% | 100% |
| Panic scream vs normal shouting / voices | 240 | 92% | 120 (274) | 25% | 58% |
| Aggression vs normal conversation | 89 | 83% | 44 (106) | 0% | 98% |
| Glass breaking vs metal impact | 35 | 63% | 16 (24) | 56% | 44% |
| Alarm vs vehicle horn | 260 | 93% | 117 (273) | 1% | 94% |
| Vehicle horn vs alarm | 117 | 94% | 260 (607) | 0% | 93% |
| Machinery fault vs normal machinery | 162 | 98% | 72 (178) | 4% | 96% |
| Help request vs ordinary speech | 66 | 100% | 105 (249) | 1% | 62% |

In the application every decision for these classes also runs the **look-alike check**: if the gap between the event and its look-alike is below the threshold (Admin → Settings → `lookalike_margin`), the event is sent to manual review with the reason “Possible look-alike: could be …”; a Background decision with a strong critical-class score is flagged as a possible hidden event.

Pairs marked *no data yet* need licensed recordings of the look-alike (e.g. Freesound CC0 “car backfire”, “metal hit”) in `downloads/extra/Background Noise/` followed by import → build → augment → train.