# Similar-event evaluation (SRS Step 14)

Python model `py-xgb-20260924-0800` · 695 unseen TEST clips (measured on the test server; run the script on the PC to add the validation clips) · look-alike gap threshold 0.20

| Pair | Event clips | Event recognised | Look-alike clips (train) | Look-alike mistaken for the event | Look-alike recognised as itself |
|---|---|---|---|---|---|
| Gunshot vs fireworks | 110 | 93% | 1 (6) | 0% | 100% |
| Gunshot vs vehicle backfire | 110 | 93% | **no data yet** | – | – |
| Gunshot vs door slam / knock | 110 | 93% | 1 (6) | 0% | 100% |
| Panic scream vs normal shouting / voices | 60 | 93% | 42 (179) | 33% | 55% |
| Aggression vs normal conversation | 45 | 89% | 26 (96) | 4% | 88% |
| Glass breaking vs metal impact | 6 | 83% | 2 (5) | 100% | 0% |
| Alarm vs vehicle horn | 71 | 92% | 55 (255) | 2% | 87% |
| Vehicle horn vs alarm | 55 | 87% | 71 (331) | 0% | 92% |
| Machinery fault vs normal machinery | 20 | 100% | 37 (177) | 14% | 86% |
| Help request vs ordinary speech | 33 | 100% | 41 (173) | 0% | 56% |

In the application every decision for these classes also runs the **look-alike check**: if the gap between the event and its look-alike is below the threshold (Admin → Settings → `lookalike_margin`), the event is sent to manual review with the reason “Possible look-alike: could be …”; a Background decision with a strong critical-class score is flagged as a possible hidden event.

Pairs marked *no data yet* need licensed recordings of the look-alike (e.g. Freesound CC0 “car backfire”, “metal hit”) in `downloads/extra/Background Noise/` followed by import → build → augment → train.