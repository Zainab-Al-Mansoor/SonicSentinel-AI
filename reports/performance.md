# Performance & scalability results

Machine: Windows-10-10.0.19045-SP0 · Python 3.14.6 · 4 CPU threads

Python model: `py-xgb-20260926-0308` · test audio: recordings from the unseen TEST split, looped to 30 s / cut to 2 s

| Requirement | Measured | Target | Result |
|---|---|---|---|
| 30-s upload analysed (Python model, images, rules, DB) | mean 7.753 s · max 9.247 s | ≤ 8 s | ❌ fail |
| Live 2-s window → prediction (server side) | mean 0.26 s · max 0.328 s | ≤ 3 s | ✅ pass |
| Event records stored and queryable | 20,020 | ≥ 20,000 | ✅ pass |
| 5 concurrent users (pages + live window) | max request 2.52 s, errors 0 | no errors | ✅ pass |

## Page times with the full record set

| Page | HTTP | Seconds |
|---|---|---|
| dashboard | 200 | 0.295 |
| history (unfiltered) | 200 | 0.088 |
| history (filtered) | 200 | 0.061 |
| timeline | 200 | 0.12 |
| admin dashboard | 200 | 0.89 |
| stats API (30 days) | 200 | 1.179 |
| events.csv export | 200 | 2.867 |

GTM inference runs in the browser and is not included; the live page shows its latency per window.