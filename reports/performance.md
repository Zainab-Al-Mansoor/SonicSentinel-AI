# Performance & scalability results

Machine: Linux-6.18.44-fc-v37-x86_64-with-glibc2.39 · Python 3.11.15 · 2 CPU threads

Python model: `py-xgb-20260924-0800` · test audio: recordings from the unseen TEST split, looped to 30 s / cut to 2 s

Measured on 2026-09-24 on a 2-CPU Linux test server (slower than the team laptop). Re-run on the demo PC with `python scripts\benchmark_performance.py`.

| Requirement | Measured | Target | Result |
|---|---|---|---|
| 30-s upload analysed (Python model, images, rules, DB) | mean 1.472 s · max 1.74 s | ≤ 8 s | ✅ pass |
| Live 2-s window → prediction (server side) | mean 0.109 s · max 0.122 s | ≤ 3 s | ✅ pass |
| Event records stored and queryable | 20,020 | ≥ 20,000 | ✅ pass |
| 5 concurrent users (pages + live window) | max request 0.8 s, errors 0 | no errors | ✅ pass |

## Page times with the full record set

| Page | HTTP | Seconds |
|---|---|---|
| dashboard | 200 | 0.066 |
| history (unfiltered) | 200 | 0.182 |
| history (filtered) | 200 | 0.018 |
| timeline | 200 | 0.041 |
| admin dashboard | 200 | 0.481 |
| stats API (30 days) | 200 | 0.284 |
| events.csv export | 200 | 1.344 |

GTM inference runs in the browser and is not included; the live page shows its latency per window.