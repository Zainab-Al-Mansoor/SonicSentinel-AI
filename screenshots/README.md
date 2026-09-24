# Screenshots

Take these on the real app (Chrome, 1440-px wide window, 100 % zoom) with the trained models, save them as PNG with
the names below, and commit them. README → *Screenshots* shows the first six.

| File | Page / what must be visible |
|---|---|
| `01_login.png` | Login page |
| `02_dashboard.png` | User dashboard with charts and recent events |
| `03_upload.png` | Upload page with a file selected (preview player) |
| `04_event_detail.png` | Event page: metadata, waveform, spectrogram, quality |
| `05_model_comparison.png` | Event page: Python vs GTM predictions, confidence bars, difference, consistency status |
| `06_live_monitor.png` | Live page while running: privacy banner, window predictions, latency |
| `07_critical_alert.png` | Critical alert toast + Alerts page with rule trace |
| `08_review.png` | Review page: reviewer override with comment |
| `09_history.png` | History with filters applied |
| `10_timeline.png` | Timeline |
| `11_admin_dashboard.png` | Admin dashboard |
| `12_admin_models.png` | Admin → Models: comparison table and confusion matrix |
| `13_admin_rules.png` | Admin → Alert rules editor |
| `14_admin_settings.png` | Admin → Settings (thresholds) |
| `15_report.png` | Downloaded event report |
| `16_validation_errors.png` | Rejected silent / corrupted / unsupported file messages |
| `17_uncertain_disagreement.png` | Low-confidence or model-disagreement event |
| `18_overlap.png` | Overlapping-sound event |
| `19_duplicate.png` | Duplicate-audio warning |
| `20_pytest.png` | Terminal: `python -m pytest -q` all passed |
| `21_training.png` | Terminal: end of `train_models` with the test metrics |
| `22_benchmark.png` | Terminal: end of `benchmark_performance.py` |
| `gtm/…` | GTM evidence: see `documentation/GTM_TRAINING_LOG.md` §7 |

Tip: Windows `Win + Shift + S` for a region, or Chrome DevTools → `Ctrl + Shift + P` → *Capture full size screenshot* for long pages.
