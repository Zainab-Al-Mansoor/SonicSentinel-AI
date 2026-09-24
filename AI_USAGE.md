# AI tool usage declaration

The SRS requires every AI tool used during development to be declared here. **Every team member must review, understand, modify where needed, and test all AI-assisted code.** Evaluators may ask any member to explain any function.

The final sound classification is produced **only** by the team's Python model and the team's Google Teachable Machine model. No generative-AI API is called by the application.

| # | Tool | Purpose | Assistance requested | Files affected | Student modifications | Testing completed | Verified by |
|---|---|---|---|---|---|---|---|
| 1 | Claude (Anthropic) | Initial project scaffold | Generate the first version of the Flask app, audio pipeline, training scripts, tests and docs from the SRS | whole repository (initial commit) | _fill in: what you changed, refactored or rewrote_ | `python -m pytest` (55 tests), manual browser test | _names_ |
| 2 | Claude (Anthropic) | Importing our downloaded datasets | Script that imports the local `downloads/` folders with label mapping, per-class and per-source caps and round-robin selection | `scripts/import_local_downloads.py`, `config/dataset_mapping.json` | _fill in_ | import run on our PC; clip counts checked in `data/dataset_statistics.json` | _names_ |
| 3 | Claude (Anthropic) | Training fixes | Skip XGBoost when a class has no data; feature-cache key with size + mtime; de-duplicate annotations; augmentation balancing (`--balance-to`); dataset clean-up script | `python_models/train_models.py`, `scripts/build_dataset.py`, `augmentation/augment.py`, `scripts/clean_dataset.py` | _fill in_ | full rebuild → augment → train on our PC (38,210 segments, XGB 86.5 % test accuracy) | _names_ |
| 4 | Claude (Anthropic) | Explaining GTM and debugging | Step-by-step Teachable Machine guidance, why GTM predictions were wrong, how to retrain | `gtm_model/README.md`, `documentation/GTM_TRAINING_LOG.md` | _fill in_ | GTM model exported and tested in the app | _names_ |
| 5 | Claude (Anthropic) | UI theme | Mocha colour theme (#3E2522, #8C6E63, #D3A376, #FFE0B2, #FFF2DF) and glass effect; plot colours | `static/css/app.css`, `templates/base.html`, `static/img/logo.svg`, several templates, `feature_extraction/visuals.py` | _fill in_ | every page checked on desktop and mobile width | _names_ |
| 6 | Claude (Anthropic) | Recording helper | Microphone device list / selection and level check in the recording script | `scripts/record_samples.py` | _fill in_ | _fill in: re-recorded Help clips_ | _names_ |
| 7 | Claude (Anthropic) | Performance evidence and deployment | Benchmark script for the NFR targets; Dockerfile, Render blueprint and deployment guide | `scripts/benchmark_performance.py`, `Dockerfile`, `.dockerignore`, `render.yaml`, `documentation/DEPLOYMENT.md` | _fill in_ | _fill in: benchmark run on our PC, deployment test_ | _names_ |
| 8 | Claude (Anthropic) | Documentation drafts | Drafts of README, project report, technical blog, demo-video script, SRS compliance checklist, submission checklist, development log | `README.md`, `documentation/*.md`, `DEVELOPMENT_LOG.md`, `screenshots/README.md` | _fill in: sections rewritten in our own words, facts checked_ | numbers checked against `reports/` | _names_ |
| 9 | Claude (Anthropic) | GTM upload archives, availability, benchmark evidence | Teachable Machine sample-archive export (`--zip`), `/healthz` health check, auto-restart server script, tests | `gtm_model/prepare_gtm_samples.py`, `gtm_model/README.md`, `src/__init__.py`, `run_server.bat`, `tests/test_gtm_samples.py`, `tests/test_app.py`, `reports/performance.md` | _fill in_ | 58 pytest tests; archives accepted by the Teachable Machine web app; benchmark with the real model | _names_ |
| 10 | Claude (Anthropic) | Standout features | "Why this prediction?" explainable-AI panel (TreeSHAP / occlusion) and Robustness Lab with live sliders and stress test | `src/services/explain.py`, `src/services/lab.py`, `src/routes/api.py`, `src/routes/main.py`, `templates/events/detail.html`, `templates/lab.html`, `templates/base.html`, `static/js/lab.js`, `static/css/app.css`, `tests/test_explain_lab.py` | _fill in_ | 66 pytest tests; checked in the browser with the real Python and GTM models | _names_ |
| 11 | | | | | | | |

## Notes for the team

* Keep adding a row whenever you use an AI tool (ChatGPT, Copilot, Claude, …).
* In "Student modifications", be specific, e.g. "tuned SVM grid, rewrote quality thresholds after testing on our recordings".
* The report and blog are drafts: read them, correct anything that does not match what you did, and add your own experience before submitting.
* Before the demo, every member should be able to explain their assigned modules. Suggested split:
  * `audio_preprocessing/` and `feature_extraction/`
  * `python_models/`, `augmentation/` and `scripts/build_dataset.py`
  * `gtm_model/` and `static/js/gtm.js`
  * `src/services/decision.py`, `rules.py` and `analysis.py`
  * `src/routes/`, `templates/` and `tests/`
