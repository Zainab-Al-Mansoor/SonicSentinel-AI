# AI tool usage declaration

The SRS requires every AI tool used during development to be declared here. **Every team member must review, understand, modify where needed, and test all AI-assisted code.** Evaluators may ask any member to explain any function.

The final sound classification is produced **only** by the team's Python model and the team's Google Teachable Machine model. No generative-AI API is called by the application.

| # | Tool | Purpose | Assistance requested | Files affected | Student modifications | Testing completed | Verified by |
|---|---|---|---|---|---|---|---|
| 1 | Claude (Anthropic) | Initial project scaffold | Generate the first version of the Flask app, audio pipeline, training scripts, tests and docs from the SRS | whole repository (initial commit) | _fill in: what you changed, refactored or rewrote_ | `python -m pytest` (55 tests), manual browser test | _names_ |
| 2 | | | | | | | |

## Notes for the team

* Keep adding a row whenever you use an AI tool (ChatGPT, Copilot, Claude, …).
* In "Student modifications", be specific, e.g. "tuned SVM grid, rewrote quality thresholds after testing on our recordings".
* Before the demo, every member should be able to explain their assigned modules. Suggested split:
  * `audio_preprocessing/` and `feature_extraction/`
  * `python_models/`, `augmentation/` and `scripts/build_dataset.py`
  * `gtm_model/` and `static/js/gtm.js`
  * `src/services/decision.py`, `rules.py` and `analysis.py`
  * `src/routes/`, `templates/` and `tests/`
