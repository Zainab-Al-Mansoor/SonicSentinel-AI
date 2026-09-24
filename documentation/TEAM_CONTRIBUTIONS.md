# Team contribution record

SRS §1.10-16 requires a team contribution record, and §1.10-11 requires commits from **all** members.
Each member fills their own row and commits this file from their own GitHub account.

| Member | GitHub username | Role | Modules owned (can explain in the viva) | Main contributions | Commits (approx.) |
|---|---|---|---|---|---|
| _name_ | | Team lead / integration | `src/services/` (analysis, decision, rules), `run.py` | | |
| _name_ | | Dataset & training | `scripts/`, `augmentation/`, `python_models/` | | |
| _name_ | | GTM & front end | `gtm_model/`, `static/js/gtm.js`, `live.js`, `templates/` | | |
| _name_ | | Audio pipeline | `audio_preprocessing/`, `feature_extraction/` | | |
| _name_ | | Testing & documentation | `tests/`, `documentation/`, README, report, blog, video | | |

## Timeline

| Date | Milestone | Members |
|---|---|---|
| 2026-09-23 | Public datasets imported, first 9-class model (MLP, 93.1 % acc, macro-F1 0.819) | |
| 2026-09-24 | GTM model (9 classes) integrated; Help class (TTS + team); 10-class XGBoost model (86.5 % acc, macro-F1 0.863); new theme | |
| _date_ | GTM retrained with 10 classes | |
| _date_ | Benchmark, model-comparison report, screenshots | |
| _date_ | Deployment, demo video, final submission | |

Details per day: [`DEVELOPMENT_LOG.md`](../DEVELOPMENT_LOG.md). AI assistance: [`AI_USAGE.md`](../AI_USAGE.md).
