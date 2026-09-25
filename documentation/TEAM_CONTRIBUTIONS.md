# Team contribution record

| Member | GitHub username | Role | Modules owned (can explain in the viva) | Main contributions | Commits (approx.) |
|---|---|---|---|---|---|
| Zainab | | Team lead / integration | `src/services/` (analysis, decision, rules), `run.py` | | |
| Zainab | | Dataset & training | `scripts/`, `augmentation/`, `python_models/` | | |
| Qirat | | GTM & front end | `gtm_model/`, `static/js/gtm.js`, `live.js`, `templates/` | | |
| Qirat | | Audio pipeline | `audio_preprocessing/`, `feature_extraction/` | | |
| Qirat | | Testing & documentation | `tests/`, `documentation/`, README, report, blog, video | | |

## Timeline

| Date | Milestone | Members |
|---|---|---|
| 2026-09-23 | Public datasets imported, first 9-class model (MLP, 93.1 % acc, macro-F1 0.819) | |
| 2026-09-24 | GTM model (9 classes) integrated; Help class (TTS + team); 10-class XGBoost model (86.5 % acc, macro-F1 0.863); new theme | |
| _2026-09-24_ | GTM retrained with 10 classes | |
| _2026-09-25_ | Benchmark, model-comparison report, screenshots | |
| _2026-09-26_ | Deployment, demo video, final submission | |

Details per day: [`DEVELOPMENT_LOG.md`](../DEVELOPMENT_LOG.md). AI assistance: [`AI_USAGE.md`](../AI_USAGE.md).

## Commit plan (SRS: meaningful commits from **every** member)

Each member commits **their own** work from **their own** GitHub account (never commit for someone else):

```powershell
git config user.name "Zainab-Al-Mansoor"
git config user.email "zainabalmansoor2@gmail.com"
git pull


git add .
git commit -m "Short description of what you changed and why"
git push

