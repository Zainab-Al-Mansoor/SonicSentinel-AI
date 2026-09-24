# Final submission checklist (SRS §1.10-16)

Legend: ✅ done in the repository · 🟡 prepared, the team must run/fill one step · 👥 only the team can do it

| # | SRS item | Status | Where / what is left |
|---|---|---|---|
| 1 | Project report | ✅ | `documentation/PROJECT_REPORT.md` (every report topic listed in the SRS; diagrams in Mermaid). Export to PDF: open in VS Code → *Markdown PDF* extension, or paste into Word. Add team names on the title page. |
| 2 | Public GitHub URL | 👥 | GitHub → Settings → *Change visibility* → Public. Put the URL in README → *Deliverables*. |
| 3 | Complete source code | ✅ | all folders required by SRS §1.10-2, incl. `screenshots/` |
| 4 | Training dataset | ✅ / 👥 | `audio_dataset/processed/train/` (built locally). Too big for GitHub: upload a zip to Google Drive and link it, or submit on USB as the evaluators prefer. |
| 5 | Validation dataset | ✅ / 👥 | `audio_dataset/processed/validation/` – same as above |
| 6 | Testing dataset | ✅ / 👥 | `audio_dataset/processed/test/` – same as above |
| 7 | Dataset metadata | ✅ | `data/dataset_metadata.csv`, `data/dataset_statistics.json`, `data/dataset_rejected.csv`, `documentation/DATA_DICTIONARY.md` |
| 8 | Python model | 🟡 | `python_models/saved/sonic_model.joblib` – commit it after training finishes (`git add -f` if ignored) |
| 9 | GTM model | 🟡 | `gtm_model/model/` (9 classes). 👥 Retrain with 10 classes – `documentation/GTM_TRAINING_LOG.md` §4 |
| 10 | Preprocessing scripts | ✅ | `audio_preprocessing/` |
| 11 | Feature-extraction scripts | ✅ | `feature_extraction/` |
| 12 | Alert rules | ✅ | `alert_rules/alert_rules.json` (+ Admin → Alert rules) |
| 13 | Model-comparison report | 👥 | Admin → *Model comparison* → *Run test-set evaluation* (keep the tab open) → *Export Excel* → save as `reports/model_comparison_report.xlsx`. Needs ≥ 10 test clips per class. |
| 14 | Installation instructions | ✅ | README §13, `RUN_AND_TRAIN_GUIDE.md`, `documentation/DEPLOYMENT.md` |
| 15 | Execution instructions | ✅ | README §15–16 |
| 16 | Deployment URL | 👥 | Render Blueprint (`render.yaml`) – `documentation/DEPLOYMENT.md` §2. If deployment is not possible, the local instructions satisfy the SRS. |
| 17 | Demonstration video | 👥 | Script and shot list: `documentation/DEMO_VIDEO_SCRIPT.md` |
| 18 | Technical blog | ✅ | `documentation/TECHNICAL_BLOG.md` (≈ 3,400 words). Optional: publish on Medium / Hashnode / dev.to and link it. |
| 19 | AI_USAGE.md | 🟡 | rows filled; 👥 each member fills *Student modifications* and *Verified by* |
| 20 | Team contribution record | 🟡 | `documentation/TEAM_CONTRIBUTIONS.md` – 👥 fill names and tasks; every member must have commits |

## Evidence still to produce (team)

- [ ] `python scripts/benchmark_performance.py` → commit `reports/performance.md` and `reports/performance.json` (NFR performance / scalability)
- [ ] Screenshots listed in `screenshots/README.md`
- [ ] Re-record *Person Asking for Help* with a working microphone (`python scripts/record_samples.py --list-devices`, then `--input N`)
- [ ] Optional: more Glass Breaking (≥ 70 clips so the test split has ≥ 10) and more MIMII abnormal clips, then rebuild → augment → train
- [ ] Model comparison Excel (item 13)
- [ ] GTM run 2 with 10 classes + screenshots (item 9)
- [ ] Deployment (item 16)
- [ ] Demo video (item 17)
- [ ] Links in README → *Deliverables*: GitHub, deployment, video, blog, dataset zip
- [ ] Everyone commits their own part (report sections, screenshots, GTM log, tests) from their own GitHub account

## Final technical check before submitting

```powershell
python -m pytest -q                         # all tests pass
python -m database.init_db                  # fresh DB with default accounts
python run.py                               # upload one clip per class, run the live monitor once
git status                                  # nothing important left uncommitted
```

Evaluator accounts (README §11): `evaluator / Eval@12345` (Normal user), `admin / Admin@12345` (change after deployment and share the new one).
