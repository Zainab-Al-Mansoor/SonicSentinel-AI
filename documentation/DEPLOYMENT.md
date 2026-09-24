# Deployment guide

SonicSentinel AI ships with a `Dockerfile` (Python 3.11 + FFmpeg + gunicorn) and a Render Blueprint (`render.yaml`).
The same image runs on Render, Railway, Fly.io or any Docker host. If deployment is not possible, the
local instructions in README §13–15 are the fallback the SRS allows.

## 1. Before you deploy

1. **Commit the trained models.** They must be in the repository:
   `python_models/saved/sonic_model.joblib` and `gtm_model/model/{model.json, metadata.json, weights.bin}`.
2. **Pin the library versions you trained with.** A joblib model should be loaded with the same
   scikit-learn / XGBoost / NumPy versions it was saved with. On the training PC run:
   ```
   python -c "import sklearn, xgboost, numpy, librosa, scipy; print('scikit-learn==' + sklearn.__version__); print('xgboost==' + xgboost.__version__); print('numpy==' + numpy.__version__); print('librosa==' + librosa.__version__); print('scipy==' + scipy.__version__)"
   ```
   and replace the matching `>=` lines in `requirements.txt` with those exact `==` versions.
3. Push to GitHub.

## 2. Render (free, HTTPS included)

1. Sign in at https://render.com with GitHub.
2. **New + → Blueprint** → choose the repository → **Apply**. Render reads `render.yaml`, builds the Docker image
   and generates a random `SONIC_SECRET_KEY`.
3. First build takes 5–10 minutes. The app is then available at `https://sonicsentinel-ai.onrender.com` (or similar).
4. Log in with `admin / Admin@12345` and **change the password** (Profile).

Notes
* HTTPS is required for the live microphone in the browser – Render provides it automatically.
* The free plan sleeps after ~15 minutes without traffic; the first request after that takes ~30–60 s.
* The free plan disk is temporary: the SQLite database and uploads reset on every redeploy.
  For persistent data add a Render **Disk** (paid) mounted at `/app/data`, or point `SQLALCHEMY_DATABASE_URI`
  in `src/__init__.py` to a hosted PostgreSQL database (and add `psycopg2-binary` to `requirements.txt`).
* Free instances have 512 MB RAM. If the build or start fails with "out of memory", use the Starter plan
  or Railway (below).

## 3. Railway

1. https://railway.app → **New Project → Deploy from GitHub repo**. Railway detects the `Dockerfile`.
2. **Variables** → add `SONIC_SECRET_KEY` (any long random string). `PORT` is set by Railway.
3. **Settings → Networking → Generate Domain**.

## 4. Any server with Docker

```bash
docker build -t sonicsentinel .
docker run -d -p 8080:10000 -e SONIC_SECRET_KEY=change-me -v sonic_data:/app/data sonicsentinel
# open http://SERVER:8080 (put it behind HTTPS for the microphone)
```

## 5. Windows server without Docker

```powershell
pip install -r requirements.txt
python -m database.init_db
waitress-serve --host 0.0.0.0 --port 5000 run:app
```

## 6. What to submit (SRS deliverable 12)

| Item | Value |
|---|---|
| Public application URL | _fill in after deployment_ |
| Evaluator credentials | `evaluator / Eval@12345` (Normal user) |
| Administrator credentials | `admin / Admin@12345` – change it and give evaluators the new one |
| Sample audio | `sample_audio/test_cases/` (+ `sample_audio/classes/<Class>/` if added) |
| Testing instructions | README §16 "How to use" and `documentation/TESTING.md` |
