# Deployment guide

SonicSentinel AI ships with a `Dockerfile` (Python 3.11 + FFmpeg + TensorFlow CPU + gunicorn), a one-command Hugging Face deploy script (`scripts/deploy_hf_space.py`) and a Render Blueprint (`render.yaml`).
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

## 2. Hugging Face Spaces (recommended – free, HTTPS, enough RAM for YAMNet)

The installed model uses YAMNet features, so the server needs TensorFlow (≈ 1–1.5 GB RAM in total).
A free Hugging Face **Docker Space** has enough memory; the free Render plan does not.

1. Create a free account at https://huggingface.co and a **Write** token
   (Settings → Access Tokens → *Create new token* → type *Write*).
2. On the training PC, in the project folder:
   ```
   pip install huggingface_hub
   python scripts/deploy_hf_space.py --space YOUR-HF-USERNAME/sonicsentinel-ai
   ```
   Paste the token when asked. The script pins the exact library versions of this PC
   (`requirements-deploy.txt`), creates the Space, stores a random `SONIC_SECRET_KEY` secret and uploads the project
   (datasets, database and uploads are skipped).
3. Open `https://huggingface.co/spaces/YOUR-HF-USERNAME/sonicsentinel-ai` → tab **Logs**. The first build takes
   10–20 minutes (TensorFlow + YAMNet download). When the status is **Running**, open the app at
   `https://YOUR-HF-USERNAME-sonicsentinel-ai.hf.space`.
4. Log in with the admin account from the README and **change the password** (Profile), then
   Admin → Settings: check *Python weight* 0.75 and *Agreement fusion* ✔.
5. To update the live app later, run the same command again.

Notes
* Use the direct `…hf.space` link (not the huggingface.co page): inside the embedded page the browser blocks the
  login cookie and the microphone.
* The Space disk is temporary: the SQLite database and uploads reset when the Space restarts or is rebuilt.
* A free Space sleeps after a period without visitors; the first visit then takes about a minute.

## 3. Render (free permanent link – light model)

Hugging Face now requires a paid PRO plan for Docker Spaces, and the free Render plan has only 512 MB RAM – too little
for TensorFlow/YAMNet. The free online version therefore uses a **light model** trained without YAMNet
(feature set v2 without YAMNet, 446 features). The full YAMNet model keeps running locally.
Measured memory of the app with a YAMNet-free model: ≈ 390 MB.

1. Train the light model into its own file and report folder (the full model and its reports stay untouched):
   ```
   set SONIC_MODEL_PATH=python_models\saved\sonic_model_light.joblib
   set SONIC_REPORTS_DIR=reports\light
   python -m python_models.train_models --fast --no-yamnet --models rf,mlp,xgb
   ```
2. Pin the library versions of the training PC (writes `requirements-deploy.txt`, also done by the Hugging Face script):
   run `python scripts/deploy_hf_space.py --space x/y` once and stop it with Ctrl+C after "requirements-deploy.txt written",
   or keep the existing file.
3. Commit and push: `git add python_models/saved/sonic_model_light.joblib requirements-deploy.txt Dockerfile.render render.yaml reports/light && git commit -m "Light model for Render" && git push`
4. https://render.com → sign in with GitHub → **New + → Blueprint** → choose the repository → **Apply**.
   Render reads `render.yaml`, builds `Dockerfile.render` and generates `SONIC_SECRET_KEY`.
5. After 5–10 minutes the app runs at `https://sonicsentinel-ai.onrender.com` (or similar). Log in and change the
   admin password; Admin → Settings: Python weight 0.75, Agreement fusion ✔.

Notes
* HTTPS is included, so the live microphone works on phones and laptops.
* The free plan sleeps after ~15 minutes without traffic; the first request then takes ~1 minute.
* The disk is temporary: the SQLite database and uploads reset on every redeploy or restart.
* Every `git push` redeploys automatically.

## 4. Railway

1. https://railway.app → **New Project → Deploy from GitHub repo**. Railway detects the `Dockerfile`.
2. **Variables** → add `SONIC_SECRET_KEY` (any long random string). `PORT` is set by Railway.
3. **Settings → Networking → Generate Domain**.

## 5. Any server with Docker

```bash
docker build -t sonicsentinel .
docker run -d -p 8080:10000 -e SONIC_SECRET_KEY=change-me -v sonic_data:/app/data sonicsentinel
# open http://SERVER:8080 (put it behind HTTPS for the microphone)
```

## 6. Windows server without Docker

```powershell
pip install -r requirements.txt
python -m database.init_db
waitress-serve --host 0.0.0.0 --port 5000 run:app
```

## 7. What to submit (SRS deliverable 12)

| Item | Value |
|---|---|
| Public application URL | _fill in after deployment_ |
| Evaluator credentials | `evaluator / Eval@12345` (Normal user) |
| Administrator credentials | `admin / Admin@12345` – change it and give evaluators the new one |
| Sample audio | `sample_audio/test_cases/` (+ `sample_audio/classes/<Class>/` if added) |
| Testing instructions | README §16 "How to use" and `documentation/TESTING.md` |
