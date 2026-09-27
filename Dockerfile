# SonicSentinel AI – production image
# Works on Hugging Face Spaces (Docker, port 7860), Render, Railway or any Docker host.
FROM python:3.11-slim

# FFmpeg decodes MP3/M4A uploads; libsndfile is used by soundfile
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Hugging Face Spaces runs the container as user 1000
RUN useradd -m -u 1000 user
WORKDIR /app

# requirements-deploy.txt (written by scripts/deploy_hf_space.py) pins the exact library versions the model
# was trained with; requirements.txt is the fallback. TensorFlow (CPU) is needed for the YAMNet features.
COPY requirements*.txt ./
RUN REQ=requirements.txt; [ -f requirements-deploy.txt ] && REQ=requirements-deploy.txt; \
    pip install --no-cache-dir -r $REQ gunicorn "tensorflow-cpu>=2.15"

COPY --chown=user . .
USER user
ENV HOME=/home/user \
    PORT=7860 \
    PYTHONUNBUFFERED=1 \
    TF_CPP_MIN_LOG_LEVEL=2 \
    MPLCONFIGDIR=/tmp/matplotlib \
    NUMBA_CACHE_DIR=/tmp/numba

# download YAMNet once while building, so the first request is fast
# (docker build --build-arg SKIP_YAMNET=1 skips it, e.g. for a model trained with --no-yamnet)
ARG SKIP_YAMNET=0
RUN if [ "$SKIP_YAMNET" = "1" ]; then echo "YAMNet download skipped"; \
    else python -c "from feature_extraction.embeddings import _load; _load(); print('YAMNet ready')"; fi
EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
  CMD python -c "import os,urllib.request;urllib.request.urlopen('http://127.0.0.1:%s/healthz' % os.environ.get('PORT','7860'),timeout=8)" || exit 1

# create tables + default accounts, then serve (1 worker keeps one copy of the models in RAM)
CMD python -m database.init_db && \
    gunicorn -w 1 --threads 4 --timeout 300 -b 0.0.0.0:${PORT} run:app
