# SonicSentinel AI – production image (Render / Railway / any Docker host)
FROM python:3.11-slim

# FFmpeg decodes MP3/M4A uploads; libsndfile is used by soundfile
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt gunicorn

COPY . .

ENV PORT=10000 \
    PYTHONUNBUFFERED=1 \
    MPLCONFIGDIR=/tmp/matplotlib \
    NUMBA_CACHE_DIR=/tmp/numba
EXPOSE 10000

# container health: /healthz checks the database and the models
HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
  CMD python -c "import os,urllib.request;urllib.request.urlopen('http://127.0.0.1:%s/healthz' % os.environ.get('PORT','10000'),timeout=8)" || exit 1

# create tables + default accounts, then serve (1 worker keeps one copy of the model in RAM)
CMD python -m database.init_db && \
    gunicorn -w 1 --threads 4 --timeout 180 -b 0.0.0.0:${PORT} run:app
