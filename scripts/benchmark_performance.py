"""
Performance + scalability benchmark for the SRS non-functional requirements.

    python scripts/benchmark_performance.py            # full run (≈ 3–8 minutes)
    python scripts/benchmark_performance.py --records 2000 --uploads 2   # quick check

NFR 1 – Performance : a 30-s uploaded clip is analysed in ≤ 8 s,
                      a live window returns a prediction in ≤ 3 s.
NFR 2 – Scalability : ≥ 20,000 event records, several concurrent users.

The benchmark runs against a TEMPORARY copy of the database and data folder
(your real data/sonicsentinel.db is never touched) but uses the REAL trained
Python model in python_models/saved/. GTM runs in the browser, so its time is
not included here (it is measured on the live page as "Last latency").

Results: reports/performance.json and reports/performance.md
"""
import argparse
import io
import json
import os
import random
import shutil
import statistics
import sys
import tempfile
import threading
import time
from datetime import timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
TMP = Path(tempfile.mkdtemp(prefix="sonic_bench_"))
os.environ["SONIC_DATA_DIR"] = str(TMP / "data")
os.environ["SONIC_DB_PATH"] = str(TMP / "bench.db")

import numpy as np
import soundfile as sf

SR = 44100


def test_clip(seconds: float, seed: int) -> np.ndarray:
    """A real test-split recording looped to the wanted length (falls back to a synthetic event)."""
    rng = np.random.default_rng(seed)
    meta = ROOT / "data" / "dataset_metadata.csv"
    if meta.exists():
        import pandas as pd
        m = pd.read_csv(meta)
        m = m[(m["split"] == "test") & (m["is_augmented"] == 0)]
        if len(m):
            row = m.sample(1, random_state=seed).iloc[0]
            p = ROOT / row["path"]
            if p.exists():
                y, sr = sf.read(str(p), dtype="float32", always_2d=True)
                y = y.mean(axis=1)
                if sr != SR:
                    import librosa
                    y = librosa.resample(y, orig_sr=sr, target_sr=SR)
                reps = int(np.ceil(seconds * SR / max(len(y), 1)))
                return np.tile(y, reps)[: int(seconds * SR)]
    t = np.arange(int(seconds * SR)) / SR
    return (0.05 * rng.standard_normal(len(t)) + 0.4 * np.sin(2 * np.pi * 900 * t) * (np.sin(2 * np.pi * 0.5 * t) > 0)).astype(np.float32)


def wav_bytes(y):
    buf = io.BytesIO()
    sf.write(buf, y, SR, format="WAV", subtype="PCM_16")
    buf.seek(0)
    return buf


def login(client, user="admin", pw="Admin@12345"):
    client.get("/login")
    with client.session_transaction() as s:
        token = s["_csrf"]
    client.post("/login", data={"username": user, "password": pw, "csrf_token": token})
    return token


def summary(xs):
    return {"n": len(xs), "mean_s": round(statistics.mean(xs), 3), "median_s": round(statistics.median(xs), 3),
            "max_s": round(max(xs), 3), "min_s": round(min(xs), 3)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--uploads", type=int, default=5, help="number of 30-s uploads to time")
    ap.add_argument("--windows", type=int, default=15, help="number of live windows to time")
    ap.add_argument("--records", type=int, default=20000, help="event records for the scalability test")
    ap.add_argument("--users", type=int, default=5, help="concurrent users for the concurrency test")
    args = ap.parse_args()

    from src import create_app
    app = create_app({"TESTING": True, "WARM_UP": False})
    from src.services.analysis import warm_up
    with app.app_context():
        t0 = time.perf_counter(); warm_up(); warm = time.perf_counter() - t0
    client = app.test_client()
    token = login(client)
    results = {"model_warm_up_s": round(warm, 2)}
    with app.app_context():
        from src.services.analysis import python_model
        try:
            results["python_model"] = python_model().version
        except Exception as e:
            results["python_model"] = f"unavailable ({e})"

    # ---- NFR 1a: 30-second upload --------------------------------------------------
    times = []
    for i in range(args.uploads):
        y = test_clip(30.0, seed=100 + i)
        t0 = time.perf_counter()
        r = client.post("/api/upload", data={"audio": (wav_bytes(y), f"bench_{i}.wav"), "csrf_token": token},
                        headers={"X-CSRFToken": token}, content_type="multipart/form-data")
        times.append(time.perf_counter() - t0)
        if r.status_code != 200:
            print("  upload failed:", r.get_json())
        print(f"  30-s upload {i+1}/{args.uploads}: {times[-1]:.2f} s")
    results["upload_30s"] = {**summary(times), "target_s": 8.0, "pass": max(times) <= 8.0}

    # ---- NFR 1b: live windows -------------------------------------------------------
    sess = client.post("/api/live/start", headers={"X-CSRFToken": token}).get_json()["session"]
    lat = []
    for i in range(args.windows):
        y = test_clip(2.0, seed=500 + i)
        t0 = time.perf_counter()
        client.post("/api/live/window", data={"session": sess, "audio": (wav_bytes(y), "w.wav")},
                    headers={"X-CSRFToken": token}, content_type="multipart/form-data")
        lat.append(time.perf_counter() - t0)
    client.post("/api/live/stop", json={"session": sess}, headers={"X-CSRFToken": token})
    print(f"  live window mean {statistics.mean(lat):.2f} s, max {max(lat):.2f} s")
    results["live_window_2s"] = {**summary(lat), "target_s": 3.0, "pass": max(lat) <= 3.0}

    # ---- NFR 2: 20,000 records -----------------------------------------------------
    from config.settings import CLASSES, SEVERITY_LEVELS, QUALITY_LEVELS
    from src.extensions import db
    from src.models import AudioEvent, User
    with app.app_context():
        admin = User.query.filter_by(username="admin").first()
        rng = random.Random(1)
        from src.models import now as _now
        base = _now()
        t0 = time.perf_counter()
        batch = []
        for i in range(args.records):
            c = rng.choice(CLASSES)
            batch.append(dict(audio_id=f"BEN-{i:06d}", user_id=admin.id, source="upload",
                              original_filename=f"bench_{i}.wav", format="wav", duration=rng.uniform(1, 30),
                              sample_rate=44100, channels=1, uploaded_at=base - timedelta(minutes=rng.randint(0, 60 * 24 * 60)),
                              quality_label=rng.choice(QUALITY_LEVELS[:3]), python_prediction=c,
                              python_confidence=rng.uniform(.3, 1), final_category=c,
                              final_confidence=rng.uniform(.3, 1), severity=rng.choice(SEVERITY_LEVELS),
                              status="Classified", consistency_status="Uncertain Result"))
            if len(batch) == 2000:
                db.session.bulk_insert_mappings(AudioEvent, batch); db.session.commit(); batch = []
        if batch:
            db.session.bulk_insert_mappings(AudioEvent, batch); db.session.commit()
        insert_s = time.perf_counter() - t0
        total = AudioEvent.query.count()
    print(f"  inserted {args.records} records in {insert_s:.1f} s (total {total})")

    pages = {"dashboard": "/dashboard", "history (unfiltered)": "/history",
             "history (filtered)": "/history?category=Gunshot&severity=Critical&conf_min=80",
             "timeline": "/timeline", "admin dashboard": "/admin/", "stats API (30 days)": "/api/stats?days=30",
             "events.csv export": "/admin/export/events.csv"}
    page_t = {}
    for name, url in pages.items():
        t0 = time.perf_counter(); r = client.get(url); dt = time.perf_counter() - t0
        page_t[name] = {"status": r.status_code, "seconds": round(dt, 3)}
        print(f"  {name:<24} {r.status_code}  {dt:.2f} s")
    results["scalability"] = {"records": total, "insert_seconds": round(insert_s, 1), "pages": page_t,
                              "pass": total >= 20000 and all(v["status"] == 200 and v["seconds"] < 5 for v in page_t.values())}

    # ---- concurrency: several users hitting pages + one live window each ----------
    errors, durs = [], []
    def worker(k):
        c = app.test_client()
        tok = login(c)
        for url in ("/dashboard", "/history", "/api/stats?days=7"):
            t0 = time.perf_counter(); r = c.get(url); durs.append(time.perf_counter() - t0)
            if r.status_code != 200:
                errors.append((url, r.status_code))
        s = c.post("/api/live/start", headers={"X-CSRFToken": tok}).get_json()["session"]
        t0 = time.perf_counter()
        r = c.post("/api/live/window", data={"session": s, "audio": (wav_bytes(test_clip(2.0, 900 + k)), "w.wav")},
                   headers={"X-CSRFToken": tok}, content_type="multipart/form-data")
        durs.append(time.perf_counter() - t0)
        if r.status_code != 200:
            errors.append(("live", r.status_code))
    threads = [threading.Thread(target=worker, args=(k,)) for k in range(args.users)]
    t0 = time.perf_counter(); [t.start() for t in threads]; [t.join() for t in threads]
    results["concurrency"] = {"users": args.users, "requests": len(durs), "errors": errors,
                              "max_request_s": round(max(durs), 2) if durs else None,
                              "wall_s": round(time.perf_counter() - t0, 2), "pass": not errors}
    print(f"  {args.users} concurrent users: {len(durs)} requests, errors={errors}")

    # ---- report ----------------------------------------------------------------------
    import platform
    results["machine"] = {"os": platform.platform(), "python": platform.python_version(), "cpu_count": os.cpu_count()}
    out = ROOT / "reports"; out.mkdir(exist_ok=True)
    (out / "performance.json").write_text(json.dumps(results, indent=2))
    ok = lambda b: "✅ pass" if b else "❌ fail"
    md = ["# Performance & scalability results", "",
          f"Machine: {results['machine']['os']} · Python {results['machine']['python']} · {os.cpu_count()} CPU threads", "",
          f"Python model: `{results['python_model']}` · test audio: recordings from the unseen TEST split, looped to 30 s / cut to 2 s", "",
          "| Requirement | Measured | Target | Result |", "|---|---|---|---|",
          f"| 30-s upload analysed (Python model, images, rules, DB) | mean {results['upload_30s']['mean_s']} s · max {results['upload_30s']['max_s']} s | ≤ 8 s | {ok(results['upload_30s']['pass'])} |",
          f"| Live 2-s window → prediction (server side) | mean {results['live_window_2s']['mean_s']} s · max {results['live_window_2s']['max_s']} s | ≤ 3 s | {ok(results['live_window_2s']['pass'])} |",
          f"| Event records stored and queryable | {results['scalability']['records']:,} | ≥ 20,000 | {ok(results['scalability']['pass'])} |",
          f"| {args.users} concurrent users (pages + live window) | max request {results['concurrency']['max_request_s']} s, errors {len(errors)} | no errors | {ok(results['concurrency']['pass'])} |",
          "", "## Page times with the full record set", "", "| Page | HTTP | Seconds |", "|---|---|---|"]
    md += [f"| {k} | {v['status']} | {v['seconds']} |" for k, v in page_t.items()]
    md += ["", "GTM inference runs in the browser and is not included; the live page shows its latency per window."]
    (out / "performance.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))
    shutil.rmtree(TMP, ignore_errors=True)


if __name__ == "__main__":
    main()
