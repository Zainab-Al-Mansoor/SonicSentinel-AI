"""Explainable-AI panel and Robustness Lab."""
import numpy as np

from conftest import class_signal, login, SR
from config.settings import CLASSES


def _upload(client, token, wav_file, ci=CLASSES.index("Gunshot"), name="gun.wav"):
    p = wav_file(class_signal(ci, 4.0, seed=3), name)
    with open(p, "rb") as f:
        r = client.post("/api/upload", data={"audio": (f, name)}, headers={"X-CSRFToken": token},
                        content_type="multipart/form-data")
    assert r.status_code == 200, r.get_json()
    return r.get_json()["event"]["audio_id"]


def test_explain_vector_contributions(app):
    from src.services.analysis import python_model
    from src.services.explain import explain_vector, GROUPS
    from audio_preprocessing import preprocess_signal, segment
    from feature_extraction import extract_features
    clean, _ = preprocess_signal(class_signal(CLASSES.index("Vehicle Horn"), 2.5), SR)
    x = extract_features(segment(clean, 22050, 2.0, 1.0)[0][2])
    with app.app_context():
        r = explain_vector(python_model(), x)
    assert r["target"] in CLASSES and 0 <= r["confidence"] <= 1
    assert len(r["groups"]) == len(GROUPS)
    assert sum(g["n_features"] for g in r["groups"]) == 299          # every feature belongs to exactly one group
    assert all(g["level"] in ("high", "low") for g in r["groups"])


def test_explain_api_and_image(client, wav_file):
    token, _ = login(client)
    aid = _upload(client, token, wav_file)
    r = client.get(f"/api/events/{aid}/explain")
    assert r.status_code == 200, r.get_json()
    x = r.get_json()
    assert x["target"] in CLASSES and x["summary"] and x["groups"]
    assert x["span"]["start"] < x["span"]["end"]
    assert len(x["segments"]) >= 1
    img = client.get(x["image_url"])
    assert img.status_code == 200 and img.mimetype == "image/png"
    assert client.get(f"/api/events/{aid}/explain").get_json()["cache_key"] == x["cache_key"]   # cached
    page = client.get(f"/events/{aid}")
    assert b"Why this prediction?" in page.data


def test_explain_requires_login(client):
    assert client.get("/api/events/AUD-000001/explain").status_code in (302, 401)


def test_lab_page_and_nav(client):
    login(client)
    r = client.get("/lab")
    assert r.status_code == 200 and b"Robustness Lab" in r.data


def test_lab_upload_run_sweep_and_audio(client, wav_file):
    token, _ = login(client)
    p = wav_file(class_signal(CLASSES.index("Alarm or Siren"), 3.0, seed=1), "siren.wav")
    with open(p, "rb") as f:
        r = client.post("/api/lab/load", data={"audio": (f, "siren.wav")}, headers={"X-CSRFToken": token},
                        content_type="multipart/form-data")
    assert r.status_code == 200, r.get_json()
    lab_token = r.get_json()["token"]

    clean = client.post("/api/lab/run", json={"token": lab_token, "params": {}}, headers={"X-CSRFToken": token}).get_json()
    assert clean["prediction"] in CLASSES and clean["audio_url"]
    worst = {"noise_snr": 0, "bg_snr": 0, "echo_rt60": 0.8, "distance_m": 30, "gain_db": -30, "device": True, "cut_start_pct": 40}
    bad = client.post("/api/lab/run", json={"token": lab_token, "params": worst}, headers={"X-CSRFToken": token}).get_json()
    assert bad["params"]["device"] is True and bad["duration"] < clean["duration"]
    assert client.get(bad["audio_url"]).status_code == 200

    sw = client.post("/api/lab/sweep", json={"token": lab_token, "params": {}}, headers={"X-CSRFToken": token}).get_json()
    assert len(sw["points"]) == 7 and sw["points"][0]["label"] == "clean"


def test_lab_rejects_bad_input(client, wav_file, tmp_path):
    token, _ = login(client)
    r = client.post("/api/lab/run", json={"token": "0123456789abcdef", "params": {}}, headers={"X-CSRFToken": token})
    assert r.status_code == 404
    bad = tmp_path / "x.wav"; bad.write_bytes(b"not audio")
    with open(bad, "rb") as f:
        r = client.post("/api/lab/load", data={"audio": (f, "x.wav")}, headers={"X-CSRFToken": token},
                        content_type="multipart/form-data")
    assert r.status_code == 422
    assert client.get("/api/lab/audio/..%2Fsecret.wav").status_code == 404
    # CSRF is required
    assert client.post("/api/lab/run", json={"token": "0123456789abcdef"}).status_code in (400, 403)


def test_lab_clip_belongs_to_its_user(client, wav_file):
    token, _ = login(client, "evaluator", "Eval@12345")
    p = wav_file(class_signal(CLASSES.index("Gunshot"), 2.0), "g.wav")
    with open(p, "rb") as f:
        lab_token = client.post("/api/lab/load", data={"audio": (f, "g.wav")}, headers={"X-CSRFToken": token},
                                content_type="multipart/form-data").get_json()["token"]
    client.get("/logout")
    token2, _ = login(client, "reviewer", "Review@12345")
    r = client.post("/api/lab/run", json={"token": lab_token, "params": {}}, headers={"X-CSRFToken": token2})
    assert r.status_code == 404


def test_degrade_is_deterministic_and_bounded():
    from src.services.lab import degrade, clean_params
    y = (0.3 * np.sin(np.arange(44100 * 2) / 10)).astype(np.float32)
    p = clean_params({"noise_snr": 5, "echo_rt60": 0.5, "device": True, "gain_db": 99})
    assert p["gain_db"] == 12.0                                        # clamped to the allowed range
    a, b = degrade(y, p, seed=7), degrade(y, p, seed=7)
    assert np.array_equal(a, b) and np.abs(a).max() <= 1.0
