"""Integration, security and database tests through the Flask test client."""
import io
import json

import numpy as np
import soundfile as sf

from config.settings import CLASSES
from conftest import tone, class_signal, csrf, login, SR


def wav_bytes(signal, sr=SR):
    buf = io.BytesIO()
    sf.write(buf, signal, sr, format="WAV", subtype="PCM_16")
    buf.seek(0)
    return buf


def upload(client, token, signal, name="clip.wav", api=True):
    url = "/api/upload" if api else "/upload"
    return client.post(url, data={"audio": (wav_bytes(signal), name), "csrf_token": token},
                       headers={"X-CSRFToken": token}, content_type="multipart/form-data")


def fake_gtm(n_segments, cls, conf=0.9):
    """GTM runs in the browser; in tests we post scores the way the browser would."""
    rest = (1 - conf) / (len(CLASSES) - 1)
    s = {c: rest for c in CLASSES}
    s[cls] = conf
    return {"segments": [s] * n_segments}


# ---------------- security ----------------
def test_login_required(client):
    assert client.get("/dashboard").status_code == 302
    assert client.get("/api/stats").status_code in (302, 401)


def test_csrf_enforced(client):
    r = client.post("/login", data={"username": "admin", "password": "Admin@12345"})
    assert r.status_code == 400


def test_wrong_password_and_lockout(client, app):
    token = csrf(client)
    for _ in range(5):
        client.post("/login", data={"username": "evaluator", "password": "wrong", "csrf_token": token})
    r = client.post("/login", data={"username": "evaluator", "password": "Eval@12345", "csrf_token": token},
                    follow_redirects=True)
    assert b"locked" in r.data
    with app.app_context():
        from src.models import User, SystemNotification
        from src.extensions import db
        u = User.query.filter_by(username="evaluator").first()
        u.failed_logins, u.locked_until = 0, None
        db.session.commit()
        assert SystemNotification.query.filter_by(kind="failed_logins").count() >= 1


def test_register_and_user_id(client):
    token = csrf(client)
    r = client.post("/register", data={"username": "newuser", "email": "n@x.com", "password": "abc12345",
                                       "confirm": "abc12345", "role": "user", "csrf_token": token}, follow_redirects=True)
    assert b"USR-" in r.data
    r = client.post("/register", data={"username": "weak", "email": "w@x.com", "password": "short",
                                       "confirm": "short", "role": "admin", "csrf_token": token}, follow_redirects=True)
    assert b"Invalid role" in r.data and b"Password needs" in r.data


def test_role_based_access(client):
    login(client, "evaluator", "Eval@12345")
    assert client.get("/admin/").status_code == 403
    assert client.get("/review/").status_code == 403
    assert client.get("/alerts/").status_code == 403
    assert client.get("/batch").status_code == 403
    assert client.get("/upload").status_code == 200


# ---------------- upload -> Python -> GTM -> decision ----------------
def test_upload_full_flow_with_gtm(client, app):
    token, _ = login(client)
    sig = class_signal(CLASSES.index("Glass Breaking"), 3.0, seed=11)
    r = upload(client, token, sig, "glass.wav")
    assert r.status_code == 200, r.data
    ev = r.get_json()["event"]
    assert ev["gtm_status"] == "unavailable"          # no GTM model installed in tests
    assert ev["python_prediction"] is None or ev["final_category"]  # finalised without GTM
    # simulate a GTM-enabled run on a fresh upload
    with app.app_context():
        from src.models import AudioEvent
        from src.extensions import db
        e = AudioEvent.query.filter_by(audio_id=ev["audio_id"]).first()
        assert e.python_prediction == "Glass Breaking"
        assert e.consistency_status == "Uncertain Result" and len(e.segments) >= 2
        e.gtm_status, e.final_category = "pending", None
        db.session.commit()
        n = len(e.segments)
    r = client.post(f"/api/events/{ev['audio_id']}/gtm", json=fake_gtm(n, "Glass Breaking"),
                    headers={"X-CSRFToken": token})
    d = r.get_json()
    assert d["gtm_prediction"] == "Glass Breaking" and d["consistency_status"] in ("Acceptable Match", "Weak Match")
    assert d["final_category"] == "Glass Breaking" and d["severity"] in ("High", "Critical")
    page = client.get(d["url"])
    assert page.status_code == 200 and b"Model prediction" in page.data
    assert client.get(f"/media/event/{ev['audio_id']}/wave").status_code == 200
    assert client.get(f"/media/event/{ev['audio_id']}/spec").status_code == 200
    rep = client.get(f"/events/{ev['audio_id']}/report")
    assert rep.status_code == 200 and b"data:image/png;base64" in rep.data


def test_model_disagreement_goes_to_review(client, app):
    token, _ = login(client)
    r = upload(client, token, class_signal(CLASSES.index("Vehicle Horn"), 2.0, seed=21), "horn.wav")
    aid = r.get_json()["event"]["audio_id"]
    with app.app_context():
        from src.models import AudioEvent
        from src.extensions import db
        e = AudioEvent.query.filter_by(audio_id=aid).first()
        e.gtm_status, e.final_category = "pending", None
        db.session.commit()
        n = len(e.segments)
    d = client.post(f"/api/events/{aid}/gtm", json=fake_gtm(n, "Animal Sound"), headers={"X-CSRFToken": token}).get_json()
    assert d["consistency_status"] == "Model Disagreement" and d["status"] == "Manual Review"
    # reviewer corrects it; original model outputs preserved
    login(client, "reviewer", "Review@12345")
    token = csrf(client)
    client.post(f"/review/{aid}", data={"category": "Vehicle Horn", "severity": "Low", "comment": "clear horn",
                                        "csrf_token": token})
    with app.app_context():
        from src.models import AudioEvent, Review
        e = AudioEvent.query.filter_by(audio_id=aid).first()
        assert e.reviewed_category == "Vehicle Horn" and e.status == "Reviewed"
        assert e.gtm_prediction == "Animal Sound"            # preserved
        assert Review.query.filter_by(event_id=e.id).first().original_category == e.final_category


def test_invalid_and_silent_uploads(client):
    token, _ = login(client)
    r = client.post("/api/upload", data={"audio": (io.BytesIO(b"not audio"), "x.txt")},
                    headers={"X-CSRFToken": token}, content_type="multipart/form-data")
    assert r.status_code == 422 and "Unsupported" in r.get_json()["error"]
    r = upload(client, token, np.zeros(SR * 2, np.float32), "silent.wav")
    assert r.status_code == 422 and "silent" in r.get_json()["error"]
    r = client.post("/api/upload", data={"audio": (io.BytesIO(b"RIFFxxxxWAVEfmt garbage"), "broken.wav")},
                    headers={"X-CSRFToken": token}, content_type="multipart/form-data")
    assert r.status_code == 422


def test_duplicate_upload(client):
    token, _ = login(client)
    sig = tone(1234, 2.0, seed=99)
    first = upload(client, token, sig, "a.wav").get_json()
    second = upload(client, token, sig, "copy_of_a.wav").get_json()
    assert second["duplicate"] and second["event"]["audio_id"] == first["event"]["audio_id"]


def test_low_quality_upload_flagged(client):
    token, _ = login(client)
    sig = np.clip(tone(700, 2.0, amp=4.0, seed=5), -1, 1)
    ev = upload(client, token, sig, "clipped.wav").get_json()["event"]
    assert ev["quality"] in ("Poor", "Acceptable")


# ---------------- live monitoring ----------------
def test_live_window_flow(client):
    token, _ = login(client, "security", "Secure@12345")
    sess = client.post("/api/live/start", headers={"X-CSRFToken": token}).get_json()["session"]
    gtm = fake_gtm(1, "Panic Scream", 0.95)["segments"][0]
    sig = class_signal(CLASSES.index("Panic Scream"), 2.0, seed=31)
    r = client.post("/api/live/window", data={"session": sess, "gtm": json.dumps(gtm),
                                              "audio": (wav_bytes(sig), "w.wav")},
                    headers={"X-CSRFToken": token}, content_type="multipart/form-data")
    d = r.get_json()
    assert r.status_code == 200 and d["final_category"] == "Panic Scream"
    assert d["alert_status"] == "Alert Generated" and d["alert"]["severity"] == "Critical"
    # silent window is not classified
    r = client.post("/api/live/window", data={"session": sess, "audio": (wav_bytes(np.zeros(SR * 2, np.float32)), "w.wav")},
                    headers={"X-CSRFToken": token}, content_type="multipart/form-data")
    assert r.get_json()["silent"]
    # acknowledge the alert
    aid = d["alert"]["id"]
    r = client.post(f"/alerts/{aid}/acknowledge", headers={"X-CSRFToken": token, "X-Requested-With": "fetch"})
    assert r.get_json()["status"] == "Acknowledged"
    client.post("/api/live/stop", json={"session": sess}, headers={"X-CSRFToken": token})
    assert client.get("/api/live/recent").status_code == 200


def test_repeated_live_detection_confirms_gunshot(client):
    token, _ = login(client, "security", "Secure@12345")
    sess = client.post("/api/live/start", headers={"X-CSRFToken": token}).get_json()["session"]
    gtm = fake_gtm(1, "Gunshot", 0.75)["segments"][0]
    sig = class_signal(CLASSES.index("Gunshot"), 2.0, seed=41)
    results = []
    for _ in range(2):
        results.append(client.post("/api/live/window", data={"session": sess, "gtm": json.dumps(gtm),
                                                            "audio": (wav_bytes(sig), "w.wav")},
                                   headers={"X-CSRFToken": token}, content_type="multipart/form-data").get_json())
    assert results[1]["repeated_count"] >= 2


# ---------------- pages, search, admin, export, database ----------------
def test_all_pages_render(client):
    login(client)
    for url in ["/dashboard", "/upload", "/batch", "/live", "/history", "/timeline", "/alerts/", "/alerts/history",
                "/review/", "/admin/", "/admin/users", "/admin/settings", "/admin/rules", "/admin/audit",
                "/admin/notifications", "/admin/models", "/admin/evaluation", "/profile",
                "/history?category=Gunshot&severity=Critical&conf_min=10&date_from=2020-01-01&review=pending"]:
        r = client.get(url)
        assert r.status_code == 200, url


def test_exports(client):
    login(client)
    for name in ["events", "alerts", "reviews", "audit", "comparison"]:
        r = client.get(f"/admin/export/{name}.csv")
        assert r.status_code == 200 and r.data
    assert client.get("/admin/export/events.xlsx").status_code == 200


def test_admin_settings_and_rules(client, app):
    token, _ = login(client)
    client.post("/admin/settings", data={"min_confidence": "0.7", "segment_seconds": "2", "segment_hop_seconds": "1",
                                         "live_window_seconds": "2", "top2_margin": "0.15", "acceptable_match_diff": "0.2",
                                         "unknown_threshold": "0.35", "overlap_threshold": "0.25", "repeat_window_seconds": "10",
                                         "noise_reduction": "on", "background_noise_limit_db": "-20",
                                         "audio_retention_days": "90", "record_retention_days": "365", "csrf_token": token})
    with app.app_context():
        from src.services.settings_service import get_settings
        assert get_settings()["min_confidence"] == 0.7
    client.post("/admin/rules", data={"action": "add", "category": "Drone Sound", "csrf_token": token})
    from src.services.rules import load_rules
    assert "Drone Sound" in load_rules()
    r = client.post("/admin/retention", data={"csrf_token": token}, follow_redirects=True)
    assert b"Retention applied" in r.data


def test_audit_trail_records_actions(app):
    with app.app_context():
        from src.models import AuditLog
        actions = {a.action for a in AuditLog.query.all()}
        assert {"login", "upload", "prediction"} <= actions


def test_database_tables(app):
    with app.app_context():
        from src.extensions import db
        tables = set(db.inspect(db.engine).get_table_names())
        assert {"users", "audio_events", "segments", "alerts", "reviews", "audit_logs", "model_versions",
                "settings", "live_sessions", "system_notifications"} <= tables
