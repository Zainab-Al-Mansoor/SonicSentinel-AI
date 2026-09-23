"""JSON API used by the browser (batch upload, GTM results, live monitoring, charts, evaluation)."""
from collections import Counter
from datetime import timedelta
from pathlib import Path

import pandas as pd
from flask import Blueprint, request, jsonify, url_for, abort
from flask_login import login_required, current_user

from config.settings import BASE_DIR, DATASET_METADATA_CSV, CLASSES
from ..extensions import db
from ..models import AudioEvent, LiveSession, Alert, now
from ..security import can_view_event, roles_required
from ..services import audit
from ..services.analysis import submit_gtm, analyze_live_window, analyze_upload, gtm_info
from .main import save_and_analyze, visible_events

bp = Blueprint("api", __name__, url_prefix="/api")


def event_json(ev: AudioEvent, include_segments=False) -> dict:
    d = {
        "audio_id": ev.audio_id, "url": url_for("main.event_detail", audio_id=ev.audio_id),
        "filename": ev.original_filename, "source": ev.source, "status": ev.status,
        "uploaded_at": ev.uploaded_at.isoformat(timespec="seconds") if ev.uploaded_at else None,
        "duration": ev.duration, "quality": ev.quality_label, "gtm_status": ev.gtm_status,
        "python_prediction": ev.python_prediction, "python_confidence": ev.python_confidence,
        "gtm_prediction": ev.gtm_prediction, "gtm_confidence": ev.gtm_confidence,
        "python_scores": ev.python_scores, "gtm_scores": ev.gtm_scores,
        "consistency_status": ev.consistency_status, "confidence_diff": ev.confidence_diff,
        "final_category": ev.final_category, "final_confidence": ev.final_confidence,
        "confidence_level": ev.confidence_level, "severity": ev.severity, "alert_status": ev.alert_status,
        "recommended_action": ev.recommended_action, "manual_review": ev.manual_review_required,
        "review_reasons": ev.review_reasons, "repeated_count": ev.repeated_count,
        "overlap": ev.overlap_detected, "overlap_classes": ev.overlap_classes,
        "trace": ev.decision_trace, "actual_class": ev.actual_class, "processing_ms": ev.processing_ms,
    }
    if include_segments:
        d["segments"] = [{"id": s.id, "index": s.index, "start": s.start_s, "end": s.end_s,
                          "url": url_for("main.segment_media", seg_id=s.id)} for s in ev.segments]
    return d


def _event(audio_id) -> AudioEvent:
    ev = AudioEvent.query.filter_by(audio_id=audio_id).first_or_404()
    if not can_view_event(current_user, ev):
        abort(403)
    return ev


@bp.route("/upload", methods=["POST"])
@login_required
def api_upload():
    f = request.files.get("audio")
    if not f or not f.filename:
        return jsonify({"error": "No file received."}), 400
    source = "batch" if current_user.role != "user" else "upload"
    ev, msgs, dup = save_and_analyze(f, source)
    if dup:
        return jsonify({"duplicate": True, "event": event_json(dup, True),
                        "message": f"Identical to existing record {dup.audio_id}"})
    if not ev:
        return jsonify({"error": " ".join(msgs)}), 422
    return jsonify({"event": event_json(ev, include_segments=True), "warnings": msgs})


@bp.route("/events/<audio_id>")
@login_required
def api_event(audio_id):
    return jsonify(event_json(_event(audio_id), include_segments=True))


@bp.route("/events/<audio_id>/gtm", methods=["POST"])
@login_required
def api_event_gtm(audio_id):
    """Browser posts GTM scores for every segment: {"segments": [{class: score}...]} or {"error": "..."}."""
    ev = _event(audio_id)
    if ev.gtm_status == "done" and ev.final_category:
        return jsonify(event_json(ev))
    data = request.get_json(silent=True) or {}
    seg_scores = data.get("segments")
    if seg_scores is not None and len(seg_scores) != len(ev.segments):
        return jsonify({"error": f"Expected {len(ev.segments)} segment results, got {len(seg_scores)}."}), 400
    ev = submit_gtm(ev, seg_scores, data.get("error"))
    return jsonify(event_json(ev))


# ---------------------------------------------------------------------------
# Live monitoring
# ---------------------------------------------------------------------------
@bp.route("/live/start", methods=["POST"])
@login_required
def live_start():
    import secrets
    code = f"LS-{now():%Y%m%d-%H%M%S}-{current_user.id}-{secrets.token_hex(2)}"
    s = LiveSession(session_code=code, user_id=current_user.id)
    db.session.add(s)
    db.session.commit()
    audit.log("mic_start", "live_session", code)
    return jsonify({"session": code})


@bp.route("/live/stop", methods=["POST"])
@login_required
def live_stop():
    code = (request.get_json(silent=True) or {}).get("session")
    s = LiveSession.query.filter_by(session_code=code, user_id=current_user.id).first()
    if s and not s.ended_at:
        s.ended_at = now()
        db.session.commit()
        audit.log("mic_stop", "live_session", code, f"{s.windows} windows analysed")
    return jsonify({"ok": True})


@bp.route("/live/window", methods=["POST"])
@login_required
def live_window():
    import json
    code = request.form.get("session")
    s = LiveSession.query.filter_by(session_code=code, user_id=current_user.id).first()
    if not s or s.ended_at:
        return jsonify({"error": "Live session not found or already stopped."}), 400
    f = request.files.get("audio")
    if not f:
        return jsonify({"error": "No audio window received."}), 400
    gtm_scores = json.loads(request.form["gtm"]) if request.form.get("gtm") else None
    try:
        res = analyze_live_window(f.read(), s, current_user, gtm_scores, request.form.get("gtm_error"))
    except Exception as exc:
        db.session.rollback()
        audit.notify("model_failure", f"Live analysis failed: {exc}")
        return jsonify({"error": f"Analysis failed: {exc}"}), 500
    if res["silent"]:
        return jsonify({"silent": True, "quality": res["quality"]["label"], "message": res["message"]})
    ev = res["event"]
    out = event_json(ev)
    out["alert"] = None
    if ev.alerts:
        a = ev.alerts[0]
        out["alert"] = {"id": a.id, "severity": a.severity, "message": a.message, "action": a.recommended_action}
    return jsonify(out)


@bp.route("/live/recent")
@login_required
def live_recent():
    evs = visible_events().filter(AudioEvent.source == "live", AudioEvent.final_category.isnot(None)) \
        .order_by(AudioEvent.uploaded_at.desc()).limit(15).all()
    return jsonify([event_json(e) for e in evs])


# ---------------------------------------------------------------------------
# Chart data
# ---------------------------------------------------------------------------
@bp.route("/stats")
@login_required
def stats():
    days = request.args.get("days", 14, type=int)
    since = now() - timedelta(days=days)
    evs = visible_events().filter(AudioEvent.uploaded_at >= since, AudioEvent.final_category.isnot(None)).all()
    cats = Counter(e.display_category for e in evs if not (e.source == "live" and e.final_category == "Background Noise"))
    trend = Counter(e.uploaded_at.strftime("%Y-%m-%d") for e in evs if e.final_category != "Background Noise")
    crit_trend = Counter(e.uploaded_at.strftime("%Y-%m-%d") for e in evs
                         if e.severity in ("High", "Critical") and e.alert_status == "Alert Generated")
    labels = [(since + timedelta(days=i + 1)).strftime("%Y-%m-%d") for i in range(days)]
    conf_bins = [0] * 10
    for e in evs:
        conf_bins[min(9, int((e.final_confidence or 0) * 10))] += 1
    return jsonify({
        "categories": {c: cats.get(c, 0) for c in CLASSES + ["Unknown"]},
        "trend": {"labels": labels, "all": [trend.get(d, 0) for d in labels], "critical": [crit_trend.get(d, 0) for d in labels]},
        "confidence_hist": conf_bins,
        "quality": Counter(e.quality_label for e in evs),
        "consistency": Counter(e.consistency_status for e in evs),
    })


# ---------------------------------------------------------------------------
# Test-set evaluation (Model Prediction & Confidence Comparison report)
# ---------------------------------------------------------------------------
@bp.route("/evaluate/testset")
@login_required
@roles_required("admin")
def testset_list():
    if not DATASET_METADATA_CSV.exists():
        return jsonify({"error": "data/dataset_metadata.csv not found – build the dataset first."}), 404
    m = pd.read_csv(DATASET_METADATA_CSV)
    m = m[(m["split"] == "test") & (m["is_augmented"] == 0)]
    per = request.args.get("per_class", 10, type=int)
    m = m.groupby("class_label", group_keys=False).head(per)
    return jsonify([{"audio_id": r.audio_id, "class_label": r.class_label, "path": r.path} for r in m.itertuples()])


@bp.route("/evaluate/file", methods=["POST"])
@login_required
@roles_required("admin")
def testset_file():
    import shutil
    data = request.get_json(silent=True) or {}
    m = pd.read_csv(DATASET_METADATA_CSV)
    row = m[m["audio_id"] == data.get("audio_id")]
    if row.empty or row.iloc[0]["split"] != "test":
        return jsonify({"error": "Only unseen TEST recordings can be evaluated."}), 400
    r = row.iloc[0]
    src = BASE_DIR / r["path"]
    from config.settings import UPLOAD_DIR
    dest = UPLOAD_DIR / f"EVAL-{r['audio_id']}.wav"
    shutil.copy2(src, dest)
    ev, msgs, dup = analyze_upload(dest, f"{r['audio_id']}.wav", current_user, source="evaluation",
                                   actual_class=r["class_label"])
    if dup:
        if dup.actual_class is None:
            dup.actual_class = r["class_label"]; db.session.commit()
        return jsonify({"duplicate": True, "event": event_json(dup, True)})
    if not ev:
        return jsonify({"error": " ".join(msgs)}), 422
    return jsonify({"event": event_json(ev, True)})
