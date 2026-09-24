"""User-facing pages: dashboard, upload, batch, live, event detail, history, timeline, media."""
import base64
from datetime import datetime, timedelta
from pathlib import Path

from flask import (Blueprint, render_template, request, redirect, url_for, flash, abort, send_file,
                   send_from_directory, Response)
from flask_login import login_required, current_user
from sqlalchemy import or_
from werkzeug.utils import secure_filename

from config.settings import UPLOAD_DIR, GTM_MODEL_DIR, CLASSES, SEVERITY_LEVELS, QUALITY_LEVELS, EVENT_STATUSES
from ..extensions import db
from ..models import AudioEvent, Alert, Segment, User, now
from ..security import can_view_event, roles_required
from ..services import audit
from ..services.analysis import analyze_upload, gtm_info, make_images, new_audio_id
from ..services.settings_service import get_settings
from python_models.inference import top_k

bp = Blueprint("main", __name__)


def visible_events():
    q = AudioEvent.query
    if current_user.role == "user":
        q = q.filter(AudioEvent.user_id == current_user.id)
    return q


def get_event_or_404(audio_id) -> AudioEvent:
    ev = AudioEvent.query.filter_by(audio_id=audio_id).first_or_404()
    if not can_view_event(current_user, ev):
        abort(403)
    return ev


@bp.route("/")
def index():
    return redirect(url_for("main.dashboard") if current_user.is_authenticated else url_for("auth.login"))


@bp.route("/dashboard")
@login_required
def dashboard():
    q = visible_events()
    recent_uploads = q.filter(AudioEvent.source != "live").order_by(AudioEvent.uploaded_at.desc()).limit(8).all()
    current = q.filter(AudioEvent.final_category.isnot(None)).order_by(AudioEvent.uploaded_at.desc()).limit(8).all()
    critical = q.filter(AudioEvent.severity.in_(["High", "Critical"]),
                        AudioEvent.alert_status == "Alert Generated").order_by(AudioEvent.uploaded_at.desc()).limit(8).all()
    quality = q.filter(AudioEvent.quality_label.in_(["Poor", "Unusable"])).order_by(AudioEvent.uploaded_at.desc()).limit(8).all()
    review = q.filter(AudioEvent.status == "Manual Review").order_by(AudioEvent.uploaded_at.desc()).limit(8).all()
    today = datetime.combine(now().date(), datetime.min.time())
    stats = {
        "total": q.count(),
        "today": q.filter(AudioEvent.uploaded_at >= today).count(),
        "alerts": q.filter(AudioEvent.alert_status == "Alert Generated").count(),
        "review": q.filter(AudioEvent.status == "Manual Review").count(),
    }
    return render_template("dashboard.html", recent_uploads=recent_uploads, current=current, critical=critical,
                           quality=quality, review=review, stats=stats, gtm=gtm_info())


@bp.route("/upload", methods=["GET", "POST"])
@login_required
def upload():
    if request.method == "POST":
        f = request.files.get("audio")
        if not f or not f.filename:
            flash("Choose an audio file first.", "error")
            return redirect(url_for("main.upload"))
        ev, msgs, dup = save_and_analyze(f, "upload")
        if dup:
            flash(f"This exact file was already analysed as {dup.audio_id}. Showing the existing result.", "warning")
            return redirect(url_for("main.event_detail", audio_id=dup.audio_id))
        if not ev:
            for m in msgs:
                flash(m, "error")
            return redirect(url_for("main.upload"))
        for m in msgs:
            flash(m, "warning")
        return redirect(url_for("main.event_detail", audio_id=ev.audio_id))
    return render_template("upload.html", gtm=gtm_info(), settings=get_settings())


def save_and_analyze(file_storage, source, actual_class=None):
    name = secure_filename(file_storage.filename) or "audio"
    ext = Path(name).suffix.lower()
    dest = UPLOAD_DIR / f"{new_audio_id('UPL')}{ext}"
    file_storage.save(dest)
    ev, msgs, dup = analyze_upload(dest, file_storage.filename, current_user, source=source, actual_class=actual_class)
    if not ev:                      # rejected or duplicate -> do not keep the file
        try:
            dest.unlink()
        except OSError:
            pass
    return ev, msgs, dup


@bp.route("/batch")
@login_required
@roles_required("reviewer", "security", "maintenance")
def batch():
    return render_template("batch.html", gtm=gtm_info())


@bp.route("/live")
@login_required
def live():
    return render_template("live.html", gtm=gtm_info(), settings=get_settings())


@bp.route("/events/<audio_id>")
@login_required
def event_detail(audio_id):
    ev = get_event_or_404(audio_id)
    if ev.stored_path and (not ev.waveform_path or not Path(ev.waveform_path).exists()):
        try:
            make_images(ev)
            db.session.commit()
        except Exception:
            db.session.rollback()
    similar = None
    if ev.near_duplicate_of_id:
        similar = db.session.get(AudioEvent, ev.near_duplicate_of_id)
    return render_template("events/detail.html", ev=ev, gtm=gtm_info(), similar=similar, top_k=top_k,
                           settings=get_settings())


@bp.route("/events/<audio_id>/report")
@login_required
def event_report(audio_id):
    ev = get_event_or_404(audio_id)
    if ev.stored_path and (not ev.waveform_path or not Path(ev.waveform_path).exists()):
        make_images(ev); db.session.commit()

    def b64(p):
        try:
            return base64.b64encode(Path(p).read_bytes()).decode() if p and Path(p).exists() else None
        except OSError:
            return None
    try:
        html = render_template("events/report.html", ev=ev, wave=b64(ev.waveform_path), spec=b64(ev.spectrogram_path),
                               top_k=top_k, generated=now())
    except Exception as exc:
        audit.notify("report_failure", f"Report failed for {ev.audio_id}: {exc}")
        flash("The report could not be generated. The error has been logged.", "error")
        return redirect(url_for("main.event_detail", audio_id=audio_id))
    audit.log("export", "report", ev.audio_id, "analysis report downloaded")
    return Response(html, mimetype="text/html",
                    headers={"Content-Disposition": f"attachment; filename=SonicSentinel_{ev.audio_id}.html"})


# ---------------------------------------------------------------------------
# Protected media (audio is never served from /static)
# ---------------------------------------------------------------------------
@bp.route("/lab")
@login_required
def lab():
    from ..services import lab as lab_service
    from ..services.analysis import gtm_info
    return render_template("lab.html", has_test_split=lab_service.test_split_available(), gtm=gtm_info(),
                           CLASSES=CLASSES)


@bp.route("/media/event/<audio_id>/<kind>")
@login_required
def event_media(audio_id, kind):
    ev = get_event_or_404(audio_id)
    if kind == "explain":
        from ..services.explain import image_path
        path = image_path(ev)
    else:
        path = {"audio": ev.stored_path, "wave": ev.waveform_path, "spec": ev.spectrogram_path}.get(kind)
    if not path or not Path(path).exists():
        abort(404)
    return send_file(path, conditional=True)


@bp.route("/media/segment/<int:seg_id>")
@login_required
def segment_media(seg_id):
    seg = db.session.get(Segment, seg_id) or abort(404)
    if not can_view_event(current_user, seg.event):
        abort(403)
    if not seg.audio_path or not Path(seg.audio_path).exists():
        abort(404)
    return send_file(seg.audio_path, mimetype="audio/wav", conditional=True)


@bp.route("/gtm/<path:filename>")
@login_required
def gtm_file(filename):
    return send_from_directory(GTM_MODEL_DIR, filename)


# ---------------------------------------------------------------------------
# History / search / timeline
# ---------------------------------------------------------------------------
def filtered_events(args):
    q = visible_events()
    if args.get("audio_id"):
        q = q.filter(AudioEvent.audio_id.contains(args["audio_id"].strip()))
    if args.get("filename"):
        q = q.filter(AudioEvent.original_filename.contains(args["filename"].strip()))
    if args.get("category"):
        c = args["category"]
        q = q.filter(or_(AudioEvent.final_category == c, AudioEvent.reviewed_category == c))
    if args.get("severity"):
        q = q.filter(AudioEvent.severity == args["severity"])
    if args.get("quality"):
        q = q.filter(AudioEvent.quality_label == args["quality"])
    if args.get("status"):
        q = q.filter(AudioEvent.status == args["status"])
    if args.get("source"):
        q = q.filter(AudioEvent.source == args["source"])
    if args.get("review"):
        if args["review"] == "reviewed":
            q = q.filter(AudioEvent.reviewer_id.isnot(None))
        elif args["review"] == "pending":
            q = q.filter(AudioEvent.status == "Manual Review")
        elif args["review"] == "none":
            q = q.filter(AudioEvent.manual_review_required.is_(False))
    if args.get("user") and current_user.role != "user":
        q = q.join(User, AudioEvent.user_id == User.id).filter(
            or_(User.username.contains(args["user"]), User.user_code == args["user"]))
    try:
        if args.get("date_from"):
            q = q.filter(AudioEvent.uploaded_at >= datetime.fromisoformat(args["date_from"]))
        if args.get("date_to"):
            q = q.filter(AudioEvent.uploaded_at < datetime.fromisoformat(args["date_to"]) + timedelta(days=1))
        if args.get("conf_min"):
            q = q.filter(AudioEvent.final_confidence >= float(args["conf_min"]) / 100)
        if args.get("conf_max"):
            q = q.filter(AudioEvent.final_confidence <= float(args["conf_max"]) / 100)
    except ValueError:
        flash("Some filter values were invalid and were ignored.", "warning")
    if not args.get("include_background"):
        q = q.filter(or_(AudioEvent.source != "live", AudioEvent.final_category != "Background Noise"))
    return q


@bp.route("/history")
@login_required
def history():
    page = max(1, request.args.get("page", 1, type=int))
    q = filtered_events(request.args).order_by(AudioEvent.uploaded_at.desc())
    pag = q.paginate(page=page, per_page=25, error_out=False)
    return render_template("history.html", pag=pag, args=request.args, QUALITY_LEVELS=QUALITY_LEVELS,
                           EVENT_STATUSES=EVENT_STATUSES)


@bp.route("/timeline")
@login_required
def timeline():
    days = request.args.get("days", 7, type=int)
    since = now() - timedelta(days=days)
    evs = visible_events().filter(AudioEvent.uploaded_at >= since,
                                  AudioEvent.final_category.isnot(None),
                                  AudioEvent.final_category != "Background Noise") \
        .order_by(AudioEvent.uploaded_at.desc()).limit(500).all()
    groups = {}
    for e in evs:
        groups.setdefault(e.uploaded_at.date(), []).append(e)
    return render_template("timeline.html", groups=groups, days=days)
